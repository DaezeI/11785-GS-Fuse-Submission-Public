"""Re-decide Fed minutes/speech/testimony audit rows with the corrected release-line parser.

The original verifier (``verify_minutes_speech_versions.py``) read "For release on delivery"
lines with a single strict pattern and missed many valid lines (two time zones, "Noon",
whole hours, capital A.M.). This re-reads each saved PDF, confirms its SHA-256 equals the
audited ``pdf_sha256``, replaces only the release-line fields, and re-runs the original
``decide`` function. Archive lookups and all other evidence are kept as recorded.
No network requests are made. Inputs are not modified; each output is written as
``<input stem>.v2.json`` next to ``--out-dir``.

    python scripts/redecide_fed_document_versions.py --raw-dirs downloads/minutes-speech-evidence-v1/raw \\
        --inputs downloads/minutes-speech-evidence-v1/slices/s000.json --out-dir downloads/redecide-v2/trainval
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys

import pypdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from release_lines import parse_release_line  # noqa: E402
from verify_minutes_speech_versions import compare_words, decide, paragraphs, pdf_info, soup  # noqa: E402

CATALOG = ROOT / "analysis/macro_releases/fed_announcements_v2.csv"
# Web-page-only lines that never appear in a released PDF, beyond the verifier's own list.
EXTRA_WEB_ONLY = re.compile(r"^\s*(The \[Tab\] key may be used|.*submitted identical remarks to the)", re.I)


def recheck_wording(row, pdf_bytes, raw_dirs, catalog):
    """Recompute the PDF-versus-HTML wording check with the final verifier's rules.

    Early slices used an older check that did not skip web-only lines such as
    "Accessible Version". Uses the saved current page; returns True when recomputed.
    """
    entry = catalog.get(row["event_id"])
    if entry is None or entry["text_format"] == "pdf":
        return False
    page = next((p for d in raw_dirs for p in d.rglob(f"{row['event_id']}.current.html")), None)
    if page is None:
        return False
    box = soup(page.read_bytes()).select_one(entry["text_selector"])
    if box is None:
        return False
    for sup in box.select("sup"):
        sup.decompose()
    paras = paragraphs(box)
    kept = [p for p in paras if not EXTRA_WEB_ONLY.match(p)]
    result = compare_words(kept, pdf_info(pdf_bytes)["text"])
    result["extra_web_only_skipped"] = [p[:80] for p in paras if EXTRA_WEB_ONLY.match(p)]
    row["word_compare_v1"] = row.get("word_compare")
    row["word_compare"] = result
    row["words_match"] = result["paragraphs_not_matched"] == 0 and not result["numbers_missing_in_pdf"]
    return True

TIME_KEYS = ("cutoff_utc", "release_utc", "pdf_mod_utc", "pdf_release_line_utc", "pdf_creation_utc",
             "release_record_utc", "pre_cutoff_capture_utc")


def as_time(value):
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def find_pdf(event_id, sha, raw_dirs):
    for directory in raw_dirs:
        for path in directory.rglob(f"{event_id}*.pdf"):
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == sha:
                return data, path
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--raw-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    catalog = {r["event_id"]: r for r in csv.DictReader(CATALOG.open(encoding="utf-8"))}
    changes, missing, rechecked = Counter(), [], Counter()
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            old = row["disposition"]
            row["disposition_v1"], row["reason_v1"] = old, row.get("reason")
            data = None
            if row.get("pdf_sha256"):
                data, found = find_pdf(row["event_id"], row["pdf_sha256"], args.raw_dirs)
                if data is None:
                    missing.append(row["event_id"])
                elif row.get("words_match") is False:
                    before = row.get("words_match")
                    if recheck_wording(row, data, args.raw_dirs, catalog):
                        rechecked[f"words_match {before} -> {row['words_match']}"] += 1
            if row.get("event_type") != "fomc_minutes" and data is not None:
                if True:
                    first = pypdf.PdfReader(found).pages[0].extract_text() or ""
                    line = parse_release_line(first)
                    row["pdf_release_line"] = line["text"] if line else None
                    row["pdf_release_line_utc"] = line["utc"] if line else None
                    row["pdf_release_line_zone"] = line["zone"] if line else None
                    row["release_line_parser"] = "release_lines.py v2"
            for key in TIME_KEYS:
                if key in row:
                    row[key] = as_time(row[key])
            row["disposition"], row["reason"] = decide(row)
            # Same fingerprint rule as the verifier, applied after the new decision.
            row["fingerprint_sha256"] = (
                row.get("pdf_sha256") if row["disposition"].startswith(("eligible", "version_conflict"))
                and row.get("pdf_sha256") else (row.get("current_body_sha256")
                if row["disposition"] == "eligible_archived_copy_before_cutoff" else None))
            changes[f"{old} -> {row['disposition']}"] += 1
        (args.out_dir / f"{path.stem}.v2.json").write_text(
            json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"transitions": dict(sorted(changes.items())), "wording_rechecks": dict(rechecked),
                      "pdf_not_found": missing}, indent=1))


if __name__ == "__main__":
    main()
