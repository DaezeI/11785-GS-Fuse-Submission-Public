"""Write the verified release text for every text-eligible Fed row, checked against its fingerprint.

For each eligible row of a cohort ledger, the text is extracted from the same evidence the audit
fingerprinted, with the same functions, and the fingerprint is recomputed. A row whose fingerprint
does not match is reported and left out; nothing is guessed.

| Evidence (fingerprint source) | Text written | Fingerprint check |
|---|---|---|
| Minutes-approved FOMC statement body | `policy_body` of the statement in the saved minutes | SHA-256 of that body |
| Released PDF bytes (minutes, speeches, testimony) | `pdf_info(...)["text"]` of the saved PDF | SHA-256 of the PDF bytes |
| Timestamped House witness PDF body | the saved UTF-8 body | SHA-256 of those bytes |
| Release-day archived Board body | normalized paragraphs of the archive replay | SHA-256 of the joined paragraphs |
| Pre-cutoff archived web page (testimony) | normalized paragraphs of the saved capture | SHA-256 of the joined paragraphs |

Saved evidence is read from the git-ignored ``downloads/`` folders. Board replays are fetched from
the recorded Wayback URLs (cached under ``--cache-dir``). No price, return or label is read.

    python scripts/build_fed_spx_text_files.py --cohort FED-PXTX-ALL-REL5-v2 \\
        --ledger results/fed-spx-news-cohort-v2/eligible_cohort.csv \\
        --ledger results/fed-spx-test-text-v2/test_text_ledger.csv \\
        --out-dir downloads/fed-spx-text-v2 --manifest-dir results/fed-spx-text-files-v2
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_fomc_statement_versions import (  # noqa: E402
    minutes_statement, norm_para, paragraphs, pdf_minutes_statement, policy_body, soup, alnum,
)
from verify_minutes_speech_versions import WEB_ONLY, clean_pdf_text, compare_words, pdf_info  # noqa: E402
from redecide_fed_document_versions import EXTRA_WEB_ONLY  # noqa: E402

DL = Path(os.environ.get("GSFUSE_DOWNLOADS", ROOT / "downloads"))  # where the git-ignored evidence lives
CATALOG = ROOT / "analysis/macro_releases/fed_announcements_v2.csv"
OVERRIDES = ROOT / "analysis/macro_releases/fomc_minutes_overrides.json"
EVIDENCE_DIRS = [
    DL / "fomc-statement-evidence-pilot", DL / "minutes-speech-evidence-v1", DL / "fed-spx-test-text-v1",
    DL / "fed-spx-premarket-v1", DL / "testimony-recheck-v1", DL / "fed-spx-board-testimony-evidence-v1",
]
BOARD_SHARDS = [DL / "board-recheck-v1/shard0.jsonl", DL / "board-recheck-v1/shard1.jsonl",
                DL / "fed-spx-test-text-v1/board/shard0.jsonl", DL / "fed-spx-premarket-v1/board/shard0.jsonl"]
MANIFEST_FIELDS = ["event_id", "event_type", "split", "market_et", "text_cutoff_utc", "evidence_class",
                   "fingerprint_sha256", "fingerprint_source", "text_source", "text_sha256", "text_chars",
                   "model_text_source", "model_text_sha256", "model_text_chars", "status"]
AGENT = "CMU-11785-GS-Fuse-research (verified text build)"


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def index_files():
    files = {}
    for directory in EVIDENCE_DIRS:
        for path in directory.rglob("*"):
            if path.is_file():
                files.setdefault(path.name, []).append(path)
    return files


def statement_text(row, files, catalog, overrides):
    eid = row["event_id"]
    ov = overrides.get(eid, {})
    for page in files.get(f"{eid}.minutes.html", []):
        content = page.read_bytes()
        if ov.get("url", "").lower().endswith(".pdf"):
            current = next(iter(files.get(f"{eid}.current.html", [])), None)
            if current is None:
                continue
            box = soup(current.read_bytes()).select_one(catalog[eid]["text_selector"])
            cur_body = policy_body(paragraphs(box)) if box is not None else ""
            clock, stream = pdf_minutes_statement(content)
            acc, keep = "", []
            for para in cur_body.split("\n"):
                if acc == stream:
                    break
                acc += alnum(para)
                keep.append(para)
            block = keep if acc == stream else []
        else:
            clock, block, _ = minutes_statement(content, ov.get("which", 0))
        if clock and block:
            body = policy_body(block)
            if sha(body) == row["fingerprint_sha256"]:
                return body, f"minutes statement body ({page.parent.name}/{page.name})"
    return None, "no saved minutes reproduce the fingerprinted statement body"


def pdf_text(row, files):
    for path in [p for name, paths in files.items() if name.startswith(row["event_id"]) and name.endswith(".pdf")
                 for p in paths]:
        data = path.read_bytes()
        if sha(data) == row["fingerprint_sha256"]:
            return pdf_info(data)["text"], f"released PDF ({path.parent.name}/{path.name})"
    return None, "no saved PDF has the fingerprinted bytes"


def house_text(row, files):
    for path in files.get(f"{row['event_id']}.txt", []):
        data = path.read_bytes()
        if sha(data) == row["fingerprint_sha256"]:
            return data.decode("utf-8"), f"House witness PDF body ({path.parent.name}/{path.name})"
    return None, "no saved House body has the fingerprint"


def archived_page_text(row, files, catalog):
    """Pre-cutoff archived HTML whose normalized body equals today's body (the fingerprint)."""
    eid = row["event_id"]
    pages = sorted(p for name, paths in files.items() if name.startswith(f"{eid}.pre-") and name.endswith(".html")
                   for p in paths)
    for path in pages:
        page = soup(path.read_bytes())
        box = page.select_one(catalog[eid]["text_selector"]) or page.select_one("#leftText")
        if box is None:
            continue
        body = "\n".join(norm_para(p) for p in paragraphs(box))
        if sha(body) == row["fingerprint_sha256"]:
            return body, f"pre-cutoff archive copy ({path.parent.name}/{path.name})"
    return None, "no saved pre-cutoff archive copy reproduces the fingerprinted body"


def model_text_for_pdf(row, text, files, catalog):
    """Readable model input for a release proven by its PDF.

    Uses the saved web page only when the audit's own wording check passes against this exact PDF
    text: every page paragraph appears in the released PDF and no page number is missing from it.
    Web-only lines and footnote markers are dropped. Otherwise falls back to the PDF text with page
    headers removed and line-end hyphenation joined.
    """
    eid = row["event_id"]
    page = next(iter(files.get(f"{eid}.current.html", [])), None)
    entry = catalog.get(eid)
    if page is not None and entry and entry["text_format"] != "pdf":
        box = soup(page.read_bytes()).select_one(entry["text_selector"])
        if box is not None:
            for sup in box.select("sup"):
                sup.decompose()
            kept = [p for p in paragraphs(box) if not WEB_ONLY.match(p) and not EXTRA_WEB_ONLY.match(p)]
            check = compare_words(kept, text)
            if kept and check["paragraphs_not_matched"] == 0 and not check["numbers_missing_in_pdf"]:
                kept = [re.sub(r"\s*(Return to (text|top)|\((PDF|HTML)\))\s*", " ", p, flags=re.I) for p in kept]
                body = "\n".join(q for q in (norm_para(p) for p in kept) if q)
                return body, "web page; every paragraph found in the released PDF"
    return clean_pdf_text(text).strip(), "released PDF, headers removed and hyphenation joined"


def board_text(row, shards, catalog, cache, session):
    rec = shards.get(row["event_id"])
    if not rec or not rec.get("replay_url"):
        return None, "no recorded archive replay"
    cached = cache / f"{row['event_id']}.html"
    if not cached.exists():
        for attempt in range(3):
            try:
                response = session.get(rec["replay_url"], timeout=60)
            except requests.RequestException:
                time.sleep(10)
                continue
            if response.status_code == 429:
                time.sleep(90)
                continue
            time.sleep(2.5)
            if response.ok:
                cached.write_bytes(response.content)
            break
    if not cached.exists():
        return None, "archive replay unavailable"
    selector = rec.get("replay_selector") or catalog[row["event_id"]]["text_selector"]
    box = soup(cached.read_bytes()).select_one(selector)
    if box is None:
        return None, "archived page body not found"
    paras = [norm_para(p) for p in paragraphs(box)]
    body = "\n".join(paras)
    if sha(body) != row["fingerprint_sha256"]:
        return None, "archived body does not match the fingerprint"
    return body, f"release-day archive copy ({rec['replay_url']})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", required=True, help="cohort ID from docs/NAMING.md")
    parser.add_argument("--ledger", action="append", required=True, type=Path,
                        help="eligible_cohort.csv or a *_text_ledger.csv; repeat for several splits")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    args = parser.parse_args()
    catalog = {r["event_id"]: r for r in csv.DictReader(CATALOG.open(encoding="utf-8"))}
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    files = index_files()
    shards = {}
    for path in BOARD_SHARDS:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    shards[rec["event_id"]] = rec
    cache = args.out_dir / "board-replays"
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = AGENT

    rows = []
    for ledger in args.ledger:
        for r in csv.DictReader((ROOT / ledger if not ledger.is_absolute() else ledger).open(encoding="utf-8-sig")):
            evidence = r.get("evidence_class") or r.get("eligibility") or ""
            if evidence.startswith("eligible"):
                rows.append({**r, "evidence_class": evidence,
                             "market_et": r.get("market_et") or r.get("release_et", ""),
                             "fingerprint_source": r.get("fingerprint_source", "")})
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate event across ledgers")

    out_dir = args.out_dir / args.cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, handles, status = [], {}, Counter()
    for row in sorted(rows, key=lambda r: (r["split"], r["market_et"], r["event_id"])):
        kind, source = row["event_type"], row["fingerprint_source"]
        if kind in ("fomc_statement", "fomc_other_policy_release"):
            text, where = statement_text(row, files, catalog, overrides)
        elif row["evidence_class"] == "eligible_archived_copy_before_cutoff":
            # the verifier fingerprints the PDF when one exists, else the archived page body
            text, where = archived_page_text(row, files, catalog)
            if not text:
                text, where = (board_text(row, shards, catalog, cache, session)
                               if kind == "fed_board_monetary_release" else pdf_text(row, files))
        elif "House" in source:
            text, where = house_text(row, files)
        elif kind == "fed_board_monetary_release":
            text, where = board_text(row, shards, catalog, cache, session)
        else:
            text, where = pdf_text(row, files)
        if text and where.startswith(("released PDF", "House witness PDF")):
            model_text, model_source = model_text_for_pdf(row, text, files, catalog)
        else:
            model_text, model_source = text, "same as text"
        state = "verified" if text else "fingerprint_not_reproduced"
        status[f"{row['split']}|{state}"] += 1
        manifest.append({**{k: row.get(k, "") for k in MANIFEST_FIELDS}, "text_source": where,
                         "text_sha256": sha(text) if text else "", "text_chars": len(text) if text else 0,
                         "model_text_source": model_source if text else "",
                         "model_text_sha256": sha(model_text) if text else "",
                         "model_text_chars": len(model_text) if text else 0, "status": state})
        if text:
            if row["split"] not in handles:
                handles[row["split"]] = (out_dir / f"{row['split']}.jsonl").open("w", encoding="utf-8", newline="\n")
            handle = handles[row["split"]]
            handle.write(json.dumps({
                "event_id": row["event_id"], "event_type": kind, "split": row["split"],
                "market_et": row["market_et"], "text_cutoff_utc": row["text_cutoff_utc"],
                "evidence_class": row["evidence_class"], "fingerprint_sha256": row["fingerprint_sha256"],
                "text_source": where, "text_sha256": sha(text), "text": text,
                "model_text_source": model_source, "model_text_sha256": sha(model_text),
                "model_text": model_text}, ensure_ascii=False) + "\n")
    for handle in handles.values():
        handle.close()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    with (args.manifest_dir / f"{args.cohort}-manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    summary = {"cohort": args.cohort, "rows": len(manifest), "status": dict(sorted(status.items())),
               "by_type_split": dict(sorted(Counter(f"{m['event_type']}|{m['split']}" for m in manifest
                                                     if m["status"] == "verified").items())),
               "model_text_source": dict(sorted(Counter(f"{m['event_type']}|{m['model_text_source']}"
                                                         for m in manifest if m["status"] == "verified").items())),
               "jsonl_sha256": {p.name: sha(p.read_bytes()) for p in sorted(out_dir.glob("*.jsonl"))}}
    (args.manifest_dir / f"{args.cohort}-summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for m in manifest:
        if m["status"] != "verified":
            print("NOT REPRODUCED:", m["event_id"], m["split"], m["evidence_class"], "|", m["text_source"])


if __name__ == "__main__":
    main()
