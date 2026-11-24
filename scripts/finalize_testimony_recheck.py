"""Write the testimony re-check (speech document rule) as a sanitized override ledger.

The input is the JSON ledger from ``verify_minutes_speech_versions.py --types fed_testimony``
on the train/validation testimony rows left ``not_verified_archive_lookup_incomplete``.
The output has the columns ``build_fed_spx_news_cohort.py`` reads from a re-check ledger.

    python scripts/finalize_testimony_recheck.py downloads/testimony-recheck-v1/ledger.json \
        results/fed-spx-testimony-recheck-v1
"""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

FIELDS = ["event_id", "split", "eligibility", "reason", "release_line", "text_cutoff_utc", "source_url",
          "capture_utc", "capture_minutes_after_cutoff", "replay_url", "fingerprint_sha256",
          "fingerprint_source", "pdf_url", "pdf_mod_utc", "pdf_written_minutes_after_release"]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads(src.read_text(encoding="utf-8"))["rows"]
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate testimony event")
    ledger = []
    for r in rows:
        eligible = r["disposition"].startswith("eligible")
        pre = r.get("pre_cutoff_capture_utc")
        ledger.append({
            "event_id": r["event_id"], "split": r["split"], "eligibility": r["disposition"], "reason": r["reason"],
            "release_line": r.get("pdf_release_line") or "",
            "text_cutoff_utc": r["cutoff_utc"].replace(" ", "T"), "source_url": r["source_url"],
            "capture_utc": pre or "", "capture_minutes_after_cutoff": "",
            "replay_url": (f"https://web.archive.org/web/{pre}/{r['pre_cutoff_capture_url']}"
                           if pre and r.get("pre_cutoff_capture_url") else r.get("pdf_url") or ""),
            "fingerprint_sha256": r.get("fingerprint_sha256") or "" if eligible else "",
            "fingerprint_source": "official PDF bytes" if eligible else "",
            "pdf_url": r.get("pdf_url") or "", "pdf_mod_utc": r.get("pdf_mod_utc") or "",
            "pdf_written_minutes_after_release": r.get("pdf_written_minutes_after_release") or "",
        })
    out.mkdir(parents=True, exist_ok=True)
    with (out / "testimony_recheck_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ledger)
    counts = Counter(f"{r['split']}|{r['eligibility']}" for r in ledger)
    (out / "counts.json").write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(sorted(counts.items())), indent=2))


if __name__ == "__main__":
    main()
