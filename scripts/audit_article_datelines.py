"""Flag C1 article folders whose opening Reuters dateline has a different date.

This is a candidate-finding audit, not proof of publication time. It reads
article summaries but writes only event IDs, dates, and aggregate counts.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re


REUTERS_DATELINE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+"
    r"(?P<day>\d{1,2})(?:,?\s+(?P<year>20\d{2}))?\s*\(Reuters\)",
    re.IGNORECASE,
)
MONTHS = {name.lower(): i for i, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
)}


def dateline_date(opening):
    """Return (MMDD, explicit year or None) for an opening Reuters dateline."""
    match = REUTERS_DATELINE.search(opening)
    if match is None:
        return None
    month = MONTHS[match.group("month")[:3].lower()]
    day = int(match.group("day"))
    if not (1 <= day <= 31):
        return None
    return f"{month:02d}{day:02d}", match.group("year")


def audit(manifest_path, event_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    event_dir = Path(event_dir)
    counts = Counter()
    candidates = []
    for role in ("train", "vali", "test"):
        for entry in manifest[f"{role}_events"]:
            parts = entry.replace("\\", "/").split("/")
            category, folder = parts[-2], parts[-1][:-4]
            event_id = f"{category}/{folder}.txt"
            summary = event_dir / category / f"{folder}.txt" / f"{folder}.txt_full_summary.txt"
            counts[f"{role}:total"] += 1
            try:
                with summary.open(encoding="utf-8", errors="ignore") as f:
                    date = dateline_date(f.read(1000))
            except FileNotFoundError:
                counts[f"{role}:missing_summary"] += 1
                continue
            if date is None:
                counts[f"{role}:no_opening_reuters_dateline"] += 1
                continue
            counts[f"{role}:opening_reuters_dateline"] += 1
            month_day, explicit_year = date
            if folder[4:] == month_day and (explicit_year is None or folder[:4] == explicit_year):
                counts[f"{role}:folder_date_matches"] += 1
            else:
                counts[f"{role}:candidate_mismatch"] += 1
                candidates.append({"role": role, "event_id": event_id,
                                   "folder_date": folder, "dateline_mmdd": month_day,
                                   "dateline_explicit_year": explicit_year})
    return {"manifest": str(manifest_path), "event_dir": str(event_dir),
            "method": "Reuters dateline in first 1000 characters of full_summary",
            "caveat": "Candidates require source verification; absent datelines are unknown.",
            "counts": dict(sorted(counts.items())), "candidates": candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.event_dir)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"], "out": args.out}))


if __name__ == "__main__":
    main()
