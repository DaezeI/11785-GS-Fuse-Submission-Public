"""Check a small official-release set against Delta summaries and price bars.

Official URLs and release times are reviewed evidence in the input JSON. This
script validates the exact local files, headline markers, C1 membership, and
that all selected targets start after the recorded release time.
"""

import argparse
from bisect import bisect_left
import csv
from collections import Counter
from datetime import datetime, time
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def event_id(entry):
    return "/".join(entry.replace("\\", "/").split("/")[-2:])


def check(evidence_path, manifest_path, event_dir, price_path):
    evidence = json.loads(Path(evidence_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    test_ids = {event_id(x) for x in manifest["test_events"]}
    other_ids = {event_id(x) for role in ("train", "vali") for x in manifest[role + "_events"]}
    with Path(price_path).open(newline="") as f:
        stamps = sorted({int(row["date_int"]) for row in csv.DictReader(f)})
    if not stamps:
        raise ValueError("Empty price timeline")
    output = []
    seen = set()
    for record in evidence["records"]:
        eid = record["event_id"]
        if eid in seen or eid not in test_ids or eid in other_ids:
            raise ValueError(f"Not a unique held-out C1 test ID: {eid}")
        seen.add(eid)
        category, filename = eid.split("/")
        date = filename[:8]
        release = datetime.fromisoformat(record["release_local"])
        if release.strftime("%Y%m%d") != date or release.time() != time(8, 30):
            raise ValueError(f"Release/date mismatch for {eid}")
        summary = Path(event_dir) / category / filename / f"{filename}_full_summary.txt"
        if file_hash(summary) != record["summary_sha256"]:
            raise ValueError(f"Summary changed for {eid}")
        content = summary.read_text(encoding="utf-8", errors="ignore").casefold()
        missing = [x for x in record["summary_match_markers"] if x.casefold() not in content]
        if missing:
            raise ValueError(f"Summary headline markers missing for {eid}: {missing}")
        release_key = int(release.strftime("%Y%m%d%H%M%S"))
        anchor = bisect_left(stamps, release_key) - 1
        start, stop = anchor - 34, anchor + 141
        if start < 0 or stop > len(stamps):
            raise ValueError(f"Incomplete 35+140 price window for {eid}")
        input_end = datetime.strptime(str(stamps[anchor]), "%Y%m%d%H%M%S")
        first_target = datetime.strptime(str(stamps[anchor + 1]), "%Y%m%d%H%M%S")
        target_end = datetime.strptime(str(stamps[stop - 1]), "%Y%m%d%H%M%S")
        if not (input_end < release < first_target):
            raise ValueError(f"Price window straddles release incorrectly for {eid}")
        if first_target.date() != release.date() or first_target.time() < time(9, 30):
            raise ValueError(f"First target is not post-release market open for {eid}")
        output.append({"event_id": eid,
                       "official_release_et": release.replace(tzinfo=ZoneInfo("America/New_York")).isoformat(),
                       "official_url": record["official_url"],
                       "summary_sha256": record["summary_sha256"],
                       "matched_headline_markers": record["summary_match_markers"],
                       "price_input_end_label": input_end.isoformat(sep=" "),
                       "price_target_first_label": first_target.isoformat(sep=" "),
                       "price_target_end_label": target_end.isoformat(sep=" "),
                       "price_window_indices": [start, anchor, stop]})
    counts = Counter(x["event_id"].split("/")[0] for x in output)
    if len(output) != 8 or counts != {"2": 4, "5": 4}:
        raise ValueError(f"Unexpected selected count or categories: {counts}")
    return {"set_name": evidence["set_name"], "checked_count": len(output),
            "category_counts": dict(counts), "evidence_sha256": file_hash(evidence_path),
            "manifest_sha256": file_hash(manifest_path), "price_sha256": file_hash(price_path),
            "price_timezone_caveat": evidence["price_label_assumption"],
            "price_check": "Input ends before official release; first target is same-day 09:30 or later",
            "records": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--event-dir", type=Path, required=True)
    p.add_argument("--price", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = check(a.evidence, a.manifest, a.event_dir, a.price)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checked_count": result["checked_count"],
                      "category_counts": result["category_counts"], "out": str(a.out)}))


if __name__ == "__main__":
    main()
