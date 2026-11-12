"""Create a versioned C1 manifest that excludes article-date candidates.

The original manifest and raw articles remain unchanged. This conservative
screen is a sensitivity protocol, not a publication-time certification.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROLES = ("train", "vali", "test")


def event_id(entry):
    return "/".join(entry.replace("\\", "/").split("/")[-2:])


def build_screened_manifest(base, candidate_ids, confirmed_ids, base_sha256):
    screened = json.loads(json.dumps(base))
    excluded = set(candidate_ids) | set(confirmed_ids)
    seen = set()
    for role in ROLES:
        old = screened[f"{role}_events"]
        new = []
        for entry in old:
            eid = event_id(entry)
            if eid in excluded:
                seen.add(eid)
            else:
                new.append(entry)
        screened[f"{role}_events"] = new
        screened[f"{role}_count"] = len(new)
        if role == "train" and len(new) != len(old):
            raise ValueError("Training exclusions require recomputing C1 block counts")
        for block in screened["schedule"]:
            if block["role"] == role:
                # C1 has one validation block and one test block.
                if role != "train":
                    block["article_date_screen_excluded"] = len(old) - len(new)
                    block["retained"] = len(new)
    if seen != excluded:
        raise ValueError(f"Exclusion IDs absent from original C1: {sorted(excluded - seen)}")
    screened["split_name"] = "C1_Macro_Eras_Date_Screen_V1"
    screened["article_date_screen"] = {
        "source_manifest_sha256": base_sha256,
        "excluded_event_ids": sorted(excluded),
        "basis": "Opening Reuters dateline differs from folder date, plus externally confirmed non-Reuters date conflict",
        "limitation": "Other summaries have unknown publication dates/times; this is a sensitivity screen."
    }
    return screened


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--confirmed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base_bytes = Path(args.base).read_bytes()
    base = json.loads(base_bytes)
    with Path(args.candidates).open(newline="") as f:
        candidates = {row["event_id"] for row in csv.DictReader(f)}
    confirmed = {row["event_id"] for row in json.loads(Path(args.confirmed).read_text())["records"]}
    screened = build_screened_manifest(base, candidates, confirmed,
                                       hashlib.sha256(base_bytes).hexdigest())
    Path(args.out).write_text(json.dumps(screened, indent=2) + "\n")
    print(json.dumps({"out": args.out, "counts": {role: screened[f"{role}_count"] for role in ROLES},
                      "excluded": len(screened["article_date_screen"]["excluded_event_ids"])}))


if __name__ == "__main__":
    main()
