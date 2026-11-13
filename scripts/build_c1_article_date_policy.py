"""Record evidence and the exclusion decision for every C1 date-screened ID.

This is a date-only evidence ledger, not a publication-time repair. The
original event ID remains the join key; neither folders nor price anchors are
renamed or inferred from a Reuters dateline.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_id(entry):
    return "/".join(entry.replace("\\", "/").split("/")[-2:])


def build(base_path, screened_path, candidates_path, confirmed_path):
    base = json.loads(Path(base_path).read_text())
    screened = json.loads(Path(screened_path).read_text())
    confirmed = {r["event_id"]: r for r in json.loads(Path(confirmed_path).read_text())["records"]}
    with Path(candidates_path).open(newline="") as f:
        candidates = {r["event_id"]: r for r in csv.DictReader(f)}
    roles = {}
    for role in ("train", "vali", "test"):
        for entry in base[f"{role}_events"]:
            eid = event_id(entry)
            if eid in roles:
                raise ValueError(f"Duplicate original event ID: {eid}")
            roles[eid] = role
    excluded = set(screened["article_date_screen"]["excluded_event_ids"])
    if excluded != set(candidates) | set(confirmed):
        raise ValueError("Screened manifest differs from candidate and verified evidence IDs")
    if excluded - roles.keys():
        raise ValueError("Evidence ID absent from original C1")
    if screened["article_date_screen"]["source_manifest_sha256"] != sha256(base_path):
        raise ValueError("Screened manifest points to another original C1")
    for role in ("train", "vali", "test"):
        actual = {event_id(x) for x in screened[f"{role}_events"]}
        expected = {eid for eid, r in roles.items() if r == role and eid not in excluded}
        if actual != expected:
            raise ValueError(f"Screened {role} IDs differ from policy")

    rows = []
    for eid in sorted(excluded):
        candidate = candidates.get(eid)
        source = confirmed.get(eid)
        folder_date = eid.split("/")[1][:8]
        if candidate and candidate["folder_date"] != folder_date:
            raise ValueError(f"Candidate folder date disagrees with ID: {eid}")
        dateline_date = ""
        if candidate:
            try:
                dateline_date = datetime.strptime(folder_date[:4] + candidate["dateline_mmdd"], "%Y%m%d").date().isoformat()
            except ValueError:
                pass
        rows.append({
            "original_event_id": eid,
            "c1_role": roles[eid],
            "folder_date": datetime.strptime(folder_date, "%Y%m%d").date().isoformat(),
            "evidence_status": "source_verified_date_only" if source else "unverified_reuters_dateline_candidate",
            "verified_publication_date": source["source_date"] if source else "",
            "verified_publication_time": "",
            "source_url": source["source_url"] if source else "",
            "summary_sha256": source["summary_sha256"] if source else "",
            "reuters_dateline_date_inferred_year": dateline_date,
            "reuters_same_year_delta_days": candidate["same_year_delta_days"] if candidate else "",
            "forecast_anchor": "",
            "decision": "exclude_until_time_and_anchor_verified",
        })
    summary = {
        "policy_version": "c1_article_date_exclusion_v1",
        "original_manifest_sha256": sha256(base_path),
        "screened_manifest_sha256": sha256(screened_path),
        "candidate_list_sha256": sha256(candidates_path),
        "confirmed_evidence_sha256": sha256(confirmed_path),
        "original_event_count": len(roles),
        "excluded_count": len(rows),
        "source_verified_date_only_count": sum(bool(r["verified_publication_date"]) for r in rows),
        "verified_publication_time_count": 0,
        "unverified_reuters_dateline_candidate_count": sum(not r["verified_publication_date"] for r in rows),
        "retained_unknown_publication_time_count": len(roles) - len(rows),
        "note": "Retained records still use assumed folder-date anchors; this is a sensitivity screen, not a timing-safe baseline.",
    }
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--screened", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--confirmed", required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    a = p.parse_args()
    rows, summary = build(a.base, a.screened, a.candidates, a.confirmed)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    a.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
