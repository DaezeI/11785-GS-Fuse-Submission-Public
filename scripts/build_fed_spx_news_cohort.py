"""Merge Fed news evidence and verify the original endpoint price windows."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fed_spx_endpoints import load_endpoint_cohorts

ET = ZoneInfo("America/New_York")
POPULATION_SHA256 = "4061fe505098babf845b2bc0547bdbe11bd732377f6e2a5face383010b024a71"
LEDGERS = (
    ("statements", {"fomc_statement", "fomc_other_policy_release"},
     "fed-spx-text-evidence-v1/fomc_statement_evidence_ledger.csv"),
    ("minutes_speeches", {"fomc_minutes", "fed_speech"},
     "fed-spx-minutes-speeches-evidence-v1/minutes_speeches_evidence_ledger.csv"),
    ("board_testimony", {"fed_board_monetary_release", "fed_testimony"},
     "fed-spx-board-testimony-evidence-v1/fed_board_testimony_evidence_ledger.csv"),
)
SPEECH_INFERRED = "eligible_document_release_line_unmodified_file"
MINUTES_INFERRED = "eligible_official_release_record_unmodified_file"
EVENT_TYPES = sorted(set().union(*(types for _, types, _ in LEDGERS)))
MANIFEST_FIELDS = (
    "event_id", "event_type", "split", "release_et", "text_cutoff_utc", "forecast_origin_et",
    "fingerprint_sha256", "fingerprint_source", "evidence_class", "evidence_url", "document_url",
    "release_instant_group", "frozen_events_at_release", "eligible_events_at_release",
    "shares_release_instant", "target_20m_complete", "target_30m_complete",
    "pdf_server_written_after_cutoff", "pdf_server_written_after_origin",
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_time(value):
    clock = datetime.fromisoformat(value)
    return clock.replace(tzinfo=ET) if clock.tzinfo is None else clock.astimezone(ET)


def disposition(row):
    value = row["eligibility"]
    if value.startswith("eligible_"):
        return "eligible"
    if value == "not_attempted":
        return "not_attempted"
    if value.startswith(("review_", "version_conflict")):
        return "version_conflict"
    if value in {"blocked_lookup", "not_verified_archive_lookup_incomplete"}:
        return "blocked_lookup"
    return "insufficient_evidence"


def load_assignments(population):
    records, hashes = {}, {}
    for label, types, relpath in LEDGERS:
        path = ROOT / "results" / relpath
        hashes[label] = sha(path)
        for row in read_csv(path):
            event = row["event_id"]
            frozen = population.get(event)
            if event in records or frozen is None or frozen["event_type"] not in types:
                raise ValueError(f"Duplicate or unexpected {label} event {event}")
            if row.get("event_type") not in (None, "", frozen["event_type"]):
                raise ValueError(f"Event type mismatch for {event}")
            if row["split"] != frozen["split"] or datetime.fromisoformat(row["text_cutoff_utc"]) != datetime.fromisoformat(
                frozen["text_cutoff_utc"]
            ) or local_time(row["forecast_origin_et"]) != local_time(frozen["origin_et"]):
                raise ValueError(f"Split or forecast clock mismatch for {event}")
            status = disposition(row)
            if status == "eligible" and not (
                re.fullmatch(r"[a-f0-9]{64}", row["fingerprint_sha256"]) and row["fingerprint_source"]
                and row["evidence_url"]
            ):
                raise ValueError(f"Missing eligible document evidence for {event}")
            records[event] = {**row, "status_class": status, "ledger": label}
    if records.keys() != population.keys():
        raise ValueError(f"Missing {len(population.keys() - records.keys())} frozen events")
    return records, hashes


def describe(rows):
    totals = {}
    for split in ("train", "val"):
        for name in ("all", *EVENT_TYPES):
            selected = [r for r in rows if r["split"] == split and (name == "all" or r["event_type"] == name)]
            totals[f"{name}/{split}"] = {
                "rows": len(selected),
                "release_instants": len({r["release_et"] for r in selected}),
                "dates": len({r["release_et"][:10] for r in selected}),
            }
    return totals


def build_manifest(pop, evidence):
    group_counts = Counter(local_time(r["market_et"]).isoformat() for r in pop.values())
    eligible_counts = Counter(local_time(r["market_et"]).isoformat() for event, r in pop.items()
                              if evidence[event]["status_class"] == "eligible")
    full, manifest = [], []
    for event, frozen in pop.items():
        row = evidence[event]
        instant = local_time(frozen["market_et"]).isoformat()
        origin = local_time(frozen["origin_et"])
        cutoff = datetime.fromisoformat(frozen["text_cutoff_utc"])
        server_written = row.get("pdf_server_written_utc")
        server_clock = datetime.fromisoformat(server_written) if server_written else None
        result = {
            **row, "event_type": frozen["event_type"], "release_et": instant,
            "forecast_origin_et": origin.isoformat(),
            "evidence_class": row["eligibility"],
            "document_url": row.get("house_document_url") or row.get("pdf_url") or row["evidence_url"],
            "release_instant_group": instant, "frozen_events_at_release": group_counts[instant],
            "eligible_events_at_release": eligible_counts[instant],
            "shares_release_instant": group_counts[instant] > 1,
            "target_20m_complete": True, "target_30m_complete": True,
            "pdf_server_written_after_cutoff": "" if server_clock is None else server_clock > cutoff,
            "pdf_server_written_after_origin": "" if server_clock is None else server_clock > origin,
        }
        full.append(result)
        if row["status_class"] == "eligible":
            manifest.append(result)
    return full, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    population = ROOT / "downloads/fed-spx-endpoints-price-v1/population.csv"
    if sha(population) != POPULATION_SHA256:
        raise ValueError("Frozen endpoint population hash mismatch")
    rows = read_csv(population)
    by_id = {r["event_id"]: r for r in rows}
    if len(rows) != len(by_id) or len(rows) != 1185:
        raise ValueError("Expected 1,185 unique endpoint events")
    cohorts, price_audit = load_endpoint_cohorts(args.price_csv)
    rebuilt = {r["event_id"]: r for cohort in cohorts.values() for r in cohort.rows.to_dict("records")}
    if rebuilt != by_id:
        raise ValueError("Rebuilt endpoint price windows do not match the frozen event IDs and clocks")
    evidence, hashes = load_assignments(by_id)
    merged, manifest = build_manifest(by_id, evidence)
    scenarios = {
        "all_audit_eligible": manifest,
        "without_speech_release_line": [r for r in manifest if r["evidence_class"] != SPEECH_INFERRED],
        "without_both_inferred_release_classes": [
            r for r in manifest if r["evidence_class"] not in {SPEECH_INFERRED, MINUTES_INFERRED}
        ],
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = list(MANIFEST_FIELDS) + sorted({key for r in merged for key in r} - set(MANIFEST_FIELDS))
    write_csv(out / "merged_evidence_ledger.csv", merged, fields)
    write_csv(out / "eligible_cohort.csv", manifest, MANIFEST_FIELDS)
    counts = {
        "population_sha256": POPULATION_SHA256, "input_ledger_sha256": hashes,
        "total_events": len(merged), "price_audit": price_audit,
        "by_type_split_disposition": dict(sorted(Counter(
            f"{r['event_type']}/{r['split']}/{r['status_class']}" for r in merged
        ).items())),
        "by_type_split_evidence_class": dict(sorted(Counter(
            f"{r['event_type']}/{r['split']}/{r['evidence_class']}" for r in merged
        ).items())),
        "scenarios": {name: describe(subset) for name, subset in scenarios.items()},
    }
    (out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
