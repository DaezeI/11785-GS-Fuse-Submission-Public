"""Compare independent minutes and speech dispositions on the frozen event IDs."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path

from audit_fed_board_testimony_versions import POPULATION_SHA, ROOT, digest, read_csv


def eligible(row):
    return row["eligibility"] in {"eligible", "verified_exact_version"} or row["eligibility"].startswith("eligible_")


def load_ledger(path, expected):
    records = {}
    for row in read_csv(path):
        event = row["event_id"]
        if event not in expected or event in records:
            raise ValueError(f"Wrong scope or duplicate ID in {path}: {event}")
        frozen = expected[event]
        if row["split"] != frozen["split"] or datetime.fromisoformat(row["text_cutoff_utc"]) != datetime.fromisoformat(
            frozen["text_cutoff_utc"]
        ):
            raise ValueError(f"Split or cutoff differs in {path}: {event}")
        if not row["eligibility"]:
            raise ValueError(f"Missing disposition in {path}: {event}")
        records[event] = row
    if records.keys() != expected.keys():
        raise ValueError(f"Missing {len(expected.keys() - records.keys())} assigned IDs in {path}")
    return records


def compare(codex, primary, expected):
    counts = Counter()
    different = []
    for event, frozen in expected.items():
        left, right = codex[event], primary[event]
        left_ok, right_ok = eligible(left), eligible(right)
        same_digest = left.get("fingerprint_sha256") == right.get("fingerprint_sha256")
        if left_ok and right_ok:
            agreement = "both_eligible_same_fingerprint" if same_digest and left.get("fingerprint_sha256") else (
                "both_eligible_review_fingerprints"
            )
        elif left_ok:
            agreement = "codex_only_eligible"
        elif right_ok:
            agreement = "primary_only_eligible"
        else:
            agreement = "neither_eligible"
        counts[f"{frozen['event_type']}/{frozen['split']}/{agreement}"] += 1
        if agreement not in {"both_eligible_same_fingerprint", "neither_eligible"}:
            different.append({
                "event_id": event, "event_type": frozen["event_type"], "split": frozen["split"],
                "comparison": agreement,
                **{f"{name}_{key}": row.get(key, "") for name, row in (("codex", left), ("primary", right))
                   for key in ("eligibility", "evidence_url", "fingerprint_sha256", "fingerprint_source")},
            })
    return counts, different


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    population = ROOT / "downloads/fed-spx-endpoints-price-v1/population.csv"
    if digest(population.read_bytes()) != POPULATION_SHA:
        raise ValueError("Frozen population SHA-256 mismatch")
    expected = {r["event_id"]: r for r in read_csv(population) if r["event_type"] in {"fomc_minutes", "fed_speech"}}
    if len(expected) != 472:
        raise ValueError("Expected exactly 472 minutes and speech events")
    counts, differences = compare(load_ledger(args.codex, expected), load_ledger(args.primary, expected), expected)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps({
        "population_sha256": POPULATION_SHA, "codex_ledger_sha256": digest(args.codex.read_bytes()),
        "primary_ledger_sha256": digest(args.primary.read_bytes()), "counts": dict(sorted(counts.items())),
        "fingerprint_note": "Different hashes can reflect extraction or normalization. Review before declaring version conflicts.",
    }, indent=2) + "\n", encoding="utf-8")
    if differences:
        with (args.out / "review.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
            writer.writeheader()
            writer.writerows(differences)


if __name__ == "__main__":
    main()
