"""Merge per-type text-version checks for one population into a sanitized ledger and counts.

Inputs are the per-type audit outputs under ``--src`` (default: the held-out test check).
Ledgers that a population does not need may be absent. The output holds dispositions,
fingerprints and URLs only. No price, return or label is read.

    python scripts/summarize_fed_spx_test_text.py --out results/fed-spx-test-text-v1
    python scripts/summarize_fed_spx_test_text.py --src downloads/fed-spx-premarket-v1         --population population.csv --out results/fed-spx-premarket-text-v1
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "downloads/fed-spx-test-text-v1"
STRICT = {"eligible_B_official_record_only", "eligible_archived_copy_before_cutoff"}
INFERRED = {
    "eligible_document_release_line_unmodified_file",
    "eligible_official_release_record_unmodified_file",
    "eligible_document_release_line_release_day_capture",
}
FIELDS = ["event_id", "event_type", "split", "market_et", "origin_et", "text_cutoff_utc", "clock_rule", "eligibility",
          "reason", "fingerprint_sha256", "evidence_url", "source_ledger"]


def json_rows(path):
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def describe(rows):
    return {"rows": len(rows), "release_instants": len({r["market_et"] for r in rows}),
            "dates": len({r["market_et"][:10] for r in rows})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--population", default="test_population.csv")
    args = parser.parse_args()
    SRC = args.src if args.src.is_absolute() else ROOT / args.src
    population_path = SRC / args.population
    population = {r["event_id"]: r for r in csv.DictReader(population_path.open(encoding="utf-8"))}
    found = {}
    for r in (json_rows(SRC / "statements/ledger.json") if (SRC / "statements/ledger.json").exists() else []):
        found[r["event_id"]] = (r["eligibility"], r.get("reason") or "", r.get("fingerprint_sha256") or
                                r.get("minutes_body_sha256") or "", r.get("minutes_url") or "", "statements")
    for name in ("minutes-speeches", "testimony"):
        if not (SRC / name / "ledger.json").exists():
            continue
        for r in json_rows(SRC / name / "ledger.json"):
            found[r["event_id"]] = (r["disposition"], r["reason"], r.get("fingerprint_sha256") or "",
                                    r.get("pdf_url") or r["source_url"], name)
    board = SRC / "board/final/board_recheck_ledger.csv"
    for r in (csv.DictReader(board.open(encoding="utf-8")) if board.exists() else []):
        found[r["event_id"]] = (r["eligibility"], r["reason"], r["fingerprint_sha256"],
                                r["replay_url"] or r["source_url"], "board")
    if found.keys() != population.keys():
        raise ValueError(f"Checked {len(found)} events for a population of {len(population)}")
    ledger = [{**population[e], "eligibility": d, "reason": why, "fingerprint_sha256": fp,
               "evidence_url": url, "source_ledger": src}
              for e, (d, why, fp, url, src) in sorted(found.items())]
    eligible = [r for r in ledger if r["eligibility"].startswith("eligible")]
    missing_fp = [r["event_id"] for r in eligible if not r["fingerprint_sha256"]]
    scenarios = {
        "all_audit_eligible": eligible,
        "without_board_release_day_capture": [
            r for r in eligible if r["eligibility"] != "eligible_document_release_line_release_day_capture"],
        "without_speech_release_line": [
            r for r in eligible if r["eligibility"] != "eligible_document_release_line_unmodified_file"],
        "without_any_inferred_release_class": [r for r in eligible if r["eligibility"] not in INFERRED],
    }
    counts = {
        "population_sha256": hashlib.sha256(population_path.read_bytes()).hexdigest(),
        "population_rows": len(population),
        "by_split": {s: describe([r for r in eligible if r["split"] == s]) for s in ("train", "val", "test")},
        "by_type_disposition": dict(sorted(Counter(f"{r['event_type']}|{r['eligibility']}" for r in ledger).items())),
        "eligible_by_type": dict(sorted(Counter(r["event_type"] for r in eligible).items())),
        "eligible_missing_fingerprint": missing_fp,
        "scenarios": {k: describe(v) for k, v in scenarios.items()},
        "strict_classes": sorted(STRICT),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "test_text_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [f for f in FIELDS if f != "clock_rule" or "clock_rule" in next(iter(population.values()))]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ledger)
    (args.out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: counts[k] for k in ("eligible_by_type", "scenarios", "eligible_missing_fingerprint")},
                     indent=2))


if __name__ == "__main__":
    main()
