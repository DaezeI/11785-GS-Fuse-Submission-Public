"""Join text-verified FOMC statements to the frozen five-minute cash SPX endpoint cohort.

Reuses data.fed_spx_endpoints.load_endpoint_cohorts, which checks the frozen
price/catalog/alignment hashes, the XNYS session, the exact five-minute grid and
zero shared bars between train and validation. It builds train and validation
only; the held-out split is never loaded.

    python scripts/check_fomc_pilot_join.py --price-csv ../fed-person3/downloads/SP500.csv \
        --ledger downloads/fomc-statement-evidence-pilot/ledger.json \
        --out downloads/fomc-statement-evidence-pilot/join.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fed_spx_endpoints import load_endpoint_cohorts  # noqa: E402
from scripts.build_fed_price_alignment import load_prices  # noqa: E402

BAR = pd.Timedelta(minutes=5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-csv", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ledger = {r["event_id"]: r for r in json.load(open(a.ledger, encoding="utf-8"))["rows"]}
    cohorts, audit = load_endpoint_cohorts(a.price_csv)
    align = pd.read_csv(ROOT / "analysis/macro_releases/price_alignment_v1/fed_price_alignment_v1.csv")
    align = align.set_index("event_id")
    bars, _ = load_prices(a.price_csv)
    times = pd.DatetimeIndex(bars.date)

    out = []
    for split, c in cohorts.items():
        for i, row in c.rows.reset_index(drop=True).iterrows():
            led = ledger.get(row.event_id)
            if led is None:
                continue
            al = align.loc[row.event_id]
            market, origin = pd.Timestamp(row.market_et), pd.Timestamp(row.origin_et)
            first, last = pd.Timestamp(al.input_first_bar), pd.Timestamp(al.input_last_bar)
            n_input = int(((times >= first) & (times <= last)).sum())
            cutoff = pd.Timestamp(led["text_cutoff_et"]).tz_localize(None)
            out.append({
                "event_id": row.event_id, "split": split, "eligibility": led["eligibility"],
                "release_et": str(market), "text_cutoff_et": str(cutoff), "forecast_origin_et": str(origin),
                "input_first_bar_start_et": str(first), "input_last_bar_start_et": str(last),
                "input_last_bar_end_et": str(last + BAR), "n_input_bars": n_input,
                "input_ends_by_release": bool(last + BAR <= market),
                "origin_close_bar_start_et": str(origin - BAR),
                "target_20m_close_bar_start_et": str(origin + 3 * BAR),
                "target_30m_close_bar_start_et": str(origin + 5 * BAR),
                "text_cutoff_le_origin_minus_5": bool(cutoff <= origin - BAR),
                "complete_input_and_target": bool(n_input == 35 and c.returns[i].shape == (2,)),
            })
    missing = sorted(set(ledger) - {r["event_id"] for r in out})
    result = {"convention": "SPX five-minute OHLC, timestamps are bar starts in America/New_York "
                            "(empirically supported assumption from the alignment audit); only 09:30-16:00 bars; "
                            "20m/30m targets use the Close of the bars starting origin+15m and origin+25m, "
                            "relative to the Close of the bar starting origin-5m.",
              "cross_split_shared_bars_train_val": audit["cross_split_shared_bars"],
              "input_sha256": audit["input_sha256"], "rows": out, "ledger_events_not_in_cohort": missing}
    Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in out:
        print(r["event_id"], r["split"], r["release_et"], r["input_last_bar_end_et"], r["forecast_origin_et"],
              r["n_input_bars"], r["input_ends_by_release"], r["complete_input_and_target"])
    print("not in cohort:", missing, "| shared train/val bars:", audit["cross_split_shared_bars"])


if __name__ == "__main__":
    main()
