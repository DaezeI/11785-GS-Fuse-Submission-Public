"""List held-out test events that pass the endpoint clock rules, from timestamps only.

Applies the same session and bar-presence rules as ``data.fed_spx_endpoints`` to
the alignment v1 **test** main rows, so their release text can be version-checked
before the comparison is frozen. Only bar timestamps are read: no price value,
return or label is loaded or computed, so the held-out outcomes stay unseen.

    python scripts/build_fed_spx_test_population.py \
        --price-csv ../fed-person3/downloads/SP500.csv \
        --out downloads/fed-spx-test-text-v1/test_population.csv
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import exchange_calendars as calendars
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fed_spx_endpoints import ALIGNMENT, BAR, ET, INPUT_SHA256  # noqa: E402
from scripts.build_fed_price_alignment import SESSION_END, SESSION_START, sha256_file  # noqa: E402

FIELDS = ["event_id", "event_type", "split", "market_et", "origin_et", "text_cutoff_utc"]


def bar_times(price_csv):
    """Regular-session bar start times, reading the date column only."""
    dates = pd.read_csv(price_csv, usecols=["date"], parse_dates=["date"]).date
    dates = dates.drop_duplicates().sort_values()
    hm = dates.dt.strftime("%H:%M")
    return pd.DatetimeIndex(dates[(hm >= SESSION_START) & (hm <= SESSION_END)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--price-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name, path in {"prices": args.price_csv, "alignment": ALIGNMENT}.items():
        if sha256_file(path) != INPUT_SHA256[name]:
            raise ValueError(f"Frozen {name} input hash mismatch")
    rows = pd.read_csv(ALIGNMENT)
    rows = rows.loc[(rows.split == "test") & rows.main_candidate].sort_values(["market_et", "event_id"])
    if not rows.event_id.is_unique:
        raise ValueError("Duplicate test event ID")
    days = pd.to_datetime(rows.market_et).dt.normalize()
    schedule = calendars.get_calendar("XNYS", start=days.min(), end=days.max()).schedule
    sessions = {d.date(): (r.open.tz_convert(ET).tz_localize(None), r.close.tz_convert(ET).tz_localize(None))
                for d, r in schedule.iterrows()}
    times = bar_times(args.price_csv)
    kept, excluded = [], Counter()
    for row in rows.itertuples(index=False):
        market, available = pd.Timestamp(row.market_et), pd.Timestamp(row.available_et)
        origin = available.ceil("5min")
        if available != market + BAR or pd.Timestamp(row.target_first_bar) != origin:
            raise ValueError(f"Unexpected v1 cutoff for {row.event_id}")
        session = sessions.get(market.date())
        if session is None or origin - BAR < session[0] or origin + 6 * BAR > session[1]:
            excluded["outside_xnys_session"] += 1
            continue
        if (times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR)) < 0).any():
            excluded["missing_anchor_or_target_bar"] += 1
            continue
        first = times.get_loc(pd.Timestamp(row.input_first_bar))
        if times[first + 34] != pd.Timestamp(row.input_last_bar) or times[first + 34] + BAR > market:
            raise ValueError(f"Unexpected history window for {row.event_id}")
        kept.append({"event_id": row.event_id, "event_type": row.event_type, "split": "test",
                     "market_et": str(market), "origin_et": str(origin),
                     "text_cutoff_utc": (origin.tz_localize(ET).tz_convert("UTC") - BAR).isoformat()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept, columns=FIELDS).to_csv(args.out, index=False)
    summary = {
        "main_test_rows": len(rows), "eligible_rows": len(kept),
        "distinct_instants": len({r["market_et"] for r in kept}),
        "dates": len({r["market_et"][:10] for r in kept}),
        "by_type": dict(Counter(r["event_type"] for r in kept)), "excluded": dict(excluded),
        "population_sha256": sha256_file(args.out), "prices_read": "date column only",
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
