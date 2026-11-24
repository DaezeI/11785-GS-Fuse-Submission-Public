"""List pre-market Fed releases usable under the open-anchored 20/30-minute rule.

Rule (Adaeze, November 23, 2026), extending the release-anchored endpoint rule
without changing alignment v1:

* A release qualifies when its verified market time ``E`` is before 09:30 ET on
  an XNYS session day (not midnight date-only entries, not press-conference
  transcripts).
* Input: the same 35 OHLC bars that alignment v1 assigns, all ending by ``E``.
* Decision time ``T0`` is 09:35, the end of the first regular bar. The text cutoff
  is ``T0 - 5 min`` (09:30), the same rule as release-anchored rows: the text must be
  public at least five minutes before the decision time.
* Labels (built later, not here): Close at 09:55 and 10:05 against the Close at 09:35.
* Splits come from alignment v1. Windows that share a bar with an earlier split
  are reported and dropped.

Only bar timestamps are read. No price value, return or label is computed, so this
can list test rows without exposing their outcomes.

    python scripts/build_fed_spx_premarket_population.py \
        --price-csv ../fed-person3/downloads/SP500.csv \
        --out downloads/fed-spx-premarket-v1/population.csv
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
from data.fed_spx_endpoints import ALIGNMENT, BAR, CATALOG, ET, INPUT_SHA256  # noqa: E402
from scripts.build_fed_price_alignment import sha256_file  # noqa: E402
from scripts.build_fed_spx_test_population import bar_times  # noqa: E402

FIELDS = ["event_id", "event_type", "split", "market_et", "origin_et", "text_cutoff_utc", "clock_rule"]
EXCLUDED_TYPES = {"fomc_press_conference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--price-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name, path in {"prices": args.price_csv, "catalog": CATALOG, "alignment": ALIGNMENT}.items():
        if sha256_file(path) != INPUT_SHA256[name]:
            raise ValueError(f"Frozen {name} input hash mismatch")
    rows = pd.read_csv(ALIGNMENT)
    catalog = pd.read_csv(CATALOG).set_index("event_id")
    times = bar_times(args.price_csv)
    rows = rows.loc[(rows.session_at_market == "pre_market") & rows.split.isin(["train", "val", "test"])].copy()
    days = pd.to_datetime(rows.market_et).dt.normalize()
    schedule = calendars.get_calendar("XNYS", start=days.min(), end=days.max()).schedule
    sessions = {d.date(): (r.open.tz_convert(ET).tz_localize(None), r.close.tz_convert(ET).tz_localize(None))
                for d, r in schedule.iterrows()}

    # Bars already used by release-anchored main rows, per split, for the overlap check.
    main = pd.read_csv(ALIGNMENT)
    main = main.loc[main.main_candidate & main.split.isin(["train", "val", "test"])]
    used = {s: set() for s in ("train", "val", "test")}
    for r in main.itertuples(index=False):
        first = times.get_loc(pd.Timestamp(r.input_first_bar))
        used[r.split].update(range(first, first + 35))
        origin = pd.Timestamp(r.available_et).ceil("5min")
        idx = times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR))
        used[r.split].update(int(i) for i in idx if i >= 0)

    kept, excluded = [], Counter()
    for r in rows.sort_values(["market_et", "event_id"]).itertuples(index=False):
        market = pd.Timestamp(r.market_et)
        info = catalog.loc[r.event_id] if r.event_id in catalog.index else None
        if r.event_type in EXCLUDED_TYPES:
            excluded["press_conference_transcript"] += 1
            continue
        if market.strftime("%H:%M") == "00:00":
            excluded["date_only_midnight_time"] += 1
            continue
        if info is None or not bool(info.get("time_verified")) or not bool(info.get("usable_intraday")):
            excluded["time_not_verified_or_not_intraday"] += 1
            continue
        session = sessions.get(market.date())
        if session is None or session[0] != market.normalize() + pd.Timedelta(hours=9, minutes=30):
            excluded["no_regular_0930_open_that_day"] += 1
            continue
        origin = session[0] + BAR
        if origin + 6 * BAR > session[1]:
            excluded["session_too_short"] += 1
            continue
        target_ids = times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR))
        if (target_ids < 0).any():
            excluded["missing_anchor_or_target_bar"] += 1
            continue
        if pd.isna(r.input_first_bar):
            excluded["no_v1_input_window"] += 1
            continue
        first = times.get_loc(pd.Timestamp(r.input_first_bar))
        if times[first + 34] != pd.Timestamp(r.input_last_bar) or times[first + 34] + BAR > market:
            raise ValueError(f"Unexpected history window for {r.event_id}")
        window = set(range(first, first + 35)) | {int(i) for i in target_ids}
        earlier = {"train": [], "val": ["train"], "test": ["train", "val"]}[r.split]
        later = {"train": ["val", "test"], "val": ["test"], "test": []}[r.split]
        if any(window & used[s] for s in earlier + later):
            excluded["shares_bar_with_other_split"] += 1
            continue
        cutoff = origin - BAR
        kept.append({"event_id": r.event_id, "event_type": r.event_type, "split": r.split,
                     "market_et": str(market), "origin_et": str(origin),
                     "text_cutoff_utc": cutoff.tz_localize(ET).tz_convert("UTC").isoformat(),
                     "clock_rule": "open_anchored_0935"})
        used[r.split].update(window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(kept, columns=FIELDS)
    frame.to_csv(args.out, index=False, lineterminator="\n")
    summary = {
        "pre_market_rows_in_alignment_v1": len(rows), "eligible_rows": len(kept),
        "by_split": {s: {"rows": int((frame.split == s).sum()),
                         "release_instants": int(frame.loc[frame.split == s, "market_et"].nunique())}
                     for s in ("train", "val", "test")},
        "by_type_split": dict(Counter(f"{k['event_type']}|{k['split']}" for k in kept)),
        "excluded": dict(excluded), "population_sha256": sha256_file(args.out), "prices_read": "date column only",
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
