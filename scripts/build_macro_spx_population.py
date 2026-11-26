"""List U.S. macro releases usable for 20/30-minute SPX forecasts, from timestamps only.

Sources (from ``results/macro-coverage-v1``): BLS Employment Situation, CPI, PPI and JOLTS;
BEA GDP and Personal Income; DOL weekly initial claims (approximate Thursday schedule, to be
confirmed by the file audit). One row per release.

Clocks (same definitions as the Fed cohorts, see docs/NAMING.md):

* ``OPEN`` for releases before 09:30 ET: decision time 09:35 (end of the first regular bar),
  text cutoff 09:30.
* ``REL5`` for releases in the session: decision time = release + 5 min rounded up to the
  5-minute grid, text cutoff five minutes before it. The anchor bar must start at or after
  09:30 and the sixth target bar must end by the close.

Input: the last 35 regular-session bars that end by the release time. Splits use the alignment
v1 cut dates; rows between splits are dropped. A row is dropped when its input or target bars
are also used by a different split (Fed REL5 rows, Fed OPEN rows or other macro rows).

Only the date column of the price file is read. No price value, return or label is computed.

    python scripts/build_macro_spx_population.py --price-csv ../fed-person3/downloads/SP500.csv \\
        --out downloads/macro-spx-v1/population.csv
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import exchange_calendars as calendars
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fed_spx_endpoints import ALIGNMENT, BAR, ET, INPUT_SHA256  # noqa: E402
from scripts.build_fed_price_alignment import sha256_file  # noqa: E402
from scripts.build_fed_spx_test_population import bar_times  # noqa: E402
from scripts.count_macro_coverage import claims_dates  # noqa: E402

COVERAGE = ROOT / "results/macro-coverage-v1"
FED_OPEN = ROOT / "downloads/fed-spx-premarket-v1/population.csv"
PRODUCT_CODE = {"EMPSIT": "empsit", "CPI": "cpi", "PPI": "ppi", "JOLTS": "jolts",
                "GDP": "gdp", "Personal Income": "pi"}
FIELDS = ["event_id", "source", "product", "split", "market_et", "origin_et", "text_cutoff_utc", "clock_rule",
          "input_first_bar", "input_last_bar", "document_hint"]


def releases():
    rows = []
    for r in csv.DictReader((COVERAGE / "bls_releases.tsv").open(encoding="utf-8"), delimiter="\t"):
        day = pd.Timestamp(r["date"])
        code = PRODUCT_CODE[r["product"]]
        rows.append(("BLS", code, pd.Timestamp(f"{r['date']} {r['time_et']}"),
                     f"www.bls.gov/news.release/archives/{code}_{day:%m%d%Y}.pdf"))
    for r in csv.DictReader((COVERAGE / "bea_releases.tsv").open(encoding="utf-8"), delimiter="\t"):
        t = pd.Timestamp(r["datetime"]).tz_convert(ET).tz_localize(None)
        rows.append(("BEA", PRODUCT_CODE[r["product"]], t, r["url"]))
    for d in claims_dates("2008-01-01", "2024-12-31", None):
        rows.append(("DOL", "claims", d + pd.Timedelta(hours=8, minutes=30),
                     f"https://oui.doleta.gov/press/{d:%Y}/{d:%m%d%y}.pdf"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--price-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name, path in {"prices": args.price_csv, "alignment": ALIGNMENT}.items():
        if sha256_file(path) != INPUT_SHA256[name]:
            raise ValueError(f"Frozen {name} input hash mismatch")
    align = pd.read_csv(ALIGNMENT)
    main_rows = align.loc[align.main_candidate & align.split.isin(["train", "val", "test"])]
    cuts = {s: (pd.Timestamp(g.market_et.min()).normalize(), pd.Timestamp(g.market_et.max()).normalize())
            for s, g in main_rows.groupby("split")}
    times = bar_times(args.price_csv)
    schedule = calendars.get_calendar("XNYS", start="2007-12-01", end="2024-12-31").schedule
    sessions = {d.date(): (r.open.tz_convert(ET).tz_localize(None), r.close.tz_convert(ET).tz_localize(None))
                for d, r in schedule.iterrows()}

    # Bars already claimed by the Fed cohorts, by split.
    used = defaultdict(set)
    for r in main_rows.itertuples(index=False):
        first = times.get_loc(pd.Timestamp(r.input_first_bar))
        used[r.split].update(range(first, first + 35))
        origin = pd.Timestamp(r.available_et).ceil("5min")
        used[r.split].update(int(i) for i in times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR)) if i >= 0)
    if FED_OPEN.exists():
        for r in csv.DictReader(FED_OPEN.open(encoding="utf-8")):
            origin = pd.Timestamp(r["origin_et"])
            used[r["split"]].update(int(i) for i in times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR))
                                    if i >= 0)
            last = times.searchsorted(pd.Timestamp(r["market_et"]) - BAR, side="right") - 1
            used[r["split"]].update(range(last - 34, last + 1))

    candidates, excluded = [], Counter()
    ends = times + BAR
    for source, product, t, hint in releases():
        split = next((s for s, (lo, hi) in cuts.items() if lo <= t.normalize() <= hi), None)
        if split is None:
            excluded["outside_or_between_splits"] += 1
            continue
        session = sessions.get(t.date())
        if session is None:
            excluded["non_session_day"] += 1
            continue
        open_, close = session
        if t < open_:
            if open_.strftime("%H:%M") != "09:30":
                excluded["no_regular_0930_open"] += 1
                continue
            origin, rule = open_ + BAR, "open_anchored_0935"
        else:
            origin, rule = (t + BAR).ceil("5min"), "release_plus_5"
            if origin - BAR < open_:
                excluded["anchor_before_open"] += 1
                continue
        if origin + 6 * BAR > close:
            excluded["target_after_close"] += 1
            continue
        target = times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR))
        if (target < 0).any():
            excluded["missing_anchor_or_target_bar"] += 1
            continue
        last = int(ends.searchsorted(t, side="right")) - 1
        if last < 34:
            excluded["insufficient_input_history"] += 1
            continue
        day = t.strftime("%Y%m%d")
        candidates.append({
            "event_id": f"{source.lower()}_{product}_{day}", "source": source, "product": product, "split": split,
            "market_et": str(t), "origin_et": str(origin),
            "text_cutoff_utc": (origin - BAR).tz_localize(ET).tz_convert("UTC").isoformat(),
            "clock_rule": rule, "input_first_bar": str(times[last - 34]), "input_last_bar": str(times[last]),
            "document_hint": hint, "_bars": set(range(last - 34, last + 1)) | {int(i) for i in target},
        })
    ids = Counter(c["event_id"] for c in candidates)
    for c in candidates:
        if ids[c["event_id"]] > 1:
            c["event_id"] += "_" + c["market_et"][11:16].replace(":", "")
    for c in candidates:
        used[c["split"]].update(c["_bars"])
    kept = []
    for c in candidates:
        if any(c["_bars"] & used[s] for s in ("train", "val", "test") if s != c["split"]):
            excluded["shares_bar_with_other_split"] += 1
            continue
        kept.append(c)
    if len({c["event_id"] for c in kept}) != len(kept):
        raise ValueError("Duplicate macro event ID")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    summary = {
        "rows": len(kept),
        "by_split_clock": dict(sorted(Counter(f"{c['split']}|{c['clock_rule']}" for c in kept).items())),
        "by_product_split": dict(sorted(Counter(f"{c['source']}_{c['product']}|{c['split']}" for c in kept).items())),
        "excluded": dict(excluded), "splits": {s: [str(lo.date()), str(hi.date())] for s, (lo, hi) in cuts.items()},
        "population_sha256": sha256_file(args.out), "prices_read": "date column only",
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
