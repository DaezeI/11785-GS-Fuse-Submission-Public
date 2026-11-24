"""Coverage count for a broader U.S. macro + Fed news cohort on frozen cash SPX.

Counts candidate forecast events, not verified text. Sources:
  * BLS Employment Situation, CPI, PPI (8:30 ET) and JOLTS (10:00 ET): archive index dates
  * BEA GDP and Personal Income: archive listing with release timestamps
  * DOL weekly initial claims (8:30 ET Thursday; holiday shifts approximated)
  * Fed events: price-alignment v1 main rows (train/val/test), already intraday by construction
Rules:
  * premarket (release before 09:30 ET on an XNYS session day): all such releases
    on that day share one forecast window at the open; forecast rule reported separately
  * intraday (09:30 <= release, and release + 5 min buffer + 30 min target <= close):
    simultaneous releases (same 5-minute origin) share one window
  * releases on non-session days, after the last usable intraday time, or outside
    frozen price coverage are counted as lost
Splits use the alignment v1 chronological cut dates. Train and validation days are
checked against the frozen price file's bar timestamps; the held-out test period is
counted from the exchange calendar only (no price data read for test dates).
"""
import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import exchange_calendars as calendars
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_fed_price_alignment import load_prices  # noqa: E402

ET = "America/New_York"
FED_HOLIDAY_THURSDAYS = "thanksgiving christmas newyear july4"


def claims_dates(start, end, xnys_holidays):
    """Thursdays; when Thursday is a federal holiday the release moves to Wednesday (approximation)."""
    out = []
    d = pd.Timestamp(start)
    while d.weekday() != 3:
        d += pd.Timedelta(days=1)
    while d <= pd.Timestamp(end):
        hol = (d.month == 11 and 22 <= d.day <= 28) or (d.month == 12 and d.day == 25) or \
              (d.month == 1 and d.day == 1) or (d.month == 7 and d.day == 4)
        out.append((d - pd.Timedelta(days=1)) if hol else d)
        d += pd.Timedelta(days=7)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bls", required=True)
    ap.add_argument("--bea", required=True)
    ap.add_argument("--price-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-train-rows", type=int, default=2000)
    a = ap.parse_args(argv)

    align = pd.read_csv(ROOT / "analysis/macro_releases/price_alignment_v1/fed_price_alignment_v1.csv")
    main_rows = align[align.main_candidate.fillna(False).astype(bool) & align.split.isin(["train", "val", "test"])]
    cuts = {s: (pd.Timestamp(g.market_et.min()), pd.Timestamp(g.market_et.max())) for s, g in main_rows.groupby("split")}

    def split_of(ts):
        for s in ("train", "val", "test"):
            lo, hi = cuts[s]
            if lo.normalize() <= ts.normalize() <= hi.normalize():
                return s
        return "between_or_outside_splits"

    bars, _ = load_prices(a.price_csv)
    bar_days = set(pd.DatetimeIndex(bars.date).normalize())
    price_end = pd.Timestamp(bars.date.max()).normalize()
    price_start = pd.Timestamp(bars.date.min()).normalize()
    cal = calendars.get_calendar("XNYS", start="2007-12-01", end="2024-12-31")
    sched = cal.schedule
    sessions = {d.normalize().tz_localize(None) if d.tzinfo else d.normalize(): (
        r.open.tz_convert(ET).tz_localize(None), r.close.tz_convert(ET).tz_localize(None)) for d, r in sched.iterrows()}

    rel = []  # (source, product, timestamp ET naive)
    for r in csv.DictReader(open(a.bls, encoding="utf-8"), delimiter="\t"):
        rel.append(("BLS", r["product"], pd.Timestamp(f"{r['date']} {r['time_et']}")))
    for r in csv.DictReader(open(a.bea, encoding="utf-8"), delimiter="\t"):
        t = pd.Timestamp(r["datetime"]).tz_convert(ET).tz_localize(None)
        rel.append(("BEA", r["product"], t))
    for d in claims_dates("2008-01-01", "2024-12-31", None):
        rel.append(("DOL", "Initial claims (approx. schedule)", d + pd.Timedelta(hours=8, minutes=30)))
    for r in main_rows.itertuples():
        rel.append(("Fed", r.event_type, pd.Timestamp(r.market_et)))

    lost = Counter()
    per_source = defaultdict(Counter)
    events = {}  # key -> dict
    for src, prod, t in rel:
        label = f"{src}: {prod}"
        day = t.normalize()
        if day < price_start or day > price_end:
            lost[("outside_price_coverage", label)] += 1
            continue
        sess = sessions.get(day)
        if sess is None:
            lost[("release_on_non_session_day", label)] += 1
            continue
        op, cl = sess
        sp = split_of(t)
        if sp in ("train", "val") and day not in bar_days:
            lost[("no_price_bars_that_day", label)] += 1
            continue
        if t < op:
            key, kind = ("premarket", day), "premarket"
        else:
            origin = (t + pd.Timedelta(minutes=5)).ceil("5min")
            if origin + pd.Timedelta(minutes=30) > cl:
                lost[("too_late_for_30m_target", label)] += 1
                continue
            key, kind = ("intraday", origin), "intraday"
        per_source[label][(kind, sp)] += 1
        ev = events.setdefault(key, {"kind": kind, "split": sp, "sources": set()})
        ev["sources"].add(label)

    by = Counter((e["kind"], e["split"]) for e in events.values())
    days = Counter((e["split"]) for e in events.values())
    macro_only = Counter((e["kind"], e["split"]) for e in events.values() if not all(s.startswith("Fed") for s in e["sources"]))
    fed_only = Counter((e["kind"], e["split"]) for e in events.values() if all(s.startswith("Fed") for s in e["sources"]))
    bundles = Counter(len(e["sources"]) for e in events.values())
    no_claims = Counter((e["kind"], e["split"]) for e in events.values()
                        if any(not x.startswith("DOL") for x in e["sources"]))
    train_total = sum(v for (k, s), v in by.items() if s == "train")
    res = {
        "splits_from_alignment_v1": {s: [str(lo), str(hi)] for s, (lo, hi) in cuts.items()},
        "price_coverage": [str(price_start.date()), str(price_end.date())],
        "releases_by_source_kind_split": {k: {f"{a_}|{b_}": n for (a_, b_), n in sorted(v.items())}
                                          for k, v in sorted(per_source.items())},
        "lost_releases": {f"{a_}|{b_}": n for (a_, b_), n in sorted(lost.items())},
        "distinct_forecast_events_by_kind_split": {f"{k}|{s}": n for (k, s), n in sorted(by.items())},
        "events_with_any_macro_release": {f"{k}|{s}": n for (k, s), n in sorted(macro_only.items())},
        "events_fed_only": {f"{k}|{s}": n for (k, s), n in sorted(fed_only.items())},
        "releases_per_event": dict(sorted(bundles.items())),
        "events_excluding_claims_only": {f"{k}|{s}": n for (k, s), n in sorted(no_claims.items())},
        "train_events_total": train_total,
        "target_train_rows": a.target_train_rows,
        "notes": ["Counts are candidate events before any exact-version text check.",
                  "BLS archive index lists start in February 2008; January 2008 BLS releases are missing.",
                  "BLS times are the standard release times, not checked per release; DOL claims dates are an approximate Thursday schedule.",
                  "Intraday events within 30 minutes of each other are counted separately but their target windows overlap.",
                  "Test period counted from the exchange calendar only."]}
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("distinct_forecast_events_by_kind_split", "events_excluding_claims_only",
                                          "events_fed_only", "releases_per_event", "train_events_total")}, indent=1))
    print(json.dumps(res["lost_releases"], indent=1))
    for k, v in res["releases_by_source_kind_split"].items():
        print(k, v)


if __name__ == "__main__":
    main()
