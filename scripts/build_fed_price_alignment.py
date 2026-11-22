#!/usr/bin/env python3
"""Person 2: align verified Fed announcements to S&P 500 5-minute price windows.

Inputs (neither is committed with source text or raw prices):
  * analysis/macro_releases/fed_announcements_v2.csv  (Person 1: what was public, and when)
  * SP500.csv, 5-minute cash-index OHLC bars, the team's frozen price file

Time contract (see docs/updates/2026-11-22-fed-price-alignment.md):
  * The ``date`` column of the price file is the START of each 5-minute bar, in New York time.
    This is inferred from the data (a 09:30 bar has its own range and closes next to the 09:35
    open); it is not stated by the data source, so Person 1 is asked to confirm it.
  * ``available_utc`` (market time + 5 minutes, from Person 1) is the earliest moment the
    announcement may influence a forecast. It is converted to New York time.
  * TARGET window starts at the first bar whose start is >= available time.
  * INPUT window ends at the last bar that had ENDED by the available time, that is
    start <= available time - 5 minutes. A bar that straddles the available time is an
    "embargo" bar: it is used neither as input nor as target.
  * Only regular-session bars (start 09:30 to 16:00 inclusive) are real prices. Bars after 16:00
    are mostly flat filler (about 61% at 16:05, 97% at 16:10, 100% at 16:15) and are dropped.

The script never reads text. Every output row is asserted to satisfy
``input_last_bar_end <= available_et <= target_first_bar_start``.

Usage:
  python scripts/build_fed_price_alignment.py \
      --events analysis/macro_releases/fed_announcements_v2.csv \
      --prices /path/to/SP500.csv --out-dir downloads/fed_price_alignment_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
BAR = pd.Timedelta(minutes=5)
FROZEN_PRICE_SHA256 = "6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a"
SESSION_START = "09:30"
SESSION_END = "16:00"  # last regular bar START (inclusive)
FULL_DAY_BARS = 79  # 09:30 ... 16:00 inclusive, every 5 minutes
SHORT_TEXT_CHARS = 200
PRICE_COLS = ["Open", "High", "Low", "Close"]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- prices
def load_prices(path, session_start=SESSION_START, session_end=SESSION_END):
    """Return (clean regular-session bars, report). Raises on conflicting duplicate bars."""
    raw = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", *PRICE_COLS} - set(raw.columns)
    if missing:
        raise ValueError(f"price file lacks columns: {sorted(missing)}")
    report = {"rows_in": int(len(raw)), "first_bar": str(raw.date.min()), "last_bar": str(raw.date.max()),
              "was_sorted": bool(raw.date.is_monotonic_increasing)}

    dup = raw[raw.date.duplicated(keep=False)]
    if len(dup):
        spread = dup.groupby("date")[PRICE_COLS].nunique().max().max()
        if spread > 1:
            raise ValueError("duplicate timestamps with different prices; refusing to guess")
    report["duplicate_rows_removed"] = int(raw.date.duplicated().sum())
    report["days_with_duplicates"] = sorted({str(d.date()) for d in dup.date})
    bars = raw.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    hm = bars.date.dt.strftime("%H:%M")
    keep = (hm >= session_start) & (hm <= session_end)
    flat = (bars.Open == bars.High) & (bars.High == bars.Low) & (bars.Low == bars.Close)
    report["bars_after_session_dropped"] = int((~keep).sum())
    report["share_flat_in_dropped"] = float(flat[~keep].mean()) if (~keep).any() else None
    report["share_flat_kept"] = float(flat[keep].mean())
    bars = bars[keep].reset_index(drop=True)

    day = bars.date.dt.normalize()
    per_day = bars.groupby(day).size()
    first = bars.groupby(day).date.min().dt.strftime("%H:%M")
    gap_min = bars.groupby(day).date.diff().dt.total_seconds() / 60
    report.update({
        "session_bars": int(len(bars)), "trading_days": int(len(per_day)),
        "full_days": int((per_day == FULL_DAY_BARS).sum()),
        "short_or_incomplete_days": {str(d.date()): int(n) for d, n in per_day[per_day != FULL_DAY_BARS].items()},
        "days_not_starting_0930": {str(d.date()): t for d, t in first[first != session_start].items()},
        "intraday_gaps_over_5min": int((gap_min > 5).sum()),
        "intraday_gap_examples": [{"bar": str(t), "gap_min": float(g)} for t, g in
                                  bars.loc[gap_min > 5, "date"].to_frame().assign(g=gap_min[gap_min > 5]).head(10)
                                  .itertuples(index=False)],
    })
    bdays = pd.bdate_range(per_day.index.min(), per_day.index.max())
    absent = bdays.difference(per_day.index)
    report["weekdays_without_bars"] = {"count": int(len(absent)),
                                       "note": "holidays or data gaps; no exchange calendar applied",
                                       "dates": [str(d.date()) for d in absent]}
    return bars, report


class BarIndex:
    """Fast window queries over the sorted regular-session bars."""

    def __init__(self, bars: pd.DataFrame):
        self.start = bars.date.values.astype("datetime64[ns]")
        day = bars.date.dt.normalize()
        self.day_id = pd.factorize(day, sort=True)[0]
        self.day = day.values.astype("datetime64[D]")
        same_day = np.r_[self.day_id[1:] == self.day_id[:-1], False]
        step = np.r_[np.diff(self.start), np.timedelta64(0, "ns")]
        # gap_after[i]: a missing bar between bar i and bar i+1 on the same day
        self.gap_after = (same_day & (step > np.timedelta64(5, "m"))).astype(int)
        self.gapcum = np.r_[0, np.cumsum(self.gap_after)]
        # bars of a day whose first bar is after 09:30 (e.g. 2016-10-27 opens at 10:00)
        first_hm = bars.groupby(day).date.transform("min").dt.strftime("%H:%M")
        self.late_open = (first_hm.values != SESSION_START).astype(int)
        self.latecum = np.r_[0, np.cumsum(self.late_open)]
        self.n = len(self.start)

    def gaps_inside(self, a: int, b: int) -> int:
        """Missing-bar gaps strictly between bars a..b (inclusive range of bars)."""
        return int(self.gapcum[b] - self.gapcum[a])

    def window_flags(self, a: int, b: int) -> dict:
        return {"crosses_day": bool(self.day_id[a] != self.day_id[b]),
                "trading_days_spanned": int(self.day_id[b] - self.day_id[a] + 1),
                "calendar_days_spanned": int((self.day[b] - self.day[a]).astype(int) + 1),
                # missing bars inside the window, or it touches a day that opened late
                "has_gap": self.gaps_inside(a, b) > 0 or int(self.latecum[b + 1] - self.latecum[a]) > 0}


# ---------------------------------------------------------------- events
def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_events(path) -> pd.DataFrame:
    ev = pd.read_csv(path, dtype={"release_date": str})
    need = {"event_id", "event_type", "available_utc", "usable_intraday", "text_chars"}
    if need - set(ev.columns):
        raise ValueError(f"events file lacks columns: {sorted(need - set(ev.columns))}")
    ev["usable"] = _to_bool(ev.usable_intraday)
    utc = pd.to_datetime(ev.available_utc, utc=True, errors="coerce")
    ev["available_et_naive"] = utc.dt.tz_convert(ET).dt.tz_localize(None)
    return ev


def classify_session(avail: pd.Timestamp, trading_days: set) -> str:
    """Where the available time falls relative to the cash session (price file days)."""
    if pd.isna(avail):
        return "unknown"
    if avail.weekday() >= 5:
        return "weekend"
    if avail.normalize() not in trading_days:
        return "closed_day_or_data_gap"
    t = avail.strftime("%H:%M")
    if t < SESSION_START:
        return "pre_market"
    if t > SESSION_END:
        return "after_hours"
    return "regular"


def align(events: pd.DataFrame, bars: pd.DataFrame, seq_len=35, pred_len=140,
          main_max_lag_min=5) -> pd.DataFrame:
    idx = BarIndex(bars)
    trading_days = set(pd.to_datetime(bars.date.dt.normalize().unique()))
    rows = []
    # events at the same instant share one price window and are not independent
    key = events.available_utc.fillna("")
    group_size = key.map(key[events.usable].value_counts()).fillna(1).astype(int)
    group_id = pd.factorize(key)[0]
    for i, e in enumerate(events.itertuples(index=False)):
        base = {"event_id": e.event_id, "event_type": e.event_type,
                "available_utc": e.available_utc, "text_chars": e.text_chars,
                "concurrent_group_id": int(group_id[i]), "n_concurrent_usable": int(group_size.iloc[i]),
                "short_text": bool(pd.notna(e.text_chars) and e.text_chars < SHORT_TEXT_CHARS)}
        if not e.usable:
            rows.append({**base, "exclude_reason": "not_usable_intraday_in_events_file"})
            continue
        av = e.available_et_naive
        if pd.isna(av):
            rows.append({**base, "exclude_reason": "unparseable_available_time"})
            continue
        base["available_et"] = av
        base["session_at_available"] = classify_session(av, trading_days)
        av64 = np.datetime64(av.to_datetime64(), "ns")
        if av64 < idx.start[0] + np.timedelta64(5, "m"):
            rows.append({**base, "exclude_reason": "before_price_coverage"})
            continue
        t = int(np.searchsorted(idx.start, av64, side="left"))      # first bar start >= available
        if t >= idx.n:
            rows.append({**base, "exclude_reason": "after_price_coverage"})
            continue
        last_in = int(np.searchsorted(idx.start, av64 - np.timedelta64(5, "m"), side="right")) - 1
        first_in = last_in - seq_len + 1
        if first_in < 0:
            rows.append({**base, "exclude_reason": "insufficient_input_bars"})
            continue
        last_t = t + pred_len - 1
        if last_t >= idx.n:
            rows.append({**base, "exclude_reason": "insufficient_target_bars"})
            continue

        in_end = pd.Timestamp(idx.start[last_in]) + BAR
        assert in_end <= av, f"LEAK: input ends {in_end} after available {av} ({e.event_id})"
        assert pd.Timestamp(idx.start[t]) >= av, f"LEAK: target starts before available ({e.event_id})"
        assert last_in < t, f"window order ({e.event_id})"

        iflag, tflag = idx.window_flags(first_in, last_in), idx.window_flags(t, last_t)
        lag = (pd.Timestamp(idx.start[t]) - av).total_seconds() / 60
        stale = (av - in_end).total_seconds() / 60
        main = (base["session_at_available"] == "regular" and lag <= main_max_lag_min
                and not iflag["has_gap"] and not tflag["has_gap"])
        rows.append({**base, "exclude_reason": "",
                     "input_first_bar": pd.Timestamp(idx.start[first_in]),
                     "input_last_bar": pd.Timestamp(idx.start[last_in]),
                     "input_last_bar_end": in_end,
                     "target_first_bar": pd.Timestamp(idx.start[t]),
                     "target_last_bar": pd.Timestamp(idx.start[last_t]),
                     "embargo_bars": int(t - last_in - 1),
                     "reaction_lag_min": lag, "input_staleness_min": stale,
                     "input_crosses_day": iflag["crosses_day"], "input_has_gap": iflag["has_gap"],
                     "input_calendar_days": iflag["calendar_days_spanned"],
                     "target_crosses_day": tflag["crosses_day"], "target_has_gap": tflag["has_gap"],
                     "target_trading_days": tflag["trading_days_spanned"],
                     "target_calendar_days": tflag["calendar_days_spanned"],
                     "main_candidate": bool(main)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- audit
def audit_summary(out: pd.DataFrame, price_report: dict, meta: dict, params: dict) -> dict:
    ok = out[out.exclude_reason == ""].copy()
    for c in ("input_crosses_day", "input_has_gap", "target_crosses_day", "target_has_gap",
              "main_candidate", "short_text"):
        ok[c] = ok[c].astype(bool)
    reasons = out.exclude_reason.replace("", "aligned").value_counts().to_dict()
    lag_bins = pd.cut(ok.reaction_lag_min, [-1, 5, 30, 120, 960, 4320, 1e9],
                      labels=["<=5min", "5-30min", "30-120min", "2-16h", "16-72h", ">72h"])
    return {
        "meta": meta, "params": params, "price_report": price_report,
        "events_total": int(len(out)),
        "rows_by_outcome": {k: int(v) for k, v in reasons.items()},
        "aligned": int(len(ok)),
        "main_candidates": int(ok.main_candidate.sum()),
        "aligned_by_session": {k: int(v) for k, v in ok.session_at_available.value_counts().items()},
        "aligned_by_event_type": {k: int(v) for k, v in ok.event_type.value_counts().items()},
        "aligned_by_year": {str(k): int(v) for k, v in ok.available_et.dt.year.value_counts().sort_index().items()},
        "reaction_lag_buckets": {str(k): int(v) for k, v in lag_bins.value_counts().sort_index().items()},
        "windows": {"input_crosses_day": int(ok.input_crosses_day.sum()),
                    "target_crosses_day": int(ok.target_crosses_day.sum()),
                    "input_has_gap": int(ok.input_has_gap.sum()),
                    "target_has_gap": int(ok.target_has_gap.sum()),
                    "embargo_bars_nonzero": int((ok.embargo_bars > 0).sum()),
                    "target_calendar_days_max": int(ok.target_calendar_days.max()) if len(ok) else 0},
        "independence": {"distinct_available_instants": int(ok.available_utc.nunique()),
                         "rows_sharing_an_instant": int((ok.n_concurrent_usable > 1).sum()),
                         "short_text_rows": int(ok.short_text.sum())},
        "leakage_check": {"rule": "input_last_bar_end <= available_et <= target_first_bar_start",
                          "violations": 0, "checked_rows": int(len(ok))},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=35)
    ap.add_argument("--pred-len", type=int, default=140)
    ap.add_argument("--main-max-lag-min", type=float, default=5)
    ap.add_argument("--expect-price-sha256", default=FROZEN_PRICE_SHA256,
                    help="empty string disables the frozen-price check")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    a = ap.parse_args(argv)

    price_sha = sha256_file(a.prices)
    if a.expect_price_sha256 and price_sha != a.expect_price_sha256:
        sys.exit(f"price file sha256 {price_sha} differs from the frozen team file")
    out_dir = Path(a.out_dir)
    targets = [out_dir / "fed_price_alignment_v1.csv", out_dir / "price_alignment_audit_v1.json"]
    if not a.force and any(t.exists() for t in targets):
        sys.exit("outputs exist; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    bars, price_report = load_prices(a.prices)
    events = load_events(a.events)
    out = align(events, bars, a.seq_len, a.pred_len, a.main_max_lag_min)
    meta = {"events_sha256": sha256_file(a.events), "prices_sha256": price_sha,
            "events_rows": int(len(events))}
    params = {"seq_len": a.seq_len, "pred_len": a.pred_len, "main_max_lag_min": a.main_max_lag_min,
              "session": [SESSION_START, SESSION_END], "bar_timestamp": "bar start, New York time"}
    summary = audit_summary(out, price_report, meta, params)
    out.to_csv(targets[0], index=False)
    targets[1].write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("events_total", "rows_by_outcome", "aligned", "main_candidates",
                                              "aligned_by_session", "reaction_lag_buckets")}, indent=2))
    print("wrote", *targets)


if __name__ == "__main__":
    main()
