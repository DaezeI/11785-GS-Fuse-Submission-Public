"""Rolling SPX price windows with the existing Fed endpoint information clock.

An anchor bar starts at E, ends at E+5 (the decision time), and is excluded
from the observed history. Targets are Close returns 20/30 minutes after that
decision time. Historical observations can span the previous trading session.
"""
from dataclasses import dataclass
from pathlib import Path

import exchange_calendars as calendars
import numpy as np
import pandas as pd

from scripts.build_fed_price_alignment import load_prices, sha256_file

PRICE_SHA256 = "6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a"
TRAIN_END = pd.Timestamp("2019-05-22 14:05:00")
# Published earliest price bar in the reserved Fed test windows; no test labels.
VALIDATION_END = pd.Timestamp("2021-08-24 11:05:00")
CHANNELS = ["Open", "High", "Low", "Close"]


@dataclass
class DenseCohort:
    history: np.ndarray
    returns: np.ndarray
    anchors: np.ndarray
    times: np.ndarray
    used_bar_mask: np.ndarray
    past_minutes: np.ndarray | None = None


def anchor_indices(times, session_open, session_close, lower, upper, stride_minutes):
    """Select rolling anchors; the entire history/target support stays in split."""
    if stride_minutes not in (5, 15):
        raise ValueError("Supported anchor strides are 5 and 15 minutes")
    times = np.asarray(times, dtype="datetime64[ns]")
    if len(times) < 42:
        return np.array([], dtype=np.int64)
    if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Price timestamps must be unique and increasing")
    j = np.arange(35, len(times) - 6)
    five = np.timedelta64(5, "m")
    day = times.astype("datetime64[D]")
    gap = np.zeros(len(times), dtype=int)
    gap[1:] = ((day[1:] == day[:-1]) & (np.diff(times) != five))
    prefix = np.cumsum(gap)
    minutes = ((times[j] - day[j]) / np.timedelta64(1, "m")).astype(int)
    valid = ((times[j - 35] >= np.datetime64(lower))
             & (times[j + 6] + five <= np.datetime64(upper))
             & (times[j] >= session_open[j])
             & (times[j] + 7 * five <= session_close[j])
             & (times[j + 6] == times[j] + 6 * five)
             & (prefix[j + 6] == prefix[j - 35])
             & ((minutes - (9 * 60 + 30)) % stride_minutes == 0))
    return j[valid]


def make_cohort(values, times, anchors, target_steps=(4, 6), include_time_offsets=False):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(times), 4) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Expected positive finite OHLC bars")
    observed = values[anchors[:, None] + np.arange(-35, 0)]
    history = (10_000 * np.log(observed / observed[:, -1:, 3:4])).transpose(0, 2, 1).astype(np.float32)
    steps = np.asarray(target_steps, dtype=int)
    if steps.ndim != 1 or len(steps) == 0 or np.any(steps < 1) or np.any(steps > 6):
        raise ValueError("Target steps must be within the existing six-bar audited support")
    returns = 10_000 * np.log(values[anchors[:, None] + steps, 3] / values[anchors, 3, None])
    used = np.zeros(len(times), dtype=bool)
    for offset in range(-35, 7):
        used[anchors + offset] = True
    offsets = None
    if include_time_offsets:
        utc = pd.DatetimeIndex(times).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None).to_numpy()
        offsets = ((utc[anchors[:, None] + np.arange(-35, 0)] - utc[anchors, None])
                   / np.timedelta64(1, "m")).astype(np.float32)
    return DenseCohort(history, returns, anchors, times[anchors], used, offsets)


def load_dense_cohorts(price_csv, stride_minutes, target_steps=(4, 6), include_time_offsets=False):
    if sha256_file(price_csv) != PRICE_SHA256:
        raise ValueError("Frozen price hash mismatch")
    bars, report = load_prices(price_csv)
    times = bars.date.to_numpy(dtype="datetime64[ns]")
    schedule = calendars.get_calendar("XNYS", start=bars.date.min().normalize(), end=VALIDATION_END.normalize()).schedule
    sessions = {day.date(): (r.open.tz_convert("America/New_York").tz_localize(None).to_datetime64(),
                             r.close.tz_convert("America/New_York").tz_localize(None).to_datetime64())
                for day, r in schedule.iterrows()}
    dates = bars.date.dt.date.to_numpy()
    opens = np.array([sessions.get(d, (np.datetime64("NaT"), np.datetime64("NaT")))[0] for d in dates], dtype="datetime64[ns]")
    closes = np.array([sessions.get(d, (np.datetime64("NaT"), np.datetime64("NaT")))[1] for d in dates], dtype="datetime64[ns]")
    train_j = anchor_indices(times, opens, closes, times[0], TRAIN_END, stride_minutes)
    val_j = anchor_indices(times, opens, closes, TRAIN_END, VALIDATION_END, stride_minutes)
    values = bars[CHANNELS].to_numpy(np.float64)
    train, val = (make_cohort(values, times, j, target_steps, include_time_offsets) for j in (train_j, val_j))
    shared = int(np.count_nonzero(train.used_bar_mask & val.used_bar_mask))
    if shared:
        raise ValueError("Rolling train and validation share price bars")
    audit = {"price_sha256": PRICE_SHA256, "stride_minutes": stride_minutes,
             "raw_price_rows": report["rows_in"], "regular_session_bars": len(bars),
             "input_bars": 35, "horizons_minutes": [int(s) * 5 for s in target_steps], "processing_lag_minutes": 5,
             "train_upper_bar_end_inclusive": str(TRAIN_END),
             "validation_upper_bar_end_inclusive": str(VALIDATION_END),
             "cross_split_shared_bars": shared, "test_loaded": False,
             "inherited_price_cleaning": "09:30 through 16:00 bar-start filter, as in the original loader; actual calendar constrains targets",
             "splits": {k: {"rows": len(c.anchors), "dates": len(np.unique(c.times.astype("datetime64[D]"))),
                              "first_anchor": str(c.times[0]), "last_anchor": str(c.times[-1]),
                              "first_used_bar": str(times[np.flatnonzero(c.used_bar_mask)[0]]),
                              "last_used_bar": str(times[np.flatnonzero(c.used_bar_mask)[-1]])}
                        for k, c in (("train", train), ("val", val))}}
    return train, val, audit
