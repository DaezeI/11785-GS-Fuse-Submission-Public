"""Build two Close-return labels from the frozen Fed/SPX train and validation windows."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import exchange_calendars as calendars
import numpy as np
import pandas as pd

from data.fed_price_dataset import CHANNELS
from scripts.build_fed_price_alignment import load_prices, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "analysis/macro_releases/fed_announcements_v2.csv"
ALIGNMENT = CATALOG.parent / "price_alignment_v1/fed_price_alignment_v1.csv"
INPUT_SHA256 = {
    "prices": "6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a",
    "catalog": "56157fe018beeba6f051f765e0f9d6763e839324b9d95ebbf49f15ec47c9d939",
    "alignment": "0ab047bf9f918d798bd1c4f4eac8c6d59e0317c8d36fcb25c81a6801cfee0270",
}
HORIZONS = (20, 30)
ET = "America/New_York"
BAR = pd.Timedelta(minutes=5)


@dataclass
class EndpointCohort:
    rows: pd.DataFrame
    history: np.ndarray
    returns: np.ndarray

    @property
    def weights(self):
        return 1.0 / self.rows.groupby("market_et")["event_id"].transform("size").to_numpy(dtype=float)


def endpoint_returns(closes):
    """The input consists of the origin Close followed by exactly six future Closes."""
    prices = np.asarray(closes, dtype=np.float64)
    if prices.shape[-1:] != (7,) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Expected seven finite positive Close prices")
    return 10_000 * np.log(prices[..., [4, 6]] / prices[..., :1])


def load_endpoint_cohorts(price_csv):
    for name, path in {"prices": price_csv, "catalog": CATALOG, "alignment": ALIGNMENT}.items():
        if sha256_file(path) != INPUT_SHA256[name]:
            raise ValueError(f"Frozen {name} input hash mismatch")
    rows = pd.read_csv(ALIGNMENT)
    rows = rows.loc[rows.split.isin(["train", "val"]) & rows.main_candidate].copy()
    if not rows.event_id.is_unique or rows.groupby("market_et").split.nunique().max() != 1:
        raise ValueError("Duplicate event ID or market instant shared across splits")
    rows = rows.sort_values(["market_et", "event_id"])
    market_days = pd.to_datetime(rows.market_et).dt.normalize()
    schedule = calendars.get_calendar("XNYS", start=market_days.min(), end=market_days.max()).schedule
    sessions = {day.date(): (row.open.tz_convert(ET).tz_localize(None), row.close.tz_convert(ET).tz_localize(None))
                for day, row in schedule.iterrows()}
    bars, _ = load_prices(price_csv)
    times = pd.DatetimeIndex(bars.date)
    values = bars[list(CHANNELS)].to_numpy(dtype=np.float64)
    cohorts, used, counts = {}, {}, {}
    for split in ("train", "val"):
        retained, inputs, targets, bar_ids = [], [], [], set()
        exclusions = Counter()
        split_rows = rows.loc[rows.split == split]
        for row in split_rows.itertuples(index=False):
            market = pd.Timestamp(row.market_et)
            available = pd.Timestamp(row.available_et)
            origin = available.ceil("5min")
            if (available != market + BAR or pd.Timestamp(row.target_first_bar) != origin
                    or available.tz_localize(ET).tz_convert("UTC") != pd.Timestamp(row.available_utc)):
                raise ValueError(f"Unexpected v1 cutoff for {row.event_id}")
            session = sessions.get(market.date())
            if session is None or origin - BAR < session[0] or origin + 6 * BAR > session[1]:
                exclusions["outside_xnys_session"] += 1
                continue
            target_ids = times.get_indexer(pd.date_range(origin - BAR, periods=7, freq=BAR))
            if np.any(target_ids < 0):
                exclusions["missing_anchor_or_target_bar"] += 1
                continue
            first = times.get_loc(pd.Timestamp(row.input_first_bar))
            last = first + 34
            if times[last] != pd.Timestamp(row.input_last_bar) or times[last] + BAR > market:
                raise ValueError(f"Unexpected history window for {row.event_id}")
            history = values[first:last + 1]
            if not np.isfinite(history).all() or np.any(history <= 0):
                raise ValueError(f"Invalid OHLC history for {row.event_id}")
            inputs.append((10_000 * np.log(history / history[-1, 3])).T)
            targets.append(endpoint_returns(values[target_ids, 3]))
            retained.append({"event_id": row.event_id, "event_type": row.event_type, "split": split,
                             "market_et": str(market), "origin_et": str(origin),
                             "text_cutoff_utc": (origin.tz_localize(ET).tz_convert("UTC") - BAR).isoformat()})
            bar_ids.update(range(first, last + 1))
            bar_ids.update(target_ids.tolist())
        if not retained:
            raise ValueError(f"No events in {split} after endpoint audit")
        cohort = EndpointCohort(pd.DataFrame(retained), np.asarray(inputs, dtype=np.float32), np.stack(targets))
        cohorts[split], used[split] = cohort, bar_ids
        counts[split] = {"main_input_rows": len(split_rows), "eligible_rows": len(cohort.rows),
                         "distinct_instants": int(cohort.rows.market_et.nunique()),
                         "dates": int(cohort.rows.market_et.str[:10].nunique()), "excluded": dict(exclusions)}
    overlap = len(used["train"] & used["val"])
    if overlap:
        raise ValueError(f"Found {overlap} price bars shared by train and validation")
    return cohorts, {"calendar": "XNYS", "exchange_calendars_version": calendars.__version__,
                     "input_sha256": INPUT_SHA256, "cross_split_shared_bars": overlap, "splits": counts}
