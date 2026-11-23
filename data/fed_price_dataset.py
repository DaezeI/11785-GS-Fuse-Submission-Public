"""Fed-announcement price-window dataset (Person 2 output), numpy only.

One example per Fed announcement: ``x`` = the 35 five-minute OHLC bars that ended before the
announcement, ``y`` = the 140 bars that start at or after ``announcement + 5 min``. The bar that
contains the announcement is in neither. Splits are chronological and purged (see
docs/updates/2026-11-22-fed-price-alignment.md).

    from data.fed_price_dataset import FedPriceDataset

    kw = dict(alignment_csv="analysis/macro_releases/price_alignment_v1/fed_price_alignment_v1.csv",
              price_csv="/path/to/SP500.csv", x_mode="return_bp", y_mode="return_bp")
    train = FedPriceDataset(split="train", **kw)
    scaler = FedPriceDataset.fit_scaler(train)          # train inputs only
    val = FedPriceDataset(split="val", scaler=scaler, **kw)
    test = FedPriceDataset(split="test", scaler=scaler, **kw)
    ex = train[0]      # {"x": (35, 4), "y": (140, 4), "event_id": ..., "text": None, ...}

Modes: "price" = raw index points; "return_bp" = 10,000 * log(price / last input Close), i.e.
basis points relative to the last observed close. Channels are Open, High, Low, Close.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("Open", "High", "Low", "Close")
MODES = ("price", "return_bp")
SUBSETS = ("main", "flagged", "all")


class FedPriceDataset:
    def __init__(self, alignment_csv, price_csv, split, subset="main", x_mode="price", y_mode="price",
                 scaler=None, text_dir=None, require_text=False, seq_len=35, pred_len=140):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}")
        if x_mode not in MODES or y_mode not in MODES:
            raise ValueError(f"modes must be one of {MODES}")
        self.split, self.subset, self.x_mode, self.y_mode = split, subset, x_mode, y_mode
        self.seq_len, self.pred_len, self.scaler = seq_len, pred_len, scaler
        self.text_dir = Path(text_dir) if text_dir else None
        self.require_text = require_text

        from scripts.build_fed_price_alignment import load_prices  # same cleaning as the alignment
        bars, _ = load_prices(price_csv)
        self._t = bars.date.values.astype("datetime64[ns]")
        self._ohlc = bars[list(CHANNELS)].to_numpy(dtype=np.float64)

        df = pd.read_csv(alignment_csv, parse_dates=["input_first_bar", "input_last_bar", "target_first_bar",
                                          "target_last_bar", "available_et"])
        df = df[df.split == split]
        if subset == "main":
            df = df[df.main_candidate.astype(bool)]
        elif subset == "flagged":
            df = df[~df.main_candidate.astype(bool)]
        self.rows = df.sort_values("available_et").reset_index(drop=True)
        self._in0 = np.searchsorted(self._t, self.rows.input_first_bar.values.astype("datetime64[ns]"))
        self._tg0 = np.searchsorted(self._t, self.rows.target_first_bar.values.astype("datetime64[ns]"))
        for i in range(len(self.rows)):  # the windows must be exactly the aligned ones
            a, b = self._in0[i], self._tg0[i]
            if (a + seq_len > len(self._t) or b + pred_len > len(self._t) or a + seq_len > b
                    or self._t[a] != self.rows.input_first_bar.values[i]
                    or self._t[a + seq_len - 1] != self.rows.input_last_bar.values[i]
                    or self._t[b] != self.rows.target_first_bar.values[i]
                    or self._t[b + pred_len - 1] != self.rows.target_last_bar.values[i]):
                raise ValueError(f"alignment file does not match this price file (row {i}); "
                                 "use the same SP500.csv and window lengths")

    def __len__(self):
        return len(self.rows)

    def _transform(self, w, last_close, mode):
        if mode == "return_bp":
            w = 1e4 * np.log(w / last_close)
        return w

    def _raw(self, i):
        a, b = self._in0[i], self._tg0[i]
        x = self._ohlc[a:a + self.seq_len]
        y = self._ohlc[b:b + self.pred_len]
        return x, y, self._ohlc[a + self.seq_len - 1, CHANNELS.index("Close")]

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        x, y, last_close = self._raw(i)
        x = self._transform(x, last_close, self.x_mode)
        y = self._transform(y, last_close, self.y_mode)
        if self.scaler is not None:
            x = (x - self.scaler["x_mean"]) / self.scaler["x_std"]
            y = (y - self.scaler["y_mean"]) / self.scaler["y_std"]
        r = self.rows.iloc[i]
        text = None
        if self.text_dir is not None:
            f = self.text_dir / f"{r.event_id}.txt"
            if f.exists():
                text = f.read_text(encoding="utf-8")
            elif self.require_text:
                raise FileNotFoundError(f)
        return {"x": x.astype(np.float32), "y": y.astype(np.float32), "last_close": float(last_close),
                "event_id": r.event_id, "event_type": r.event_type,
                "available_et": str(r.available_et), "is_main": bool(r.main_candidate),
                "n_concurrent": int(r.n_concurrent_usable), "text": text}

    def window_bar_times(self, i):
        """Start times of every bar used by example i (input and target) - for audits."""
        a, b = self._in0[i], self._tg0[i]
        return self._t[a:a + self.seq_len], self._t[b:b + self.pred_len]

    def as_arrays(self):
        """Stacked (N, seq_len, 4) and (N, pred_len, 4) arrays after the chosen modes and scaler."""
        items = [self[i] for i in range(len(self))]
        return (np.stack([it["x"] for it in items]), np.stack([it["y"] for it in items]))

    @staticmethod
    def fit_scaler(train_ds: "FedPriceDataset"):
        """Per-channel mean/std of the TRAIN windows (inputs and targets, after the modes)."""
        if train_ds.split != "train":
            raise ValueError("fit the scaler on the train split only")
        if train_ds.scaler is not None:
            raise ValueError("pass an unscaled train dataset")
        xs, ys = [], []
        for i in range(len(train_ds)):
            x, y, last = train_ds._raw(i)
            xs.append(train_ds._transform(x, last, train_ds.x_mode))
            ys.append(train_ds._transform(y, last, train_ds.y_mode))
        x, y = np.concatenate(xs), np.concatenate(ys)
        return {"x_mean": x.mean(0), "x_std": np.maximum(x.std(0), 1e-8),
                "y_mean": y.mean(0), "y_std": np.maximum(y.std(0), 1e-8)}
