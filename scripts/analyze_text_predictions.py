#!/usr/bin/env python3
"""Paired, CPU-only text-control analysis from saved per-event predictions.

Required NPZ fields (channel-last arrays):
  targets [N,140,4], real_text [N,140,4], zero_text [N,140,4],
  shuffled_text [S,N,140,4], seqs [N,35,4], raw_seqs [N,35,4],
  event_ids [N], event_dates [N], event_types [N], train_volatility [M],
  metadata_json: scalar JSON string with checkpoint and protocol information.

All prediction/target/seq arrays must share one scale. raw_seqs are unscaled
OHLC used only for pre-event log-return volatility. Multiple shuffled runs use
the same examples and checkpoint. No model retraining occurs here.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np


def pre_event_volatility(raw):
    close = np.asarray(raw, dtype=float)[..., 3]
    if not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("raw_seqs must contain positive finite unscaled prices")
    return np.std(np.diff(np.log(close), axis=1), axis=1)


def validate(blob):
    required = ("targets", "real_text", "zero_text", "shuffled_text", "seqs", "raw_seqs",
                "event_ids", "event_dates", "event_types", "train_volatility", "metadata_json")
    for key in required:
        if key not in blob:
            raise ValueError(f"Missing field: {key}")
    y = blob["targets"]
    if y.ndim != 3 or y.shape[1:] != (140, 4):
        raise ValueError("targets must be channel-last [N,140,4]")
    n = len(y)
    if n == 0:
        raise ValueError("Supply at least one evaluation event")
    if len(set(blob["event_ids"].astype(str))) != n:
        raise ValueError("event_ids must be unique; matching dates may repeat")
    for key in ("real_text", "zero_text"):
        if blob[key].shape != y.shape:
            raise ValueError(f"{key} shape must match targets")
    s = blob["shuffled_text"]
    if s.ndim != 4 or s.shape[1:] != y.shape or len(s) < 1:
        raise ValueError("shuffled_text must have shape [S,N,140,4]")
    for key in ("seqs", "raw_seqs"):
        if blob[key].shape != (n, 35, 4):
            raise ValueError(f"{key} must have shape [N,35,4]")
    for key in ("event_dates", "event_types"):
        if blob[key].shape != (n,):
            raise ValueError(f"{key} must have length N")
    for date in blob["event_dates"].astype(str):
        datetime.strptime(date, "%Y%m%d")
    for key in ("targets", "real_text", "zero_text", "shuffled_text", "seqs", "raw_seqs", "train_volatility"):
        if not np.isfinite(blob[key]).all():
            raise ValueError(f"Non-finite values in {key}")
    if blob["train_volatility"].ndim != 1 or len(blob["train_volatility"]) < 3:
        raise ValueError("Supply a training-only volatility reference")
    if np.any(blob["train_volatility"] < 0):
        raise ValueError("Training volatility must be nonnegative")
    return json.loads(str(blob["metadata_json"].item()))


def block_bootstrap_improvement(real_loss, control_loss, dates, mask, *, reps=1000, block=20, seed=23):
    """Resample contiguous blocks of global event dates, keeping paired rows.

    Resampling whole dates keeps reports sharing a price window together.
    Blocks reduce false precision from overlapping neighboring forecast paths.
    The block length is a sensitivity parameter, not a universal guarantee.
    """
    all_dates, inverse = np.unique(np.asarray(dates).astype(str), return_inverse=True)
    group_dates = np.unique(inverse[mask])
    if len(group_dates) < 20:
        return {"ci95_pct": None, "reason": "fewer than 20 distinct dates"}
    n_dates = len(all_dates)
    block = min(block, n_dates)
    if n_dates < 3 * block:
        return {"ci95_pct": None, "reason": "fewer than three blocks at the requested block length",
                "block_event_dates": int(block)}
    real_sums = np.bincount(inverse[mask], weights=real_loss[mask], minlength=n_dates)
    control_sums = np.bincount(inverse[mask], weights=control_loss[mask], minlength=n_dates)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        starts = rng.integers(0, n_dates, size=int(np.ceil(n_dates / block)))
        selected = ((starts[:, None] + np.arange(block)) % n_dates).reshape(-1)[:n_dates]
        counts = np.bincount(selected, minlength=n_dates)
        den = counts @ control_sums
        if den > 0:
            vals.append(100 * (1 - (counts @ real_sums) / den))
    if not vals:
        return {"ci95_pct": None, "reason": "control error is zero in all bootstrap replicates"}
    return {"ci95_pct": np.quantile(vals, [0.025, 0.975]).tolist(),
            "bootstrap_replicates": len(vals), "block_event_dates": int(block),
            "note": "conditional on checkpoint and supplied shuffle draws; inspect block-length sensitivity"}


def analyze(blob, reps=1000, block=20):
    meta = validate(blob)
    cuts = np.quantile(blob["train_volatility"], [1 / 3, 2 / 3])
    volatility = pre_event_volatility(blob["raw_seqs"])
    labels = np.array(["low", "medium", "high"])[np.searchsorted(cuts, volatility, side="right")]
    n = len(blob["targets"])
    masks = {"all": np.ones(n, dtype=bool)}
    masks.update({"event_type:" + t: blob["event_types"].astype(str) == t
                  for t in np.unique(blob["event_types"].astype(str))})
    masks.update({"volatility:" + t: labels == t for t in ("low", "medium", "high")})
    out = {"metadata": meta, "events": n, "shuffles": len(blob["shuffled_text"]),
           "volatility_train_cutoffs": cuts.tolist(), "horizons": {},
           "missing_provenance": [k for k in ("checkpoint_id", "training_split", "evaluation_split",
                "no_text_definition", "shuffle_description")
                if str(meta.get(k, "")).strip().lower() in ("", "none", "unknown", "unverified")],
           "interpretation": "positive gain means real text has lower MSE than the named control"}
    for h in (35, 70, 140):
        truth = blob["targets"][:, :h]
        losses = {k: np.mean((blob[k][:, :h] - truth) ** 2, axis=(1, 2))
                  for k in ("real_text", "zero_text")}
        shuffle_losses = np.mean((blob["shuffled_text"][:, :, :h] - truth[None]) ** 2, axis=(2, 3))
        losses["shuffled_text"] = shuffle_losses.mean(axis=0)
        groups = {}
        for group, mask in masks.items():
            if not mask.any():
                continue
            row = {"events": int(mask.sum()), "distinct_dates": len(np.unique(blob["event_dates"][mask])),
                   "mse": {k: float(v[mask].mean()) for k, v in losses.items()}, "comparisons": {}}
            for control in ("zero_text", "shuffled_text"):
                denom = losses[control][mask].mean()
                row["comparisons"][control] = {
                    "real_text_mse_gain_pct": float(100 * (1 - losses["real_text"][mask].mean() / denom)) if denom else None,
                    **block_bootstrap_improvement(losses["real_text"], losses[control], blob["event_dates"], mask,
                                                  reps=reps, block=block)}
            last = blob["seqs"][:, -1, 3]
            direction = np.sign(truth[:, -1, 3] - last)
            row["endpoint_close_dhr"] = {k: float(np.mean(
                np.sign(blob[k][mask, h - 1, 3] - last[mask]) == direction[mask]))
                for k in ("real_text", "zero_text")}
            row["endpoint_close_dhr"]["always_up"] = float(np.mean(direction[mask] > 0))
            groups[group] = row
        out["horizons"][str(h)] = groups
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bootstrap-reps", default=1000, type=int)
    p.add_argument("--block-dates", default=20, type=int)
    args = p.parse_args()
    if args.bootstrap_reps < 1 or args.block_dates < 1:
        p.error("bootstrap-reps and block-dates must be positive")
    with np.load(args.predictions, allow_pickle=False) as blob:
        result = analyze(blob, args.bootstrap_reps, args.block_dates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"events": result["events"], "shuffles": result["shuffles"],
                      "missing_provenance": result["missing_provenance"], "saved": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
