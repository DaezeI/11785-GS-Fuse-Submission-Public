#!/usr/bin/env python3
"""Fit and evaluate news-free baselines on Fed/SPX alignment v1 validation."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from data.fed_price_dataset import CHANNELS, FedPriceDataset
from scripts.analyze_text_predictions import block_bootstrap_improvement
from scripts.build_fed_price_alignment import sha256_file

ALIGNMENT_DIR = ROOT / "analysis/macro_releases/price_alignment_v1"
ALIGNMENT_CSV = ALIGNMENT_DIR / "fed_price_alignment_v1.csv"
ALIGNMENT_AUDIT = ALIGNMENT_DIR / "price_alignment_audit_v1.json"
ALIGNMENT_COMMIT = "a4269dce74eb0cc829fa6772b049eed7ed463abd"
HORIZONS = (35, 70, 140)
RIDGE_ALPHA = 10.0
EXPECTED_COUNTS = {("train", "main"): 1013, ("val", "main"): 215, ("val", "flagged"): 85}


def instant_weights(rows):
    """Give each announcement instant unit weight within its own cohort."""
    sizes = rows.groupby("available_et")["event_id"].transform("size")
    return 1.0 / sizes.to_numpy(dtype=np.float64)


def fit_ridge(x, y, weights):
    """Standardize flattened train inputs and fit one multioutput ridge."""
    features = x.reshape(len(x), -1).astype(np.float64)
    targets = y.reshape(len(y), -1).astype(np.float64)
    mean = np.average(features, axis=0, weights=weights)
    centered = features - mean
    scale = np.sqrt(np.average(centered ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1.0  # The last input close is identically zero in bp.
    features = centered / scale
    intercept = np.average(targets, axis=0, weights=weights)
    weighted_features = features * weights[:, None]
    covariance = weighted_features.T @ features
    covariance += RIDGE_ALPHA * np.eye(features.shape[1])
    coefficients = np.linalg.solve(covariance, weighted_features.T @ (targets - intercept))
    return dict(x_mean=mean, x_scale=scale, intercept=intercept, coefficients=coefficients)


def predict_ridge(x, model):
    features = (x.reshape(len(x), -1) - model["x_mean"]) / model["x_scale"]
    return (features @ model["coefficients"] + model["intercept"]).reshape(len(x), 140, 4)


def score_paths(prediction, target, weights):
    """Score channel-last log-return paths in bp with matched row weights."""
    if prediction.shape != target.shape or target.shape != (len(weights), 140, 4):
        raise ValueError("Predictions and targets must have matched [N, 140, 4] shapes")
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError("Predictions and targets must be finite")
    error = prediction - target
    metrics = {}
    for horizon in HORIZONS:
        path_error = error[:, :horizon]
        pred_close = prediction[:, horizon - 1, 3]
        true_close = target[:, horizon - 1, 3]
        nonflat = true_close != 0
        hits = np.sign(pred_close[nonflat]) == np.sign(true_close[nonflat])
        metrics[str(horizon)] = {
            "path_mae_bp": float(np.average(np.abs(path_error).mean(axis=(1, 2)), weights=weights)),
            "path_rmse_bp": float(np.sqrt(np.average((path_error ** 2).mean(axis=(1, 2)), weights=weights))),
            "endpoint_close_mae_bp": float(np.average(np.abs(pred_close - true_close), weights=weights)),
            "direction_accuracy_nonflat": (
                float(np.average(hits, weights=weights[nonflat])) if nonflat.any() else None
            ),
            "direction_eligible_rows": int(nonflat.sum()),
            "predicted_flat_fraction": float(np.average(pred_close == 0, weights=weights)),
        }
    return metrics


def cohort_metadata(rows):
    # Sorting the IDs here leaves the original row order intact for paired predictions.
    ids = sorted(rows.event_id.astype(str))
    return {
        "events": len(ids),
        "distinct_instants": int(rows.available_et.nunique()),
        "distinct_announcement_dates": int(rows.market_et.str[:10].nunique()),
        "event_id_set_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def load_cohort(prices, split, subset):
    expected = EXPECTED_COUNTS[(split, subset)]  # This runner reads train/validation only.
    data = FedPriceDataset(ALIGNMENT_CSV, prices, split=split, subset=subset,
                           x_mode="return_bp", y_mode="return_bp")
    if len(data) != expected or not data.rows.event_id.is_unique:
        raise ValueError(f"Unexpected alignment membership for {split}/{subset}")
    x, y = data.as_arrays()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"Non-finite returns in {split}/{subset}")
    return data.rows, x, y


def evaluate_cohort(rows, x, target, model):
    weights = instant_weights(rows)
    pred = {"last_close": np.zeros_like(target), "ridge": predict_ridge(x, model)}
    dates = rows.market_et.str[:10].to_numpy(dtype=str)
    mask = np.ones(len(weights), dtype=bool)
    intervals = {}
    for horizon in HORIZONS:
        losses = {
            name: weights * np.abs(values[:, :horizon] - target[:, :horizon]).mean(axis=(1, 2))
            for name, values in pred.items()
        }
        intervals[str(horizon)] = {
            str(block): block_bootstrap_improvement(losses["ridge"], losses["last_close"], dates, mask,
                                                    block=block, reps=2000, seed=23)
            for block in (5, 10)
        }
    result = {
        **cohort_metadata(rows),
        "models": {name: score_paths(values, target, weights) for name, values in pred.items()},
        "ridge_mae_improvement_pct_intervals": intervals,
    }
    paired = dict(event_ids=rows.event_id.to_numpy(dtype=str), announcement_dates=dates,
                  weights=weights, targets=target, **pred)
    return result, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh local run directory")
    args = parser.parse_args()

    audit = json.loads(ALIGNMENT_AUDIT.read_text(encoding="utf-8"))
    price_hash = sha256_file(args.price_csv)
    if price_hash != audit["meta"]["prices_sha256"]:
        parser.error("Price CSV SHA-256 does not match the frozen alignment input")
    train_rows, train_x, train_y = load_cohort(args.price_csv, "train", "main")
    model = fit_ridge(train_x, train_y, instant_weights(train_rows))
    validation, paired = {}, {}
    for subset in ("main", "flagged"):
        rows, x, y = load_cohort(args.price_csv, "val", subset)
        validation[subset], paired[subset] = evaluate_cohort(rows, x, y, model)

    metrics = {
        "split": "val",
        "channels": list(CHANNELS),
        "target": "10000 * log(OHLC / last input close)",
        "training": {"subset": "main", **cohort_metadata(train_rows)},
        "validation": validation,
        "ridge_alpha": RIDGE_ALPHA,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "alignment_commit": ALIGNMENT_COMMIT,
        "input_sha256": {
            "prices": price_hash,
            "alignment_csv": sha256_file(ALIGNMENT_CSV),
            "alignment_audit": sha256_file(ALIGNMENT_AUDIT),
            "fed_events": audit["meta"]["events_sha256"],
        },
        "environment": {
            "os": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out_dir / "ridge.npz", **model)
    for subset, values in paired.items():
        np.savez_compressed(args.out_dir / f"val-{subset}-paired.npz", **values)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(f"Scored {validation['main']['events']} main and {validation['flagged']['events']} flagged validation rows.")


if __name__ == "__main__":
    main()
