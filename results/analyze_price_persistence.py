#!/usr/bin/env python3
"""Compare the last observed OHLC bar with saved, aligned model forecasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_text_predictions import block_bootstrap_improvement, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--sha256", required=True, help="Recorded hash of the prediction archive")
    args = parser.parse_args()

    digest = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    if digest != args.sha256:
        parser.error(f"Prediction archive hash mismatch: {digest}")

    with np.load(args.predictions, allow_pickle=False) as blob:
        metadata = validate(blob)
        target = blob["targets"].astype(float)
        real = blob["real_text"].astype(float)
        shuffled = blob["shuffled_text"].astype(float)
        last_bar = blob["seqs"][:, -1:, :].astype(float)
        dates = blob["event_dates"]
        date_by_id = dict(zip(blob["event_ids"], dates))
        donor_dates = np.array([
            [date_by_id[donor] for donor in draw] for draw in blob["shuffle_donor_ids"]
        ])

    mask = np.ones(len(dates), dtype=bool)
    rows = []
    for horizon in (35, 70, 140):
        targets = target[:, :horizon]
        predictions = real[:, :horizon]
        model_loss = np.square(predictions - targets).mean(axis=(1, 2))
        persistence_loss = np.square(last_bar - targets).mean(axis=(1, 2))
        model_mse = float(model_loss.mean())
        persistence_mse = float(persistence_loss.mean())
        swap_rms = float(np.sqrt(np.square(predictions - shuffled[:, :, :horizon]).mean()))

        rows.append({
            "horizon": horizon,
            "model_mse": model_mse,
            "persistence_mse": persistence_mse,
            "persistence_gain_pct": 100 * (1 - persistence_mse / model_mse),
            "persistence_ci95_pct_by_date_block": {
                block: block_bootstrap_improvement(
                    persistence_loss, model_loss, dates, mask, block=block
                )["ci95_pct"]
                for block in (5, 10, 20, 40)
            },
            "swap_rms_pct_of_model_rmse": 100 * swap_rms / np.sqrt(model_mse),
        })

    print(json.dumps({
        "archive_sha256": digest,
        "checkpoint_sha256": metadata["checkpoint_id"],
        "source_commit": metadata["reference_source_commit"],
        "manifest_sha256": metadata["c1_manifest_sha256"],
        "price_sha256": metadata["price_sha256"],
        "events": len(dates),
        "distinct_dates": len(np.unique(dates)),
        "future_donor_fractions": (donor_dates > dates).mean(axis=1).tolist(),
        "horizons": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
