#!/usr/bin/env python3
"""Export paired prediction-only and per-event loss artifacts for a public PR.

The source NPZ files contain price inputs and targets. This script deliberately
omits those arrays from the public output. The full files stay on shared Delta.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zeroed", type=Path, required=True)
    p.add_argument("--bypass", type=Path, required=True)
    p.add_argument("--predictions-out", type=Path, required=True)
    p.add_argument("--losses-out", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.zeroed, allow_pickle=False) as zero, np.load(args.bypass, allow_pickle=False) as bypass:
        for key in ("event_ids", "event_dates", "event_types", "shuffle_donor_ids"):
            if not np.array_equal(zero[key], bypass[key]):
                raise ValueError(f"Paired runs have different {key}")
        for key in ("real_text", "shuffled_text", "targets", "seqs"):
            if not np.array_equal(zero[key], bypass[key]):
                raise ValueError(f"Paired runs have different {key}")
        m_zero = json.loads(str(zero["metadata_json"]))
        m_bypass = json.loads(str(bypass["metadata_json"]))
        if m_zero["checkpoint_id"] != m_bypass["checkpoint_id"]:
            raise ValueError("Paired runs have different checkpoint hashes")
        if zero["real_text"].shape != zero["targets"].shape:
            raise ValueError("Predictions and targets have different shapes")
        seeds = [x["seed"] for x in m_zero["shuffle_protocols"]]
        if len(seeds) != zero["shuffled_text"].shape[0]:
            raise ValueError("Shuffle seed count does not match predictions")
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        args.losses_out.parent.mkdir(parents=True, exist_ok=True)
        # Explicit allowlist: no targets, price inputs, raw price windows,
        # train volatility array, or article text enters this artifact.
        np.savez_compressed(
            args.predictions_out,
            real_text=zero["real_text"], shuffled_text=zero["shuffled_text"],
            zeroed_article_tokens=zero["zero_text"],
            fusion_bypass_ts_inst=bypass["zero_text"],
            event_ids=zero["event_ids"], event_dates=zero["event_dates"],
            event_types=zero["event_types"],
            shuffle_donor_ids=zero["shuffle_donor_ids"],
            metadata_json=np.array(json.dumps({
                "checkpoint_sha256": m_zero["checkpoint_id"],
                "manifest_sha256": m_zero["c1_manifest_sha256"],
                "price_sha256": m_zero["price_sha256"],
                "training_split": m_zero["training_split"],
                "evaluation_split": m_zero["evaluation_split"],
                "timing_policy": m_zero["c1_data_audit"]["timing_policy"],
                "shuffle_protocols": m_zero["shuffle_protocols"],
                "note": "prediction arrays only; full targets and price windows remain on Delta",
            })),
        )
        fields = ["event_id", "event_date", "event_type", "horizon", "real_mse",
                  "zeroed_mse", "bypass_mse"] + [f"shuffle_seed_{seed}_mse" for seed in seeds]
        with args.losses_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            target = zero["targets"]
            for h in (35, 70, 140):
                if h > target.shape[1]:
                    continue
                def per_event(pred):
                    return np.mean((pred[:, :h, :] - target[:, :h, :]) ** 2, axis=(1, 2))
                real = per_event(zero["real_text"])
                zeroed = per_event(zero["zero_text"])
                bypass_loss = per_event(bypass["zero_text"])
                shuffles = [per_event(pred) for pred in zero["shuffled_text"]]
                for i, event_id in enumerate(zero["event_ids"]):
                    row = {"event_id": str(event_id), "event_date": str(zero["event_dates"][i]),
                           "event_type": str(zero["event_types"][i]), "horizon": h,
                           "real_mse": float(real[i]), "zeroed_mse": float(zeroed[i]),
                           "bypass_mse": float(bypass_loss[i])}
                    row.update({f"shuffle_seed_{seed}_mse": float(loss[i])
                                for seed, loss in zip(seeds, shuffles)})
                    writer.writerow(row)
                print(f"H={h}: real={real.mean():.8f}, zeroed={zeroed.mean():.8f}, "
                      f"bypass={bypass_loss.mean():.8f}, shuffled={np.mean(shuffles):.8f}")
    print(f"Saved public predictions: {args.predictions_out}")
    print(f"Saved per-event losses: {args.losses_out}")


if __name__ == "__main__":
    main()
