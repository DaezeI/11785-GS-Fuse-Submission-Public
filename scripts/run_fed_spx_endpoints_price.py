#!/usr/bin/env python3
"""Fit the compact Fed/SPX endpoint model using only pre-release prices."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from data.fed_spx_endpoints import HORIZONS, load_endpoint_cohorts
from model.compact_return import CompactReturn
from scripts.build_fed_price_alignment import sha256_file
from scripts.score_fed_spx_paired import bootstrap_gain, endpoint_metrics  # noqa: F401  (re-exported)


def run_seed(train, val, input_scale, target_scale, config, seed, device):
    torch.manual_seed(seed)
    model = CompactReturn(output_dim=len(HORIZONS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    rand = torch.Generator().manual_seed(seed)
    histories = torch.as_tensor(train.history / input_scale, dtype=torch.float32, device=device)
    targets = torch.as_tensor(train.returns / target_scale, dtype=torch.float32, device=device)
    weights = torch.as_tensor(train.weights / train.weights.mean(), dtype=torch.float32, device=device)
    val_histories = torch.as_tensor(val.history / input_scale, dtype=torch.float32, device=device)
    val_weights = val.weights
    best = float("inf")
    trace = []
    for epoch in range(config["epochs"] + 1):
        if epoch:
            model.train()
            for selection in torch.randperm(len(histories), generator=rand).split(config["batch_size"]):
                selection = selection.to(device)
                optimizer.zero_grad(set_to_none=True)
                errors = F.smooth_l1_loss(model(histories[selection]), targets[selection], reduction="none")
                loss = (errors.mean(dim=1) * weights[selection]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(val_histories).cpu().numpy().astype(np.float64) * target_scale
        score = float(np.average(np.abs(prediction[:, 0] - val.returns[:, 0]), weights=val_weights))
        if not np.isfinite(score):
            raise ValueError(f"Non-finite validation MAE for seed {seed}")
        trace.append(score)
        if score < best:
            best, chosen_epoch, chosen_forecast = score, epoch, prediction
            chosen_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return chosen_state, chosen_forecast, chosen_epoch, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/fed-spx-endpoints-price-v1/config.json")
    args = parser.parse_args()

    start = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in the selected environment")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    cohorts, audit = load_endpoint_cohorts(args.price_csv)
    train, val = cohorts["train"], cohorts["val"]
    input_scale = np.sqrt(np.average(np.square(train.history.astype(np.float64)).mean(axis=(1, 2)), weights=train.weights))
    target_scale = np.sqrt(np.average(np.square(train.returns), weights=train.weights, axis=0))
    if not np.isfinite(input_scale) or input_scale <= 0 or not np.all(np.isfinite(target_scale) & (target_scale > 0)):
        raise ValueError("Expected finite positive training-only RMS scales")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    population = args.out_dir / "population.csv"
    pd.concat([train.rows, val.rows], ignore_index=True).to_csv(population, index=False, lineterminator="\n")

    records, forecasts = [], []
    for seed in config["seeds"]:
        state, prediction, epoch, trace = run_seed(train, val, input_scale, target_scale, config, seed, device)
        forecasts.append(prediction)
        checkpoint = args.out_dir / f"compact-price-seed-{seed}.pt"
        torch.save({"state_dict": state, "input_scale": float(input_scale), "target_scale": target_scale.tolist(),
                    "horizons_minutes": HORIZONS, "seed": seed, "epoch": epoch}, checkpoint)
        records.append({"seed": seed, "selected_epoch": epoch, "checkpoint_sha256": sha256_file(checkpoint),
                        "validation_mae20_trace_bp": trace,
                        "metrics": endpoint_metrics(val.returns, prediction, val.weights)})
        print(f"Seed {seed}: selected epoch {epoch}; MAE 20 min {trace[epoch]:.4f} bp", flush=True)

    forecasts = np.stack(forecasts)
    zero_loss = np.abs(val.returns)
    gain = zero_loss - np.abs(forecasts - val.returns).mean(axis=0)
    dates = val.rows.market_et.str[:10].to_numpy(dtype=str)
    np.savez_compressed(args.out_dir / "val-paired.npz", forecasts=forecasts, targets=val.returns, weights=val.weights,
                        event_ids=val.rows.event_id.to_numpy(dtype=str), dates=dates)
    result = {"split": "val", "horizons_minutes": HORIZONS, "price_audit": audit, "config": config,
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "config_sha256": sha256_file(args.config), "population_sha256": sha256_file(population),
              "input_scale_bp": float(input_scale), "target_scales_bp": target_scale.tolist(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "pandas": pd.__version__, "device": str(device),
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
              "runtime_seconds": time.perf_counter() - start,
              "zero_return": endpoint_metrics(val.returns, np.zeros_like(val.returns), val.weights),
              "price_only_seeds": records,
              "gain_over_zero_bp_20_30": np.average(gain, weights=val.weights, axis=0).tolist(),
              "paired_gain_ci95_bp_by_block_20_30": bootstrap_gain(gain, dates, val.weights, config)}
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
