#!/usr/bin/env python3
"""Compare completed depth 4/8/12 runs on identical six-horizon validation rows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_fed_spx_paired import bootstrap_gain


def validate_pairing(reference, current):
    for key in ("truth", "anchor_times"):
        if not np.array_equal(reference[key], current[key]):
            raise ValueError(f"Depth comparison has mismatched {key}")
    if current["forecast"].shape != current["truth"].shape or current["truth"].shape != (40548, 6):
        raise ValueError("Expected 40,548 paired six-horizon validation forecasts")
    if not np.isfinite(current["forecast"]).all():
        raise ValueError("Nonfinite forecasts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for depth in (4, 8, 12):
        p.add_argument(f"--depth{depth}", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-id", default="transformer-depth-sweep-v1")
    a = p.parse_args()
    runs = {d: getattr(a, f"depth{d}") for d in (4, 8, 12)}
    metrics = {d: json.loads((path / "metrics.json").read_text()) for d, path in runs.items()}
    forecasts = {d: dict(np.load(path / "validation-forecasts.npz")) for d, path in runs.items()}
    fixed = ("d_model", "nhead", "ffn", "dropout", "shared_query_parameters", "linear_readout",
             "maximum_batch_size", "max_updates", "activation_checkpointing", "attention_batch_chunk",
             "precision", "gradient_accumulation")
    baseline_scales = torch.load(runs[4] / "best.pt", map_location="cpu", weights_only=True)
    rows, curves, errors = [], [], {}
    for depth, path in runs.items():
        m = metrics[depth]
        if m["status"] != "completed" or m["completed_updates"] != 1000 or m["test_scored"]:
            raise ValueError(f"Depth {depth} did not complete the fixed training-only budget")
        if m["configuration"]["depth"] != depth:
            raise ValueError("Architecture depth mismatch")
        for key in fixed:
            if m["configuration"][key] != metrics[4]["configuration"][key]:
                raise ValueError(f"Depth {depth} changed fixed configuration {key}")
        for key in ("audit", "horizons_minutes", "train_cohort", "score_cohort", "model_sha256"):
            if m[key] != metrics[4][key]:
                raise ValueError(f"Depth {depth} changed fixed inputs {key}")
        scales = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        for key in ("input_scale", "target_scale"):
            if not np.array_equal(scales[key], baseline_scales[key]):
                raise ValueError("Preprocessing scales changed")
        validate_pairing(forecasts[4], forecasts[depth])
        errors[depth] = np.abs(forecasts[depth]["forecast"] - forecasts[depth]["truth"])
        mae = errors[depth].mean(0)
        np.testing.assert_allclose(mae, [m["validation"]["by_horizon"][str(h)]["mae_bp"]
                                       for h in m["horizons_minutes"]], rtol=1e-12, atol=1e-12)
        curve = pd.read_csv(path / "learning-curves.csv")
        if curve["update"].tolist() != list(range(0, 1001, 25)):
            raise ValueError("Evaluation cadence/budget changed")
        selected = curve.loc[curve["update"] == m["selected_update"]].iloc[0]
        np.testing.assert_allclose(selected["val_mean_mae_bp"], mae.mean(), rtol=1e-12)
        curves.append(curve.assign(depth=depth))
        row = dict(depth=depth, parameters=m["parameters"], selected_update=m["selected_update"],
                   best_mean_val_mae_bp=float(mae.mean()),
                   final_mean_val_mae_bp=float(curve.iloc[-1]["val_mean_mae_bp"]),
                   final_train_smooth_l1=float(curve.iloc[-1]["train_smooth_l1"]),
                   runtime_seconds=m["runtime_seconds"], code_commit=m["code_commit"],
                   checkpoint_sha256=m["checkpoint_sha256"])
        row.update({f"best_val_mae_{h}m_bp": float(mae[i]) for i, h in enumerate(m["horizons_minutes"])})
        rows.append(row)
    dates = pd.to_datetime(forecasts[4]["anchor_times"]).strftime("%Y-%m-%d").to_numpy()
    paired = {}
    for old, new in ((4, 8), (4, 12), (8, 12)):
        gain = errors[old] - errors[new]
        both = np.column_stack((gain.mean(1), gain))
        paired[f"depth{new}_vs_depth{old}"] = dict(mean_gain_bp=float(gain.mean()),
            per_horizon_gain_bp=gain.mean(0).tolist(),
            ci95_mean_then_six_horizons=bootstrap_gain(both, dates, np.ones(len(dates)),
                {"bootstrap_seed": 23, "bootstrap_replicates": 2000}))
    a.out.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(a.out / "depth-comparison.csv", index=False)
    pd.concat(curves).to_csv(a.out / "learning-curves.csv", index=False)
    result = dict(id=a.run_id, status="completed", rows=rows,
                  persistence=metrics[4]["persistence"], paired=paired,
                  score_cohort="SPX-PX6-GRID5-v1 val", test_scored=False,
                  limitations=["Each depth has one fit; this is a bounded comparison under one optimization recipe.",
                               "Reused validation selects checkpoints and depth; bootstrap intervals do not adjust selection.",
                               "Equal update/sample budgets do not imply equal compute time."])
    (a.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    text = ["# Transformer depth comparison", "", "Same SPX-PX6-GRID5-v1 windows, 128d width, six targets, batch 204,694, 1,000 updates. No reserved test scored.", "",
            "| Depth | Parameters | Selected update | Best mean validation MAE, bp | Final validation MAE, bp |",
            "|---|---:|---:|---:|---:|"]
    text += [f"| {r['depth']} | {r['parameters']:,} | {r['selected_update']} | {r['best_mean_val_mae_bp']:.6f} | {r['final_mean_val_mae_bp']:.6f} |" for r in rows]
    best = min(rows, key=lambda r: r["best_mean_val_mae_bp"])
    text += ["", f"Lowest selected validation MAE in this comparison: depth {best['depth']}. This is exploratory validation, not independent evidence of a general depth benefit.", "",
             "Positive paired gain means lower MAE for the deeper fit. Intervals and per-horizon results are in metrics.json. All checkpoint/depth selection uses reused validation and is not adjusted in those intervals."]
    (a.out / "summary.md").write_text("\n".join(text) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for curve in curves:
        depth = int(curve["depth"].iloc[0]); tail = curve[curve["update"] >= 100]
        for ax, frame, key in ((axes[0, 0], curve, "train_smooth_l1"),
                               (axes[0, 1], tail, "train_smooth_l1"),
                               (axes[1, 0], curve, "val_mean_mae_bp"),
                               (axes[1, 1], tail, "val_mean_mae_bp")):
            ax.plot(frame["update"], frame[key], label=f"depth {depth}")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update"); ax.legend()
    for ax in axes[0]: ax.set_ylabel("Full training Smooth L1")
    for ax in axes[1]:
        ax.set_ylabel("Validation mean six-horizon MAE (bp)")
        ax.axhline(metrics[4]["persistence"]["mean_mae_bp"], ls="--", color="gray")
    axes[0, 0].set_title("Complete training curve"); axes[0, 1].set_title("Training after update 100")
    axes[1, 0].set_title("Complete validation curve"); axes[1, 1].set_title("Validation after update 100")
    fig.suptitle("Transformer128 depth comparison | SPX-PX6-GRID5-v1 | reused validation")
    fig.savefig(a.out / "learning-curves.png", dpi=180); plt.close(fig)
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
