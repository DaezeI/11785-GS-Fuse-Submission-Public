"""Regenerate the aggregate depth-comparison figure from saved numeric evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parent
curves = pd.read_csv(root / "learning-curves.csv")
metrics = json.loads((root / "metrics.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
for depth, frame in curves.groupby("depth"):
    tail = frame[frame["update"] >= 100]
    for ax, data, key in ((axes[0, 0], frame, "train_smooth_l1"),
                           (axes[0, 1], tail, "train_smooth_l1"),
                           (axes[1, 0], frame, "val_mean_mae_bp"),
                           (axes[1, 1], tail, "val_mean_mae_bp")):
        ax.plot(data["update"], data[key], label=f"depth {depth}")
for ax in axes[0]:
    ax.set_ylabel("Full training Smooth L1")
for ax in axes[1]:
    ax.set_ylabel("Validation mean six-horizon MAE (bp)")
    ax.axhline(metrics["persistence"]["mean_mae_bp"], ls="--", color="gray", label="Persistence")
for ax in axes.flat:
    ax.set_xlabel("Optimizer update")
    ax.legend()
axes[0, 0].set_title("Complete training curve")
axes[0, 1].set_title("Training after update 100")
axes[1, 0].set_title("Complete validation curve")
axes[1, 1].set_title("Validation after update 100")
fig.suptitle("Transformer128 depth comparison | SPX-PX6-GRID5-v1 | reused validation")
fig.savefig(root / "learning-curves.png", dpi=180)
plt.close(fig)
