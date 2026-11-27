"""Render the aggregate CPU replay diagnostics; no private inputs required."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "training-traces.csv")
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
for row, (arm, budget, title) in enumerate((("price", 480, "Price CNN"), ("text", 90, "Text correction head"))):
    subset = d[d.arm.eq(arm)]
    ax = axes[row, 0]
    ax.plot(subset.step, subset.train_smooth_l1, label="Full training set", color="#20639b")
    ax.plot(subset.step, subset.val_smooth_l1, label="Validation set", color="#c44735")
    ax.axvline(budget, color="0.4", ls="--", lw=1)
    ax.set(title=title + ": original budget and extension", ylabel="Standardized Smooth L1", xlabel="Optimizer updates")
    ax.legend(frameon=False, fontsize=8)
    tail = subset[subset.step.ge(budget)]
    ax = axes[row, 1]
    ax.plot(tail.step, tail.train_smooth_l1, "o-", color="#20639b")
    change = (tail.train_smooth_l1.iloc[-1] / tail.train_smooth_l1.iloc[0] - 1) * 100
    ax.set(title=f"Ten extra updates: training loss {change:+.1f}%", xlabel="Optimizer updates", ylabel="Full-training Smooth L1")
for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
fig.suptitle("CPU replay of recorded configurations; not original W&B histories", fontsize=13)
fig.savefig(HERE / "training-curves.png", dpi=170)
