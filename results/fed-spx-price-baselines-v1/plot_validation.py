"""Regenerate the two-panel Fed/SPX validation figure from aggregate metrics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_validation(metrics_path, output_path):
    main = json.loads(metrics_path.read_text(encoding="utf-8"))["validation"]["main"]
    horizons = ["35", "70", "140"]
    last = np.array([main["models"]["last_close"][h]["path_mae_bp"] for h in horizons])
    ridge = np.array([main["models"]["ridge"][h]["path_mae_bp"] for h in horizons])
    improvement = 100 * (1 - ridge / last)
    limits = np.array([main["ridge_mae_improvement_pct_intervals"][h]["10"]["ci95_pct"] for h in horizons])
    blue, ochre, ink = "#214F76", "#9D501D", "#1B2A39"
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": ink, "axes.labelcolor": ink,
                         "xtick.color": ink, "ytick.color": ink, "axes.titlecolor": ink})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.5), width_ratios=[1.1, 1])
    fig.subplots_adjust(left=.092, right=.976, top=.76, bottom=.20, wspace=.35)
    fig.text(.087, .94, "Price-only forecasts around Fed announcements", fontsize=17, weight="bold")
    fig.text(.089, .874,
             f"Main validation  |  {main['events']} rows across {main['distinct_instants']} announcement instants",
             fontsize=11)
    x = np.arange(len(horizons))
    left.bar(x - .17, last, .32, color=blue, label="Last close")
    left.bar(x + .17, ridge, .32, color=ochre, label="Ridge")
    left.set(xticks=x, xticklabels=horizons, ylim=(0, 94), ylabel="OHLC path MAE (bp)",
             xlabel="Forecast horizon (bars)")
    left.legend(frameon=False, loc="upper left", fontsize=9)
    for i, (last_mae, ridge_mae) in enumerate(zip(last, ridge)):
        left.text(i - .17, last_mae + 1.3, f"{last_mae:.1f}", ha="center", fontsize=8)
        left.text(i + .17, ridge_mae + 1.3, f"{ridge_mae:.1f}", ha="center", fontsize=8)
    right.axvline(0, color=ink, lw=1, ls="--")
    right.errorbar(improvement, x, xerr=[improvement - limits[:, 0], limits[:, 1] - improvement],
                   fmt="o", color=ochre, capsize=4, lw=1.6)
    right.set(yticks=x, yticklabels=horizons, xlim=(-16, 1.6), ylim=(2.55, -.55),
              ylabel="Forecast horizon (bars)", xlabel="Ridge improvement in MAE (%)")
    for ax, axis in [(left, "y"), (right, "x")]:
        ax.set_axisbelow(True)
        ax.grid(axis=axis, color="#E1E7EC", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#A4B1BD")
    fig.text(.089, .054, "Intervals use paired bootstrap blocks of ten announcement dates.", fontsize=9)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=folder / "metrics.json")
    parser.add_argument("--output", type=Path, default=folder / "baseline-validation.png")
    args = parser.parse_args()
    plot_validation(args.metrics, args.output)
