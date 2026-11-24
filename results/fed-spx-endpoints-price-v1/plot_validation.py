"""Plot the price-only comparison and paired validation intervals."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=folder / "metrics.json")
    parser.add_argument("--output", type=Path, default=folder / "validation-mae.png")
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    horizons = [str(h) for h in metrics["horizons_minutes"]]
    zero = [metrics["zero_return"][h]["mae_bp"] for h in horizons]
    cnn = [np.mean([r["metrics"][h]["mae_bp"] for r in metrics["price_only_seeds"]]) for h in horizons]
    sample = metrics["price_audit"]["splits"]["val"]
    gain = metrics["gain_over_zero_bp_20_30"]
    limits = np.asarray(metrics["paired_gain_ci95_bp_by_block_20_30"]["10"])
    positions = np.arange(len(horizons))

    fig, (scores, paired) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": .36})
    fig.subplots_adjust(left=.09, right=.975, top=.73, bottom=.22)
    fig.text(.09, .92, "Short-horizon SPX forecasts from prices alone", fontsize=17, weight="bold")
    fig.text(.09, .854, f"Validation · {sample['eligible_rows']} events across {sample['distinct_instants']} release instants",
             fontsize=11)
    scores.bar(positions - .17, zero, .32, label="Zero return", color="#738394")
    scores.bar(positions + .17, cnn, .32, label="Compact CNN", color="#236A83")
    scores.set(xticks=positions, xticklabels=[f"{h} min" for h in horizons], ylabel="Close-return MAE (bp)",
               ylim=(0, max(*zero, *cnn) * 1.29))
    scores.legend(loc="upper left", frameon=False, ncols=2, fontsize=8.5)
    for x, a, b in zip(positions, zero, cnn):
        scores.text(x - .17, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        scores.text(x + .17, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    paired.axvline(0, color="#536679", lw=1, ls="--")
    paired.hlines(positions, limits[:, 0], limits[:, 1], color="#236A83", lw=2)
    paired.scatter(gain, positions, color="#236A83", zorder=3)
    paired.set(yticks=positions, yticklabels=[f"{h} min" for h in horizons], ylim=(1.5, -.5),
               xlabel="Zero-return MAE − CNN MAE (bp)")
    for panel, axis in [(scores, "y"), (paired, "x")]:
        panel.spines[["top", "right"]].set_visible(False)
        panel.set_axisbelow(True)
        panel.grid(axis=axis, color="#E0E6EB", linewidth=.8)
    fig.text(.09, .065, "CNN averages three seeds. Intervals pair forecasts in blocks of ten announcement dates.", fontsize=9)
    fig.savefig(args.output, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
