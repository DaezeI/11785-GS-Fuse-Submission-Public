#!/usr/bin/env python3
"""Render the matched-news gains against three R3 inference controls."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
metrics = json.loads((root / "metrics.json").read_text())
labels = ["Shuffled articles", "Article tokens zeroed", "Fusion bypassed"]
keys = ["shuffled_text", "zeroed_article_tokens", "fusion_bypass_ts_inst"]
comparisons = [metrics["paired_comparisons"][key] for key in keys]
points = [row["mae_gain_pct"] for row in comparisons]
intervals = [row["mae_date_block_ci"]["ci95_pct"] for row in comparisons]
lower = [point - bounds[0] for point, bounds in zip(points, intervals)]
upper = [bounds[1] - point for point, bounds in zip(points, intervals)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "#213631"})
fig, ax = plt.subplots(figsize=(9.3, 4.0), dpi=150)
background = "#f8faf9"
fig.patch.set_facecolor(background)
ax.set_facecolor(background)
fig.subplots_adjust(left=0.26, right=0.93, top=0.66, bottom=0.28)
fig.text(0.07, 0.88, "Correct and shuffled articles gave almost identical error",
         fontsize=17, fontweight="bold")
fig.text(0.07, 0.76, "R3 return forecasts on the same 2,236 test events",
         fontsize=11, color="#49645d")
ax.axvline(0, color="#546d65", linewidth=1.2)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#dce6e1")
for position, point, left, right, color in zip(
    (2, 1, 0), points, lower, upper, ("#087565", "#8a6152", "#396784")
):
    ax.errorbar(point, position, xerr=[[left], [right]], fmt="o", color=color,
                capsize=4, markersize=7, elinewidth=2, zorder=3)
ax.annotate(f"{points[0]:+.4f}%", xy=(points[0], 2), xytext=(12, -2),
            textcoords="offset points", va="center", color="#087565", fontweight="bold")
ax.set_yticks([2, 1, 0], labels)
ax.set_ylim(-0.55, 2.55)
ax.set_xlim(-0.48, 0.34)
ticks = [-0.4, -0.2, 0, 0.2]
ax.set_xticks(ticks, [f"{tick:+.1f}%" if tick else "0%" for tick in ticks])
ax.tick_params(axis="both", length=0, pad=8, colors="#49645d")
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(0.07, 0.115, "Positive values favor correct articles. Lines show paired 95% intervals.",
         fontsize=10, color="#49645d")
fig.text(0.07, 0.066, "Error is mean absolute error across all price channels and forecast steps.",
         fontsize=9, color="#49645d")
fig.savefig(root / "article-comparison.png", facecolor=background)
