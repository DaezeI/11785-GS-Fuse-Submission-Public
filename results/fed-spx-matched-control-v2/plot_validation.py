"""Plot matched-cohort validation MAE for the frozen control and the baselines."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
scores = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))["results"]["all_text_verified"]["scores"]
methods = [("Persistence (zero return)", scores["zero_return"]),
           ("none+CNN~none~none (frozen)", scores["price_cnn_control"]["metrics_seed_average_forecast"]),
           ("none+DLinear~none~none", scores["dlinear"]["metrics_seed_average_forecast"]),
           ("none+Ridge~none~none", scores["ridge"]["metrics_seed_average_forecast"])]
colors = ["#8a8f98", "#2f6db5", "#4f9a6a", "#c7772b"]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
width = 0.2
for i, (label, m) in enumerate(methods):
    values = [m["20"]["mae_bp"], m["30"]["mae_bp"]]
    bars = ax.bar([x + (i - 1.5) * width for x in (0, 1)], values, width, label=label, color=colors[i])
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks([0, 1], ["20 minutes", "30 minutes"])
ax.set_ylabel("Weighted MAE (bp)")
ax.set_title("FED-PXTX-ALL-REL5-v2 validation: 149 releases, 139 release times")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "validation-mae.png", dpi=160)
