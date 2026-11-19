"""Plot mean direction accuracy change by endpoint from the completed result."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
d = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
values = d["means"]["direction_hit_rate_change_pp"]
fig, ax = plt.subplots(figsize=(6.8, 3.6), layout="constrained")
ax.barh([2, 1, 0], values, color=["#a3473b", "#14766e", "#a3473b"], height=.5, zorder=3)
for y, v in zip([2,1,0], values):
    ax.text(v + (.03 if v >= 0 else -.03), y, f"{v:+.3f}",
            ha="left" if v >= 0 else "right", va="center", fontsize=10)
ax.set_yticks([2, 1, 0], ["H35", "H70", "H140"])
ax.axvline(0, color="#263b49", lw=1.2)
ax.set_xlim(-1.44, .30)
ax.set_xlabel("Change in close-direction hit rate (percentage points)")
ax.set_title("Direction penalty did not help validation accuracy", loc="left", fontweight="bold")
ax.grid(axis="x", color="#e4e9ed", zorder=0)
ax.spines[["top","right","left"]].set_visible(False)
fig.savefig(root / "direction-changes.png", dpi=165, facecolor="white")
