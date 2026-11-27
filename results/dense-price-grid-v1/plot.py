"""Render training and common-validation curves from aggregate CSV metrics."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

HERE = Path(__file__).resolve().parent
curves = pd.read_csv(HERE / "learning-curves.csv")
metrics = json.loads((HERE / "metrics.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
for stride, color in ((5, "#20639b"), (15, "#c44735")):
    part = curves[curves.stride_minutes.eq(stride)]
    label = f"{stride}-minute training grid"
    axes[0].plot(part.epoch, part.train_smooth_l1, color=color, label=label)
    axes[1].plot(part.epoch, part.val_mae20_bp, color=color, label=label)
    axes[2].plot(part.epoch, part.val_mae30_bp, color=color, label=label)
for ax, horizon in ((axes[1], 20), (axes[2], 30)):
    baseline = curves.loc[curves.epoch.eq(0), f"val_mae{horizon}_bp"].iloc[0]
    ax.axhline(baseline, color="0.4", ls="--", lw=1, label="Persistence")
axes[0].set(title="Full-training objective", ylabel="Standardized Smooth L1")
axes[1].set(title="Common validation: 20 minutes", ylabel="MAE (bp)")
axes[2].set(title="Common validation: 30 minutes", ylabel="MAE (bp)")
for ax in axes:
    ax.set_xlabel("Training epochs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.grid(alpha=.15)
axes[0].legend(frameon=False, fontsize=8)
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("none+CNN~none~none: rolling-price training; validation only", fontsize=12)
if "--chinese" in sys.argv:
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if not font.is_file():
        raise FileNotFoundError("Install a CJK font and configure its path for the Chinese export")
    properties = font_manager.FontProperties(fname=str(font))
    translations = {"Full-training objective": "完整训练集目标损失",
                    "Standardized Smooth L1": "标准化 Smooth L1",
                    "Common validation: 20 minutes": "共同验证集：预测 20 分钟",
                    "Common validation: 30 minutes": "共同验证集：预测 30 分钟",
                    "Training epochs": "训练轮数",
                    "5-minute training grid": "训练起点步长：5 分钟",
                    "15-minute training grid": "训练起点步长：15 分钟",
                    "Persistence": "持续性预测（Persistence）",
                    "none+CNN~none~none: rolling-price training; validation only": "CNN 全价格窗口训练：训练与验证曲线"}
    from matplotlib.text import Text
    for item in fig.findobj(match=Text):
        size = item.get_fontsize()
        item.set_text(translations.get(item.get_text(), item.get_text()))
        item.set_fontproperties(properties)
        item.set_fontsize(size)
    filename = "learning-curves-zh.png"
else:
    filename = "learning-curves.png"
fig.savefig(HERE / filename, dpi=180)
