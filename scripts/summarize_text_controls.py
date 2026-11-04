#!/usr/bin/env python3
"""Reproduce descriptive text/no-text findings from the committed aggregates.

No price CSV, raw reports, model weights, GPU, or private Drive access needed.
This summarizes an existing experiment; it does not generate new predictions.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL_SOURCE = Path("task2_shuffled_text/outputs/task2_shuffled_text_control.json")
REFERENCE_SHA256 = "f8cc4db714057d213e8433ed3fd4edcae2ae0ad6ab1a4cf7b193586be84fffab"
REFERENCE_COMMIT = "02f2b742960384161a7ea0c79aacabb9eea43d9b"


def decompose(real, shuffled, zero):
    if not all(math.isfinite(v) for v in (real, shuffled, zero)):
        raise ValueError("MSE values must be finite")
    if min(real, shuffled, zero) < 0:
        raise ValueError("MSE values must be nonnegative")
    total = zero - real
    retained, matching = zero - shuffled, shuffled - real
    return {
        "real_mse": real, "shuffled_mse": shuffled, "zero_mse": zero,
        "real_vs_zero_gain_pct": 100 * total / zero if zero else None,
        "shuffle_vs_real_error_increase_pct": 100 * matching / real if real else None,
        "zero_minus_real_mse": total, "zero_minus_shuffled_mse": retained,
        "shuffled_minus_real_mse": matching,
        "benefit_retained_with_shuffled_pct": 100 * retained / total if total > 0 else None,
        "matching_gap_share_pct": 100 * matching / total if total > 0 else None,
    }


def summarize(data):
    rows = [{"metric": "overall", **decompose(*(data[k]["overall_mse"]
        for k in ("real_text", "shuffled_text", "zero_text")))}]
    for h in (35, 70, 140):
        vals = [data[k]["horizons"][f"H={h}"] for k in ("real_text", "shuffled_text", "zero_text")]
        rows.append({"metric": f"H={h}", **decompose(*(v["mse"] for v in vals)),
            "real_dhr_pct": 100 * vals[0]["dhr"], "shuffled_dhr_pct": 100 * vals[1]["dhr"],
            "zero_dhr_pct": 100 * vals[2]["dhr"],
            "zero_minus_real_dhr_pp": 100 * (vals[2]["dhr"] - vals[0]["dhr"])})
    return rows


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "gsfuse-text-value"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    y = np.arange(len(rows))
    retained = [100 * r["zero_minus_shuffled_mse"] / r["zero_mse"] if r["zero_mse"] else float("nan") for r in rows]
    matching = [100 * r["shuffled_minus_real_mse"] / r["zero_mse"] if r["zero_mse"] else float("nan") for r in rows]
    axes[0].barh(y, retained, label="Improvement retained after shuffling", color="#426C8B")
    axes[0].barh(y, matching, left=retained, label="Extra gap from correct matching", color="#D88A3A")
    axes[0].set(yticks=y, yticklabels=[r["metric"] for r in rows],
        xlabel="MSE reduction versus no text (%)", title="Most observed MSE benefit survives shuffling")
    axes[0].invert_yaxis()
    xx = np.arange(3)
    for offset, key, label, color in [(-.2, "real_dhr_pct", "Real text", "#426C8B"),
                                     (.2, "zero_dhr_pct", "No text", "#D88A3A")]:
        bars = axes[1].bar(xx + offset, [r[key] for r in rows[1:]], .4, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    axes[1].set(xticks=xx, xticklabels=[r["metric"] for r in rows[1:]], ylim=(0, 60),
        ylabel="Reported direction accuracy (%)", title="No text has higher reported direction accuracy")
    for ax, x, columns in ((axes[0], .065, 1), (axes[1], .56, 2)):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(x, .065),
                   fontsize=8, ncol=columns, frameon=False)
    fig.text(.5, .025, "Reanalysis of the uploaded aggregates; descriptive gaps, no new inference or confidence intervals.",
        ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .20, 1, 1))
    svg = out / "text_value.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(out / "text_value.png", dpi=160); plt.close(fig)


def main():
    default = ROOT / REL_SOURCE
    if not default.exists(): default = ROOT / "analysis/nov03/source" / REL_SOURCE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=default)
    p.add_argument("--out", type=Path, default=ROOT / "analysis/text_value")
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--source-commit", help="Optional revision for a custom input; otherwise hash-matched or unverified")
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text()); rows = summarize(data)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    try: source_path = str(args.input.resolve().relative_to(ROOT))
    except ValueError: source_path = args.input.name
    # The local audit snapshot holds the same public file at the pinned hash.
    if source_hash == REFERENCE_SHA256: source_path = str(REL_SOURCE)
    report = {"source_path": source_path,
        "source_commit": args.source_commit or (REFERENCE_COMMIT if source_hash == REFERENCE_SHA256 else "unverified"),
        "source_sha256": source_hash,
        "execution": "CPU arithmetic on supplied aggregate results; no new model inference",
        "rows": rows,
        "interpretation_limits": [
            "The decomposition is arithmetic sensitivity, not causal or semantic attribution.",
            "Overall and horizon MSE definitions are not verified; compare conditions within a metric only.",
            "No individual predictions or repeated shuffles were provided; significance and subgroup gains are unknown.",
            "The uploaded no-text implementation is not verified; it is distinct from the new explicit no-text controls."]}
    (args.out / "reported_text_analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (args.out / "reported_text_analysis.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    if not args.skip_plot: plot(rows, args.out)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
