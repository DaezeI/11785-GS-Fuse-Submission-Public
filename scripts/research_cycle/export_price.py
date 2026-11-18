#!/usr/bin/env python3
"""Export public aggregates from verified price-control analyses; never copies raw predictions.

Example: python scripts/research_cycle/export_price.py --zero ANALYSIS.json
 --bypass ANALYSIS.json --config resolved_config.json --checkpoint-sha SHA
 --seed 1 --output results/screened-price-seed1-v1
Use --historical-replay only for testing against the completed seed-0 evidence;
this cannot produce a seed-1 record.
"""
import argparse, csv, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path


def main():
    P = argparse.ArgumentParser(description=__doc__)
    for name in (
        "zero",
        "bypass",
        "config",
        "fingerprints",
        "completion",
        "checkpoint-sha",
        "output",
    ):
        P.add_argument("--" + name, required=True)
    P.add_argument("--seed", type=int, default=1)
    P.add_argument("--historical-replay", action="store_true")
    a = P.parse_args()
    if (a.historical_replay and a.seed != 0) or (
        not a.historical_replay and a.seed <= 0
    ):
        P.error("expected a positive seed, or seed 0 with --historical-replay")
    config = json.loads(Path(a.config).read_text())
    expected_config = {
        "seed": a.seed,
        "forecast_task": "price",
        "c1_timing_policy": "category4_1400_strict_before",
        "ts_backbone": "moment",
        "text_backbone": "llama",
        "raw_ts_residual": True,
        "learnable_last_value": False,
        "batch_size": 8,
        "sw_stride": 10,
        "stage1_epochs": 2,
        "stage2_epochs": 2,
        "num_epochs": 2,
        "tf32": True,
        "no_4bit": True,
        "fusion_type": "crossattn",
        "gate_type": "softmax",
        "source_commit": "bb09396881cc92c5e654c4210944ce5c6da17ced",
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(f"resolved config mismatch for {key}")
    expected_hashes = {
        "snapshot_content_sha256": "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf",
        "manifest_sha256": "1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1",
        "price_sha256": "6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a",
        "code_commit": expected_config["source_commit"],
    }
    fingerprints = json.loads(Path(a.fingerprints).read_text())
    if any(fingerprints.get(k) != v for k, v in expected_hashes.items()):
        raise ValueError("frozen input fingerprints mismatch")
    completion = json.loads(Path(a.completion).read_text())
    if completion.get("stages") != {"smoke": 0, "full": 0, "zero": 0, "bypass": 0}:
        raise ValueError("all four job exit statuses must be verified zero")
    sha = a.checkpoint_sha.strip()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid checkpoint SHA-256")
    reports = {
        k: json.loads(Path(p).read_text())
        for k, p in [
            ("zeroed_article_tokens", a.zero),
            ("fusion_bypass_ts_inst", a.bypass),
        ]
    }
    for mode, report in reports.items():
        m = report["metadata"]
        if (
            report["events"] != 2236
            or report["shuffles"] != 3
            or report["missing_provenance"]
            or m["checkpoint_id"] != sha
            or m["no_text_definition"] != mode
            or m["seed"] != 23
            or m["reference_source_commit"]
            != "bb09396881cc92c5e654c4210944ce5c6da17ced"
            or m["smoke_subset"]
        ):
            raise ValueError("incomplete or mismatched control provenance: " + mode)
        if [x["seed"] for x in m["shuffle_protocols"]] != [23, 24, 25]:
            raise ValueError("unexpected shuffle seeds")
    source = {"events": 2236, "distinct_dates": 845, "shuffles": 3, "horizons": {}}
    for h in ("35", "70", "140"):
        z, b = (
            reports[k]["horizons"][h]["all"]
            for k in ("zeroed_article_tokens", "fusion_bypass_ts_inst")
        )
        if any(x["events"] != 2236 or x["distinct_dates"] != 845 for x in (z, b)):
            raise ValueError("event population mismatch")
        if any(
            not math.isclose(z["mse"][k], b["mse"][k], rel_tol=1e-7, abs_tol=1e-12)
            for k in ("real_text", "shuffled_text")
        ):
            raise ValueError(
                "real/shuffle predictions differ between control evaluations"
            )
        source["horizons"][h] = {"zeroed_article_tokens": z, "fusion_bypass_ts_inst": b}
    out = Path(a.output)
    if a.historical_replay and "screened-price-seed" in str(out):
        raise ValueError("historical replay cannot write a replication result")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_metrics.json").write_text(json.dumps(source, indent=2) + "\n")
    rows = []
    for h, modes in source["horizons"].items():
        for mode, key in [
            ("zeroed_article_tokens", "shuffled_text"),
            ("zeroed_article_tokens", "zero_text"),
            ("fusion_bypass_ts_inst", "zero_text"),
        ]:
            v = modes[mode]
            c = v["comparisons"][key]
            low, high = c["ci95_pct"]
            rows.append(
                {
                    "horizon": int(h),
                    "control": "shuffled_text" if key == "shuffled_text" else mode,
                    "events": v["events"],
                    "distinct_dates": v["distinct_dates"],
                    "real_mse": v["mse"]["real_text"],
                    "control_mse": v["mse"][key],
                    "real_mse_gain_pct": c["real_text_mse_gain_pct"],
                    "ci95_low_pct": low,
                    "ci95_high_pct": high,
                }
            )
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    import os, tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gsfuse-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    for ax, (control, title) in zip(
        axes,
        [
            ("shuffled_text", "vs shuffled articles"),
            ("zeroed_article_tokens", "vs zero article tokens"),
            ("fusion_bypass_ts_inst", "vs fusion bypass"),
        ],
    ):
        for r in [x for x in rows if x["control"] == control]:
            y = r["real_mse_gain_pct"]
            lo = r["ci95_low_pct"]
            hi = r["ci95_high_pct"]
            ax.errorbar(
                r["horizon"],
                y,
                yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                fmt="o",
                color="#165b86",
                capsize=4,
            )
        ax.axhline(0, color="#555", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Forecast prefix (bars)", xticks=[35, 70, 140])
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel(
        "Real-text MSE improvement (%)\nPositive favors correctly matched text"
    )
    fig.suptitle(f"Screened C1 price model: seed {a.seed} paired controls")
    fig.supxlabel(
        "95% date-block bootstrap intervals; one checkpoint; 2,236 events", fontsize=8
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"text-control-gains.{ext}", dpi=180)
    plt.close(fig)
    h = source["horizons"]["140"]["zeroed_article_tokens"]
    comp = h["comparisons"]["shuffled_text"]
    lo, hi = comp["ci95_pct"]
    label = "Historical seed-0 replay" if a.historical_replay else "Seed-1 replication"
    finding = f"{label}: at H140, real-text MSE {h['mse']['real_text']:.8f}, shuffled MSE {h['mse']['shuffled_text']:.8f}, real-text gain {comp['real_text_mse_gain_pct']:.4f}% (95% date-block interval [{lo:.4f}%, {hi:.4f}%])."
    if lo <= 0 <= hi:
        finding += " The interval includes zero, so this checkpoint does not establish a text advantage."
    else:
        finding += " The interval excludes zero for this checkpoint; this does not establish generality across seeds."
    limitations = [
        "Intervals are conditional on one checkpoint and three shuffle draws; two seeds do not establish generality.",
        "Retained article publication times remain unverified.",
        "Shuffled donors can come from later test dates; this is a diagnostic placebo.",
        "Article-free controls are inference interventions, not separately trained baselines.",
    ]
    provenance = {
        "run_id": (
            "historical-seed0-replay"
            if a.historical_replay
            else f"screened-price-seed{a.seed}-v1"
        ),
        "status": "historical_replay" if a.historical_replay else "completed",
        "code_commit": "bb09396881cc92c5e654c4210944ce5c6da17ced",
        "checkpoint_sha256": sha,
        "snapshot_sha256": "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf",
        "seed": a.seed,
        "shuffle_seeds": [23, 24, 25],
        "hardware": "NVIDIA H200",
        "training": "batch 8; two epochs per stage; stride 10; TF32; no 4-bit; cross-attention; softmax gate; raw TS residual; no learnable last value",
        "events": 2236,
        "distinct_dates": 845,
        "uncertainty": "1000 date-block bootstrap replicates, block length 20 distinct dates",
        "limitations": limitations,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "summary.md").write_text(
        "# " + label + "\n\n" + finding + "\n\n" + " ".join(limitations) + "\n"
    )
    if not a.historical_replay:
        root = f"https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/main/results/screened-price-seed{a.seed}-v1/"
        raw = f"https://raw.githubusercontent.com/DaezeI/11785-GS-Fuse-Submission-Public/main/results/screened-price-seed{a.seed}-v1/"
        record = {
            "id": f"screened-price-seed{a.seed}-v1",
            "title": f"Screened price seed-{a.seed} replication",
            "kind": "Experiment",
            "status": "completed",
            "owner": "Nicole",
            "question": "Does the near-zero correct-versus-shuffled text gap persist with a second training seed?",
            "configuration": provenance["training"],
            "finding": finding,
            "limitations": " ".join(limitations),
            "figure": {
                "url": raw + "text-control-gains.png",
                "alt": "Paired text-control improvements with date-block intervals across three forecast horizons",
            },
            "evidence": [
                {"label": "Result summary", "url": root + "summary.md"},
                {"label": "Metrics", "url": root + "metrics.csv"},
                {"label": "Provenance", "url": root + "provenance.json"},
            ],
            "related": ["screened-price-v1"],
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        (out / "candidate-record.json").write_text(json.dumps(record, indent=2) + "\n")
    checks = [
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "SHA256SUMS"
    ]
    (out / "SHA256SUMS").write_text("\n".join(checks) + "\n")
    print(finding)


if __name__ == "__main__":
    main()
