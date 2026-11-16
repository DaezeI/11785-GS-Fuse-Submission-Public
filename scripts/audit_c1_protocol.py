#!/usr/bin/env python3
"""Inspect the exact C1 samples emitted by training, without loading a model."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.c1_protocol import build_c1_dataloaders
from data.forecast_task import reconstruct_prices
from scripts.freeze_c1_inputs import verify as verify_snapshot
from scripts.audit_article_datelines import audit as audit_datelines


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_returns(prices, anchor):
    return 10_000 * (np.log(prices.astype(np.float64)) - np.log(anchor))


def check_return_window(seq, target, raw_seq, raw_target, anchor, label):
    # The loader stores float32. Check against an independent float64 formula.
    for actual, prices in ((seq, raw_seq), (target, raw_target)):
        if not np.allclose(actual, expected_returns(prices, anchor), rtol=1e-6, atol=0.002):
            raise ValueError(f"Return transform mismatch: {label}")
        if not np.allclose(reconstruct_prices(actual, anchor), prices, rtol=1e-6, atol=0):
            raise ValueError(f"Price reconstruction mismatch: {label}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="splits/manifests/purged_split_c1_macro_eras.json")
    p.add_argument("--event-dir", default="data/event")
    p.add_argument("--series", default="data/series/SP500.csv")
    p.add_argument("--snapshot-root", type=Path,
                   help="Verify and audit a frozen snapshot instead of live inputs")
    p.add_argument("--timing-policy", choices=["legacy_all0830_floor",
                   "category4_1400_strict_before"], required=True)
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--forecast-task", choices=["price", "return_bp"], default="price")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--windows-csv", type=Path,
                   help="Save one row per exact emitted event window")
    args = p.parse_args()
    if args.out.exists() or (args.windows_csv and args.windows_csv.exists()):
        p.error("Audit output already exists; choose fresh versioned output paths")
    snapshot = None
    if args.snapshot_root:
        snapshot = verify_snapshot(args.snapshot_root)
        args.manifest = args.snapshot_root / "manifest.json"
        args.event_dir = args.snapshot_root / "event"
        args.series = args.snapshot_root / "series" / "SP500.csv"
    data = build_c1_dataloaders(args.manifest, args.event_dir, args.series,
                                timing_policy=args.timing_policy,
                                skip_missing=args.skip_missing, task_mode=args.forecast_task)
    report = {**data.audit,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "snapshot_content_sha256": snapshot["content_sha256"] if snapshot else None,
              "manifest_sha256": digest(args.manifest),
              "price_sha256": digest(args.series),
              "stage1_train_window_count": len(data.sw_train_set),
              "stage1_vali_window_count": len(data.sw_vali_set),
              "emitted_windows": {r: getattr(data, r + "_set").windows
                                  for r in ("train", "vali", "test")}}
    if snapshot:
        date_audit = audit_datelines(args.manifest, args.event_dir)
        summary = date_audit["counts"]
        if date_audit["candidates"] or any(key.endswith(":missing_summary") for key in summary):
            ids = [row["event_id"] for row in date_audit["candidates"][:10]]
            raise ValueError(f"Frozen snapshot has new date-screen findings: {summary}, {ids}")
        report["reuters_dateline_screen"] = summary
    if args.forecast_task == "return_bp":
        max_roundtrip_error = 0.0
        eligible = {role: {str(h): 0 for h in data.task_config.direction_horizons}
                    for role in ("train", "vali", "test")}
        for role in ("train", "vali", "test"):
            ds = getattr(data, role + "_set")
            for item, window in zip(ds.list, ds.windows):
                raw_seq, raw_target = item[3].numpy(), item[4].numpy()
                seq, target = item[6].numpy(), item[7].numpy()
                anchor = window["anchor_close"]
                if not np.isclose(raw_seq[-1, data.task_config.close_channel], anchor):
                    raise ValueError(f"Wrong close anchor: {window['event_id']}")
                check_return_window(seq, target, raw_seq, raw_target, anchor, window["event_id"])
                if abs(float(seq[-1, data.task_config.close_channel])) > 1e-4:
                    raise ValueError(f"Last input close is not zero: {window['event_id']}")
                max_roundtrip_error = max(max_roundtrip_error,
                    float(np.max(np.abs(reconstruct_prices(seq, anchor) - raw_seq))),
                    float(np.max(np.abs(reconstruct_prices(target, anchor) - raw_target))))
                for h in data.task_config.direction_horizons:
                    actual = target[int(h) - 1, data.task_config.close_channel]
                    eligible[role][str(h)] += int(abs(actual) > data.task_config.direction_deadband_bp)
        report["return_checks"] = {
            "event_windows_checked": sum(len(getattr(data, r + "_set")) for r in ("train", "vali", "test")),
            "stage1_train_windows_checked": len(data.sw_train_set),
            "stage1_vali_windows_checked": len(data.sw_vali_set),
            "max_price_roundtrip_abs_error": max_roundtrip_error,
            "direction_eligible_events": eligible,
        }
        for role in ("train", "vali"):
            ds = getattr(data, "sw_" + role + "_set")
            for i in range(len(ds)):
                raw_seq, raw_target, seq, target = ds[i]
                anchor = float(raw_seq[data.task_config.close_channel, -1])
                check_return_window(seq.T.numpy(), target.T.numpy(), raw_seq.T.numpy(),
                                    raw_target.T.numpy(), anchor, f"Stage-1 {role} {i}")
        train_targets = np.stack([item[7].numpy() for item in data.train_set.list])
        endpoints = [h - 1 for h in data.task_config.direction_horizons]
        scale = max(1.0, float(np.median(np.abs(
            train_targets[:, endpoints, data.task_config.close_channel]))))
        if data.task_config.scale_bp != scale:
            raise ValueError("Resolved return scale differs from the training-only median")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k != "emitted_windows"}
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.windows_csv:
        args.windows_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.windows_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("role", "event_id", "category", "anchor_close",
                                                   "start", "anchor", "stop", "input_start",
                                                   "target_end"), lineterminator="\n")
            writer.writeheader()
            for role, windows in report["emitted_windows"].items():
                writer.writerows({"role": role, **w} for w in windows)
    print(json.dumps(summary, indent=2))
    print(f"Saved C1 summary: {args.out}")
    if args.windows_csv:
        print(f"Saved exact emitted windows: {args.windows_csv}")


if __name__ == "__main__":
    main()
