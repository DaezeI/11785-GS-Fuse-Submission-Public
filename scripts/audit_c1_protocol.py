#!/usr/bin/env python3
"""Inspect the exact C1 samples emitted by training, without loading a model."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.c1_protocol import build_c1_dataloaders


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="splits/manifests/purged_split_c1_macro_eras.json")
    p.add_argument("--event-dir", default="data/event")
    p.add_argument("--series", default="data/series/SP500.csv")
    p.add_argument("--timing-policy", choices=["legacy_all0830_floor",
                   "category4_1400_strict_before"], required=True)
    p.add_argument("--skip-missing", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--windows-csv", type=Path,
                   help="Save one row per exact emitted event window")
    args = p.parse_args()
    data = build_c1_dataloaders(args.manifest, args.event_dir, args.series,
                                timing_policy=args.timing_policy,
                                skip_missing=args.skip_missing)
    report = {**data.audit,
              "manifest_sha256": digest(args.manifest),
              "price_sha256": digest(args.series),
              "stage1_train_window_count": len(data.sw_train_set),
              "stage1_vali_window_count": len(data.sw_vali_set),
              "emitted_windows": {r: getattr(data, r + "_set").windows
                                  for r in ("train", "vali", "test")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k != "emitted_windows"}
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.windows_csv:
        args.windows_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.windows_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("role", "event_id", "start", "anchor", "stop",
                                                   "input_start", "target_end"), lineterminator="\n")
            writer.writeheader()
            for role, windows in report["emitted_windows"].items():
                writer.writerows({"role": role, **w} for w in windows)
    print(json.dumps(summary, indent=2))
    print(f"Saved C1 summary: {args.out}")
    if args.windows_csv:
        print(f"Saved exact emitted windows: {args.windows_csv}")


if __name__ == "__main__":
    main()
