#!/usr/bin/env python3
"""Paired price-only L35/L140 ablation on screened C1. Validation only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.improvement.compact_path_metrics import CHANNELS, FORECAST_BARS, HORIZONS, metrics

SHORT, LONG = 35, 140
ROLES = ("train", "vali", "test")
FIT_ROLES = ROLES[:2]
COUNTS = {"train": 6167, "vali": 2302, "test": 2236}
EXPECTED_SNAPSHOT = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"
EXPECTED_MANIFEST = "1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1"
TIMING = "category4_1400_strict_before"
WIDTH = 32


def cover(rows, n, start_field):
    delta = np.zeros(n + 1, dtype=np.int32)
    for row in rows:
        delta[row[start_field]] += 1
        delta[row["stop"]] -= 1
    return np.cumsum(delta[:-1], dtype=np.int64) > 0


def shared_bars(coverages):
    return {f"{a}/{b}": int(np.count_nonzero(coverages[a] & coverages[b]))
            for i, a in enumerate(ROLES) for b in ROLES[i + 1:]}


def eligible_rows(canonical, n):
    """Retain L140 windows that do not touch any other role's original L35 window.

    Test windows are protected, unmodified and never scored. Original windows in
    the other role are protected even if those rows are dropped from the ablation.
    The L35 arm is trained and scored on exactly the retained L140 cohort.
    """
    original = {role: cover(canonical[role], n, "short_start") for role in ROLES}
    canonical_shared = shared_bars(original)
    if any(canonical_shared.values()):
        raise ValueError(f"Canonical L35 windows already overlap: {canonical_shared}")
    selected, excluded = {}, {}
    for role in FIT_ROLES:
        selected[role], excluded[role] = [], []
        other = [r for r in ROLES if r != role]
        prefix = {r: np.r_[0, np.cumsum(original[r], dtype=np.int64)] for r in other}
        for row in canonical[role]:
            start, stop = row["anchor"] - LONG + 1, row["stop"]
            reasons = (["insufficient_price_history"] if start < 0 else
                       [f"touches_canonical_{r}_bars" for r in other
                        if prefix[r][stop] != prefix[r][start]])
            if reasons:
                excluded[role].append({"event_id": row["event_id"], "reasons": reasons})
            else:
                selected[role].append(dict(row, long_start=start))
        if not selected[role]:
            raise ValueError(f"No {role} events eligible for paired L35/L140 context evaluation")
    overlaps, bars = {}, {}
    for label, start_field in (("L35", "short_start"), ("L140", "long_start")):
        coverage = {r: cover(selected[r], n, start_field) for r in FIT_ROLES}
        coverage["test"] = original["test"]
        overlaps[label] = shared_bars(coverage)
        bars[label] = {r: int(np.count_nonzero(coverage[r])) for r in ROLES}
        if any(overlaps[label].values()):
            raise ValueError(f"Paired {label} windows overlap: {overlaps[label]}")
    return selected, excluded, canonical_shared, overlaps, bars


def event_windows(manifest, stamps, event_key, window_for_event):
    canonical, ids = {}, set()
    for role in ROLES:
        canonical[role] = []
        for entry in manifest[role + "_events"]:
            category, date, event_id = event_key(entry)
            if event_id in ids:
                raise ValueError(f"Duplicate canonical event: {event_id}")
            ids.add(event_id)
            window = window_for_event(category, date, stamps, SHORT, FORECAST_BARS, TIMING)
            if window is None:
                raise ValueError(f"Canonical event missing its baseline price window: {event_id}")
            start, anchor, stop = window
            canonical[role].append(dict(event_id=event_id, short_start=start, anchor=anchor, stop=stop))
        if len(canonical[role]) != COUNTS[role]:
            raise ValueError(f"Unexpected canonical event count: {role}")
    return canonical


def make_arrays(values, rows, prices_to_returns):
    inputs, targets = [], []
    for row in rows:
        start, anchor, stop = row["long_start"], row["anchor"], row["stop"]
        if anchor - start + 1 != LONG or anchor - row["short_start"] + 1 != SHORT or stop - anchor - 1 != FORECAST_BARS:
            raise ValueError("Forecast origin or target window has inconsistent geometry")
        close = float(values[anchor, 3])
        inputs.append(prices_to_returns(values[start:anchor + 1], close).T)
        targets.append(prices_to_returns(values[anchor + 1:stop], close))
    history, truth = np.stack(inputs), np.stack(targets)
    if (history.shape != (len(rows), CHANNELS, LONG)
            or truth.shape != (len(rows), FORECAST_BARS, CHANNELS)
            or not np.isfinite(history).all() or not np.isfinite(truth).all()):
        raise ValueError("Unexpected paired price tensor dimensions or values")
    return history, truth


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    if not (1 <= args.epochs <= 30) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 1–30 epochs and unique seeds from 0, 1, 2")
    args.seeds.sort()
    snapshot, out = args.snapshot_root.resolve(), args.out.resolve()
    if out.exists() or out == snapshot or snapshot in out.parents:
        parser.error("Choose a fresh output directory outside the frozen snapshot")

    import torch
    from data.c1_protocol import event_key, load_price_timeline, window_for_event
    from data.forecast_task import prices_to_log_returns_bp, resolve_scale_bp
    from scripts.freeze_c1_inputs import verify
    from scripts.improvement.train_compact_path import CompactPath, PathForecast, fit
    from scripts.improvement.train_compact_raw import git_revision, predict, state_cpu

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    start = time.perf_counter()
    print("Verifying canonical C1 inputs and auditing both context lengths (no test scoring).", flush=True)
    receipt = verify(snapshot)
    manifest_bytes = (snapshot / "manifest.json").read_bytes()
    if (receipt["content_sha256"] != EXPECTED_SNAPSHOT
            or receipt["manifest_sha256"] != EXPECTED_MANIFEST
            or sha(manifest_bytes) != EXPECTED_MANIFEST):
        raise ValueError("Ablation requires canonical screened C1 snapshot and manifest bytes")
    manifest = json.loads(manifest_bytes)
    stamps, values, quality = load_price_timeline(snapshot / "series" / "SP500.csv")
    canonical = event_windows(manifest, stamps, event_key, window_for_event)
    paired, excluded, original_overlap, paired_overlap, covered_bars = eligible_rows(canonical, len(stamps))
    arrays = {r: make_arrays(values, paired[r], prices_to_log_returns_bp) for r in FIT_ROLES}
    scale = resolve_scale_bp(arrays["train"][1], horizons=HORIZONS, close_channel=3)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid paired-training return scale")
    role_hashes = {r: sha("\n".join(x["event_id"] for x in paired[r]).encode("utf-8")) for r in FIT_ROLES}
    selection = {
        "rule": "Protect all original L35+H140 bars of every other role, keep complete L140 histories "
                "with no collision, and use the exact same retained rows for both contexts.",
        "canonical_counts": COUNTS,
        "retained": {r: len(paired[r]) for r in FIT_ROLES},
        "excluded_counts": {r: len(excluded[r]) for r in FIT_ROLES},
        "excluded_reason_counts": {r: dict(Counter(reason for e in excluded[r] for reason in e["reasons"]))
                                   for r in FIT_ROLES},
        "event_ids_sha256": role_hashes,
        "canonical_shared_bars": original_overlap,
        "matched_context_shared_bars": paired_overlap,
        "matched_context_covered_bars": covered_bars,
        "scope": "Matched eligible subset of canonical C1; not necessarily the full canonical population.",
        "test_protection": "All 2,236 canonical L35+H140 test windows are protected. "
                           "No test samples or targets were constructed and no test score was computed.",
    }
    cohort = dict(**selection, excluded=excluded, rows={
        r: [dict(row, forecast_origin=int(stamps[row["anchor"]]),
                 first_target=int(stamps[row["anchor"] + 1]), last_target=int(stamps[row["stop"] - 1]))
            for row in paired[r]] for r in FIT_ROLES
    })
    out.mkdir(parents=True, exist_ok=False)
    cohort_path = out / "context-ablation-cohort.json"
    write_json(cohort_path, cohort)
    print(f"Common cohort: {len(paired['train'])}/{COUNTS['train']} train; "
          f"{len(paired['vali'])}/{COUNTS['vali']} validation. Training both contexts from scratch.", flush=True)
    source_paths = (
        Path(__file__), ROOT / "model/compact_return.py", ROOT / "scripts/improvement/train_compact_path.py",
        ROOT / "scripts/improvement/compact_path_metrics.py", ROOT / "scripts/improvement/train_compact_raw.py",
        ROOT / "scripts/freeze_c1_inputs.py", ROOT / "data/c1_protocol.py", ROOT / "data/forecast_task.py",
    )
    source_sha256 = {path.relative_to(ROOT).as_posix(): sha(path.read_bytes().replace(b"\r\n", b"\n"))
                     for path in source_paths}
    commit, dirty = git_revision()
    tensor_pairs = {role: (torch.from_numpy(x).float() / scale, torch.from_numpy(y).float() / scale)
                    for role, (x, y) in arrays.items()}
    truth = arrays["vali"][1]
    runs, saved = [], {SHORT: [], LONG: []}
    for seed in args.seeds:
        for length in (SHORT, LONG):
            print(f"Training L{length} / seed {seed} / price only", flush=True)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            arm_start = time.perf_counter()
            model = PathForecast(CompactPath(WIDTH)).to(args.device)
            arm = {r: (x[..., -length:].contiguous(), y) for r, (x, y) in tensor_pairs.items()}
            opts = SimpleNamespace(mode=f"L{length}", device=args.device, epochs=args.epochs, seed=seed,
                                   batch_size=32)
            epochs_run, best_epoch, _ = fit(model, arm["train"], arm["vali"], None, None, opts)
            forecast = predict(model, arm["vali"][0], args.device).numpy() * scale
            if not np.isfinite(forecast).all():
                raise ValueError("Non-finite validation forecasts")
            saved[length].append(forecast)
            run = dict(seed=seed, context_bars=length, epochs_run=epochs_run, best_epoch=best_epoch,
                       validation=metrics(truth, forecast), elapsed_seconds=time.perf_counter() - arm_start,
                       cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                                  if args.device == "cuda" else None))
            checkpoint = out / f"l{length}-seed-{seed}.pt"
            torch.save({"state_dict": state_cpu(model), "config": dict(
                seed=seed, context_bars=length, width=WIDTH, scale_bp=scale, forecast_bars=FORECAST_BARS,
                snapshot_content_sha256=EXPECTED_SNAPSHOT, manifest_sha256=EXPECTED_MANIFEST,
                cohort_event_ids_sha256=role_hashes, test_scored=False)}, checkpoint)
            run["checkpoint_sha256"] = sha(checkpoint.read_bytes())
            runs.append(run)
            del model
    comparisons = [
        dict(seed=seed, l35_mae_bp=short["validation"]["path_mae_bp"],
             l140_mae_bp=long["validation"]["path_mae_bp"],
             improvement_bp=short["validation"]["path_mae_bp"] - long["validation"]["path_mae_bp"])
        for seed, short, long in zip(args.seeds, runs[::2], runs[1::2])
    ]
    means = {k: float(np.mean([r[k] for r in comparisons]))
             for k in ("l35_mae_bp", "l140_mae_bp", "improvement_bp")}
    summary = dict(schema="gs-fuse-c1-price-context-ablation-v1", validation_only=True, test_scored=False,
                   forecast_bars=FORECAST_BARS, channels=["open", "high", "low", "close"], device=args.device,
                   torch_version=str(torch.__version__), timing_policy=TIMING, release_timestamps_verified=False,
                   snapshot_content_sha256=EXPECTED_SNAPSHOT, manifest_sha256=EXPECTED_MANIFEST,
                   price_timeline=quality, scale_bp=scale, scale_fit="retained training rows only",
                   seeds=args.seeds, batch_size=32, max_epochs=args.epochs, width=WIDTH,
                   source_commit=commit, source_dirty=dirty, source_sha256=source_sha256,
                   cohort_sha256=sha(cohort_path.read_bytes()), cohort=selection,
                   run_selection="minimum validation whole-path OHLC log-return MAE; epoch 0 eligible",
                   forecast_origin="same last observed close and target window in both paired contexts",
                   reporting_units="basis-point log return relative to last observed close",
                   validation_zero=metrics(truth, np.zeros_like(truth)), runs=runs, comparisons=comparisons,
                   means=means, elapsed_seconds=time.perf_counter() - start)
    rows = cohort["rows"]["vali"]
    np.savez_compressed(out / "validation-predictions.npz",
                        event_ids=np.asarray([r["event_id"] for r in rows]),
                        forecast_origins=np.asarray([r["forecast_origin"] for r in rows], dtype=np.int64),
                        targets_bp=truth, l35_bp=np.stack(saved[SHORT]), l140_bp=np.stack(saved[LONG]),
                        seed_order=np.asarray(args.seeds))
    write_json(out / "context-ablation-summary.json", summary)
    print(json.dumps(means, indent=2), flush=True)
    print(f"Saved {out / 'context-ablation-summary.json'}", flush=True)


if __name__ == "__main__":
    main()
