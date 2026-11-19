#!/usr/bin/env python3
"""Compare compact price-only training with and without a close-direction penalty."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.improvement.compact_path_metrics import FORECAST_BARS, HORIZONS, metrics

SCHEMA = "gs-fuse-compact-direction-ablation-v1"
MODES = ("smooth_l1", "smooth_l1_plus_direction")
DIRECTION_WEIGHT = 1.0
BATCH_SIZE = 32
WIDTH = 32
CONTEXT = 35
SNAPSHOT = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"
MANIFEST = "1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1"
TIMING = "category4_1400_strict_before"
SOURCE_PATHS = (
    "scripts/improvement/train_compact_direction_ablation.py",
    "scripts/improvement/train_compact_path.py",
    "scripts/improvement/train_compact_raw.py",
    "scripts/improvement/compact_path_metrics.py",
    "scripts/improvement/prepare_raw_inputs.py",
    "scripts/freeze_c1_inputs.py",
    "data/c1_protocol.py",
    "data/forecast_task.py",
    "model/compact_return.py",
)


def direction_penalty(pred, target):
    """Project-style BCE on scaled close returns, restricted to nonzero actual signs."""
    import torch.nn.functional as F
    endpoints = [h - 1 for h in HORIZONS]
    actual = target[:, endpoints, 3]
    forecast = pred[:, endpoints, 3]
    eligible = actual.ne(0)
    # Equivalent to BCEWithLogits(forecast, actual > 0); target sign zero is excluded.
    terms = F.softplus(-actual.sign() * forecast).masked_fill(~eligible, 0)
    return (terms.sum(dim=1) / eligible.sum(dim=1).clamp_min(1)).mean()


def objective(pred, target, mode):
    import torch.nn.functional as F
    base = F.smooth_l1_loss(pred, target)
    if mode == MODES[0]:
        return base
    if mode == MODES[1]:
        return base + DIRECTION_WEIGHT * direction_penalty(pred, target)
    raise ValueError(f"Unsupported objective: {mode}")


def fit(model, train_xy, val_xy, args, seed, mode, torch, predict, state_cpu):
    from torch.utils.data import DataLoader, TensorDataset
    tx, ty = train_xy
    vx, vy = val_xy
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    best = float((predict(model, vx, args.device) - vy).abs().mean())
    best_state, best_epoch, stale = state_cpu(model), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for px, target in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(px.to(args.device))
            loss = objective(pred, target.to(args.device), mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        score = float((predict(model, vx, args.device) - vy).abs().mean())
        print(f"{mode} seed={seed} epoch={epoch} scaled_val_path_mae={score:.6f}", flush=True)
        if not math.isfinite(score):
            raise ValueError("Non-finite validation score")
        if score < best:
            best, best_state, best_epoch, stale = score, state_cpu(model), epoch, 0
        else:
            stale += 1
        if stale == 5:
            break
    model.load_state_dict(best_state)
    return epoch, best_epoch


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_out(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(truth, scores, seeds):
    baseline, direction = scores[MODES[0]], scores[MODES[1]]
    paired = []
    for seed, p0, p1 in zip(seeds, baseline, direction, strict=True):
        m0, m1 = metrics(truth, p0), metrics(truth, p1)
        eligible = np.count_nonzero(truth[:, [h - 1 for h in HORIZONS], 3], axis=0)
        hit_delta = [100 * (b - a) if a is not None and b is not None else None
                     for a, b in zip(m0["close_endpoint_direction_hit_rate"],
                                     m1["close_endpoint_direction_hit_rate"], strict=True)]
        paired.append(dict(seed=seed, smooth_l1=m0, smooth_l1_plus_direction=m1,
                           improvement_bp=m0["path_mae_bp"] - m1["path_mae_bp"],
                           direction_hit_rate_change_pp=hit_delta, direction_eligible=eligible.tolist()))
    means = {mode: {"path_mae_bp": float(np.mean([pair[mode]["path_mae_bp"] for pair in paired])),
                    "close_endpoint_direction_hit_rate": [
                        float(np.mean(values)) if all(x is not None for x in values) else None
                        for values in zip(*[pair[mode]["close_endpoint_direction_hit_rate"] for pair in paired])
                    ]}
             for mode in MODES}
    means["improvement_bp"] = float(np.mean([p["improvement_bp"] for p in paired]))
    means["direction_hit_rate_change_pp"] = [
        float(np.mean(values)) if all(x is not None for x in values) else None
        for values in zip(*[pair["direction_hit_rate_change_pp"] for pair in paired])
    ]
    return paired, means


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 30 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use 1–30 epochs and unique seeds from 0, 1, 2")
    args.seeds.sort()
    snapshot, out = args.snapshot_root.resolve(), args.out.resolve()
    if out.exists() or out == snapshot or snapshot in out.parents:
        parser.error("Use a fresh output directory outside the frozen snapshot")

    import torch
    from scripts.improvement.train_compact_raw import git_revision, load_c1, predict, state_cpu
    from scripts.improvement.train_compact_path import CompactPath, PathForecast, price_tensors

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    started = time.perf_counter()
    print("Verifying canonical C1 and preparing common train/validation price tensors.", flush=True)
    data, snapshot_sha, manifest_sha = load_c1(snapshot, seed=0, exploratory=False)
    if (snapshot_sha, manifest_sha) != (SNAPSHOT, MANIFEST):
        raise ValueError("Canonical screened C1 snapshot and manifest are required")
    scale = float(data.task_config.scale_bp)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training-only return scale")
    train, val = (price_tensors(ds, scale) for ds in (data.train_set, data.vali_set))
    val_ids = list(data.vali_set.event_ids)
    if (len(data.train_set) != 6167 or len(val_ids) != 2302 or len(set(val_ids)) != 2302):
        raise ValueError("Unexpected canonical train/validation cohort")
    truth = val[1].numpy() * scale
    cohort = {
        "train_events": len(data.train_set), "validation_events": len(val_ids),
        "validation_event_ids_sha256": sha("\n".join(val_ids).encode("utf-8")),
        "train_event_ids_sha256": sha("\n".join(data.train_set.event_ids).encode("utf-8")),
    }
    source_sha256 = {path: sha((ROOT / path).read_bytes().replace(b"\r\n", b"\n"))
                     for path in SOURCE_PATHS}
    commit, dirty = git_revision()
    out.mkdir(parents=True, exist_ok=False)
    results, scores = [], {mode: [] for mode in MODES}
    settings = dict(task="c1-ohlc-return-path-v1", context_bars=CONTEXT, forecast_bars=FORECAST_BARS,
                    channels=["open", "high", "low", "close"], horizons=list(HORIZONS), width=WIDTH,
                    snapshot_content_sha256=snapshot_sha, manifest_sha256=manifest_sha, timing_policy=TIMING,
                    device=args.device, torch_version=str(torch.__version__), batch_size=BATCH_SIZE,
                    max_epochs=args.epochs, optimizer="AdamW", learning_rate=0.001, weight_decay=0.001,
                    gradient_clip=1.0, patience=5, scale_bp=scale, scale_fit="canonical training targets only",
                    source_commit=commit, source_dirty=dirty, source_sha256=source_sha256,
                    reporting_units="basis-point log return relative to last observed close",
                    direction_logits="endpoint prediction in train-scale units; temperature is scale_bp",
                    direction_weight=DIRECTION_WEIGHT,
                    direction_loss="mean per-event softplus(-sign(actual)*logit), excluding zero actuals",
                    selection="minimum validation full-path OHLC return MAE, epoch 0 eligible",
                    release_timestamps_verified=False, validation_only=True, test_scored=False,
                    uses_text=False, **cohort)
    for seed in args.seeds:
        same_init = None
        for mode in MODES:
            print(f"Training {mode} / seed {seed}", flush=True)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            run_started = time.perf_counter()
            model = PathForecast(CompactPath(WIDTH))
            if same_init is None:
                same_init = state_cpu(model)
            else:
                if not all(torch.equal(value, same_init[key]) for key, value in model.state_dict().items()):
                    raise ValueError("Paired objectives did not start with identical parameters")
            model.to(args.device)
            epochs_run, best_epoch = fit(model, train, val, args, seed, mode, torch, predict, state_cpu)
            if args.device == "cuda":
                torch.cuda.synchronize()
            forecast = predict(model, val[0], args.device).numpy() * scale
            if not np.isfinite(forecast).all():
                raise ValueError("Non-finite validation forecasts")
            scores[mode].append(forecast)
            checkpoint = out / f"{mode}-seed-{seed}.pt"
            run_record = dict(mode=mode, seed=seed, epochs_run=epochs_run, best_epoch=best_epoch,
                              elapsed_seconds=time.perf_counter() - run_started,
                              cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                                         if args.device == "cuda" else None))
            torch.save({"state_dict": state_cpu(model), "configuration": settings | run_record}, checkpoint)
            run_record["checkpoint_sha256"] = sha(checkpoint.read_bytes())
            results.append(run_record)
            del model
    paired, means = compare(truth, scores, args.seeds)
    np.savez_compressed(out / "validation-predictions.npz", event_ids=np.asarray(val_ids),
                        forecast_anchor_index=np.asarray([r["anchor"] for r in data.vali_set.windows], dtype=np.int64),
                        targets_bp=truth, seed_order=np.asarray(args.seeds),
                        **{mode + "_bp": np.stack(scores[mode]) for mode in MODES})
    summary = dict(schema=SCHEMA, **settings, seeds=args.seeds, runs=results, comparisons=paired, means=means,
                   validation_zero=metrics(truth, np.zeros_like(truth)), elapsed_seconds=time.perf_counter()-started)
    json_out(out / "direction-ablation-summary.json", summary)
    print(json.dumps(means, indent=2), flush=True)
    print(f"Saved {out / 'direction-ablation-summary.json'}", flush=True)


if __name__ == "__main__":
    main()
