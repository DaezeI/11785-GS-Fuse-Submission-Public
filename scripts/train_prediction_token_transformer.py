#!/usr/bin/env python3
"""Single-H200 batch calibration and bounded six-horizon Transformer training."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dense_price_endpoints import load_dense_cohorts
from model.prediction_token_transformer import PredictionTokenTransformer
from scripts.score_fed_spx_paired import bootstrap_gain

HORIZONS = (5, 10, 15, 20, 25, 30)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def largest_batch(limit, probe, quantum=256):
    """Find the largest accepted batch at the declared search resolution."""
    low, high = 0, None
    candidate = min(1024, limit)
    while True:
        if probe(candidate):
            low = candidate
            if candidate == limit:
                return candidate
            candidate = min(limit, candidate * 2)
        else:
            high = candidate
            break
    if low == 0:
        candidate = max(1, high // 2)
        while candidate:
            if probe(candidate):
                low = candidate
                break
            high = candidate
            if candidate == 1:
                raise RuntimeError("No training batch fits")
            candidate = max(1, candidate // 2)
    while high - low > quantum:
        candidate = ((low + high) // (2 * quantum)) * quantum
        if candidate <= low:
            break
        if probe(candidate):
            low = candidate
        else:
            high = candidate
    return low


def horizon_metrics(truth, prediction):
    if truth.shape != prediction.shape or truth.shape[1] != len(HORIZONS):
        raise ValueError("Expected matched six-horizon arrays")
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    mae = np.abs(truth - prediction).mean(0)
    return {"mean_mae_bp": float(mae.mean()), "by_horizon": {
        str(h): {"mae_bp": float(mae[i]), "direction_accuracy": float(np.mean(
            np.sign(truth[truth[:, i] != 0, i]) == np.sign(prediction[truth[:, i] != 0, i])))}
        for i, h in enumerate(HORIZONS)}}


def evaluate(model, x, t, truth, scale, batch=8192):
    model.eval()
    pieces = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(x), batch):
            pieces.append(model(x[start:start + batch], t[start:start + batch]).float().cpu().numpy())
    normalized = np.concatenate(pieces).astype(np.float64)
    prediction = normalized * scale
    error = np.abs(normalized - truth / scale)
    objective = np.where(error < 1, .5 * error ** 2, error - .5).mean()
    return prediction, {"smooth_l1": float(objective), **horizon_metrics(truth, prediction)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--max-updates", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--max-training-seconds", type=int, default=5400)
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    train, val, audit = load_dense_cohorts(a.prices, 5, target_steps=range(1, 7), include_time_offsets=True)
    audit.update(train_cohort="SPX-PX6-GRID5-v1", score_cohort="SPX-PX6-GRID5-v1 val",
                 time_embedding="sinusoidal UTC elapsed minutes / five, measured from forecast origin",
                 test_loaded=False)
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if a.audit_only:
        print(json.dumps(audit), flush=True)
        return
    torch.set_num_threads(8)
    input_scale = float(np.sqrt(np.mean(np.square(train.history.astype(np.float64)))))
    target_scale = np.sqrt(np.mean(np.square(train.returns), axis=0))
    normalized = a.out / "training-tensors.npz"
    np.savez(normalized, x=(train.history / input_scale).astype(np.float32),
             t=train.past_minutes, y=(train.returns / target_scale).astype(np.float32))
    initial = PredictionTokenTransformer(activation_checkpointing=True, attention_batch_chunk=8192)
    initial_path = a.out / "initial-state.pt"
    torch.save(initial.state_dict(), initial_path)
    architecture = "none+Transformer128x4~none~none"
    calibration = []

    def probe(batch, updates=3, activation_checkpointing=False):
        path = a.out / "calibration" / f"checkpoint-{int(activation_checkpointing)}-batch-{batch}-updates-{updates}.json"
        command = [sys.executable, str(ROOT / "scripts/profile_prediction_token_batch.py"),
                        "--data", str(normalized), "--initial", str(initial_path),
                        "--batch-size", str(batch), "--updates", str(updates), "--out", str(path)]
        if activation_checkpointing:
            command.append("--activation-checkpointing")
        subprocess.run(command, check=True)
        report = json.loads(path.read_text())
        calibration.append(report)
        (a.out / "batch-calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        return report["accepted"]

    chosen_batch = largest_batch(len(train.history), probe)
    activation_checkpointing = False
    if chosen_batch < len(train.history):
        with_checkpoint = largest_batch(len(train.history), lambda batch: probe(batch, activation_checkpointing=True))
        if with_checkpoint > chosen_batch:
            chosen_batch, activation_checkpointing = with_checkpoint, True
    while not probe(chosen_batch, updates=10, activation_checkpointing=activation_checkpointing):
        chosen_batch = (int(chosen_batch * .95) // 256) * 256
        if chosen_batch < 1:
            raise RuntimeError("No stable confirmed batch")
    print(json.dumps({"selected_batch_size": chosen_batch, "data_limited": chosen_batch == len(train.history),
                      "activation_checkpointing": activation_checkpointing}), flush=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU is required")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    arrays = np.load(normalized)
    x, t, y = (torch.as_tensor(arrays[k], device=device) for k in ("x", "t", "y"))
    vx = torch.as_tensor((val.history / input_scale).astype(np.float32), device=device)
    vt = torch.as_tensor(val.past_minutes, device=device)
    model = PredictionTokenTransformer(activation_checkpointing=activation_checkpointing, attention_batch_chunk=8192).to(device)
    model.load_state_dict(torch.load(initial_path, map_location="cpu", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, fused=True)
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(x), generator=generator)
    cursor = 0
    curves = []
    best, best_step = float("inf"), 0
    started = time.perf_counter()
    stream = (a.out / "optimizer-steps.csv").open("w", newline="")
    writer = csv.writer(stream)
    writer.writerow(["update", "examples_seen", "batch_size", "loss", "gradient_norm", "learning_rate", "seconds"])
    torch.cuda.reset_peak_memory_stats()
    examples_seen, completed = 0, 0
    stop_reason = "max_updates"

    def assess(step):
        nonlocal best, best_step
        _, tm = evaluate(model, x, t, train.returns, target_scale)
        _, vm = evaluate(model, vx, vt, val.returns, target_scale)
        row = {"update": step, "examples_seen": examples_seen, "train_smooth_l1": tm["smooth_l1"],
               "val_smooth_l1": vm["smooth_l1"], "train_mean_mae_bp": tm["mean_mae_bp"],
               "val_mean_mae_bp": vm["mean_mae_bp"], "elapsed_seconds": time.perf_counter() - started}
        row.update({f"val_mae_{h}m_bp": vm["by_horizon"][str(h)]["mae_bp"] for h in HORIZONS})
        curves.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(curves).to_csv(a.out / "learning-curves.csv", index=False)
        if vm["mean_mae_bp"] < best:
            best, best_step = vm["mean_mae_bp"], step
            torch.save({"state_dict": model.state_dict(), "input_scale": input_scale,
                        "target_scale": target_scale.tolist(), "architecture": architecture,
                        "selected_update": step}, a.out / "best.pt")
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                    "order_generator_state": generator.get_state(), "order": order, "cursor": cursor,
                    "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(),
                    "update": step, "examples_seen": examples_seen, "input_scale": input_scale,
                    "target_scale": target_scale.tolist()}, a.out / "resume-last.pt")

    assess(0)
    for step in range(1, a.max_updates + 1):
        if time.perf_counter() - started > a.max_training_seconds:
            stop_reason = "training_time_budget"
            break
        if cursor == len(order):
            order, cursor = torch.randperm(len(x), generator=generator), 0
        selection = order[cursor:cursor + chosen_batch].to(device)
        cursor += len(selection)
        warmup = min(1., step / 50)
        phase = max(0., (step - 50) / max(1, a.max_updates - 50))
        lr = (3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * phase))) * warmup
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        begin = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.smooth_l1_loss(model(x[selection], t[selection]), y[selection])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        completed, examples_seen = step, examples_seen + len(selection)
        writer.writerow([step, examples_seen, len(selection), loss.item(), float(norm), lr, time.perf_counter() - begin])
        stream.flush()
        if step % a.eval_every == 0 or step == a.max_updates:
            assess(step)
    if curves[-1]["update"] != completed:
        assess(completed)
    stream.close()
    peak = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}
    model.load_state_dict(torch.load(a.out / "best.pt", map_location="cpu", weights_only=True)["state_dict"])
    forecast, score = evaluate(model, vx, vt, val.returns, target_scale)
    persistence = horizon_metrics(val.returns, np.zeros_like(val.returns))
    gains = np.abs(val.returns) - np.abs(forecast - val.returns)
    intervals = bootstrap_gain(gains, pd.DatetimeIndex(val.times).strftime("%Y-%m-%d").to_numpy(),
                               np.ones(len(val.times)), {"bootstrap_seed": 23, "bootstrap_replicates": 2000})
    np.savez_compressed(a.out / "validation-forecasts.npz", forecast=forecast, truth=val.returns,
                        anchor_times=val.times.astype(str))
    result = {"id": "shared-token-transformer-v1", "architecture": architecture,
              "status": "completed" if completed == a.max_updates else "budget_limited",
              "train_cohort": "SPX-PX6-GRID5-v1", "score_cohort": "SPX-PX6-GRID5-v1 val",
              "audit": audit, "horizons_minutes": HORIZONS, "test_scored": False,
              "configuration": {"d_model": 128, "depth": 4, "nhead": 4, "ffn": 512, "dropout": .1,
                  "shared_query_parameters": 128, "linear_readout": "128 -> 1 per future position",
                  "maximum_batch_size": chosen_batch, "max_updates": a.max_updates,
                  "activation_checkpointing": activation_checkpointing, "attention_batch_chunk": 8192,
                  "precision": "BF16 autocast; FP32 parameters/AdamW states", "gradient_accumulation": False},
              "parameters": sum(p.numel() for p in model.parameters()), "completed_updates": completed,
              "examples_seen": examples_seen, "selected_update": best_step, "stop_reason": stop_reason,
              "runtime_seconds": time.perf_counter() - started, "memory": peak,
              "validation": score, "persistence": persistence,
              "paired_gain_bp": gains.mean(0).tolist(), "paired_ci95_by_date_block": intervals,
              "code_commit": a.code_commit, "model_sha256": sha(ROOT / "model/prediction_token_transformer.py"),
              "runner_sha256": sha(__file__), "checkpoint_sha256": sha(a.out / "best.pt"),
              "environment": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                              "gpu": props.name, "total_memory_bytes": props.total_memory},
              "limitations": ["Validation selected the checkpoint; paired intervals do not adjust selection.",
                              "Six-target supervision, architecture, batch and optimizer settings change together.",
                              "Overlapping windows are correlated; no test was scored.",
                              "Largest confirmed batch is capped at the number of distinct training windows."]}
    (a.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("SHARED_TOKEN_TRANSFORMER_FINISHED", flush=True)


if __name__ == "__main__":
    main()
