#!/usr/bin/env python3
"""Train a small price-only endpoint model on audited frozen C1 inputs.

No Llama/MOMENT weights, token caches, or GPU downloads. This first executable
arm is deliberately price-only; the separate pooled-text extension needs its
own feature provenance and matched controls before use.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from data.c1_protocol import build_c1_dataloaders
from model.compact_return import CompactReturn
from scripts.freeze_c1_inputs import verify

SNAPSHOT_SHA = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"


def tensors(dataset, scale):
    # Read existing numeric tensors directly: no article/sentence-report I/O per batch.
    history = torch.stack([row[6].T for row in dataset.list]).float() / scale
    target = torch.stack([row[7][[34, 69, 139], 3] for row in dataset.list]).float() / scale
    return history, target


@torch.no_grad()
def predictions(model, history, device, batch=128):
    model.eval()
    return torch.cat([model(part.to(device)).cpu() for part in history.split(batch)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, choices=[35, 70, 140, 280], default=35)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--score-test", action="store_true",
                        help="Only after committing the validation-based model selection")
    parser.add_argument("--selection-json", type=Path,
                        help="Locked architecture selection and validation evidence required for test scoring")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or args.batch_size < 1:
        parser.error("Use 1-100 epochs and a positive batch size")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("This Python environment has no CUDA support; install a CUDA-enabled PyTorch or use cpu")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.score_test:
        if args.selection_json is None:
            parser.error("--score-test requires --selection-json")
        selection = json.loads(args.selection_json.read_text())
        if (selection.get("source_commit") != commit or selection.get("seq_len") != args.seq_len or
                selection.get("width") != args.width or not selection.get("validation_evidence") or
                not selection.get("rule")):
            parser.error("Test selection must lock this code/architecture and validation evidence")
    snapshot = verify(args.snapshot_root)
    if snapshot["content_sha256"] != SNAPSHOT_SHA:
        raise ValueError("Expected the existing screened C1 snapshot; derived cohorts need a new explicit config")
    args.out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    data = build_c1_dataloaders(args.snapshot_root / "manifest.json", args.snapshot_root / "event",
        args.snapshot_root / "series/SP500.csv", seq_len=args.seq_len, pred_len=140,
        timing_policy="category4_1400_strict_before", task_mode="return_bp", seed=args.seed)
    if data.audit["emitted_counts"] != {"train": 6167, "vali": 2302, "test": 2236}:
        raise ValueError("Population changed; prepare a matched-cohort comparison")
    scale = data.task_config.scale_bp
    history, target = tensors(data.train_set, scale)
    val_history, val_target = tensors(data.vali_set, scale)
    model = CompactReturn(args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(history, target), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    best, state, patience = float("inf"), None, 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_pred = predictions(model, val_history, args.device)
        score = float((val_pred[:, 2] - val_target[:, 2]).abs().mean())
        if score < best:
            best, patience = score, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        print(f"epoch={epoch + 1} validation_h140_close_mae_bp={score * scale:.6f}", flush=True)
        if patience >= 5:
            break
    model.load_state_dict(state)
    model.eval()
    record = {**vars(args), "snapshot_content_sha256": SNAPSHOT_SHA,
        "source_commit": commit,
        "scale_bp": scale, "parameter_count": sum(p.numel() for p in model.parameters()),
        "selection": "minimum validation H140 endpoint close MAE; patience=5",
        "endpoint_horizons": [35, 70, 140], "epochs_run": epoch + 1,
        "training_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "release_timestamps_verified": False, "task": "close endpoint log returns, bp"}
    record = {key: str(value) if isinstance(value, Path) else value for key, value in record.items()}
    torch.save(dict(state_dict=state, configuration=record), args.out / "best.pt")
    for split in ("vali", "test") if args.score_test else ("vali",):
        dataset = getattr(data, split + "_set")
        x, y = tensors(dataset, scale)
        pred = predictions(model, x, args.device).numpy() * scale
        truth = y.numpy() * scale
        np.savez_compressed(args.out / (split + ".npz"), predictions_bp=pred, targets_bp=truth,
            event_ids=np.asarray(dataset.event_ids), event_dates=np.asarray([i.split('/')[-1][:8] for i in dataset.event_ids]))
        record[split] = dict(events=len(dataset), endpoint_mae_bp=np.abs(pred - truth).mean(0).tolist(),
                            zero_return_mae_bp=np.abs(truth).mean(0).tolist())
    (args.out / "resolved_config.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
