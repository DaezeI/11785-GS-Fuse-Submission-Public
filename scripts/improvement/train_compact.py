#!/usr/bin/env python3
"""Train compact price forecasts and an optional pooled-text residual on frozen C1.

The price-only baseline is fitted first and frozen before text learning. No
Llama/MOMENT weights, token caches or pretrained downloads enter this loop.
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
from scripts.improvement.compact_features import load_features, past_donors, load_numeric_pack
from scripts.analyze_text_predictions import block_bootstrap_improvement

SNAPSHOT_SHA = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"


def tensors(dataset, scale):
    # Read existing numeric tensors directly: no article/sentence-report I/O per batch.
    if hasattr(dataset, "history_bp"):
        return dataset.history_bp / scale, dataset.targets_bp / scale
    history = torch.stack([row[6].T for row in dataset.list]).float() / scale
    target = torch.stack([row[7][[34, 69, 139], 3] for row in dataset.list]).float() / scale
    return history, target


@torch.no_grad()
def predictions(model, history, device, batch=128, features=None):
    model.eval()
    return torch.cat([model(history[start:start + batch].to(device),
        features[start:start + batch].to(device) if features is not None else None).cpu()
        for start in range(0, len(history), batch)])


def fit(model, history, target, val_history, val_target, args, features=None, val_features=None):
    """Allow the unchanged zero/base forecast as a validation candidate."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-3)
    items = (history, target) if features is None else (history, target, features)
    loader = DataLoader(TensorDataset(*items), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    initial = predictions(model, val_history, args.device, features=val_features)
    best = float((initial[:, 2] - val_target[:, 2]).abs().mean())
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            x, y = batch[:2]
            text = batch[2].to(args.device) if features is not None else None
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(model(x.to(args.device), text), y.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        val_pred = predictions(model, val_history, args.device, features=val_features)
        score = float((val_pred[:, 2] - val_target[:, 2]).abs().mean())
        if score < best:
            best, patience = score, 0
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        print(f"stage={'text' if features is not None else 'price'} epoch={epoch + 1} scaled_val_h140_mae={score:.6f}", flush=True)
        if patience >= 5:
            break
    model.load_state_dict(state)
    return epoch + 1


def evaluate_controls(model, history, target, features, article_hashes, dataset, args, scale, split):
    draws = [past_donors(dataset.event_ids, article_hashes, seed) for seed in (23, 24, 25)]
    eligible = np.flatnonzero(np.logical_and.reduce([draw >= 0 for draw in draws]))
    if not len(eligible):
        return dict(events=0, note="No eligible past-date donors; no matching claim")
    x, text = history[eligible], features[eligible]
    truth = target[eligible].numpy() * scale
    real = predictions(model, x, args.device, features=text).numpy() * scale
    base = predictions(model, x, args.device).numpy() * scale
    shuffled = np.stack([predictions(model, x, args.device, features=features[draw[eligible]]).numpy() * scale for draw in draws])
    ids = np.asarray(dataset.event_ids)[eligible]
    dates = np.asarray([key.split('/')[-1][:8] for key in ids])
    real_error = np.abs(real[:, 2] - truth[:, 2])
    control_error = dict(frozen_price=np.abs(base[:, 2] - truth[:, 2]),
        zero_return=np.abs(truth[:, 2]), shuffled_text=np.abs(shuffled[:, :, 2] - truth[None, :, 2]).mean(0))
    np.savez_compressed(args.out / (split + "-text-controls.npz"), event_ids=ids, event_dates=dates,
        targets_bp=truth, real_text_bp=real, frozen_price_bp=base, shuffled_text_bp=shuffled,
        donor_event_ids=np.stack([np.asarray(dataset.event_ids)[draw[eligible]] for draw in draws]))
    return dict(events=len(eligible), excluded_no_donor=len(dataset) - len(eligible),
        primary="H140 endpoint close MAE, bp", real_mae_bp=float(real_error.mean()),
        controls={name: dict(mae_bp=float(error.mean()), paired_20date_ci=block_bootstrap_improvement(
            real_error, error, dates, np.ones(len(dates), dtype=bool), reps=1000, block=20))
            for name, error in control_error.items()},
        donor_policy="same split/category/year, strictly earlier recorded date; exclude exact normalized content; near duplicates unassessed; times unverified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--snapshot-root", type=Path)
    inputs.add_argument("--input-pack", type=Path, help="Portable audited input/feature export; no raw articles or backbones needed")
    parser.add_argument("--input-pack-sha256")
    parser.add_argument("--use-text", action="store_true", help="Use pooled text features inside the portable input pack")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, choices=[35, 70, 140, 280], default=35)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--text-features", type=Path, help="Content-matched pooled feature NPZ; enables multimodal residual training")
    parser.add_argument("--text-feature-sha256", help="Exact checksum from the pooled-feature export receipt")
    parser.add_argument("--score-test", action="store_true",
                        help="Only after committing the validation-based model selection")
    parser.add_argument("--selection-json", type=Path,
                        help="Locked architecture selection and validation evidence required for test scoring")
    args = parser.parse_args()
    if args.input_pack and not args.input_pack_sha256:
        parser.error("--input-pack requires --input-pack-sha256")
    if args.use_text:
        if not args.input_pack:
            parser.error("--use-text requires --input-pack; use --text-features with a raw snapshot")
        args.text_features, args.text_feature_sha256 = args.input_pack, args.input_pack_sha256
    if bool(args.text_features) != bool(args.text_feature_sha256):
        parser.error("--text-features and --text-feature-sha256 must be supplied together")
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
                not selection.get("rule") or selection.get("text_feature_sha256") != args.text_feature_sha256 or
                selection.get("input_pack_sha256") != args.input_pack_sha256):
            parser.error("Test selection must lock this code/architecture and validation evidence")
    args.out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.input_pack:
        data = load_numeric_pack(args.input_pack, args.input_pack_sha256, args.seq_len)
    else:
        snapshot = verify(args.snapshot_root)
        if snapshot["content_sha256"] != SNAPSHOT_SHA:
            raise ValueError("Expected the existing screened C1 snapshot")
        data = build_c1_dataloaders(args.snapshot_root / "manifest.json", args.snapshot_root / "event",
            args.snapshot_root / "series/SP500.csv", seq_len=args.seq_len, pred_len=140,
            timing_policy="category4_1400_strict_before", task_mode="return_bp", seed=args.seed)
        if data.audit["emitted_counts"] != {"train": 6167, "vali": 2302, "test": 2236}:
            raise ValueError("Population changed; prepare a matched-cohort comparison")
    scale = data.task_config.scale_bp
    history, target = tensors(data.train_set, scale)
    val_history, val_target = tensors(data.vali_set, scale)
    feature_rows, article_hashes, feature_metadata = {}, {}, None
    feature_mean = feature_std = None
    if args.text_features:
        for role in ("train", "vali", "test"):
            dataset = getattr(data, role + "_set")
            values, hashes, feature_metadata = load_features(args.text_features,
                args.text_feature_sha256, dataset.event_ids, SNAPSHOT_SHA)
            feature_rows[role], article_hashes[role] = values, hashes
        feature_mean = feature_rows["train"].mean(0)
        feature_std = np.maximum(feature_rows["train"].std(0), 1e-6)
        feature_rows = {role: torch.from_numpy((values - feature_mean) / feature_std)
                        for role, values in feature_rows.items()}
    model = CompactReturn(args.width, text_dim=feature_rows["train"].shape[1] if feature_rows else 0).to(args.device)
    started = time.perf_counter()
    price_epochs = fit(model, history, target, val_history, val_target, args)
    text_epochs = 0
    if feature_rows:
        model.freeze_price()
        text_epochs = fit(model, history, target, val_history, val_target, args,
                           feature_rows["train"], feature_rows["vali"])
    model.eval()
    record = {**vars(args), "snapshot_content_sha256": SNAPSHOT_SHA,
        "source_commit": commit,
        "scale_bp": scale, "parameter_count": sum(p.numel() for p in model.parameters()),
        "selection": "minimum validation H140 endpoint close MAE; patience=5",
        "endpoint_horizons": [35, 70, 140], "price_epochs_run": price_epochs, "text_epochs_run": text_epochs,
        "feature_provenance": feature_metadata,
        "training_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "release_timestamps_verified": False, "task": "close endpoint log returns, bp"}
    record = {key: str(value) if isinstance(value, Path) else value for key, value in record.items()}
    torch.save(dict(state_dict={key: value.cpu() for key, value in model.state_dict().items()}, configuration=record,
        feature_mean=torch.from_numpy(feature_mean) if feature_mean is not None else None,
        feature_std=torch.from_numpy(feature_std) if feature_std is not None else None), args.out / "best.pt")
    for split in ("vali", "test") if args.score_test else ("vali",):
        dataset = getattr(data, split + "_set")
        x, y = tensors(dataset, scale)
        pred = predictions(model, x, args.device, features=feature_rows.get(split)).numpy() * scale
        price_pred = predictions(model, x, args.device).numpy() * scale
        truth = y.numpy() * scale
        np.savez_compressed(args.out / (split + ".npz"), predictions_bp=pred,
            frozen_price_predictions_bp=price_pred, targets_bp=truth,
            event_ids=np.asarray(dataset.event_ids), event_dates=np.asarray([i.split('/')[-1][:8] for i in dataset.event_ids]))
        record[split] = dict(events=len(dataset), endpoint_mae_bp=np.abs(pred - truth).mean(0).tolist(),
                            frozen_price_endpoint_mae_bp=np.abs(price_pred - truth).mean(0).tolist(),
                            zero_return_mae_bp=np.abs(truth).mean(0).tolist())
        if feature_rows:
            record[split]["paired_text_controls"] = evaluate_controls(model, x, y, feature_rows[split],
                article_hashes[split], dataset, args, scale, split)
    (args.out / "resolved_config.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
