#!/usr/bin/env python3
"""Train the independently text-free screened C1 price reference."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.c1_protocol import build_c1_dataloaders
from model.gs_fuse import (
    FUSE_DIM,
    PRED_DROPOUT,
    RAW_TS_RESIDUAL_WEIGHT,
    MultiScaleTemporalExtractor,
    TinyDecoderV3,
    TrainingConfig,
)
from model.ts_encoders import build_ts_encoder
from scripts.analyze_text_predictions import block_bootstrap_improvement, validate

REFERENCE_COMMIT = "bb09396881cc92c5e654c4210944ce5c6da17ced"
SNAPSHOT_SHA = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"
INPUT_HASHES = {
    "manifest_sha256": "1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1",
    "price_sha256": "6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a",
}
COUNTS = (6167, 2302, 2236)
HORIZONS = (35, 70, 140)
BATCH = 8
EPOCHS = 2


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class PriceOnlyForecaster(nn.Module):
    """The released model's strict pre-attention price-to-decoder path."""

    def __init__(self, moment_path, device):
        super().__init__()
        self.ts_encoder = build_ts_encoder(
            "moment", device, moment_path=str(moment_path), seq_len=35, d=4
        )
        self.ts_encoder.freeze()
        width = self.ts_encoder.output_dim
        self.ts_inst_to_fuse = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, FUSE_DIM), nn.GELU()
        )
        self.residual_project = nn.Sequential(
            nn.Linear(140, 140), nn.GELU(),
            nn.Linear(140, 140), nn.GELU(),
            nn.Linear(140, FUSE_DIM),
        )
        self.multi_scale_extractor = MultiScaleTemporalExtractor(
            d_input=4, d_output=FUSE_DIM, seq_len=35
        )
        self.decoder = TinyDecoderV3(
            context_dim=FUSE_DIM, d=4, pred_len=140, width=FUSE_DIM,
            n_heads=8, n_layers=3, dropout=PRED_DROPOUT, seq_len=35,
        )
        self.to(device)

    def forward(self, seq):
        _, price_instance = self.ts_encoder.encode_per_timestep(seq)
        features = self.ts_inst_to_fuse(price_instance)
        features = features + RAW_TS_RESIDUAL_WEIGHT * self.residual_project(seq.flatten(1))
        features = features + 0.5 * self.multi_scale_extractor(seq)
        return self.decoder(F.normalize(features, p=2, dim=-1), x_cf=seq)

    def load_reference(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        expected = {
            "ts_backbone": "moment", "seq_len": 35, "pred_len": 140, "d": 4,
            "decoder_layers": 3, "decoder_heads": 8, "use_raw_ts_residual": True,
            "use_multi_scale_ts": True, "use_learnable_last_value": False,
        }
        for key, value in expected.items():
            if checkpoint["metadata"].get(key) != value:
                raise ValueError(f"Incompatible reference checkpoint: {key}")
        for name, module in self.named_children():
            module.load_state_dict(checkpoint["state_dicts"][name], strict=True)


def read_reference(args):
    full = args.reference_dir / "full"
    config = json.loads((full / "resolved_config.json").read_text())
    audit = json.loads((full / "c1_data_audit.json").read_text())
    inputs = json.loads((full / "input-fingerprints.json").read_text())
    expected = {
        "seed": args.seed, "source_commit": REFERENCE_COMMIT,
        "batch_size": BATCH, "num_epochs": EPOCHS, "stage1_epochs": EPOCHS,
        "forecast_task": "price", "tf32": True, "sw_stride": 10,
        "no_4bit": True, "finetune_ts_last_block": False, "use_ts_memory": False,
        "use_raw_ts_residual": True, "use_learnable_last_value": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Original run config differs: {key}")
    if inputs["snapshot_content_sha256"] != SNAPSHOT_SHA or any(
        inputs.get(name) != sha for name, sha in INPUT_HASHES.items()
    ):
        raise ValueError("Original frozen snapshot fingerprint differs")
    if audit.get("timing_policy") != "category4_1400_strict_before":
        raise ValueError("Original timing rule differs")
    return full, audit


def price_loader(dataset, seed, shuffle, limit):
    rows = dataset.list[:limit] if limit else dataset.list
    seq, target = zip(*((row[-2].T, row[-1].T) for row in rows), strict=True)
    prices = TensorDataset(torch.stack(seq), torch.stack(target))
    return DataLoader(
        prices, batch_size=BATCH, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def forecast(model, loader, device):
    model.eval()
    outputs = []
    for seq, _ in loader:
        prediction = model(seq.to(device=device, dtype=torch.float32))
        outputs.append(prediction.cpu().numpy().transpose(0, 2, 1))
    output = np.concatenate(outputs)
    if not np.isfinite(output).all():
        raise ValueError("Forecasts contain non-finite values")
    return output


def train(model, loaders, device, output, epochs):
    train_loader, validation_loader = loaders
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=TrainingConfig.BASE_LR, weight_decay=0.01)
    steps = epochs * len(train_loader)
    schedule = get_cosine_schedule_with_warmup(
        optimizer, int(TrainingConfig.WARMUP_RATIO * steps), steps
    )
    validation_targets = validation_loader.dataset.tensors[1].numpy().transpose(0, 2, 1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for step, (seq, target) in enumerate(train_loader, 1):
            seq = seq.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(seq)
            loss = (0.5 * F.mse_loss(pred, target) + 0.3 * F.l1_loss(pred, target)
                    + 0.2 * F.smooth_l1_loss(pred, target, beta=0.5))
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}, step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, TrainingConfig.GRAD_CLIP_NORM, error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if step % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, batch {step}/{len(train_loader)}", flush=True)
        prediction = forecast(model, validation_loader, device)
        mse = float(np.square(prediction - validation_targets, dtype=np.float64).mean())
        if not history or mse < min(row["validation_mse"] for row in history):
            torch.save(model.state_dict(), output)
        history.append({"epoch": epoch, "validation_mse": mse})
        print(f"Epoch {epoch}/{epochs}, validation MSE {mse:.9f}", flush=True)
    return history


def compare(pred, reference, scaler):
    date = reference["event_dates"]
    target = reference["targets"].astype(np.float64)
    real = reference["real_text"].astype(np.float64)
    seqs = reference["seqs"].astype(np.float64)
    bypass = reference["zero_text"].astype(np.float64)
    keep = np.ones(len(date), dtype=bool)
    rows = []
    for h in HORIZONS:
        losses = {
            label: np.square(model[:, :h] - target[:, :h]).mean((1, 2))
            for label, model in {
                "news": real, "price_only": pred, "persistence": seqs[:, -1:],
                "inference_bypass": bypass,
            }.items()
        }
        mse = {label: float(loss.mean()) for label, loss in losses.items()}
        gains = {}
        for model, control in (
            ("news", "price_only"), ("news", "persistence"), ("price_only", "persistence")
        ):
            gains[f"{model}_over_{control}"] = {
                "gain_pct": 100 * (1 - mse[model] / mse[control]),
                "intervals_pct": {
                    b: block_bootstrap_improvement(
                        losses[model], losses[control], date, keep, block=b
                    )["ci95_pct"]
                    for b in (5, 10, 20, 40)
                },
            }
        # Close return differences use the last observed close as the origin.
        mean, scale = scaler.mean_[3], scaler.scale_[3]
        anchor = seqs[:, -1, 3] * scale + mean
        endpoint_prices = {
            label: values[:, h - 1, 3] * scale + mean
            for label, values in {"target": target, "news": real, "price_only": pred}.items()
        }
        if any(np.any(values <= 0) for values in (anchor, *endpoint_prices.values())):
            raise ValueError("Close prices must be positive for log return scoring")
        realized = 1e4 * np.log(endpoint_prices.pop("target") / anchor)
        eligible = realized != 0
        endpoint = {}
        for label, values in endpoint_prices.items():
            returns = 1e4 * np.log(values / anchor)
            endpoint[label] = {
                "mae_bp": float(np.abs(returns - realized).mean()),
                "direction_accuracy": float((np.sign(returns[eligible]) == np.sign(realized[eligible])).mean()),
            }
        endpoint["always_up"] = {"direction_accuracy": float((realized[eligible] > 0).mean())}
        endpoint["zero_return"] = {"mae_bp": float(np.abs(realized).mean())}
        rows.append({
            "horizon": h, "mse": mse, "comparison": gains,
            "endpoint_close": endpoint, "direction_eligible_events": int(eligible.sum()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--moment-path", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint-sha", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("The matched experiment requires a CUDA device")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    full, original_audit = read_reference(args)
    manifest = args.snapshot_dir / "manifest.json"
    series = args.snapshot_dir / "series" / "SP500.csv"
    if any(file_hash(path) != INPUT_HASHES[key] for path, key in (
        (manifest, "manifest_sha256"), (series, "price_sha256")
    )):
        raise ValueError("Frozen manifest or price file differs")

    events = build_c1_dataloaders(
        manifest, args.snapshot_dir / "event", series, batch_size=BATCH,
        sw_stride=10, timing_policy="category4_1400_strict_before", seed=args.seed,
    )
    datasets = (events.train_set, events.vali_set, events.test_set)
    if tuple(map(len, datasets)) != COUNTS:
        raise ValueError("Frozen event counts differ")
    for key in ("manifest_sha256", "price_sha256", "scaler_mean", "scaler_scale", "covered_bars"):
        if events.audit[key] != original_audit[key]:
            raise ValueError(f"Comparison input differs from original run: {key}")

    archive = args.reference_dir / "paired" / "fusion_bypass_ts_inst.npz"
    hash_file = archive.with_suffix(".sha256")
    expected_archive = next(line.split()[0] for line in hash_file.read_text().splitlines()
                            if line.split()[-1].endswith(archive.name))
    if file_hash(archive) != expected_archive:
        raise ValueError("Saved prediction archive hash differs")
    with np.load(archive, allow_pickle=False) as saved:
        metadata = validate(saved)
        if metadata["checkpoint_id"] != args.reference_checkpoint_sha:
            raise ValueError("Saved predictions use a different news checkpoint")
        if metadata["reference_source_commit"] != REFERENCE_COMMIT:
            raise ValueError("Saved predictions use a different source commit")
        if (metadata["price_sha256"] != INPUT_HASHES["price_sha256"]
                or metadata["c1_manifest_sha256"] != INPUT_HASHES["manifest_sha256"]
                or metadata["no_text_definition"] != "fusion_bypass_ts_inst"):
            raise ValueError("Saved predictions use different inputs or control")
        reference = {key: saved[key] for key in (
            "event_ids", "event_dates", "targets", "seqs", "real_text", "zero_text"
        )}

    train_loader, val_loader = (
        price_loader(dataset, args.seed, shuffle, 16 if args.smoke else 0)
        for dataset, shuffle in zip(datasets[:2], (True, False), strict=True)
    )
    full_test = price_loader(datasets[2], args.seed, False, 0)
    if not np.array_equal(datasets[2].event_ids, reference["event_ids"]):
        raise ValueError("Test event ID order differs from the archived forecast")
    for tensor, name in zip(full_test.dataset.tensors, ("seqs", "targets"), strict=True):
        if not np.array_equal(tensor.numpy().astype(np.float32).transpose(0, 2, 1), reference[name]):
            raise ValueError(f"Test data do not match saved {name}")

    init = full / "stage1_best_model.pth"
    checkpoint = args.output / "best_price_only.pth"
    model = PriceOnlyForecaster(args.moment_path, device)
    if args.smoke:
        model.load_reference(full / "best_model.pth")
        smoke_test = price_loader(datasets[2], args.seed, False, 16)
        parity = forecast(model, smoke_test, device)
        np.testing.assert_allclose(parity, reference["zero_text"][:16], rtol=2e-4, atol=5e-4)
    model.load_reference(init)
    history = train(model, (train_loader, val_loader), device, checkpoint, 1 if args.smoke else EPOCHS)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True))
    prediction = forecast(model, smoke_test if args.smoke else full_test, device)
    result = {
        "smoke": args.smoke, "seed": args.seed, "base_code_commit": base,
        "runner_sha256": file_hash(Path(__file__)),
        "init_sha256": file_hash(init), "checkpoint_sha256": file_hash(checkpoint),
        "reference_checkpoint_sha256": args.reference_checkpoint_sha,
        "reference_archive_sha256": expected_archive, "snapshot_sha256": SNAPSHOT_SHA,
        **INPUT_HASHES, "counts": list(COUNTS), "validation_history": history,
        "selected_epoch": min(history, key=lambda x: x["validation_mse"])["epoch"],
        "events": len(prediction), "hardware": torch.cuda.get_device_name(0),
    }
    if args.smoke:
        result["reference_parity_max_abs_delta"] = float(np.abs(parity - reference["zero_text"][:16]).max())
    else:
        result["dates"] = len(np.unique(reference["event_dates"]))
        result["results"] = compare(prediction, reference, events.scaler)
        np.savez_compressed(args.output / "price_only_predictions.npz",
                            prediction=prediction, event_ids=reference["event_ids"])
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
