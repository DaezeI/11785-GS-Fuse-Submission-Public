#!/usr/bin/env python3
"""Screened C1 compact close-return experiments using raw news, validation only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.compact_raw_text import RawTextEncoder, TrainVocabulary

CANONICAL = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"
COUNTS = {"train": 6167, "vali": 2302, "test": 2236}
TIMING = "category4_1400_strict_before"
HORIZONS = (35, 70, 140)
WIDTH = 32
TEXT_DIM = 64
MANIFEST = ROOT / "splits/manifests/purged_split_c1_date_screen_v1.json"


class RawForecast(nn.Module):
    def __init__(self, compact, encoder=None):
        super().__init__()
        self.compact = compact
        self.encoder = encoder

    def forward(self, prices, token_ids=None):
        if token_ids is None:
            return self.compact(prices)
        if self.encoder is None:
            raise ValueError("Text IDs require a trainable encoder")
        encoded = self.encoder(token_ids)
        # Explicitly mask the correction too: text_head can learn a bias.
        offset = self.compact.text_head(encoded)
        return self.compact(prices) + offset * token_ids.ne(0).any(1, keepdim=True)


def load_c1(snapshot_root, seed, exploratory):
    from data.c1_protocol import build_c1_dataloaders, event_key
    from scripts.freeze_c1_inputs import verify
    from scripts.improvement.prepare_raw_inputs import MANIFEST_SHA256
    receipt = verify(snapshot_root)
    sha = receipt["content_sha256"]
    if sha != CANONICAL and not exploratory:
        raise ValueError("Snapshot differs from canonical C1; use --exploratory only after auditing provenance")
    expect = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
    if receipt["manifest_sha256"] != MANIFEST_SHA256 or actual != expect:
        raise ValueError("Snapshot manifest differs from the checked-in screened C1 manifest")
    data = build_c1_dataloaders(
        snapshot_root / "manifest.json", snapshot_root / "event", snapshot_root / "series/SP500.csv",
        seq_len=35, pred_len=140, timing_policy=TIMING, task_mode="return_bp",
        skip_missing=False, num_workers=0, seed=seed,
    )
    if data.audit["emitted_counts"] != COUNTS:
        raise ValueError("Screened C1 population changed")
    for role in COUNTS:
        ids = [event_key(p)[2] for p in expect[role + "_events"]]
        if list(getattr(data, role + "_set").event_ids) != ids:
            raise ValueError(f"Screened C1 {role} article IDs or ordering changed")
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return data, sha, manifest_sha


def price_tensors(dataset, scale):
    x = torch.stack([row[6].T for row in dataset.list]).float() / scale
    y = torch.stack([row[7][[h - 1 for h in HORIZONS], 3] for row in dataset.list]).float() / scale
    if x.shape != (len(dataset), 4, 35) or y.shape != (len(dataset), 3):
        raise ValueError("Incorrect screened C1 return tensor dimensions")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite price history or endpoint targets")
    return x, y


def articles(dataset):
    # One screened event summary per row; do not read its negative examples.
    return [Path(row[0]).read_text(encoding="utf-8", errors="ignore") for row in dataset.list]


def article_hash(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def past_donors(ids, hashes, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    parts = []
    for idx, event_id in enumerate(ids):
        category, filename = event_id.split("/")
        date = filename[:8]
        group = (category, date[:4])
        parts.append((group, date))
        groups.setdefault(group, []).append((idx, date))
    donors = np.full(len(ids), -1, dtype=np.int64)
    for i, (group, date) in enumerate(parts):
        pool = [j for j, other_date in groups[group] if other_date < date and hashes[i] != hashes[j]]
        if pool:
            donors[i] = rng.choice(pool)
    return donors


@torch.no_grad()
def predict(model, x, device, ids=None):
    model.eval()
    return torch.cat([
        model(x[i:i+128].to(device), None if ids is None else ids[i:i+128].to(device)).cpu()
        for i in range(0, len(x), 128)
    ])


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(model, train, val, train_ids, val_ids, args):
    x, y = train
    vx, vy = val
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(x, y, *(() if train_ids is None else (train_ids,))),
                        batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    # Keep the zero/base forecast as a valid validation-selected checkpoint.
    best = float((predict(model, vx, args.device, val_ids)[:, 2] - vy[:, 2]).abs().mean())
    best_state, best_epoch, stale = state_cpu(model), 0, 0
    saw_embedding_grad = False
    embedding = model.encoder.embedding.weight if model.encoder is not None else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for px, target, *tokens in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(px.to(args.device), tokens[0].to(args.device) if tokens else None)
            loss = torch.nn.functional.smooth_l1_loss(pred, target.to(args.device))
            loss.backward()
            if embedding is not None and not saw_embedding_grad and embedding.grad is not None:
                saw_embedding_grad = bool(torch.count_nonzero(embedding.grad[1:]))
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
        score = float((predict(model, vx, args.device, val_ids)[:, 2] - vy[:, 2]).abs().mean())
        if not math.isfinite(score):
            raise ValueError("Non-finite validation score")
        print(f"{args.mode} epoch={epoch} scaled_val_h140_mae={score:.6f}", flush=True)
        if score < best:
            best, best_state, best_epoch, stale = score, state_cpu(model), epoch, 0
        else:
            stale += 1
        if stale == 5:
            break
    if embedding is not None and not saw_embedding_grad:
        raise ValueError("Forecasting loss never reached the news embedding weights")
    model.load_state_dict(best_state)
    return epoch, best_epoch, saw_embedding_grad


def metrics(truth, pred):
    return {
        "endpoint_mae_bp": np.abs(truth - pred).mean(axis=0).tolist(),
        "endpoint_direction_hit_rate": (np.sign(truth) == np.sign(pred)).mean(axis=0).tolist(),
    }


def controls(model, vx, truth, matched, paired_price, val_ids, texts, event_ids, args, scale):
    hashes = [article_hash(t) for t in texts]
    draws = np.stack([past_donors(event_ids, hashes, seed) for seed in (23, 24, 25)])
    rows = np.flatnonzero((draws >= 0).all(0))
    info = {
        "events": len(rows), "excluded_no_donor": len(event_ids) - len(rows),
        "coverage": len(rows) / len(event_ids),
        "policy": "same validation split/category/calendar year; strictly earlier recorded date; "
                  "normalized exact duplicates excluded; near duplicates and publication times unverified",
    }
    if not len(rows):
        return info
    shuffled = np.stack([predict(model, vx[rows], args.device, val_ids[draw[rows]]).numpy() * scale
                         for draw in draws])
    no_text = predict(model, vx[rows], args.device).numpy() * scale
    y = truth[rows]
    info.update(matched=metrics(y, matched[rows]), price_reference=metrics(y, paired_price[rows]),
                zero=metrics(y, np.zeros_like(y)), no_text=metrics(y, no_text),
                shuffled_by_seed={str(s): metrics(y, p) for s, p in zip((23, 24, 25), shuffled)})
    ids = np.asarray(event_ids)
    np.savez_compressed(args.out / "vali-text-controls.npz", event_ids=ids[rows],
                        donor_event_ids=ids[draws[:, rows]], targets_bp=y, matched_bp=matched[rows],
                        price_reference_bp=paired_price[rows], no_text_bp=no_text, shuffled_bp=shuffled)
    return info


def git_revision():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return commit, dirty


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=("price", "residual", "joint"))
    parser.add_argument("--price-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exploratory", action="store_true", help="Allow noncanonical bytes, never a different split")
    args = parser.parse_args(argv)
    if bool(args.price_checkpoint) != (args.mode != "price"):
        parser.error("Specify --price-checkpoint for residual and joint only")
    if not 1 <= args.epochs <= 30 or args.batch_size < 1:
        parser.error("Use 1–30 epochs and a positive batch size")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    if args.out.exists():
        parser.error("Choose a new output directory")

    from model.compact_return import CompactReturn

    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    else:
        torch.cuda.reset_peak_memory_stats()
    data, snapshot_sha, manifest_sha = load_c1(args.snapshot_root, args.seed, args.exploratory)
    scale = float(data.task_config.scale_bp)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid train-only C1 return scale")
    train, val = (price_tensors(ds, scale) for ds in (data.train_set, data.vali_set))
    commit, dirty = git_revision()
    source_sha256 = {
        path: hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in ("model/compact_return.py", "model/compact_raw_text.py", "model/compact_news_vocab.py",
                     "scripts/improvement/train_compact_raw.py", "scripts/freeze_c1_inputs.py",
                     "data/c1_protocol.py", "data/forecast_task.py")
    }
    common = dict(snapshot_content_sha256=snapshot_sha, manifest_sha256=manifest_sha, scale_bp=scale,
                  seed=args.seed, width=WIDTH, seq_len=35, horizons=list(HORIZONS),
                  timing_policy=TIMING, epochs=args.epochs, batch_size=args.batch_size,
                  source_commit=commit, source_dirty=dirty, source_sha256=source_sha256)
    vocab = train_ids = val_ids = val_text = reference = reference_sha = None
    if args.mode != "price":
        payload = args.price_checkpoint.read_bytes()
        reference_sha = hashlib.sha256(payload).hexdigest()
        # Hash and load the same bytes. Accept tensor/state metadata only.
        from io import BytesIO
        checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        cfg = checkpoint["configuration"]
        if cfg.get("mode") != "price" or any(cfg.get(k) != v for k, v in common.items()
                                           if k != "source_dirty"):
            raise ValueError("Reference must be this C1 input/setup's same-seed price-only checkpoint")
        reference = RawForecast(CompactReturn(WIDTH))
        reference.load_state_dict(checkpoint["state_dict"], strict=True)
        reference.to(args.device).eval().requires_grad_(False)
        train_text = articles(data.train_set)
        vocab = TrainVocabulary.fit(train_text, max_vocab=16000, max_tokens=512)
        train_ids = torch.tensor([vocab.encode(t) for t in train_text], dtype=torch.long)
        del train_text
        val_text = articles(data.vali_set)
        val_ids = torch.tensor([vocab.encode(t) for t in val_text], dtype=torch.long)
    # Reset after loading the separate comparator; joint starts randomly, not from its weights.
    torch.manual_seed(args.seed)
    compact = CompactReturn(WIDTH, text_dim=TEXT_DIM if vocab is not None else 0)
    encoder = RawTextEncoder(len(vocab), output_dim=TEXT_DIM, embedding_dim=64) if vocab is not None else None
    model = RawForecast(compact, encoder)
    if args.mode == "residual":
        model.compact.price_encoder.load_state_dict(reference.compact.price_encoder.state_dict())
        model.compact.price_head.load_state_dict(reference.compact.price_head.state_dict())
        model.compact.freeze_price()
    model.to(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    fit_start = time.perf_counter()
    epochs_run, best_epoch, saw_grad = fit(model, train, val, train_ids, val_ids, args)
    if args.device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - fit_start
    vx, vy = val
    truth = vy.numpy() * scale
    pred = predict(model, vx, args.device, val_ids).numpy() * scale
    paired = predict(reference, vx, args.device).numpy() * scale if reference is not None else pred
    if args.mode == "residual" and not np.array_equal(paired, predict(model, vx, args.device).numpy() * scale):
        raise ValueError("Frozen reference no longer reproduces its original price predictions")
    event_ids = data.vali_set.event_ids
    np.savez_compressed(args.out / "vali.npz", event_ids=np.asarray(event_ids),
                        targets_bp=truth, predictions_bp=pred, price_reference_bp=paired)
    record = dict(**common, mode=args.mode, device=args.device, torch_version=str(torch.__version__),
                  input_class="canonical_bytes_timing_unverified" if snapshot_sha == CANONICAL
                              else "exploratory_noncanonical_input",
                  release_timestamps_verified=False, test_scored=False, scale_fit="training_only",
                  selection="minimum validation H140 close MAE (epoch 0 eligible), patience 5",
                  news_source="screened event full_summary raw text" if vocab is not None else None,
                  epochs_run=epochs_run, best_epoch=best_epoch, encoder_gradient_observed=saw_grad,
                  price_checkpoint_sha256=reference_sha,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  training_seconds=train_seconds, vali=dict(events=len(event_ids), matched=metrics(truth, pred),
                  price_reference=metrics(truth, paired), zero=metrics(truth, np.zeros_like(truth))))
    if val_ids is not None:
        record["paired_text_controls"] = controls(
            model, vx, truth, pred, paired, val_ids, val_text, event_ids, args, scale)
    record.update(elapsed_seconds=time.perf_counter() - start,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else None)
    torch.save(dict(state_dict=state_cpu(model), configuration=common | {"mode": args.mode},
                    tokenizer=None if vocab is None else vocab.to_dict()), args.out / "best.pt")
    record["artifact_bytes_excluding_summary"] = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    (args.out / "resolved_config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
