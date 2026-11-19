#!/usr/bin/env python3
"""Train compact 140-bar OHLC log-return forecasts on canonical C1, validation only."""
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model.compact_return import CompactReturn
from model.compact_raw_text import PAD_ID, RawTextEncoder, TrainVocabulary
from scripts.improvement.compact_path_metrics import CHANNELS, FORECAST_BARS, HORIZONS, metrics
from scripts.improvement.train_compact_raw import (
    CANONICAL, TEXT_DIM, TIMING, WIDTH, article_hash, articles,
    git_revision, load_c1, past_donors, predict, state_cpu,
)

TASK_ID = "c1-ohlc-return-path-v1"


class CompactPath(CompactReturn):
    """Reuse the compact causal price encoder and emit a complete OHLC path."""
    def __init__(self, width=WIDTH, text_dim=0):
        super().__init__(width, text_dim)
        for head in (self.price_head, self.text_head):
            if head is not None:
                head[-1] = nn.Linear(head[-1].in_features, FORECAST_BARS * CHANNELS)
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, history, text=None):
        flat = super().forward(history, text)
        return flat.reshape(-1, FORECAST_BARS, CHANNELS)


class PathForecast(nn.Module):
    def __init__(self, compact, encoder=None):
        super().__init__()
        self.compact, self.encoder = compact, encoder

    def forward(self, prices, token_ids=None):
        base = self.compact(prices)
        if token_ids is None:
            return base
        if self.encoder is None:
            raise ValueError("Text IDs require a trainable encoder")
        offset = self.compact.text_head(self.encoder(token_ids)).reshape_as(base)
        return base + offset * token_ids.ne(PAD_ID).any(1).view(-1, 1, 1)


def price_tensors(dataset, scale):
    x = torch.stack([row[6].T for row in dataset.list]).float() / scale
    y = torch.stack([row[7] for row in dataset.list]).float() / scale
    if x.shape != (len(dataset), CHANNELS, 35) or y.shape != (len(dataset), FORECAST_BARS, CHANNELS):
        raise ValueError("Incorrect screened C1 whole-path return tensor dimensions")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Non-finite price history or target returns")
    return x, y


def fit(model, train, val, train_ids, val_ids, args):
    x, y = train
    vx, vy = val
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(x, y, *(() if train_ids is None else (train_ids,))),
                        batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    # Keep the zero/base forecast as a valid validation-selected checkpoint.
    best = float((predict(model, vx, args.device, val_ids) - vy).abs().mean())
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
        score = float((predict(model, vx, args.device, val_ids) - vy).abs().mean())
        if not math.isfinite(score):
            raise ValueError("Non-finite validation score")
        print(f"{args.mode} epoch={epoch} scaled_val_path_mae={score:.6f}", flush=True)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=("price", "residual"))
    parser.add_argument("--price-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    if bool(args.price_checkpoint) != (args.mode != "price"):
        parser.error("Specify --price-checkpoint for residual only")
    if not 1 <= args.epochs <= 30 or args.batch_size < 1:
        parser.error("Use 1–30 epochs and a positive batch size")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    if args.out.exists():
        parser.error("Choose a new output directory")

    start = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    else:
        torch.cuda.reset_peak_memory_stats()
    data, snapshot_sha, manifest_sha = load_c1(args.snapshot_root, args.seed, False)
    scale = float(data.task_config.scale_bp)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid train-only C1 return scale")
    train, val = (price_tensors(ds, scale) for ds in (data.train_set, data.vali_set))
    commit, dirty = git_revision()
    source_sha256 = {
        path: hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in ("model/compact_return.py", "model/compact_raw_text.py", "model/compact_news_vocab.py",
                     "scripts/improvement/train_compact_path.py", "scripts/improvement/compact_path_metrics.py",
                     "scripts/improvement/train_compact_raw.py",
                     "scripts/improvement/prepare_raw_inputs.py", "scripts/freeze_c1_inputs.py",
                     "data/c1_protocol.py", "data/forecast_task.py")
    }
    common = dict(task=TASK_ID, forecast_bars=FORECAST_BARS, channels=["open", "high", "low", "close"],
                  snapshot_content_sha256=snapshot_sha, manifest_sha256=manifest_sha, scale_bp=scale,
                  seed=args.seed, width=WIDTH, seq_len=35, horizons=list(HORIZONS),
                  timing_policy=TIMING, epochs=args.epochs, batch_size=args.batch_size,
                  source_commit=commit, source_dirty=dirty, source_sha256=source_sha256)
    vocab = train_ids = val_ids = val_text = reference = reference_sha = None
    if args.mode != "price":
        payload = args.price_checkpoint.read_bytes()
        reference_sha = hashlib.sha256(payload).hexdigest()
        # Hash and load the same bytes. Accept tensor/state metadata only.
        checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
        cfg = checkpoint["configuration"]
        if cfg.get("mode") != "price" or any(cfg.get(k) != v for k, v in common.items()
                                           if k != "source_dirty"):
            raise ValueError("Reference must be this C1 input/setup's same-seed price-only checkpoint")
        reference = PathForecast(CompactPath(WIDTH))
        reference.load_state_dict(checkpoint["state_dict"], strict=True)
        reference.to(args.device).eval().requires_grad_(False)
        train_text = articles(data.train_set)
        vocab = TrainVocabulary.fit(train_text, max_vocab=16000, max_tokens=512)
        train_ids = torch.tensor([vocab.encode(t) for t in train_text], dtype=torch.long)
        del train_text
        val_text = articles(data.vali_set)
        val_ids = torch.tensor([vocab.encode(t) for t in val_text], dtype=torch.long)
    # Seed residual initialization independently of reference deserialization.
    torch.manual_seed(args.seed)
    compact = CompactPath(WIDTH, text_dim=TEXT_DIM if vocab is not None else 0)
    encoder = RawTextEncoder(len(vocab), output_dim=TEXT_DIM, embedding_dim=64) if vocab is not None else None
    model = PathForecast(compact, encoder)
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
                              else "unexpected_noncanonical_input",
                  release_timestamps_verified=False, test_scored=False, scale_fit="training_only",
                  training_loss="SmoothL1 on all 140 OHLC return steps divided by training-only C1 scale",
                  reporting_units="basis-point log return relative to last observed close",
                  comparison_scope="R3 forecast target and scoring metric, not its model or training objective",
                  selection="minimum validation whole-path OHLC return MAE (epoch 0 eligible), patience 5",
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
