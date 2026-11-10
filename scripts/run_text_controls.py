#!/usr/bin/env python3
"""Save paired real/shuffled/no-article predictions for an existing checkpoint.

By default this runner uses the original event loader/scaler to diagnose an
existing checkpoint. --c1-manifest selects the shared C1 loader and scaler,
but C1 performance also requires a checkpoint trained under that protocol.
Default 64 events is
an evenly spaced smoke sample; --max-events 0 evaluates the complete split.

Example (in the team's model environment):
 python scripts/run_text_controls.py --model-path outputs/best_model.pth \
   --text-model-path pretrained/Llama-3.2-3B --moment-path pretrained/MOMENT-1-large \
   --out outputs/text_controls.npz --max-events 64 --shuffles 3
 python scripts/analyze_text_predictions.py --predictions outputs/text_controls.npz \
   --out outputs/text_control_analysis.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def shuffle_donors(texts, dates, types, seed, within_type=False):
    """Find an index derangement, then minimize retained date/content matches.

    Exact distinct-date/content matching may be impossible. Fractions and full
    donor IDs are saved rather than silently treating those pairs as shuffled.
    A singleton in a within-type shuffle remains fixed and is reported.
    """
    n = len(texts)
    if n < 2: raise ValueError("At least two events are needed for a shuffle control")
    rng = np.random.default_rng(seed)
    digests = np.array([hashlib.sha256(t.encode()).hexdigest() for t in texts])
    dates, types = np.asarray(dates), np.asarray(types)
    donors = np.arange(n)
    groups = [np.flatnonzero(types == t) for t in np.unique(types)] if within_type else [np.arange(n)]
    for ids in groups:
        if len(ids) < 2: continue
        best, score = None, float("inf")
        for _ in range(128):
            perm = rng.permutation(ids)
            if np.any(perm == ids): continue
            cost = np.sum(dates[perm] == dates[ids]) + np.sum(digests[perm] == digests[ids])
            if cost < score: best, score = perm, cost
            if cost == 0: break
        if best is None:
            # A cyclic shift is always an index derangement for size >= 2.
            best = np.roll(ids, 1)
        donors[ids] = best
    return donors, {
        "seed": int(seed), "within_type": bool(within_type),
        "fixed_index_fraction": float(np.mean(donors == np.arange(n))),
        "same_date_fraction": float(np.mean(dates[donors] == dates)),
        "identical_content_fraction": float(np.mean(digests[donors] == digests)),
    }


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--text-model-path", required=True)
    p.add_argument("--moment-path", required=True)
    p.add_argument("--event-dir", default="data/event")
    p.add_argument("--series-dir", default="data/series")
    p.add_argument("--c1-manifest")
    p.add_argument("--c1-timing-policy", choices=["legacy_all0830_floor",
                   "category4_1400_strict_before"], default="category4_1400_strict_before")
    p.add_argument("--c1-skip-missing", action="store_true")
    p.add_argument("--split", choices=["test", "vali"], default="test")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-events", type=int, default=64)
    p.add_argument("--shuffles", type=int, default=3)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--shuffle-within-type", action="store_true")
    p.add_argument("--no-text-mode", choices=["zeroed_article_tokens", "fusion_bypass_ts_inst"], default="zeroed_article_tokens")
    p.add_argument("--checkpoint-training-split", default="unverified")
    p.add_argument("--fusion-type", choices=["crossattn", "qformer"], default="crossattn")
    p.add_argument("--gate-type", choices=["softmax", "moe"], default="softmax")
    p.add_argument("--text-cache")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.batch_size < 1 or args.max_events < 0 or args.shuffles < 1:
        p.error("batch-size/shuffles must be positive; max-events must be nonnegative")
    return args


def main():
    args = parse_args()
    import torch
    from types import SimpleNamespace
    from data.dataloader import event_set
    from model.no_text_control import predict_without_article
    from test_pretrained_model import load_pretrained_model
    import model.gs_fuse as module

    if not torch.cuda.is_available():
        raise SystemExit("This full-checkpoint runner needs a CUDA GPU; the aggregate analyzer and split audit use CPU.")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    module.USE_4BIT = not args.no_4bit; module.TEXT_CACHE_PATH = args.text_cache
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    if args.c1_manifest:
        from data.c1_protocol import build_c1_dataloaders
        events = build_c1_dataloaders(
            args.c1_manifest, args.event_dir, Path(args.series_dir) / "SP500.csv",
            batch_size=args.batch_size, timing_policy=args.c1_timing_policy,
            skip_missing=args.c1_skip_missing)
    else:
        events = event_set(35, 140, event_id=0, series_id="SP500", shuffle=False,
            batch_size=args.batch_size, event_dir=args.event_dir, series_dir=args.series_dir)
    ds = events.test_set if args.split == "test" else events.vali_set
    ids = np.arange(len(ds.list))
    if args.max_events and args.max_events < len(ids):
        ids = np.linspace(0, len(ids) - 1, args.max_events, dtype=int)
    if len(ids) < 2: raise SystemExit("Need at least two selected examples")
    items = [ds.list[i] for i in ids]
    # Only main articles are needed for inference. Don't read contrastive
    # negatives or sentence reports from held-out/future folders.
    texts = [Path(it[0]).read_text(encoding="utf-8", errors="ignore") for it in items]
    def category(it): return str(it[0]).replace("\\", "/").split("/")[-3]
    types = np.array([str(it[5]) if len(it) == 8 else category(it) for it in items])
    dates = np.array([Path(it[0]).name[:8] for it in items])
    event_ids = np.array([t + "/" + d + ".txt" for t, d in zip(types, dates)])
    if len(set(event_ids)) != len(event_ids): raise SystemExit("Event IDs are not unique")
    donors, protocols = [], []
    for s in range(args.shuffles):
        donor, info = shuffle_donors(texts, dates, types, args.seed + s, args.shuffle_within_type)
        donors.append(donor); protocols.append(info)

    config = SimpleNamespace(model_path=args.model_path, ts_backbone="moment", text_backbone="llama",
        text_model_path=args.text_model_path, moment_path=args.moment_path,
        kronos_model_path=None, kronos_tokenizer_path=None, seq_len=35, pred_len=140,
        window=500, stride=400, batch_size=args.batch_size, decoder_layers=3,
        decoder_heads=8, max_token_num=1024, fusion_type=args.fusion_type, gate_type=args.gate_type,
        num_experts=4, moe_top_k=2, num_queries=16, qformer_layers=2)
    checkpoint_sha = file_hash(args.model_path)
    model = load_pretrained_model(config, d=events.d); model.eval()
    # The legacy encoder's disk cache is keyed only by text. Isolate this run
    # from cached features created under a different backbone/precision.
    cache_tag = f"{checkpoint_sha[:12]}-4bit{not args.no_4bit}-tf32{args.tf32}"
    encoder_cache = args.out.parent / "text_control_cache" / cache_tag
    encoder_cache.mkdir(parents=True, exist_ok=True)
    model.text_encoder.cache_dir = str(encoder_cache)
    conditions = [texts] + [[texts[int(j)] for j in donor] for donor in donors]
    predictions = [[] for _ in conditions]; zero = []
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            seq = torch.stack([it[-2].T for it in batch]).float()
            for c, articles in enumerate(conditions):
                pred, *_ = model.predict_single_case(articles[start:start + len(batch)], seq)
                predictions[c].append(pred.detach().cpu().numpy().transpose(0, 2, 1))
            pred = predict_without_article(model, seq, args.no_text_mode)
            zero.append(pred.detach().cpu().numpy().transpose(0, 2, 1))
            print(f"Paired predictions: {min(start + len(batch), len(items))}/{len(items)}", flush=True)
    train_vol = np.array([np.std(np.diff(np.log(it[3].numpy()[:, 3]))) for it in events.train_set.list])
    metadata = {"checkpoint_id": checkpoint_sha,
        "training_split": args.checkpoint_training_split,
        "evaluation_split": f"c1_{args.split}" if args.c1_manifest else f"original_positional_{args.split}",
        "no_text_definition": args.no_text_mode,
        "shuffle_description": "seeded index derangement over selected evaluation texts; future donors allowed as a diagnostic placebo",
        "shuffle_protocols": protocols, "smoke_subset": len(ids) != len(ds.list),
        "reference_source_commit": "02f2b742960384161a7ea0c79aacabb9eea43d9b",
        "runner_sha256": file_hash(__file__),
        "no_text_control_sha256": file_hash(ROOT / "model/no_text_control.py"),
        "batch_size": args.batch_size, "seed": args.seed, "tf32": args.tf32,
        "selected_source_indices": ids.tolist(),
        "article_content_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in texts],
        "token_cache_policy": "isolated by checkpoint SHA, quantization, and TF32 under output directory",
        "price_sha256": file_hash(Path(args.series_dir) / "SP500.csv"),
        "backbone": Path(args.text_model_path).name, "four_bit": not args.no_4bit,
        "note": ("C1 evaluation; corrected C1 performance requires verified C1 checkpoint training"
                 if args.c1_manifest else "diagnosis of an existing checkpoint; not a leakage-corrected C1 result")}
    if args.c1_manifest:
        metadata["c1_manifest_sha256"] = file_hash(args.c1_manifest)
        metadata["c1_data_audit"] = events.audit
        metadata["c1_selected_windows"] = [ds.windows[int(i)] for i in ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, real_text=np.concatenate(predictions[0]),
        shuffled_text=np.stack([np.concatenate(p) for p in predictions[1:]]), zero_text=np.concatenate(zero),
        targets=np.stack([it[-1].numpy() for it in items]), seqs=np.stack([it[-2].numpy() for it in items]),
        raw_seqs=np.stack([it[3].numpy() for it in items]), train_volatility=train_vol,
        event_ids=event_ids, event_dates=dates, event_types=types,
        shuffle_donor_ids=np.stack([event_ids[p] for p in donors]),
        metadata_json=np.array(json.dumps(metadata)))
    print(f"Saved paired predictions to {args.out}")


if __name__ == "__main__": main()
