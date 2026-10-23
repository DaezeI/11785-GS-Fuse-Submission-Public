#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Encode every document the event dataset can feed to GSFuse.text_embedding, once.

Covers main summaries, sentiment reports and same-type negatives of all splits. The output
is resumable: rerunning skips documents already in the file.

  python scripts/precompute_text_cache.py --text-model-path pretrained/Llama-3.2-1B \
      --out cache/text_inst_llama1b.pt [--tf32]
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import event_set  # noqa: E402
from model.text_cache import TextEmbeddingCache, cache_signature  # noqa: E402
from model.text_encoders import build_text_encoder  # noqa: E402


def read(path):
    # Must match the dataset's __getitem__ so cache keys line up with training-time text.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-backbone", choices=["llama", "phi"], default="llama")
    p.add_argument("--text-model-path", required=True)
    p.add_argument("--event-dir", default="data/event")
    p.add_argument("--series-dir", default="data/series")
    p.add_argument("--series-id", default="SP500")
    p.add_argument("--event-id", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=35)
    p.add_argument("--pred-len", type=int, default=140)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--stride", type=int, default=400)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--use-4bit", action="store_true", help="Load the text backbone in 4-bit NF4 (must match training)")
    p.add_argument("--out", required=True)
    p.add_argument("--save-every", type=int, default=5000)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signature = cache_signature(args.text_model_path, args.window, args.stride, args.tf32, use_4bit=args.use_4bit)

    ev = event_set(args.seq_len, args.pred_len, event_id=args.event_id, series_id=args.series_id,
                   shuffle=False, batch_size=1, scale=True,
                   event_dir=args.event_dir, series_dir=args.series_dir)
    paths = set()
    for ds in (ev.train_set, ev.vali_set, ev.test_set):
        for item in ds.list:
            paths.add(item[0])
            paths.update(item[1])
            paths.update(item[2])
    paths = sorted(paths)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        cache = TextEmbeddingCache.load(args.out, signature)
        print(f"Resuming: {len(cache)} embeddings already in {args.out}")
    else:
        cache = TextEmbeddingCache(signature)

    encoder = build_text_encoder(args.text_backbone, args.text_model_path, use_4bit=args.use_4bit)
    todo = [p for p in paths if read(p) not in cache]
    print(f"{len(paths)} unique documents, {len(todo)} to encode | signature {signature}")

    start = time.time()
    for i, path in enumerate(todo, 1):
        text = read(path)
        cache.put(text, encoder.encode_instance(text, args.window, args.stride, device))
        if i % args.save_every == 0 or i == len(todo):
            cache.save(args.out)
        if i % 500 == 0 or i == len(todo):
            rate = i / (time.time() - start)
            print(f"[{i}/{len(todo)}] {rate:.1f} docs/s | ETA {(len(todo) - i) / rate / 3600:.2f} h", flush=True)
    print(f"Done: {len(cache)} embeddings in {args.out}")


if __name__ == "__main__":
    main()
