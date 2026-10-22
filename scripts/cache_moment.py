#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline MOMENT-1-large Representation Pre-Extractor & Caching Script.

Pre-computes and saves temporal representations for all unique lookback windows
in the dataset to `data/cache_moment_emb/`. This completely eliminates the need
to run MOMENT forward passes during training and evaluation, reducing epoch
times by up to 5x-10x.
"""

import os
import sys
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

from model.ts_encoders import build_ts_encoder
from data.dataloader import event_set


def parse_args():
    p = argparse.ArgumentParser(description="Pre-extract and cache MOMENT representations")
    p.add_argument("--moment-path", type=str, default="AutonLab/MOMENT-1-large")
    p.add_argument("--series-id", type=str, default="SP500")
    p.add_argument("--seq-len", type=int, default=35)
    p.add_argument("--pred-len", type=int, default=140)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="data/cache_moment_emb")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[MOMENT Cache] Target directory: {args.output_dir}")
    print(f"[MOMENT Cache] Loading encoder: {args.moment_path} on {device}")

    ts_encoder = build_ts_encoder(
        backbone="moment",
        model_path=args.moment_path,
        device=device
    )
    ts_encoder.freeze()
    ts_encoder.eval()

    print("[MOMENT Cache] Loading event dataset to extract all lookback windows...")
    events = event_set(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        series_id=args.series_id,
        shuffle=False,
        batch_size=args.batch_size,
        scale=True,
    )

    all_loaders = [
        ("train", events.train_loader),
        ("vali", events.vali_loader),
        ("test", events.test_loader),
    ]

    total_cached = 0
    total_skipped = 0

    with torch.no_grad():
        for split_name, loader in all_loaders:
            print(f"\n[MOMENT Cache] Processing {split_name} loader ({len(loader)} batches)...")
            for batch in tqdm(loader, desc=f"Caching {split_name}"):
                # Dataloader returns: text, sent_reports, neg_reports, seq, pred, scaled_seq, scaled_pred
                scaled_seq = batch[5]  # [B, 4, L] or [B, L, 4]
                if scaled_seq.dim() == 3 and scaled_seq.size(1) != 4 and scaled_seq.size(2) == 4:
                    scaled_seq = scaled_seq.transpose(1, 2)
                scaled_seq = scaled_seq.float().to(device)

                for i in range(scaled_seq.size(0)):
                    sample = scaled_seq[i:i+1] # [1, 4, L]
                    h_key = hashlib.md5(sample.cpu().numpy().tobytes()).hexdigest()
                    disk_path = os.path.join(args.output_dir, f"{h_key}.pt")

                    if os.path.exists(disk_path):
                        total_skipped += 1
                        continue

                    tok_emb, inst_emb = ts_encoder.encode_per_timestep(sample)
                    torch.save({
                        "tok_emb": tok_emb.cpu(),
                        "inst_emb": inst_emb.cpu(),
                    }, disk_path)
                    total_cached += 1

    print("\n" + "=" * 80)
    print(f"[MOMENT Cache] Complete! Newly cached: {total_cached}, Already cached: {total_skipped}")
    print(f"[MOMENT Cache] Total files in {args.output_dir}: {len(os.listdir(args.output_dir))}")
    print("=" * 80)


if __name__ == "__main__":
    main()
