#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small-subset end-to-end run of the official GS-FUSE three-stage pipeline.

Builds the same datasets and model as train.py, then truncates every loader to a
few samples so the full Stage1 -> Stage2 -> Stage3 path can be exercised on a
laptop without a GPU. Subsets are spread evenly over each split because the
event data is grouped by event type.
"""

import argparse
import json
import os
import warnings

import torch
from torch.utils.data import DataLoader, Subset

from data.dataloader import event_set
from data.sliding_window_dataloader import sliding_window_set
from interpret import evenly_spaced, run_and_log
import model.gs_fuse as gs_fuse_mod
from model import tracking
from model.gs_fuse import GSFuse, train_three_stage

warnings.simplefilter("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="GS-FUSE smoke run on small data subsets")
    p.add_argument("--ts-backbone", choices=["moment", "kronos"], default="moment")
    p.add_argument("--text-backbone", choices=["llama", "phi"], default="llama")
    p.add_argument("--moment-path", type=str, default=None)
    p.add_argument("--kronos-model-path", type=str, default=None)
    p.add_argument("--kronos-tokenizer-path", type=str, default=None)
    p.add_argument("--text-model-path", type=str, required=True)
    p.add_argument("--event-dir", type=str, default="data/event")
    p.add_argument("--series-dir", type=str, default="data/series")
    p.add_argument("--output-path", type=str, required=True)
    p.add_argument("--series-id", type=str, default="SP500")
    p.add_argument("--seq-len", type=int, default=35)
    p.add_argument("--pred-len", type=int, default=140)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-token-num", type=int, default=1024)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-epochs", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--n-event-train", type=int, default=8)
    p.add_argument("--n-event-eval", type=int, default=4)
    p.add_argument("--n-sw-train", type=int, default=64)
    p.add_argument("--n-sw-eval", type=int, default=32)
    p.add_argument("--n-interpret", type=int, default=24)
    p.add_argument("--skip-checkpoints", action="store_true")
    p.add_argument("--fusion-type", choices=["crossattn", "qformer"], default="crossattn")
    p.add_argument("--gate-type", choices=["softmax", "moe"], default="softmax")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--moe-top-k", type=int, default=2)
    p.add_argument("--num-queries", type=int, default=16)
    p.add_argument("--qformer-layers", type=int, default=2)
    p.add_argument("--text-cache", type=str, default=None)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--no-4bit", action="store_true", help="Load the text backbone in full precision instead of 4-bit")
    tracking.add_cli_args(p)
    return p.parse_args()


def subset_loader(dataset, n, batch_size, shuffle=False):
    return DataLoader(Subset(dataset, evenly_spaced(len(dataset), n)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():
    args = parse_args()
    torch.manual_seed(0)
    gs_fuse_mod.TEXT_CACHE_PATH = args.text_cache
    if args.no_4bit:
        gs_fuse_mod.USE_4BIT = False
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    events = event_set(
        args.seq_len, args.pred_len, event_id=0, series_id=args.series_id,
        shuffle=False, batch_size=args.batch_size, scale=True,
        event_dir=args.event_dir, series_dir=args.series_dir,
    )
    sw_data = sliding_window_set(
        seq_len=args.seq_len, pred_len=args.pred_len, series_id=args.series_id,
        series_dir=args.series_dir, stride=1, batch_size=args.batch_size, num_workers=0,
    )

    bs = args.batch_size
    train_loader = subset_loader(events.train_set, args.n_event_train, bs)
    vali_loader = subset_loader(events.vali_set, args.n_event_eval, bs)
    test_loader = subset_loader(events.test_set, args.n_event_eval, bs)
    sw_train_loader = subset_loader(sw_data.train_set, args.n_sw_train, bs, shuffle=True)
    sw_vali_loader = subset_loader(sw_data.vali_set, args.n_sw_eval, bs)

    model = GSFuse(
        text_model_path=args.text_model_path,
        ts_backbone=args.ts_backbone,
        text_backbone=args.text_backbone,
        moment_path=args.moment_path,
        kronos_model_path=args.kronos_model_path,
        kronos_tokenizer_path=args.kronos_tokenizer_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d=events.d,
        batch_size=bs,
        decoder_layers=3,
        decoder_heads=8,
        use_ts_memory=False,
        max_token_num=args.max_token_num,
        fusion_type=args.fusion_type,
        gate_type=args.gate_type,
        num_experts=args.num_experts,
        moe_top_k=args.moe_top_k,
        num_queries=args.num_queries,
        qformer_layers=args.qformer_layers,
    )
    if args.skip_checkpoints:
        model.save_model_combined = lambda save_path=None, **kw: print(f"[smoke] checkpoint skipped: {save_path}")

    name = f"fusion={args.fusion_type}, gate={args.gate_type}"
    tracking.init_from_args(args, default_name=f"smoke_{args.fusion_type}_{args.gate_type}")
    try:
        train_three_stage(
            model, train_loader, test_loader, vali_loader,
            sw_train_loader=sw_train_loader,
            sw_vali_loader=sw_vali_loader,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            num_epochs=args.num_epochs,
            output_path=args.output_path,
            contrastive_lambda=1.0,
            eval_every_steps=None,
        )
        if args.n_interpret > 0:
            summary = run_and_log(model, events.test_set, args.n_interpret,
                                  os.path.join(args.output_path, "interpret"), name, bs)
            print(json.dumps(summary, ensure_ascii=False, indent=1))
    finally:
        tracking.finish()


if __name__ == "__main__":
    main()
