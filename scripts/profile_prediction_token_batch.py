#!/usr/bin/env python3
"""One isolated, single-GPU training-memory trial for a candidate batch."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.prediction_token_transformer import PredictionTokenTransformer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--depth", type=int, choices=(4, 8, 12), default=4)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This trial requires exactly one visible GPU")
    torch.set_num_threads(8)
    props = torch.cuda.get_device_properties(0)
    report = {"batch_size": a.batch_size, "updates_requested": a.updates, "depth": a.depth,
              "device_name": props.name, "device_total_bytes": props.total_memory,
              "precision": "BF16 autocast, FP32 parameters/AdamW states",
              "activation_checkpointing": a.activation_checkpointing, "attention_batch_chunk": 8192,
              "gradient_accumulation": False, "accepted": False}
    try:
        data = np.load(a.data)
        x = torch.as_tensor(data["x"], device="cuda")
        t = torch.as_tensor(data["t"], device="cuda")
        y = torch.as_tensor(data["y"], device="cuda")
        if a.batch_size > len(x):
            raise ValueError("Do not duplicate observations to inflate a training batch")
        model = PredictionTokenTransformer(depth=a.depth, activation_checkpointing=a.activation_checkpointing,
                                            attention_batch_chunk=8192).cuda().train()
        model.load_state_dict(torch.load(a.initial, map_location="cpu", weights_only=True))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01, fused=True)
        free, total = torch.cuda.mem_get_info()
        untracked = max(0, total - free - torch.cuda.memory_reserved())
        reserve = max(4 * 1024 ** 3, int(.05 * total))
        torch.cuda.reset_peak_memory_stats()
        ix = torch.arange(a.batch_size, device="cuda")
        timings = []
        for _ in range(a.updates):
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(x[ix], t[ix])
                loss = F.smooth_l1_loss(prediction, y[ix])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        report.update(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                      untracked_context_bytes=untracked, reserved_headroom_bytes=reserve,
                      accepted=reserved + untracked <= total - reserve,
                      reason="fits" if reserved + untracked <= total - reserve else "headroom",
                      seconds_per_update=timings, last_loss=loss.item(), last_gradient_norm=float(norm),
                      samples_per_second=a.batch_size / float(np.median(timings[1:] or timings)))
    except torch.OutOfMemoryError:
        report.update(reason="cuda_oom")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            report.update(reason="cuda_oom")
        elif "invalid configuration" in str(exc).lower() or "not supported" in str(exc).lower():
            report.update(reason="kernel_limit", error=str(exc)[:400])
        else:
            raise
    finally:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
