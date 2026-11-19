#!/usr/bin/env python3
"""Synthetic engineering benchmark only: no dataset, quality or trading claim."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from model.compact_return import CompactReturn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("This Python environment has no CUDA support")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()
    model = CompactReturn().to(args.device)
    x, y = torch.randn(64, 4, 140, device=args.device), torch.randn(64, 3, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    times = []
    for step in range(13):
        sync()
        started = time.perf_counter()
        optimizer.zero_grad()
        (model(x) - y).square().mean().backward()
        optimizer.step()
        sync()
        if step >= 3:
            times.append(time.perf_counter() - started)
    model.eval()
    inference = []
    with torch.no_grad():
        for step in range(23):
            sync()
            started = time.perf_counter()
            model(x[:1])
            sync()
            if step >= 3:
                inference.append(time.perf_counter() - started)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    record = dict(kind="synthetic engineering check; no market-quality result", device=args.device,
        torch_version=torch.__version__, cpu_threads=2, seed=0, batch_size=64, seq_len=140,
        parameters=sum(p.numel() for p in model.parameters()), state_bytes=len(stream.getvalue()),
        model_source_sha256=hashlib.sha256((ROOT / "model/compact_return.py").read_bytes()).hexdigest(),
        median_train_step_ms=1000 * statistics.median(times),
        median_batch1_inference_ms=1000 * statistics.median(inference),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
