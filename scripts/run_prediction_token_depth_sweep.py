#!/usr/bin/env python3
"""Run depth 8/12 concurrently inside one GPU allocation, then compare to depth 4."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth4", type=Path, required=True)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--run-id", default="transformer-depth-sweep-v2")
    a = p.parse_args()
    # Both children inherit exactly this one visible GPU. No second allocation.
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Depth sweep requires exactly one visible GPU")
    a.out.mkdir(parents=True, exist_ok=True)
    processes, logs, commands = {}, {}, {}
    started = time.monotonic()
    try:
        for depth in (8, 12):
            command = [sys.executable, "-u", str(ROOT / "scripts/train_prediction_token_transformer.py"),
                "--prices", str(a.prices), "--out", str(a.out / f"depth{depth}"),
                "--code-commit", a.code_commit, "--depth", str(depth),
                "--run-id", f"shared-token-transformer-depth{depth}-v2", "--fixed-batch-size", "204694",
                "--max-updates", "1000", "--eval-every", "25", "--max-training-seconds", "10800"]
            logs[depth] = (a.out / f"depth{depth}.log").open("w")
            processes[depth] = subprocess.Popen(command, stdout=logs[depth], stderr=subprocess.STDOUT,
                                                 start_new_session=True)
            commands[depth] = command
        (a.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        (a.out / "stage.txt").write_text("training-depth-8-and-12-concurrently\n")
        pending = set(processes)
        while pending:
            for depth in list(pending):
                code = processes[depth].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"Depth {depth} process failed with exit {code}; see depth{depth}.log")
                m = json.loads((a.out / f"depth{depth}/metrics.json").read_text())
                if m["status"] != "completed" or m["completed_updates"] != 1000:
                    raise RuntimeError(f"Depth {depth} did not finish the fixed update budget")
                pending.remove(depth)
            if pending:
                time.sleep(2)
        (a.out / "stage.txt").write_text("comparison\n")
        subprocess.run([sys.executable, str(ROOT / "scripts/compare_prediction_token_depths.py"),
            "--depth4", str(a.depth4), "--depth8", str(a.out / "depth8"),
            "--depth12", str(a.out / "depth12"), "--out", str(a.out / "comparison"),
            "--run-id", a.run_id], check=True)
        (a.out / "stage.txt").write_text("completed\n")
        print("CONCURRENT_DEPTH_SWEEP_FINISHED", flush=True)
    except BaseException as exc:
        (a.out / "controller-failure.json").write_text(json.dumps(
            {"error": str(exc), "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in logs.values():
            stream.close()


if __name__ == "__main__":
    main()
