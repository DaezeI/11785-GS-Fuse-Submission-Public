#!/usr/bin/env python3
"""Read-only disk/GPU inventory and failure-signature triage; never deletes files."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def classify_log(text):
    lower = text.lower()
    signatures = {
        "disk_space": ("no space left on device", "errno 28"),
        "disk_quota": ("disk quota exceeded", "storage quota exceeded", "edquot"),
        "gpu_memory": ("cuda out of memory", "torch.outofmemoryerror"),
        "walltime": ("walltime exceeded", "due to time limit", "walltime limit"),
    }
    found = [kind for kind, needles in signatures.items() if any(n in lower for n in needles)]
    return found or ["unknown; inspect traceback and scheduler accounting"]


def disk_record(path):
    requested = Path(path).resolve()
    existing = requested
    while not existing.exists():
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return dict(requested_path=str(requested), inspected_path=str(existing),
                free_bytes=usage.free, used_bytes=usage.used, total_bytes=usage.total,
                quota_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, action="append", required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = dict(disks=[disk_record(p) for p in args.path], failures=[], gpu=None,
                  note="Private diagnostic. Filesystem free space does not verify user/group quotas.")
    for path in args.log:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
            tail = stream.read().decode("utf-8", errors="replace")
        record["failures"].append(dict(log=str(path), signatures=classify_log(tail)))
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                                 "--format=csv,noheader"], capture_output=True, text=True)
        record["gpu"] = result.stdout.strip() if result.returncode == 0 else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(f"Saved private read-only diagnostics to {args.out}")


if __name__ == "__main__":
    main()
