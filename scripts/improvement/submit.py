#!/usr/bin/env python3
"""Submit a rendered PBS batch once, recording every scheduler response.

Nicole checks the live queue first and supplies the currently free campaign lanes.
This program is copied into each private rendered campaign directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(directory, available_slots):
    directory = directory.resolve()
    plan = json.loads((directory / "plan.json").read_text())
    if str(directory) != plan["resource_dir"]:
        raise ValueError("Install the complete package at its recorded resource_dir first")
    if available_slots not in (1, 2) or available_slots > plan["slots"]:
        raise ValueError("Available lanes must respect the planned and global two-job limit")
    for task in plan["tasks"]:
        if digest(directory / task["script"]) != task["sha256"] or digest(
                directory / task["task_file"]) != task["task_sha256"]:
            raise ValueError(f"Rendered package changed: {task['id']}")
    previous = [None] * available_slots
    # An exclusive journal prevents a second batch submission after partial failure.
    with (directory / "submission.jsonl").open("x", encoding="utf-8") as journal:
        journal.write(json.dumps(dict(status="submission_started", available_slots=available_slots,
                                      time=datetime.now(timezone.utc).isoformat())) + "\n")
        journal.flush()
        for index, task in enumerate(plan["tasks"]):
            lane = index % available_slots
            command = ["qsub"]
            if previous[lane]:
                command.extend(["-W", "depend=afterok:" + previous[lane]])
            command.append(str(directory / task["script"]))
            response = subprocess.run(command, text=True, capture_output=True)
            raw = response.stdout.strip()
            entry = dict(task=task["id"], lane=lane, dependency=previous[lane],
                         command=command, returncode=response.returncode, response=raw,
                         stderr=response.stderr.strip(), time=datetime.now(timezone.utc).isoformat())
            # Save even unexpected successful responses before checking their syntax.
            journal.write(json.dumps(entry) + "\n")
            journal.flush()
            if response.returncode or not re.fullmatch(r"[0-9]+\.[A-Za-z0-9._-]+", raw):
                raise RuntimeError("Submission stopped; inspect the journal and queue before recovery")
            previous[lane] = raw
            print(f"{task['id']}: {raw} lane={lane}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--available-slots", type=int, choices=[1, 2], required=True,
                        help="Free slots after checking all Nicole-owned live GPU jobs")
    args = parser.parse_args()
    submit(args.directory, args.available_slots)


if __name__ == "__main__":
    main()
