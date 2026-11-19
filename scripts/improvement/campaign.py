#!/usr/bin/env python3
"""Render a bounded two-lane PBS campaign; run immutable tasks inside allocations.

Rendering never submits. Submit the generated submit.py once on Nicole's resource.
The site-owned profile supplies PBS directives, paths and environment setup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "experiments/improvement-v1/matrix.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def options(config):
    result = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                result.append("--" + key)
        else:
            result.extend(["--" + key, str(value)])
    return result


def resolved(matrix, variant):
    if variant not in matrix["variants"]:
        raise ValueError(f"Unknown variant: {variant}")
    return {**matrix["defaults"], **matrix["variants"][variant]["overrides"]}


def validate_readiness(receipt, commit, matrix):
    if receipt.get("source_commit") != commit or receipt.get("snapshot_sha256") != matrix["snapshot_sha256"]:
        raise ValueError("Readiness evidence must match this source commit and frozen snapshot")
    required = ("failure_diagnosed", "disk_and_quota_budget_checked", "window_audits_passed",
                "stage_handoff_smoke_passed", "gradient_smokes_passed")
    if any(receipt.get("checks", {}).get(key) is not True for key in required):
        raise ValueError("Full training requires verified failure, storage, data and model readiness checks")
    if receipt.get("timing_scope") not in ("date-screened-sensitivity", "publication-verified"):
        raise ValueError("Declare the timing scope explicitly")
    if not receipt.get("evidence"):
        raise ValueError("Readiness receipt needs evidence paths and checksums")
    if receipt["timing_scope"] == "publication-verified" and not receipt.get("timestamp_ledger_sha256"):
        raise ValueError("Verified timing requires a versioned timestamp ledger")


def render(args):
    matrix, profile = read(MATRIX), read(args.profile)
    required = ("checkout", "python", "snapshot", "moment", "text_model", "run_root",
                "pbs_directives", "setup", "walltime_hours")
    if any(k not in profile for k in required):
        raise ValueError(f"Profile requires {required}")
    if not 0 < profile["walltime_hours"] <= 8:
        raise ValueError("Each job must have a bounded walltime of at most 8 hours")
    if args.phase == "smoke" and profile["walltime_hours"] > 1:
        raise ValueError("Audit-first smoke jobs are capped at one hour each")
    # Site directives are private and reviewed by Nicole before submission.
    if not any("walltime=" in line for line in profile["pbs_directives"]):
        raise ValueError("PBS directives must include a walltime")
    walltimes = [line.split("walltime=", 1)[1].strip() for line in profile["pbs_directives"]
                 if "walltime=" in line]
    expected_walltime = f"{int(profile['walltime_hours']):02}:00:00"
    if walltimes != [expected_walltime] or profile["walltime_hours"] != int(profile["walltime_hours"]):
        raise ValueError("Use exactly one PBS walltime matching the integer walltime_hours cap")
    for key in ("checkout", "python", "snapshot", "moment", "text_model", "run_root"):
        if not str(profile[key]).startswith("/"):
            raise ValueError(f"Use an absolute Linux resource path for {key}")
    variants = args.variants or matrix["screen"]
    if len(variants) != len(set(variants)) or not variants:
        raise ValueError("Variant list must be nonempty and unique")
    if len(args.seeds) != len(set(args.seeds)) or any(s not in (0, 1, 2) for s in args.seeds):
        raise ValueError("Use unique prespecified seeds 0, 1, 2")
    if len(variants) * len(args.seeds) > 18:
        raise ValueError("At most 18 training tasks per explicit batch")
    if not 1 <= args.slots <= 2:
        raise ValueError("Nicole's limit is two concurrent single-GPU jobs")
    if args.split == "test" and not args.selection:
        raise ValueError("Test scoring requires a locked validation selection JSON")
    selection = read(args.selection) if args.selection else None
    if selection and (set(selection["variants"]) != set(variants) or
                      not selection.get("validation_evidence") or not selection.get("rule")):
        raise ValueError("Selection must lock exactly these variants, evidence paths, and rule")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit the campaign before rendering a source-pinned batch")
    readiness = read(args.readiness) if args.readiness else None
    if args.phase == "full":
        if readiness is None:
            raise ValueError("Full training requires --readiness; default phase runs smokes only")
        validate_readiness(readiness, commit, matrix)
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    resource_dir = PurePosixPath(profile["run_root"]) / destination.name
    tasks = []
    for variant in variants:
        for seed in args.seeds:
            config = resolved(matrix, variant)
            task_id = f"{variant}-s{seed}"
            task = dict(id=task_id, variant=variant, seed=seed, split=args.split, phase=args.phase,
                        source_commit=commit, config=config, profile=profile,
                        snapshot_sha256=matrix["snapshot_sha256"],
                        manifest_sha256=matrix["manifest_sha256"],
                        expected_counts=matrix["expected_counts"],
                        run_dir=str(resource_dir / task_id))
            task_path = destination / (task_id + ".json")
            task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
            argv = [profile["python"], str(PurePosixPath(profile["checkout"]) / "scripts/improvement/campaign.py"),
                    "worker", "--task", str(resource_dir / task_path.name)]
            job = ["#!/bin/bash", "set -euo pipefail", "umask 007", profile["setup"],
                   "cd " + shlex.quote(profile["checkout"]), shlex.join(argv)]
            # PBS requires directives before the first executable statement.
            job[1:1] = profile["pbs_directives"]
            script = destination / (task_id + ".pbs")
            script.write_text("\n".join(job) + "\n", encoding="utf-8", newline="\n")
            tasks.append(dict(id=task_id, script=script.name, sha256=sha(script),
                              task_file=task_path.name, task_sha256=sha(task_path)))
    plan = dict(source_commit=commit, slots=args.slots, tasks=tasks, selection=selection,
                phase=args.phase, readiness=readiness,
                resource_dir=str(resource_dir), matrix_sha256=sha(MATRIX),
                max_gpu_hours=len(tasks) * profile["walltime_hours"],
                status="rendered; no jobs submitted")
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    # Copy the standard-library submitter so the rendered package is self-contained.
    (destination / "submit.py").write_text((ROOT / "scripts/improvement/submit.py").read_text(), encoding="utf-8")
    print(json.dumps({k: v for k, v in plan.items() if k != "tasks"}, indent=2))
    print(f"Rendered {len(tasks)} jobs. Copy this directory to {resource_dir}; review then run submit.py.")


def run(command, log, env):
    with Path(log).open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def worker(args):
    task = read(args.task)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != task["source_commit"] or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Worker needs the clean pinned code checkout")
    output = Path(task["run_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    profile, config = task["profile"], task["config"]
    python = profile["python"]
    snapshot = Path(profile["snapshot"])
    env = {**os.environ, "C1_SNAPSHOT_SHA256": task["snapshot_sha256"],
           "WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"}
    log = output / "execution.log"
    try:
        minimum_free = float(profile.get("min_free_disk_gib", 20)) * 1024 ** 3
        if shutil.disk_usage(output).free < minimum_free:
            raise ValueError("Insufficient output filesystem space for the declared disk budget")
        run([python, "scripts/freeze_c1_inputs.py", "--verify", str(snapshot),
             "--expect-content-sha256", task["snapshot_sha256"]], log, env)
        if sha(snapshot / "manifest.json") != task["manifest_sha256"]:
            raise ValueError("Snapshot manifest is not the approved screened C1 manifest")
        audit = output / "audit.json"
        run([python, "scripts/audit_c1_protocol.py", "--snapshot-root", str(snapshot),
             "--forecast-task", "return_bp", "--timing-policy", config["c1-timing-policy"],
             "--seq-len", str(config["seq-len"]), "--pred-len", "140", "--out", str(audit),
             "--windows-csv", str(output / "windows.csv")], log, env)
        if read(audit)["emitted_counts"] != task["expected_counts"]:
            raise ValueError("Input population changed; new matched-population design required")
        shared = options({**config, "seed": task["seed"], "moment-path": profile["moment"],
                          "text-model-path": profile["text_model"],
                          "c1-manifest": str(snapshot / "manifest.json"),
                          "event-dir": str(snapshot / "event"),
                          "series-dir": str(snapshot / "series")})
        # Each configuration must pass its own three-stage/reload smoke first.
        run([python, "scripts/improvement/smoke.py", *shared,
             "--output-path", str(output / "smoke")], log, env)
        if task["phase"] == "smoke":
            (output / "status.json").write_text(json.dumps(dict(status="smoke_passed",
                source_commit=commit, note="Execution only; no full model or performance result"), indent=2) + "\n")
            return
        train = output / "full"
        run([python, "train.py", *shared, "--output-path", str(train)], log, env)
        checkpoint = train / "best_model.pth"
        if not checkpoint.is_file():
            raise ValueError("Full run did not produce a validation-selected checkpoint")
        controls = [python, "scripts/run_text_controls.py", "--model-path", str(checkpoint),
                    "--task-config-json", str(train / "resolved_config.json"),
                    "--text-model-path", profile["text_model"], "--moment-path", profile["moment"],
                    "--c1-manifest", str(snapshot / "manifest.json"), "--event-dir", str(snapshot / "event"),
                    "--series-dir", str(snapshot / "series"), "--forecast-task", "return_bp",
                    "--c1-timing-policy", config["c1-timing-policy"], "--no-4bit", "--tf32",
                    "--fusion-type", config["fusion-type"], "--gate-type", config["gate-type"],
                    "--checkpoint-training-split", "improvement-v1-screened-c1",
                    "--split", task["split"], "--max-events", "0", "--batch-size", "2",
                    "--shuffles", "3", "--seed", "23"]
        for mode in ("zeroed_article_tokens", "fusion_bypass_ts_inst"):
            predictions = output / (task["split"] + "-" + mode + ".npz")
            run([*controls, "--no-text-mode", mode, "--out", str(predictions)], log, env)
            run([python, "scripts/analyze_return_predictions.py", "--predictions", str(predictions),
                 "--out", str(predictions.with_suffix(".analysis.json"))], log, env)
        status = dict(status="completed_execution_pending_review", checkpoint_sha256=sha(checkpoint),
                      split=task["split"], source_commit=commit,
                      note="Verify counts, gradients, controls and uncertainty before publishing a finding")
    except Exception as error:
        (output / "status.json").write_text(json.dumps(dict(status="failed", error=str(error)), indent=2))
        raise
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render_parser = commands.add_parser("render")
    render_parser.add_argument("--profile", type=Path, required=True)
    render_parser.add_argument("--out", type=Path, required=True)
    render_parser.add_argument("--variants", nargs="+")
    render_parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    render_parser.add_argument("--slots", type=int, default=1)
    render_parser.add_argument("--split", choices=["vali", "test"], default="vali")
    render_parser.add_argument("--selection", type=Path)
    render_parser.add_argument("--phase", choices=["smoke", "full"], default="smoke")
    render_parser.add_argument("--readiness", type=Path)
    work_parser = commands.add_parser("worker")
    work_parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    (render if args.command == "render" else worker)(args)


if __name__ == "__main__":
    main()
