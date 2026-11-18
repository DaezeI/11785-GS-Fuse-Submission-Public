#!/usr/bin/env python3
"""Collect verified small price-run evidence, export a pack, and upsert locally.

The SSH host/options and private resource root are operator-supplied. The
scheduler must retain completed accounting for all four explicit job IDs.
This command never uploads or publishes anything.
"""
import argparse
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", required=True, help="SSH host or locally configured alias"
    )
    parser.add_argument("--ssh-config", help="Local SSH configuration file")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="SSH -o option; repeat as needed",
    )
    parser.add_argument(
        "--remote-root", required=True, help="Private absolute resource project root"
    )
    parser.add_argument(
        "--jobs", nargs=4, required=True, metavar=("SMOKE", "FULL", "ZERO", "BYPASS")
    )
    parser.add_argument(
        "--private-evidence", required=True, help="Fresh local private directory"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        help="Designated result path; defaults to the seed-specific result directory",
    )
    parser.add_argument("--registry", help="Optional local registry path for a dry run")
    args = parser.parse_args()
    if not args.remote_root.startswith("/") or not all(
        re.fullmatch(r"[0-9]+\.pbs[0-9]+", job) for job in args.jobs
    ):
        parser.error("provide an absolute resource root and four PBS job IDs")
    if args.seed <= 0:
        parser.error("seed must be positive")
    run = f"runs/screened-price-seed{args.seed}-v1"
    files = [
        f"{run}/full/resolved_config.json",
        f"{run}/full/input-fingerprints.json",
        f"{run}/full/checkpoint.sha256",
        f"{run}/paired/zeroed_article_tokens.analysis.json",
        f"{run}/paired/fusion_bypass_ts_inst.analysis.json",
    ]
    evidence = Path(args.private_evidence).resolve()
    evidence.mkdir(mode=0o700, parents=False, exist_ok=False)
    ssh = ["ssh", "-o", "BatchMode=yes"]
    if args.ssh_config:
        ssh.extend(["-F", args.ssh_config])
    for option in args.ssh_option:
        ssh.extend(["-o", option])
    ssh.append(args.host)
    accounting = subprocess.check_output(
        ssh + ["qstat -x -f " + " ".join(args.jobs)], text=True
    )
    blocks = re.split(r"(?m)^Job Id:\s*", accounting)[1:]
    states = {}
    for block in blocks:
        job = block.splitlines()[0].strip()
        state = re.search(r"(?m)^\s*job_state\s*=\s*(\S+)", block)
        status = re.search(r"(?m)^\s*Exit_status\s*=\s*(-?\d+)", block)
        if state and status and state.group(1) == "F":
            states[job] = int(status.group(1))
    if any(states.get(job) != 0 for job in args.jobs):
        raise RuntimeError(
            "jobs are incomplete or failed; inspect scheduler accounting and logs"
        )
    (evidence / "completion.json").write_text(
        json.dumps(
            {
                "stages": dict(
                    zip(
                        ("smoke", "full", "zero", "bypass"),
                        (states[j] for j in args.jobs),
                    )
                )
            },
            indent=2,
        )
        + "\n"
    )
    root = shlex.quote(args.remote_root)
    # Verify on the resource before copying the small aggregate files.
    checks = [
        f"{run}/full/checkpoint.sha256",
        f"{run}/paired/zeroed_article_tokens.sha256",
        f"{run}/paired/fusion_bypass_ts_inst.sha256",
    ]
    command = (
        "cd "
        + root
        + " && "
        + " && ".join("sha256sum -c " + shlex.quote(c) + " >/dev/null" for c in checks)
    )
    subprocess.run(ssh + [command], check=True)
    command = "tar -C " + root + " -cf - " + " ".join(shlex.quote(f) for f in files)
    archive = subprocess.check_output(ssh + [command])
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members = {m.name: m for m in tar.getmembers()}
        if set(members) != set(files) or any(not m.isfile() for m in members.values()):
            raise ValueError("unexpected archive contents")
        for name, member in members.items():
            (evidence / Path(name).name).write_bytes(tar.extractfile(member).read())
    checkpoint_sha = (evidence / "checkpoint.sha256").read_text().split()[0]
    expected_out = REPO / f"results/screened-price-seed{args.seed}-v1"
    out = (REPO / (args.output or expected_out)).resolve()
    if out != expected_out:
        raise ValueError("output must be the designated public result directory")
    command = [
        sys.executable,
        str(REPO / "scripts/research_cycle/export_price.py"),
        "--zero",
        str(evidence / "zeroed_article_tokens.analysis.json"),
        "--bypass",
        str(evidence / "fusion_bypass_ts_inst.analysis.json"),
        "--config",
        str(evidence / "resolved_config.json"),
        "--fingerprints",
        str(evidence / "input-fingerprints.json"),
        "--completion",
        str(evidence / "completion.json"),
        "--checkpoint-sha",
        checkpoint_sha,
        "--seed",
        str(args.seed),
        "--output",
        str(out),
    ]
    subprocess.run(command, check=True, cwd=REPO)
    upsert = [sys.executable, str(REPO / "scripts/research_home.py")]
    if args.registry:
        upsert += ["--registry", str(Path(args.registry).resolve())]
    subprocess.run(
        upsert + ["upsert", str(out / "candidate-record.json")], check=True, cwd=REPO
    )
    print(
        "Verified and exported the completed result; local registry updated. Review before publication."
    )


if __name__ == "__main__":
    main()
