#!/usr/bin/env python3
"""Copy screened C1 inputs to a new, content-verified, immutable run source."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def event_folders(manifest):
    folders = set()
    for role in ("train", "vali", "test"):
        for item in manifest[role + "_events"]:
            parts = str(item).replace("\\", "/").split("/")
            category, filename = parts[-2:]
            if not category.isdigit() or not filename.endswith(".txt"):
                raise ValueError(f"Invalid event path: {item}")
            folders.add((category, filename))
    return sorted(folders)


def content_inventory(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p != root / "snapshot.json")
    h = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=16) as workers:
        for path, digest in zip(files, workers.map(sha256_file, files)):
            relative = path.relative_to(root).as_posix()
            h.update(f"{relative}\t{digest}\n".encode())
    return {"file_count": len(files), "content_sha256": h.hexdigest()}


def verify(root):
    root = Path(root)
    record = json.loads((root / "snapshot.json").read_text())
    actual = content_inventory(root)
    if actual != {k: record[k] for k in ("file_count", "content_sha256")}:
        raise ValueError("Frozen C1 snapshot content differs from its recorded fingerprint")
    for name, relative in (("manifest_sha256", "manifest.json"),
                           ("price_sha256", "series/SP500.csv")):
        if sha256_file(root / relative) != record[name]:
            raise ValueError(f"Frozen C1 snapshot {relative} differs from its recorded hash")
    return record


def freeze(manifest_path, event_dir, series_path, destination):
    manifest_path, event_dir, series_path, destination = map(
        Path, (manifest_path, event_dir, series_path, destination))
    if os.path.lexists(destination):
        raise FileExistsError(f"Snapshot already exists: {destination}")
    manifest = json.loads(manifest_path.read_text())
    folders = event_folders(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        shutil.copy2(manifest_path, temp / "manifest.json")
        (temp / "series").mkdir()
        shutil.copy2(series_path, temp / "series" / "SP500.csv")
        for category in {category for category, _ in folders}:
            (temp / "event" / category).mkdir(parents=True)

        def copy_event(folder):
            category, filename = folder
            source = event_dir / category / filename
            target = temp / "event" / category / filename
            # Content and relative paths are the snapshot contract. Avoid
            # copying source ACLs and timestamps onto every small report file.
            shutil.copytree(source, target, copy_function=shutil.copyfile)
            for path in target.rglob("*"):
                if path.is_file():
                    path.chmod(0o440)

        with ThreadPoolExecutor(max_workers=16) as workers:
            for count, _ in enumerate(workers.map(copy_event, folders), 1):
                if count % 1000 == 0:
                    print(f"Copied {count}/{len(folders)} event folders", file=sys.stderr, flush=True)
        record = {
            "schema": "c1-screened-input-snapshot-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "event_folder_count": len(folders),
            "manifest_sha256": sha256_file(temp / "manifest.json"),
            "price_sha256": sha256_file(temp / "series" / "SP500.csv"),
            **content_inventory(temp),
        }
        (temp / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")
        for path in (temp / "manifest.json", temp / "series" / "SP500.csv", temp / "snapshot.json"):
            path.chmod(0o440)
        if os.path.lexists(destination):
            raise FileExistsError(f"Snapshot already exists: {destination}")
        os.rename(temp, destination)
        for path in sorted(destination.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o2550)
        destination.chmod(0o2550)
        return record
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--event-dir", type=Path)
    p.add_argument("--series", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--verify", type=Path, help="Check an existing snapshot without changing it")
    p.add_argument("--expect-content-sha256", help="Require the exact frozen content fingerprint")
    args = p.parse_args()
    if args.expect_content_sha256 and not args.verify:
        p.error("--expect-content-sha256 requires --verify")
    if args.verify:
        if any(x is not None for x in (args.manifest, args.event_dir, args.series, args.out)):
            p.error("--verify cannot be combined with copy arguments")
        record = verify(args.verify)
    else:
        if any(x is None for x in (args.manifest, args.event_dir, args.series, args.out)):
            p.error("copying requires --manifest, --event-dir, --series, and --out")
        record = freeze(args.manifest, args.event_dir, args.series, args.out)
    if args.expect_content_sha256 and record["content_sha256"] != args.expect_content_sha256:
        raise ValueError("Frozen C1 snapshot has a different content fingerprint")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
