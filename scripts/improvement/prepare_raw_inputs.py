#!/usr/bin/env python3
"""Stage an immutable, independently verified screened-C1 raw-input snapshot.

Run from the campaign checkout with the existing GS-Fuse dependencies installed.
This does not download data, modify the source tree, or load a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_RELATIVE = "splits/manifests/purged_split_c1_date_screen_v1.json"
MANIFEST_SHA256 = "1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1"
CANONICAL_CONTENT_SHA256 = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"
EXPECTED_PRICE_SHA256 = "6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a"
COUNTS = {"train": 6167, "vali": 2302, "test": 2236}
TIMING_POLICY = "category4_1400_strict_before"


def fail_if_exists(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"Destination already exists; refusing to replace it: {path}")


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def canonical_manifest_bytes(repo_root: Path) -> bytes:
    """Read the exact pinned bytes, unaffected by Windows Git EOL conversion."""
    tracked = subprocess.run(
        ["git", "show", f"HEAD:{MANIFEST_RELATIVE}"],
        cwd=repo_root, check=True, capture_output=True,
    ).stdout
    if sha256(tracked) == MANIFEST_SHA256:
        return tracked
    # In case the committed blob itself uses CRLF, permit only the deterministic
    # normalization that reproduces the independently published SHA-256.
    lf = tracked.replace(b"\r\n", b"\n")
    if sha256(lf) == MANIFEST_SHA256:
        return lf
    raise ValueError(
        f"Campaign checkout manifest does not match pinned SHA-256 {MANIFEST_SHA256}; "
        "switch to the correct checkout. Exploratory mode does not relax the manifest."
    )


def manifest_ids(manifest: dict, event_key) -> dict[str, list[str]]:
    ids = {
        role: [event_key(item)[2] for item in manifest[f"{role}_events"]]
        for role in COUNTS
    }
    if {role: len(keys) for role, keys in ids.items()} != COUNTS:
        raise ValueError("Screened C1 manifest has unexpected split counts")
    all_ids = [key for keys in ids.values() for key in keys]
    if len(set(all_ids)) != sum(COUNTS.values()):
        raise ValueError("Duplicate screened C1 event ID")
    return ids


def validate_raw_summaries(event_dir: Path, ids: dict[str, list[str]]) -> None:
    """Reject placeholder or empty article bodies without reporting their text."""
    invalid = []
    for key in (item for split in ids.values() for item in split):
        folder = event_dir / key
        summary = folder / f"{folder.name}_full_summary.txt"
        try:
            # Match C1's decoder; check that decoded, non-whitespace text exists.
            if not summary.read_text(encoding="utf-8", errors="ignore").strip():
                invalid.append(f"{key} (empty summary)")
        except FileNotFoundError:
            invalid.append(f"{key} (missing summary)")
        if len(invalid) == 10:
            break
    if invalid:
        raise ValueError("Raw C1 summary check failed (first 10): " + ", ".join(invalid))


def _unlock_staging_for_cleanup(staged: Path) -> None:
    """Release permissions only on the newly made disposable scratch copy."""
    if staged.exists():
        staged.chmod(stat.S_IRWXU)
        for path in staged.rglob("*"):
            if not path.is_symlink() and (os.name == "nt" or path.is_dir()):
                path.chmod(stat.S_IRWXU if path.is_dir() else stat.S_IRUSR | stat.S_IWUSR)


def prepare(source_root: Path, destination: Path, exploratory: bool) -> dict:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from data.c1_protocol import build_c1_dataloaders, event_key
    from scripts.freeze_c1_inputs import freeze, verify

    source_root = source_root.resolve(strict=True)
    # Resolve the parent, not the final segment: dangling links count as occupied.
    destination = destination.absolute()
    destination = destination.parent.resolve() / destination.name
    sidecar = destination.with_name(destination.name + ".receipt.json")
    for target in (destination, sidecar):
        fail_if_exists(target)
        if target == source_root or source_root in target.parents or target in source_root.parents:
            raise ValueError("Snapshot and receipt must be outside the source tree")

    event_dir = source_root / "event"
    price_path = source_root / "series" / "SP500.csv"
    if not event_dir.is_dir() or not price_path.is_file():
        raise FileNotFoundError(
            "Expected event/ and series/SP500.csv under --source-root. "
            "If the archive has a wrapper folder, point to the inner data folder."
        )

    manifest_bytes = canonical_manifest_bytes(REPO_ROOT)
    expected_ids = manifest_ids(json.loads(manifest_bytes), event_key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.stage-", dir=destination.parent) as tmp:
        scratch = Path(tmp)
        staged = scratch / "snapshot"
        manifest = scratch / "manifest.json"
        manifest.write_bytes(manifest_bytes)
        reserved_receipt = False
        promoted = False
        try:
            old_cwd = Path.cwd()
            try:
                os.chdir(REPO_ROOT)  # native freeze records git HEAD of the campaign checkout
                freeze(manifest, event_dir, price_path, staged)
            finally:
                os.chdir(old_cwd)

            record = verify(staged)  # recompute hashes from the copied bytes
            if record["manifest_sha256"] != MANIFEST_SHA256:
                raise ValueError("Staged manifest does not match the pinned C1 manifest")
            if record["event_folder_count"] != sum(COUNTS.values()):
                raise ValueError("Frozen snapshot has unexpected event count")

            actual_hash = record["content_sha256"]
            canonical = actual_hash == CANONICAL_CONTENT_SHA256
            if not canonical and not exploratory:
                raise ValueError(
                    "Staged input fingerprint differs from the canonical snapshot; "
                    f"expected {CANONICAL_CONTENT_SHA256}, found {actual_hash}. "
                    "No snapshot was published. If this is an authorized alternate "
                    "source, --exploratory requires the same C1 membership and loader checks."
                )

            validate_raw_summaries(staged / "event", expected_ids)
            loaders = build_c1_dataloaders(
                staged / "manifest.json", staged / "event", staged / "series" / "SP500.csv",
                seq_len=35, pred_len=140, timing_policy=TIMING_POLICY, skip_missing=False,
            )
            emitted = {"train": loaders.train_set, "vali": loaders.vali_set, "test": loaders.test_set}
            for role, dataset in emitted.items():
                if list(dataset.event_ids) != expected_ids[role]:
                    raise ValueError(f"Emitted C1 {role} membership/order differs from pinned manifest")
            audit = loaders.audit
            if audit["emitted_counts"] != COUNTS or audit["excluded_counts"]:
                raise ValueError("C1 loader integrity check did not emit all events without exclusions")
            receipt = {
                "schema": "gs-fuse-improvement-stage-v1",
                "classification": "canonical" if canonical else "exploratory",
                "snapshot": record,
                "reference_content_sha256": CANONICAL_CONTENT_SHA256,
                "reference_price_sha256": EXPECTED_PRICE_SHA256,
                "matches_reference_price_bytes": record["price_sha256"] == EXPECTED_PRICE_SHA256,
                "timing_policy": TIMING_POLICY,
                "release_time_verified": False,
                "cross_split_price_bars": 0,
                "emitted_counts": audit["emitted_counts"],
                "excluded_counts": audit["excluded_counts"],
                "covered_bars": audit["covered_bars"],
                "event_ids_sha256": {
                    role: sha256("\n".join(keys).encode()) for role, keys in expected_ids.items()
                },
            }
            fail_if_exists(destination)
            fail_if_exists(sidecar)
            # Use an exclusive write and roll it back if promotion fails.
            with sidecar.open("x", encoding="utf-8", newline="\n") as handle:
                reserved_receipt = True
                handle.write(json.dumps(receipt, indent=2) + "\n")
            # Linux requires a writable directory when changing its parent.
            # Unfreeze only the disposable root during the move; data files and
            # all child directories remain read-only throughout.
            frozen_mode = stat.S_IMODE(staged.stat().st_mode)
            staged.chmod(frozen_mode | stat.S_IWUSR)
            staged.rename(destination)
            destination.chmod(frozen_mode)
            promoted = True
            return receipt
        finally:
            if reserved_receipt and not promoted:
                sidecar.unlink(missing_ok=True)
            _unlock_staging_for_cleanup(staged)
            # Native freeze may leave a partial copy if its own error cleanup
            # encounters read-only attributes (notably when running on Windows).
            for partial in scratch.glob("snapshot.partial-*"):
                _unlock_staging_for_cleanup(partial)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Extracted root with event/ and series/SP500.csv")
    parser.add_argument("--snapshot-root", required=True, type=Path, help="Fresh, local output directory")
    parser.add_argument("--exploratory", "--allow-exploratory-inputs", action="store_true",
                        help="Allow a different content fingerprint, not different C1 events")
    args = parser.parse_args()
    try:
        receipt = prepare(args.source_root, args.snapshot_root, args.exploratory)
    except (ValueError, OSError, subprocess.CalledProcessError, ImportError) as error:
        parser.exit(2, f"Input staging failed: {error}\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
