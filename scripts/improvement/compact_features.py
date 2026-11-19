#!/usr/bin/env python3
"""Export small, content-matched pooled features from an existing frozen cache.

No LLM is loaded, no embeddings are generated, and missing entries fail closed.
Use only an operator-verified cache; its checksum and signature are recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from model.text_cache import TextEmbeddingCache, cache_signature
from scripts.freeze_c1_inputs import verify, sha256_file
from data.c1_protocol import build_c1_dataloaders
from data.c1_protocol import _covered
import torch

SNAPSHOT_SHA = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"


class NumericDataset:
    def __init__(self, ids, history, targets):
        self.event_ids = ids
        self.history_bp = torch.from_numpy(history)
        self.targets_bp = torch.from_numpy(targets)

    def __len__(self):
        return len(self.event_ids)


def load_numeric_pack(path, expected_sha, seq_len, expected_counts=None):
    counts = expected_counts or {"train": 6167, "vali": 2302, "test": 2236}
    if sha256_file(Path(path)) != expected_sha:
        raise ValueError("Portable input checksum differs from the export receipt")
    with np.load(path, allow_pickle=False) as pack:
        metadata = json.loads(str(pack["metadata_json"].item()))
        spec = metadata.get("numeric", {})
        if metadata.get("snapshot_sha256") != SNAPSHOT_SHA or spec.get("seq_len") != seq_len:
            raise ValueError("Portable inputs have a different snapshot or lookback")
        ids, roles = pack["event_ids"].astype(str), pack["roles"].astype(str)
        history, targets = pack["history_bp"].astype(np.float32), pack["targets_bp"].astype(np.float32)
        windows = pack["bar_windows"].astype(np.int64)
    n = len(ids)
    if (len(set(ids)) != n or history.shape != (n, 4, seq_len) or targets.shape != (n, 3) or
            windows.shape != (n, 3) or roles.shape != (n,) or set(roles) != set(counts) or
            not np.isfinite(history).all() or not np.isfinite(targets).all()):
        raise ValueError("Invalid portable input shapes, roles, IDs or values")
    if (np.any(windows[:, 0] < 0) or np.any(windows[:, 1] - windows[:, 0] + 1 != seq_len) or
            np.any(windows[:, 2] - windows[:, 1] - 1 != 140)):
        raise ValueError("Portable window spans do not match L/H")
    coverage = {role: _covered(windows[roles == role], int(windows[:, 2].max())) for role in counts}
    for a, b in (("train", "vali"), ("train", "test"), ("vali", "test")):
        if np.any(coverage[a] & coverage[b]):
            raise ValueError("Portable input price bars overlap across partitions")
    result = {}
    for role, count in counts.items():
        mask = roles == role
        if int(mask.sum()) != count:
            raise ValueError("Portable population differs from the approved counts")
        result[role + "_set"] = NumericDataset(ids[mask].tolist(), history[mask], targets[mask])
    computed = max(1., float(np.median(np.abs(targets[roles == "train"]))))
    scale = float(spec["scale_bp"])
    if not np.isfinite(scale) or not np.isclose(scale, computed, rtol=1e-5):
        raise ValueError("Portable return scale differs from the training-only endpoints")
    return SimpleNamespace(**result, task_config=SimpleNamespace(scale_bp=scale))


def load_features(path, expected_sha, event_ids, snapshot_sha):
    if sha256_file(Path(path)) != expected_sha:
        raise ValueError("Pooled feature file checksum differs from the operator receipt")
    with np.load(path, allow_pickle=False) as pack:
        metadata = json.loads(str(pack["metadata_json"].item()))
        if metadata.get("snapshot_sha256") != snapshot_sha or not metadata.get("cache_sha256"):
            raise ValueError("Feature provenance does not match the frozen snapshot")
        ids = pack["event_ids"].astype(str).tolist()
        values = pack["embeddings"].astype(np.float32)
        hashes = pack["normalized_article_sha256"].astype(str)
    if len(ids) != len(set(ids)) or values.ndim != 2 or len(values) != len(ids) or len(hashes) != len(ids):
        raise ValueError("Duplicate IDs or invalid pooled feature shape")
    if not np.isfinite(values).all() or values.shape[1] < 1:
        raise ValueError("Non-finite or empty pooled features")
    positions = {key: index for index, key in enumerate(ids)}
    if any(key not in positions for key in event_ids):
        raise ValueError("Feature file is missing required event IDs")
    indices = [positions[key] for key in event_ids]
    return values[indices], hashes[indices], metadata


def past_donors(event_ids, article_hashes, seed):
    """Same-category/year, strictly earlier recorded date; exclude exact normalized content."""
    rng = np.random.default_rng(seed)
    parts = [key.split("/") for key in event_ids]
    donors = np.full(len(parts), -1, dtype=np.int64)
    for row, (category, filename) in enumerate(parts):
        pool = [i for i, (other, date) in enumerate(parts) if other == category and
                date[:4] == filename[:4] and date[:8] < filename[:8] and article_hashes[i] != article_hashes[row]]
        if pool:
            donors[row] = rng.choice(pool)
    return donors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--text-model-name", default="Llama-3.2-1B")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-prices", action="store_true",
                        help="Also export audited numeric inputs/targets for a laptop without the raw snapshot")
    args = parser.parse_args()
    snapshot = verify(args.snapshot_root)
    if snapshot["content_sha256"] != SNAPSHOT_SHA or sha256_file(args.cache) != args.cache_sha256:
        raise ValueError("Frozen snapshot or source cache checksum differs")
    cache = TextEmbeddingCache.load(args.cache, cache_signature(args.text_model_name, 500, 400, True))
    data = build_c1_dataloaders(args.snapshot_root / "manifest.json", args.snapshot_root / "event",
        args.snapshot_root / "series/SP500.csv", timing_policy="category4_1400_strict_before", task_mode="return_bp")
    ids, vectors, hashes, roles, histories, targets, windows = [], [], [], [], [], [], []
    for role in ("train", "vali", "test"):
        dataset = getattr(data, role + "_set")
        for event_id, item in zip(dataset.event_ids, dataset.list, strict=True):
            text = Path(item[0]).read_text(encoding="utf-8", errors="ignore")
            vector = cache.get(text)
            if vector is None:
                raise ValueError(f"Missing frozen feature for {event_id}; do not fill with a different encoder")
            ids.append(event_id)
            vectors.append(vector.numpy())
            hashes.append(hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest())
            if args.include_prices:
                roles.append(role)
                histories.append(item[6].numpy().T)
                targets.append(item[7].numpy()[[34, 69, 139], 3])
        if args.include_prices:
            windows.extend([[w["start"], w["anchor"], w["stop"]] for w in dataset.windows])
    metadata = dict(snapshot_sha256=SNAPSHOT_SHA, cache_sha256=args.cache_sha256,
                    signature=cache.signature, pooling="existing frozen instance cache",
                    storage_dtype="float16", release_time_verified=False)
    numeric = {}
    if args.include_prices:
        metadata["numeric"] = dict(seq_len=35, pred_len=140, scale_bp=data.task_config.scale_bp,
                                    audit=data.audit, endpoint_horizons=[35, 70, 140])
        numeric = dict(roles=np.asarray(roles), history_bp=np.asarray(histories, dtype=np.float32),
                       targets_bp=np.asarray(targets, dtype=np.float32), bar_windows=np.asarray(windows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, event_ids=np.asarray(ids), embeddings=np.asarray(vectors, dtype=np.float16),
                            normalized_article_sha256=np.asarray(hashes), metadata_json=np.asarray(json.dumps(metadata)), **numeric)
    receipt = dict(path=str(args.out), sha256=sha256_file(args.out), bytes=args.out.stat().st_size,
                   rows=len(ids), dimensions=len(vectors[0]), includes_prices=args.include_prices)
    with Path(str(args.out) + ".receipt.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
