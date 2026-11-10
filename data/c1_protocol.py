"""C1 event and Stage-1 loaders with one audited price timeline and scaler.

The manifest stores Windows paths such as ``data/event\\4\\20200101.txt``.
Those paths identify an event directory, not a summary file.  No publication
timestamp is supplied by the manifest, so ``timing_policy`` is an explicit
assumption rather than a claim about historical release times.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left
import hashlib
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data.sliding_window_dataloader import SlidingWindowDataset

ROLES = ("train", "vali", "test")
POLICIES = ("legacy_all0830_floor", "category4_1400_strict_before")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_key(manifest_path):
    parts = str(manifest_path).replace("\\", "/").split("/")
    if len(parts) < 2 or not parts[-2].isdigit() or not parts[-1].endswith(".txt"):
        raise ValueError(f"Invalid manifest event path: {manifest_path}")
    date = parts[-1][:-4]
    pd.to_datetime(date, format="%Y%m%d", errors="raise")
    return parts[-2], date, f"{parts[-2]}/{date}.txt"


def load_price_timeline(series_path):
    raw = pd.read_csv(series_path)
    required = ("date", "date_int", "Open", "High", "Low", "Close")
    if any(col not in raw for col in required):
        raise ValueError(f"Price CSV must have {required}")
    if raw["date_int"].isna().any() or raw[list(required[2:])].isna().any().any():
        raise ValueError("Price CSV contains missing timestamps or prices")
    conflicts = raw.groupby("date_int", sort=False)[list(required[2:])].nunique()
    if (conflicts > 1).any().any():
        raise ValueError("Price CSV has conflicting duplicate timestamps")
    duplicates = int(raw.duplicated("date_int").sum())
    clean = raw.drop_duplicates("date_int").sort_values("date_int").reset_index(drop=True)
    stamps = clean["date_int"].to_numpy(dtype=np.int64)
    values = clean[list(required[2:])].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Price CSV contains non-finite or non-positive OHLC values")
    return stamps, values, {"raw_rows": len(raw), "clean_rows": len(clean),
                            "exact_duplicate_rows": duplicates}


def window_for_event(category, date, stamps, seq_len, pred_len, timing_policy):
    if timing_policy not in POLICIES:
        raise ValueError(f"Unknown timing policy: {timing_policy}")
    hour = "140000" if category == "4" and timing_policy == POLICIES[1] else "083000"
    side = "left" if timing_policy == POLICIES[1] else "right"
    anchor = int(np.searchsorted(stamps, int(date + hour), side=side)) - 1
    start, stop = anchor - seq_len + 1, anchor + pred_len + 1  # stop is exclusive
    return (start, anchor, stop) if start >= 0 and stop <= len(stamps) else None


def _report_files(folder):
    files = sorted(folder.glob("*.txt"))
    summary = folder / (folder.name[:-4] + ".txt_full_summary.txt")
    sent = []
    for n in range(11):
        matches = [p for p in files if p.name.endswith(f".txt_report_sent{n}.txt")]
        if matches:
            sent.append(matches[-1])
    return summary, sent[:10]


class C1EventDataset(Dataset):
    """Seven returned values match the existing GS-Fuse training batch contract."""

    def __init__(self, items, event_ids, windows):
        self.list = items
        self.event_ids = event_ids
        self.windows = windows

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        summary, sent, negatives, seq, target, _category, scaled_seq, scaled_target = self.list[idx]
        def read(path):
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        return (read(summary), [read(p) for p in sent], [read(p) for p in negatives],
                seq.T, target.T, scaled_seq.T, scaled_target.T)


@dataclass
class C1Loaders:
    train_set: C1EventDataset
    vali_set: C1EventDataset
    test_set: C1EventDataset
    train_loader: DataLoader
    vali_loader: DataLoader
    test_loader: DataLoader
    sw_train_set: SlidingWindowDataset
    sw_vali_set: SlidingWindowDataset
    sw_train_loader: DataLoader
    sw_vali_loader: DataLoader
    scaler: StandardScaler
    d: int
    audit: dict


def _covered(windows, n):
    delta = np.zeros(n + 1, dtype=np.int32)
    for start, _anchor, stop in windows:
        delta[start] += 1
        delta[stop] -= 1
    return np.cumsum(delta[:-1]) > 0


def _stage1_indices(coverage, total, stride):
    prefix = np.r_[0, np.cumsum(coverage, dtype=np.int64)]
    starts = np.arange(0, len(coverage) - total + 1, dtype=np.int64)
    valid = starts[prefix[starts + total] - prefix[starts] == total]
    return valid[::stride].tolist()


def build_c1_dataloaders(manifest_path, event_dir, series_path, seq_len=35,
                         pred_len=140, batch_size=8, sw_stride=10,
                         timing_policy=None,
                         num_workers=0, skip_missing=False):
    """Build all three event splits and Stage-1 windows from the same C1 protocol.

    Missing summaries, incomplete sentence reports, missing price windows, and
    insufficient in-split negative summaries fail by default. ``skip_missing``
    permits explicit exclusions; counts and example IDs remain in ``audit``.
    """
    if sw_stride < 1 or batch_size < 1:
        raise ValueError("sw_stride and batch_size must be positive")
    if timing_policy not in POLICIES:
        raise ValueError(f"C1 requires an explicit timing_policy from {POLICIES}")
    manifest = json.loads(Path(manifest_path).read_text())
    stamps, values, price_audit = load_price_timeline(series_path)
    event_dir = Path(event_dir)
    records, excluded = {}, Counter()
    examples = {}
    seen = set()
    for role in ROLES:
        records[role] = []
        for path in manifest[role + "_events"]:
            category, date, key = event_key(path)
            if key in seen:
                raise ValueError(f"Duplicate C1 event ID across manifest: {key}")
            seen.add(key)
            folder = event_dir / category / f"{date}.txt"
            summary, sent = _report_files(folder)
            window = window_for_event(category, date, stamps, seq_len, pred_len, timing_policy)
            reason = ("missing_summary" if not summary.is_file() else
                      "incomplete_sent_reports" if len(sent) != 10 else
                      "missing_price_window" if window is None else None)
            if reason:
                excluded[(role, reason)] += 1
                examples.setdefault(f"{role}:{reason}", []).append(key)
                continue
            records[role].append({"id": key, "category": category, "date": date,
                                  "day_ordinal": datetime.strptime(date, "%Y%m%d").toordinal(),
                                  "summary": summary, "sent": sent, "window": window})
    for role in ROLES:
        # Deterministic nearby negative reports, restricted to this role.
        by_type = {}
        for category in {r["category"] for r in records[role]}:
            options = sorted((r for r in records[role] if r["category"] == category),
                             key=lambda x: (x["date"], x["id"]))
            by_type[category] = (options, [x["date"] for x in options])
        for record in records[role]:
            candidates = []
            for category, (options, dates) in by_type.items():
                if category == record["category"]:
                    continue
                at = bisect_left(dates, record["date"])
                candidates.extend(options[max(0, at - 5):at + 5])
            candidates.sort(key=lambda x: (abs(x["day_ordinal"] - record["day_ordinal"]), x["id"]))
            record["negatives"] = [x["summary"] for x in candidates[:5]]
            if len(record["negatives"]) < 5:
                excluded[(role, "insufficient_negatives")] += 1
                examples.setdefault(f"{role}:insufficient_negatives", []).append(record["id"])
        records[role] = [r for r in records[role] if len(r["negatives"]) == 5]
    audit = {"manifest_counts": {r: len(manifest[r + "_events"]) for r in ROLES},
             "emitted_counts": {r: len(records[r]) for r in ROLES},
             "excluded_counts": {f"{r}:{why}": n for (r, why), n in sorted(excluded.items())},
             "excluded_examples": {k: v[:10] for k, v in examples.items()},
             "price": price_audit, "timing_policy": timing_policy,
             "manifest_sha256": sha256_file(manifest_path),
             "price_sha256": sha256_file(series_path),
             "release_time_verified": False}
    if excluded and not skip_missing:
        raise ValueError("C1 inputs incomplete; use skip_missing only after reviewing: " + json.dumps(audit))
    if any(not records[r] for r in ROLES):
        raise ValueError("At least one C1 split has no usable events: " + json.dumps(audit))
    coverage = {r: _covered([x["window"] for x in records[r]], len(stamps)) for r in ROLES}
    for a, b in (("train", "vali"), ("train", "test"), ("vali", "test")):
        shared = int(np.sum(coverage[a] & coverage[b]))
        if shared:
            raise ValueError(f"C1 emitted full windows overlap: {a}/{b} share {shared} bars")
    scaler = StandardScaler().fit(values[coverage["train"]])
    scaled = scaler.transform(values).astype(np.float32)
    audit["scaler_fit_unique_bars"] = int(coverage["train"].sum())
    audit["scaler_mean"] = scaler.mean_.tolist()
    audit["scaler_scale"] = scaler.scale_.tolist()
    audit["covered_bars"] = {r: int(coverage[r].sum()) for r in ROLES}
    datasets = {}
    for role in ROLES:
        items, ids, windows = [], [], []
        for r in records[role]:
            start, anchor, stop = r["window"]
            seq = torch.from_numpy(values[start:anchor + 1].copy())
            target = torch.from_numpy(values[anchor + 1:stop].copy())
            scale_seq = torch.from_numpy(scaled[start:anchor + 1].copy())
            scale_target = torch.from_numpy(scaled[anchor + 1:stop].copy())
            items.append((str(r["summary"]), [str(p) for p in r["sent"]],
                          [str(p) for p in r["negatives"]], seq, target,
                          r["category"], scale_seq, scale_target))
            ids.append(r["id"])
            windows.append({"event_id": r["id"], "start": start, "anchor": anchor,
                            "stop": stop, "input_start": int(stamps[start]),
                            "target_end": int(stamps[stop - 1])})
        datasets[role] = C1EventDataset(items, ids, windows)
    total = seq_len + pred_len
    sw = {}
    for role in ("train", "vali"):
        indices = _stage1_indices(coverage[role], total, sw_stride)
        if not indices:
            raise ValueError(f"No Stage-1 {role} windows inside C1 event coverage")
        sw[role] = SlidingWindowDataset(values, scaled, seq_len, pred_len, indices)
        audit[f"stage1_{role}_windows"] = len(indices)
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers)
    return C1Loaders(datasets["train"], datasets["vali"], datasets["test"],
                     loader(datasets["train"], True), loader(datasets["vali"], False),
                     loader(datasets["test"], False), sw["train"], sw["vali"],
                     loader(sw["train"], True), loader(sw["vali"], False),
                     scaler, values.shape[1], audit)
