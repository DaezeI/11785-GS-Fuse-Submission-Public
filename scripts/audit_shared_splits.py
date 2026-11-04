#!/usr/bin/env python3
"""Audit committed split lists against actual bars, with no model/GPU dependency.

python3 scripts/audit_shared_splits.py

Only exact duplicate price rows are removed in the derived data; conflicting
duplicates stop the audit. The downloaded source CSV is never modified.
"""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "vali", "test")
SOURCE_COMMIT = "02f2b742960384161a7ea0c79aacabb9eea43d9b"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prices(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    stamps = np.array([int(r["date_int"]) for r in rows], dtype=np.int64)
    values = np.array([[float(r[k]) for k in ("Open", "High", "Low", "Close")]
                       for r in rows], dtype=np.float64)
    mismatch = sum(r["date"].replace("-", "").replace(":", "").replace(" ", "")
                   != r["date_int"] for r in rows)
    groups = defaultdict(list)
    for i, stamp in enumerate(stamps):
        groups[int(stamp)].append(i)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    conflicts = [k for k, ids in dup.items()
                 if not np.all(values[ids] == values[ids[0]])]
    if conflicts:
        raise ValueError(f"Conflicting OHLC rows at {len(conflicts)} timestamps")
    keep = np.array([groups[k][0] for k in sorted(groups)], dtype=int)
    clean_t, clean_x = stamps[keep], values[keep]
    if not np.isfinite(clean_x).all():
        raise ValueError("Non-finite prices; investigate before evaluating")
    if np.any(clean_x <= 0):
        raise ValueError("Non-positive prices; investigate before evaluating")
    backwards = np.flatnonzero(np.diff(stamps) < 0)
    stats = {
        "source_sha256": sha256(path), "raw_rows": len(stamps),
        "clean_rows": len(clean_t), "timestamp_text_mismatches": mismatch,
        "extra_duplicate_rows": sum(len(ids) - 1 for ids in dup.values()),
        "duplicate_dates": dict(Counter(str(k)[:8] for k in dup)),
        "conflicting_duplicate_timestamps": len(conflicts),
        "backward_jumps": [{"row": int(i), "from": int(stamps[i]),
                             "to": int(stamps[i + 1])} for i in backwards],
        "first_timestamp": int(clean_t[0]), "last_timestamp": int(clean_t[-1]),
        "ohlc_invariant_violations": int(np.sum(
            (clean_x[:, 1] < np.max(clean_x[:, [0, 2, 3]], axis=1)) |
            (clean_x[:, 2] > np.min(clean_x[:, [0, 1, 3]], axis=1)))),
        "timezone": "not recorded in CSV; must be confirmed",
        "bar_label": "start/end meaning not recorded in CSV; must be confirmed",
    }
    if mismatch:
        raise ValueError("date and date_int disagree")
    return clean_t, clean_x, stats


def event_key(path):
    parts = str(path).replace("\\", "/").split("/")
    category, date = parts[-2], parts[-1].split(".")[0]
    datetime.strptime(date, "%Y%m%d")
    return category, date, f"{category}/{date}.txt"


def window_for_event(path, stamps, policy, lookback=35, horizon=140):
    if policy not in ("legacy_all0830_floor", "category4_1400_strict_before"):
        raise ValueError(f"Unknown timestamp policy: {policy}")
    category, date, key = event_key(path)
    # Neither policy establishes the real publication time. The second is a
    # sensitivity check of the category-4 assumption in the uploaded builder.
    time = "140000" if policy == "category4_1400_strict_before" and category == "4" else "083000"
    origin = int(date + time)
    side = "left" if policy == "category4_1400_strict_before" else "right"
    anchor = int(np.searchsorted(stamps, origin, side=side)) - 1
    start, end = anchor - lookback + 1, anchor + horizon
    valid = start >= 0 and end < len(stamps)
    return {"event_id": key, "category": category, "event_date": date,
            "assumed_origin": origin, "anchor": anchor, "start": start,
            "end": end, "valid": valid,
            "input_start": int(stamps[start]) if valid else None,
            "input_end": int(stamps[anchor]) if valid else None,
            "target_start": int(stamps[anchor + 1]) if valid else None,
            "target_end": int(stamps[end]) if valid else None}


def interval_coverage(windows, size, target_only=False):
    diff = np.zeros(size + 1, dtype=np.int64)
    for w in windows:
        if w["valid"]:
            lo = w["anchor"] + 1 if target_only else w["start"]
            diff[lo] += 1
            diff[w["end"] + 1] -= 1
    return np.cumsum(diff[:-1]) > 0


def pair_overlap(a, b, size):
    ca, cb = interval_coverage(a, size), interval_coverage(b, size)
    ta, tb = interval_coverage(a, size, True), interval_coverage(b, size, True)
    pref_a = np.r_[0, np.cumsum(ca)]
    pref_b = np.r_[0, np.cumsum(cb)]
    def affected(events, pref):
        return sum(w["valid"] and pref[w["end"] + 1] > pref[w["start"]]
                   for w in events)
    dates_a = {w["event_date"] for w in a}
    dates_b = {w["event_date"] for w in b}
    return {"shared_dates": len(dates_a & dates_b),
            "shared_full_window_bars": int(np.sum(ca & cb)),
            "shared_target_bars": int(np.sum(ta & tb)),
            "a_events_touching_b": int(affected(a, pref_b)),
            "b_events_touching_a": int(affected(b, pref_a))}


def evaluate_predictions(pred, truth, history):
    """Channel-last [event, time, OHLC] arrays; report errors and close direction."""
    out = {}
    for h in (35, 70, 140):
        p, y = pred[:, :h], truth[:, :h]
        last = history[:, -1, 3]
        pmove, ymove = p[:, -1, 3] - last, y[:, -1, 3] - last
        nonzero = ymove != 0
        out[str(h)] = {
            "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "endpoint_close_dhr": float(np.mean(np.sign(pmove) == np.sign(ymove))) if np.any(pmove != 0) else None,
            "predicted_direction_events": int(np.sum(pmove != 0)),
            "always_up_endpoint_dhr": float(np.mean(ymove > 0)),
            "always_up_dhr_excluding_zero": float(np.mean(ymove[nonzero] > 0)) if nonzero.any() else None,
            "zero_actual_move_events": int(np.sum(~nonzero)),
        }
    return out


def price_only_benchmarks(stamps, x, event_groups, out_dir):
    """Train-only scaling; tune a small ridge price baseline on validation only.

    Ridge forecasts changes from the last price. It is a new reference model,
    not the team's DLinear, and uses the same event weighting for all splits.
    """
    groups = {k: [w for w in v if w["valid"]] for k, v in event_groups.items()}
    train_mask = interval_coverage(groups["train"], len(stamps))
    mu, sigma = x[train_mask].mean(axis=0), x[train_mask].std(axis=0)
    sigma[sigma == 0] = 1
    scaled = (x - mu) / sigma
    arrays = {}
    for role, events in groups.items():
        anchors = np.array([w["anchor"] for w in events])
        seq = scaled[anchors[:, None] + np.arange(-34, 1)]
        target = scaled[anchors[:, None] + np.arange(1, 141)]
        arrays[role] = (seq, target)
    # Price pattern relative to its last value removes dependence on price level.
    def features(seq):
        f = (seq[:, :-1] - seq[:, -1:]).reshape(len(seq), -1)
        return np.c_[np.ones(len(f)), f]
    f_train = features(arrays["train"][0])
    center, scale = f_train.mean(axis=0), f_train.std(axis=0)
    center[0], scale[0] = 0, 1
    scale[scale == 0] = 1
    feats = {r: (features(a[0]) - center) / scale for r, a in arrays.items()}
    target_delta = (arrays["train"][1] - arrays["train"][0][:, -1:]).reshape(len(f_train), -1)
    gram, cross = feats["train"].T @ feats["train"], feats["train"].T @ target_delta
    penalty = np.eye(gram.shape[0]); penalty[0, 0] = 0
    candidates, best = [], None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        coef = np.linalg.solve(gram + alpha * penalty, cross)
        seq, truth = arrays["vali"]
        pred = (feats["vali"] @ coef).reshape(truth.shape) + seq[:, -1:]
        mse = float(np.mean((pred - truth) ** 2))
        candidates.append({"alpha": alpha, "validation_mse": mse})
        if best is None or mse < best[0]:
            best = (mse, alpha, coef)
    result = {"scaler": {"mean": mu.tolist(), "std": sigma.tolist(),
                          "fit_unique_train_bars": int(train_mask.sum())},
              "ridge_selection": {"candidates": candidates, "selected_alpha": best[1]},
              "settings": {"lookback": 35, "horizon": 140, "features": "OHLC",
                           "scoring": "all-channel standardized MSE/MAE; close endpoint direction",
                           "test_used_for_selection": False}, "splits": {}}
    for role, (seq, truth) in arrays.items():
        persistence = np.broadcast_to(seq[:, -1:], truth.shape)
        ridge = (feats[role] @ best[2]).reshape(truth.shape) + seq[:, -1:]
        result["splits"][role] = {
            "events": len(seq), "persistence": evaluate_predictions(persistence, truth, seq),
            "ridge_price_only": evaluate_predictions(ridge, truth, seq)}
        if role in ("vali", "test"):
            np.savez_compressed(out_dir / f"c1_{role}_price_predictions.npz",
                event_ids=np.array([w["event_id"] for w in groups[role]]),
                event_dates=np.array([w["event_date"] for w in groups[role]]),
                seqs=seq, targets=truth, persistence=persistence, ridge=ridge)
    return result


def summarize_controls(path):
    d = json.loads(Path(path).read_text())
    real, shuffled, zero = (d[k] for k in ("real_text", "shuffled_text", "zero_text"))
    return {"source_sha256": sha256(path), "conditions": d,
            "shuffled_mse_increase_pct": 100 * (shuffled["overall_mse"] / real["overall_mse"] - 1),
            "real_mse_reduction_vs_zero_pct": 100 * (1 - real["overall_mse"] / zero["overall_mse"]),
            "zero_minus_real_dhr_h140_pp": 100 * (zero["horizons"]["H=140"]["dhr"] - real["horizons"]["H=140"]["dhr"]),
            "limitations": ["aggregate metrics only; no paired uncertainty calculation possible",
                "checkpoint, sample IDs, split, shuffle seeds and inference code not supplied",
                "zero-text inference is not a separately trained price-only model",
                "directional accuracy and MSE answer different questions"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--price", type=Path, default=ROOT / "data/series/SP500.csv")
    default_source = ROOT if (ROOT / "splits/manifests").exists() else ROOT / "analysis/nov03/source"
    p.add_argument("--source", type=Path, default=default_source)
    p.add_argument("--out", type=Path, default=ROOT / "analysis/nov03/results")
    p.add_argument("--skip-benchmarks", action="store_true")
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    stamps, prices, quality = load_prices(args.price)
    print(f"Loaded {len(stamps):,} unique bars; removed {quality['extra_duplicate_rows']} exact duplicates", flush=True)
    # Derived sanitized data are separate from the original source.
    with (args.out / "SP500_sorted_unique.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["date", "date_int", "Open", "High", "Low", "Close"])
        for t, values in zip(stamps, prices):
            s = str(t)
            date_text = f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"
            writer.writerow([date_text, int(t), *values])
    report = {"source_commit": SOURCE_COMMIT, "price_quality": quality, "splits": {},
              "publication_times_verified": False}
    table_rows, c1_groups = [], None
    manifest_paths = sorted((args.source / "splits/manifests").glob("purged_split_*.json"))
    if not manifest_paths:
        raise ValueError(f"No split manifests found under {args.source}; use --source with the repository root")
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text())
        print(f"Checking {manifest['split_name']}", flush=True)
        result = {"sha256": sha256(manifest_path), "name": manifest["split_name"], "policies": {}}
        for policy in ("legacy_all0830_floor", "category4_1400_strict_before"):
            groups = {r: [window_for_event(e, stamps, policy) for e in manifest[r + "_events"]] for r in ROLES}
            pairs = {f"{a}_{b}": pair_overlap(groups[a], groups[b], len(stamps))
                     for a, b in itertools.combinations(ROLES, 2)}
            result["policies"][policy] = {"roles": {
                r: {"events": len(v), "invalid_windows": sum(not w["valid"] for w in v),
                    "unique_full_windows": len({(w["start"], w["end"]) for w in v if w["valid"]}),
                    "first_event_date": min(w["event_date"] for w in v),
                    "last_event_date": max(w["event_date"] for w in v),
                    "categories": dict(Counter(w["category"] for w in v))} for r, v in groups.items()},
                "pairwise_overlap": pairs,
                "full_windows_disjoint": all(x["shared_full_window_bars"] == 0 for x in pairs.values())}
            if "c1_" in manifest_path.name and policy == "category4_1400_strict_before":
                c1_groups = groups
            for role, events in groups.items():
                table_rows.extend({"manifest": manifest_path.name, "policy": policy, "role": role, **w} for w in events)
        result["forward_chronology"] = (
            max(event_key(e)[1] for e in manifest["train_events"]) < min(event_key(e)[1] for e in manifest["vali_events"]) and
            max(event_key(e)[1] for e in manifest["vali_events"]) < min(event_key(e)[1] for e in manifest["test_events"]))
        report["splits"][manifest_path.name] = result
    with (args.out / "event_windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0])); writer.writeheader(); writer.writerows(table_rows)
    controls = args.source / "task2_shuffled_text/outputs/task2_shuffled_text_control.json"
    report["text_controls"] = summarize_controls(controls)
    audit = json.loads((args.source / "splits/manifests/data_leakage_audit.json").read_text())
    report["reported_populations"] = {"audit": audit,
        "control_test_events_in_notes": 2329,
        "count_difference": audit["n_test"] - 2329,
        "explanation_verified": False}
    if not args.skip_benchmarks and c1_groups:
        if any(not w["valid"] for v in c1_groups.values() for w in v):
            raise ValueError("C1 has incomplete windows; resolve before scoring")
        print("Fitting the price-only baseline using training and validation only", flush=True)
        report["c1_price_only_benchmarks"] = price_only_benchmarks(stamps, prices, c1_groups, args.out)
    (args.out / "audit_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = {"prices": quality, "splits": {k: {"forward": v["forward_chronology"],
        "checks": {p: {"disjoint": s["full_windows_disjoint"], "overlap": s["pairwise_overlap"]}
                   for p, s in v["policies"].items()}} for k, v in report["splits"].items()},
        "controls": {k: v for k, v in report["text_controls"].items() if k not in ("conditions", "limitations")}}
    if "c1_price_only_benchmarks" in report:
        summary["c1_test_benchmarks"] = report["c1_price_only_benchmarks"]["splits"]["test"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
