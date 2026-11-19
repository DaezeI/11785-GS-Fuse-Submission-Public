#!/usr/bin/env python3
"""Summarize aligned return-task predictions without loading model backbones."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_text_predictions import block_bootstrap_improvement


def analyze(blob, reps=1000, block=20):
    meta = json.loads(str(blob["metadata_json"].item()))
    config = meta.get("forecast_task_config")
    if not isinstance(config, dict) or config.get("task") != "return_bp":
        raise ValueError("Expected an aligned return_bp prediction file")
    y = np.asarray(blob["targets"], dtype=np.float64)
    n, horizon, channels = y.shape
    if (n == 0 or channels != 4 or len(set(blob["event_ids"].astype(str))) != n or
            blob["seqs"].ndim != 3 or blob["seqs"].shape[0] != n or
            blob["seqs"].shape[2] != 4 or not 1 <= blob["seqs"].shape[1] <= 280 or
            blob["anchor_close"].shape != (n,) or
            np.any(blob["anchor_close"] <= 0)):
        raise ValueError("Return prediction population, shape, or close anchors are invalid")
    conditions = {name: np.asarray(blob[name], dtype=np.float64) for name in
                  ("real_text", "zero_text", "fixed_gate", "strict_ts")}
    conditions["zero_return"] = np.zeros_like(y)
    shuffled = np.asarray(blob["shuffled_text"], dtype=np.float64)
    if shuffled.ndim != 4 or shuffled.shape[1:] != y.shape or not len(shuffled):
        raise ValueError("shuffled_text must have shape [seeds, events, horizon, channels]")
    for name, pred in conditions.items():
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f"{name} does not match finite return targets")
    if not np.isfinite(y).all() or not np.isfinite(shuffled).all():
        raise ValueError("Non-finite return predictions or targets")
    for name in ("gate_text_weight", "full_task_loss", "strict_ts_task_loss", "gate_target"):
        if blob[name].shape != (n,) or not np.isfinite(blob[name]).all():
            raise ValueError(f"Missing or invalid per-event {name}")

    dates = blob["event_dates"].astype(str)
    types = blob["event_types"].astype(str)
    groups = {"all": np.ones(n, dtype=bool)}
    groups.update({f"event_type:{t}": types == t for t in np.unique(types)})
    loss = {name: {"mae": np.abs(pred - y).mean(axis=(1, 2)),
                   "mse": ((pred - y) ** 2).mean(axis=(1, 2))}
            for name, pred in conditions.items()}
    loss["shuffled_text"] = {
        "mae": np.abs(shuffled - y[None]).mean(axis=(2, 3)).mean(axis=0),
        "mse": ((shuffled - y[None]) ** 2).mean(axis=(2, 3)).mean(axis=0),
    }
    out = {"task": "return_bp", "metadata": meta, "events": n,
           "distinct_dates": len(np.unique(dates)), "shuffles": len(shuffled),
           "all_channel": {}, "close_endpoint_dhr": {}, "close_endpoint_error": {}, "gate": {},
           "note": "Return errors are in bp/bp²; date-block intervals are conditional on this checkpoint."}
    for group, mask in groups.items():
        rows = {name: {"mae_bp": float(vals["mae"][mask].mean()),
                       "mse_bp2": float(vals["mse"][mask].mean())}
                for name, vals in loss.items()}
        if group == "all":
            comparisons = {}
            for control in ("zero_text", "shuffled_text", "fixed_gate", "strict_ts"):
                comparisons[control] = {}
                for metric in ("mae", "mse"):
                    control_loss = loss[control][metric]
                    real_loss = loss["real_text"][metric]
                    denominator = control_loss.mean()
                    comparisons[control][metric + "_gain_pct"] = (
                        float(100 * (1 - real_loss.mean() / denominator)) if denominator else None)
                    comparisons[control][metric + "_date_block_ci"] = block_bootstrap_improvement(
                        real_loss, control_loss, dates, mask, reps=reps, block=block)
            out["comparisons"] = comparisons
        out["all_channel"][group] = {"events": int(mask.sum()), "conditions": rows}

    close = int(config["close_channel"])
    deadband = float(config["direction_deadband_bp"])
    majority = meta["training_majority_direction"]
    for h in config["direction_horizons"]:
        h = int(h)
        if not 1 <= h <= horizon:
            raise ValueError("Direction horizon is outside saved forecasts")
        actual = y[:, h - 1, close]
        errors = {name: np.abs(pred[:, h - 1, close] - actual)
                  for name, pred in conditions.items()}
        errors["shuffled_text"] = np.abs(shuffled[:, :, h - 1, close] - actual[None]).mean(0)
        out["close_endpoint_error"][str(h)] = {
            "mae_bp": {name: float(error.mean()) for name, error in errors.items()},
            "real_vs_control": {name: block_bootstrap_improvement(
                errors["real_text"], errors[name], dates, groups["all"], reps=reps, block=block)
                for name in ("shuffled_text", "zero_text", "zero_return", "strict_ts")},
        }
        eligible = (actual != 0) & (np.abs(actual) > deadband)
        signs = {name: np.sign(pred[:, h - 1, close]) for name, pred in conditions.items()}
        shuffle_sign = np.sign(shuffled[:, :, h - 1, close])
        endpoint = {}
        for group, mask in groups.items():
            chosen = eligible & mask
            count = int(chosen.sum())
            if not count:
                endpoint[group] = {"eligible_events": 0, "hit_rate": None}
                continue
            rates = {name: float((sign[chosen] == np.sign(actual[chosen])).mean())
                     for name, sign in signs.items() if name != "zero_return"}
            rates["shuffled_text"] = float(
                (shuffle_sign[:, chosen] == np.sign(actual[chosen])[None]).mean())
            rates["training_majority_direction"] = float(
                (int(majority[str(h)]) == np.sign(actual[chosen])).mean())
            endpoint[group] = {"eligible_events": count, "hit_rate": rates}
        out["close_endpoint_dhr"][str(h)] = endpoint

    utility = blob["strict_ts_task_loss"] - blob["full_task_loss"]
    weight = blob["gate_text_weight"]
    correlation = (float(np.corrcoef(weight, utility)[0, 1])
                   if np.std(weight) > 0 and np.std(utility) > 0 else None)
    out["gate"] = {"text_weight_mean": float(np.mean(weight)),
                   "text_weight_std": float(np.std(weight)),
                   "strict_ts_minus_full_task_mean": float(np.mean(utility)),
                   "full_better_fraction": float(np.mean(utility > 0)),
                   "gate_target_mean": float(np.mean(blob["gate_target"])),
                   "weight_utility_correlation": correlation,
                   "warning": "Adaptive gate targets depend on evaluation batch composition."}
    return out



def compare_runs(candidate, reference, reps=1000, block=20):
    """Compare real-text forecasts for two variants on the same events."""
    candidate_meta = json.loads(str(candidate["metadata_json"].item()))
    reference_meta = json.loads(str(reference["metadata_json"].item()))
    for key in ("event_ids", "event_dates", "targets", "anchor_close", "shuffle_donor_ids"):
        if not np.array_equal(candidate[key], reference[key]):
            raise ValueError(f"Paired return runs differ in {key}")
    for key in ("evaluation_split", "no_text_definition", "reference_source_commit",
                "seed", "batch_size", "tf32", "four_bit", "article_content_sha256"):
        if key not in candidate_meta or candidate_meta[key] != reference_meta.get(key):
            raise ValueError(f"Paired return runs differ in {key}")
    snapshot_key = "snapshot_content_sha256"
    candidate_sha = candidate_meta["c1_data_audit"].get(snapshot_key)
    if not candidate_sha or candidate_sha != reference_meta["c1_data_audit"].get(snapshot_key):
        raise ValueError("Paired return runs require the same frozen snapshot")

    target = candidate["targets"].astype(np.float64)
    candidate_error = candidate["real_text"] - target
    reference_error = reference["real_text"] - target
    dates = candidate["event_dates"]
    mask = np.ones(len(dates), dtype=bool)
    result = {"candidate_checkpoint": candidate_meta["checkpoint_id"],
              "reference_checkpoint": reference_meta["checkpoint_id"],
              "events": len(dates), snapshot_key: candidate_sha, "errors": {}, "close_dhr": {}}
    for name, transform in (("mae_bp", np.abs), ("mse_bp2", np.square)):
        new = transform(candidate_error).mean(axis=(1, 2))
        old = transform(reference_error).mean(axis=(1, 2))
        result["errors"][name] = {
            "candidate": float(new.mean()),
            "reference": float(old.mean()),
            "candidate_minus_reference": float((new - old).mean()),
            "relative_improvement": block_bootstrap_improvement(
                new, old, dates, mask, reps=reps, block=block),
        }
    config = candidate_meta["forecast_task_config"]
    reference_config = reference_meta["forecast_task_config"]
    for key in ("direction_horizons", "direction_deadband_bp", "close_channel"):
        if config[key] != reference_config[key]:
            raise ValueError(f"Paired directional metrics differ in {key}")
    for horizon in config["direction_horizons"]:
        index = int(horizon) - 1
        close = config["close_channel"]
        actual = target[:, index, close]
        eligible = np.abs(actual) > config["direction_deadband_bp"]
        row = {"eligible_events": int(eligible.sum())}
        for name, predictions in (("candidate", candidate), ("reference", reference)):
            predicted = predictions["real_text"][:, index, close][eligible]
            row[name] = float((np.sign(predicted) == np.sign(actual[eligible])).mean()) if eligible.any() else None
        row["candidate_minus_reference"] = (
            row["candidate"] - row["reference"] if row["eligible_events"] else None
        )
        result["close_dhr"][str(horizon)] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Aligned return predictions from the comparison run")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--block-dates", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap_reps < 1 or args.block_dates < 1:
        parser.error("bootstrap-reps and block-dates must be positive")
    with np.load(args.predictions, allow_pickle=False) as blob:
        result = analyze(blob, args.bootstrap_reps, args.block_dates)
        if args.reference:
            with np.load(args.reference, allow_pickle=False) as reference:
                result["paired_reference"] = compare_runs(
                    blob, reference, args.bootstrap_reps, args.block_dates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {result['events']} aligned return events to {args.out}")


if __name__ == "__main__":
    main()
