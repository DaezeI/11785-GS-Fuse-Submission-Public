#!/usr/bin/env python3
"""Summarize paired gate predictions; never substitutes simulated measurements."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(blob):
    metadata = json.loads(str(blob["metadata_json"]))
    target = blob["targets"]
    n = len(target)
    if target.ndim != 3 or target.shape[1:] != (4, 140) or n == 0:
        raise ValueError("Expected nonempty targets [event, 4, 140]")
    ids, dates = blob["event_ids"], blob["event_dates"]
    if ids.shape != (n,) or dates.shape != (n,) or len(set(ids.tolist())) != n:
        raise ValueError("Missing or duplicate event IDs")
    if not np.array_equal(dates, np.array([x.split("/")[1][:8] for x in ids])):
        raise ValueError("Event dates do not match IDs")
    gate = blob["gate_text"]
    fixed = blob["fixed_gate_text"]
    if gate.ndim != 2 or gate.shape[0] != n or fixed.shape != gate.shape:
        raise ValueError("Gate dimensions do not align with events")
    if not np.allclose(fixed, 0.5, atol=0, rtol=0):
        raise ValueError("The fixed gate was not exactly 0.5")
    if np.any((gate < 0) | (gate > 1)):
        raise ValueError("Gate values outside [0,1]")
    for key in ("targets", "gate_text", "fixed_gate_text", "recomputed_gate_target",
                "legacy_reference_delta", "learned", "fixed_050", "zeroed_article_tokens"):
        if not np.isfinite(blob[key]).all():
            raise ValueError(f"Non-finite data: {key}")
    for key in ("recomputed_gate_target", "legacy_reference_delta"):
        if blob[key].shape != (n,):
            raise ValueError(f"Wrong dimensions: {key}")
    if np.any((blob["recomputed_gate_target"] < 0) | (blob["recomputed_gate_target"] > 1)):
        raise ValueError("Supervision targets outside [0,1]")
    rows, horizons = [], {}
    gate_mean = gate.mean(axis=1)
    for h in (35, 70, 140):
        losses = {}
        for condition in ("learned", "fixed_050", "zeroed_article_tokens"):
            if blob[condition].shape != target.shape:
                raise ValueError(f"Prediction alignment/shape mismatch: {condition}")
            difference = blob[condition][:, :, :h] - target[:, :, :h]
            losses[condition] = (difference ** 2).mean(axis=(1, 2))
        benefit = losses["zeroed_article_tokens"] - losses["learned"]
        corr = None
        if n >= 3 and gate_mean.std() > 1e-12 and benefit.std() > 1e-12:
            corr = float(np.corrcoef(gate_mean, benefit)[0, 1])
        horizons[str(h)] = {
            "mse": {key: float(value.mean()) for key, value in losses.items()},
            "fixed_minus_learned_mse": float((losses["fixed_050"] - losses["learned"]).mean()),
            "gate_vs_zero_token_benefit_pearson_r": corr}
        for i in range(n):
            rows.append({"event_id": str(ids[i]), "date": str(dates[i]), "horizon": h,
                         "gate_text_mean": float(gate_mean[i]),
                         "gate_text_channel_std": float(gate[i].std()),
                         "recomputed_gate_target": float(blob["recomputed_gate_target"][i]),
                         **{key + "_mse": float(value[i]) for key, value in losses.items()}})
    summary = {"run": metadata, "distinct_dates": len(set(dates.tolist())),
               "gate": {"event_mean": float(gate_mean.mean()), "event_std": float(gate_mean.std()),
                        "minimum_event_mean": float(gate_mean.min()),
                        "maximum_event_mean": float(gate_mean.max())},
               "horizons": horizons,
               "limits": ["Descriptive paired metrics; no significance or causal claim",
                          "Events can share dates/windows; no independent-event uncertainty assumption",
                          "Fixed gate changes fusion only; zero-token removal is a separate intervention",
                          "Recomputed targets use the current batch scale, not historical training batches"]}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as blob:
        summary, rows = summarize(blob)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "event_losses.csv"):
        if (args.out / name).exists():
            raise FileExistsError(args.out / name)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.out / "event_losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"gate": summary["gate"], "horizons": summary["horizons"]}, indent=2))


if __name__ == "__main__":
    main()
