#!/usr/bin/env python3
"""Replay the fixed Fed train/validation budgets and measure ten extra updates.

No test text or predictions are opened. Cached features avoid loading an LLM.
Private checkpoints are read; only aggregate diagnostics are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fed_spx_endpoints import load_endpoint_cohorts
import data.fed_spx_endpoints as endpoint_data
from model.compact_return import CompactReturn
from scripts.build_fed_price_alignment import load_prices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(arm, x, target, w, scale, text=None):
    arm.eval()
    with torch.inference_mode():
        prediction = arm(x, text)
        objective = (F.smooth_l1_loss(prediction, target, reduction="none").mean(1) * w).mean().item()
        errors = np.abs((prediction - target).numpy().astype(np.float64) * scale)
    mae = np.average(errors, axis=0, weights=w.numpy())
    return objective, mae.tolist()


def replay(arm, train, val, scale, features, output_name):
    x, y, w = train
    vx, vy, vw = val
    text, vtext = features
    parameters = list(arm.text_head.parameters()) if text is not None else list(arm.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.001)
    order = torch.Generator().manual_seed(0)
    rows, epoch_val, step = [], [], 0

    def snapshot(epoch, grad=None, minibatch_loss=None):
        loss, trmae = measure(arm, x, y, w, scale, text)
        vloss, vmae = measure(arm, vx, vy, vw, scale, vtext)
        rows.append({"arm": output_name, "step": step, "epoch": epoch,
                     "train_smooth_l1": loss, "val_smooth_l1": vloss,
                     "train_mae20_bp": trmae[0], "train_mae30_bp": trmae[1],
                     "val_mae20_bp": vmae[0], "val_mae30_bp": vmae[1],
                     "last_minibatch_grad_norm": grad, "last_minibatch_loss": minibatch_loss})
        return vmae[0]

    def update(ix):
        nonlocal step
        arm.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = arm(x[ix], text[ix] if text is not None else None)
        loss = (F.smooth_l1_loss(prediction, y[ix], reduction="none").mean(1) * w[ix]).mean()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        return float(grad), loss.item()

    epoch_val.append(snapshot(0))
    for epoch in range(1, 31):
        for ix in torch.randperm(len(y), generator=order).split(64):
            grad, minibatch = update(ix)
        epoch_val.append(snapshot(epoch, grad, minibatch))
    original = dict(rows[-1])
    extra = 0
    while extra < 10:
        for ix in torch.randperm(len(y), generator=order).split(64):
            grad, minibatch = update(ix)
            extra += 1
            snapshot(30 + extra / ((len(y) + 63) // 64), grad, minibatch)
            if extra == 10:
                break
    final = dict(rows[-1])
    assert step == 30 * ((len(y) + 63) // 64) + 10
    return rows, {"original_budget": original, "after_10_extra_updates": final,
                  "train_loss_relative_reduction_extra": (original["train_smooth_l1"] - final["train_smooth_l1"]) / original["train_smooth_l1"],
                  "validation_mae20_trace": epoch_val,
                  "trainable_parameters": sum(p.numel() for p in parameters)}


def tensors(cohort, input_scale, scale):
    return (torch.tensor(cohort.history / input_scale, dtype=torch.float32),
            torch.tensor(cohort.returns / scale, dtype=torch.float32),
            torch.tensor(cohort.weights / cohort.weights.mean(), dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, help="Optional byte-identical frozen alignment copy")
    parser.add_argument("--frozen-price", type=Path, required=True)
    parser.add_argument("--text-inputs", type=Path, required=True)
    parser.add_argument("--completed-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if a.alignment is not None:
        endpoint_data.ALIGNMENT = a.alignment
    cohorts, audit = load_endpoint_cohorts(a.prices)
    baseline_metrics = json.loads((a.frozen_price / "metrics.json").read_text())
    saved_text = json.loads((a.completed_text / "metrics.json").read_text())
    frozen_path = a.frozen_price / "compact-price-seed-0.pt"
    assert sha(frozen_path) == baseline_metrics["price_only_seeds"][0]["checkpoint_sha256"]
    state = torch.load(frozen_path, map_location="cpu", weights_only=False)
    scale = np.asarray(state["target_scale"], dtype=np.float64)
    input_scale = state["input_scale"]
    train, val = (tensors(cohorts[k], input_scale, scale) for k in ("train", "val"))
    torch.manual_seed(0)
    price = CompactReturn(output_dim=2)
    price_rows, price_summary = replay(price, train, val, scale, (None, None), "price")
    price_summary["published_val_trace_max_abs_difference_bp"] = float(np.max(np.abs(
        np.asarray(price_summary["validation_mae20_trace"]) - baseline_metrics["price_only_seeds"][0]["validation_mae20_trace_bp"])))

    expected = {"train": "7d34c2a1f5a34ded2bc459cf1894f390048b2cb760479cd5475dc70aa7fc5760",
                "val": "b2e52208741d3b1a4f32d55e5cd06292b275eaa92028b81b1d4aff8abd0495be"}
    cache = np.load(a.completed_text / "features.npz")
    text_cohorts, features = {}, {}
    for split in ("train", "val"):
        path = a.text_inputs / (split + ".jsonl")
        assert sha(path) == expected[split]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if r["event_id"] not in {"stmt_20081029a", "stmt_20140430a"}]
        cohort = cohorts[split]
        lookup = dict(zip(cohort.rows.event_id, range(len(cohort.rows))))
        ix = [lookup[r["event_id"]] for r in rows]
        full_x, full_y, _ = tensors(cohort, input_scale, scale)
        matched_rows = cohort.rows.iloc[ix].reset_index(drop=True)
        matched_weights = 1 / matched_rows.groupby("market_et").event_id.transform("size").to_numpy(float)
        matched_weights /= matched_weights.mean()
        text_cohorts[split] = (full_x[ix], full_y[ix], torch.tensor(matched_weights, dtype=torch.float32))
        features[split] = cache[split]
        assert features[split].shape == (len(rows), 4096)
        assert len(rows) == {"train": 171, "val": 149}[split]
    mean = features["train"].mean(0)
    std = np.maximum(features["train"].std(0), 1e-6)
    z = {k: torch.tensor((v - mean) / std, dtype=torch.float32) for k, v in features.items()}
    torch.manual_seed(0)
    base = CompactReturn(output_dim=2)
    base.load_state_dict(state["state_dict"])
    text_arm = CompactReturn(text_dim=4096, output_dim=2)
    missing, unexpected = text_arm.load_state_dict(state["state_dict"], strict=False)
    assert not unexpected and all(k.startswith("text_head.") for k in missing)
    text_arm.freeze_price()
    text_rows, text_summary = replay(text_arm, text_cohorts["train"], text_cohorts["val"], scale,
                                     (z["train"], z["val"]), "text")
    text_summary["published_val_trace_max_abs_difference_bp"] = float(np.max(np.abs(
        np.asarray(text_summary["validation_mae20_trace"]) - saved_text["seeds"][0]["val_mae20_trace"])))
    assert abs(text_summary["validation_mae20_trace"][0] - saved_text["seeds"][0]["val_mae20_trace"][0]) < 1e-3
    assert all(torch.equal(v, text_arm.state_dict()[k]) for k, v in state["state_dict"].items())
    bars, price_report = load_prices(a.prices)
    cutoff = pd.Timestamp("2019-05-22 14:05:00")
    training_bars = bars.loc[bars.date < cutoff]
    population = pd.read_csv(ROOT / "analysis/macro_releases/price_alignment_v1/fed_price_alignment_v1.csv")
    train_rows = population.loc[population.split.eq("train")]
    funnel = {"raw_price_rows": price_report["rows_in"], "regular_session_price_rows_all_years": len(bars),
              "price_first": price_report["first_bar"], "price_last": price_report["last_bar"],
              "training_period_regular_session_bars": len(training_bars),
              "training_bar_cutoff_exclusive": str(cutoff),
              "training_catalog_rows": len(train_rows), "main_candidate_rows": int(train_rows.main_candidate.sum()),
              "endpoint_eligible_training_rows": audit["splits"]["train"]["eligible_rows"],
              "endpoint_exclusions": audit["splits"]["train"]["excluded"],
              "distinct_training_instants": audit["splits"]["train"]["distinct_instants"],
              "distinct_training_dates": audit["splits"]["train"]["dates"],
              "continuous_windows_trained_in_this_run": False}
    result = {"id": "fed-training-diagnostics-v1", "status": "completed", "test_loaded": False,
              "execution": "Local CPU replay, first recorded configuration, original budgets plus 10 updates",
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                              "pandas": pd.__version__, "device": "cpu", "threads": torch.get_num_threads()},
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": sha(__file__), "price_sha256": sha(a.prices),
              "frozen_price_sha256": sha(frozen_path), "feature_cache_sha256": sha(a.completed_text / "features.npz"),
              "data_audit": audit, "selection_funnel": funnel, "price": price_summary, "text": text_summary,
              "limits": ["CPU replay is not the original CUDA trajectory; trace differences are reported.",
                         "Original final optimizer states were not retained, so this is replay rather than exact resume.",
                         "Ten additional updates diagnose local loss movement and cannot certify convergence.",
                         "No new checkpoint was selected for deployment or reserved-test scoring."]}
    pd.DataFrame(price_rows + text_rows).to_csv(a.out / "training-traces.csv", index=False)
    (a.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selection_funnel": funnel, "price": price_summary, "text": text_summary}, indent=2))


if __name__ == "__main__":
    main()
