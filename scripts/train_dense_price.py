#!/usr/bin/env python3
"""Train none+CNN~none~none on all eligible training-period price anchors.

The 5- and 15-minute anchor grids share architecture, optimization, the 5-minute
validation grid and the original 35-bar / 20,30-minute label clock. No test
labels or text are loaded. Full per-step histories stay in private run storage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data.fed_spx_endpoints as fed_data
from data.dense_price_endpoints import load_dense_cohorts
from model.compact_return import CompactReturn
from scripts.score_fed_spx_paired import endpoint_metrics, bootstrap_gain, instant_weights


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(arm, x, truth, scale, weights=None, batch=2048):
    arm.eval()
    with torch.inference_mode():
        outputs = torch.cat([arm(part) for part in x.split(batch)]).cpu().numpy().astype(np.float64)
    prediction = outputs * scale
    weights = np.ones(len(truth)) if weights is None else weights
    normalized = (prediction - truth) / scale
    abs_errors = np.abs(normalized)
    huber = np.where(abs_errors < 1, .5 * normalized ** 2, abs_errors - .5)
    return prediction, {"smooth_l1": float(np.average(huber.mean(1), weights=weights)),
                        **endpoint_metrics(truth, prediction, weights)}


def paired(truth, forecast, weights, dates):
    gain = np.abs(truth) - np.abs(forecast - truth)
    return {"gain_over_persistence_bp_20_30": np.average(gain, weights=weights, axis=0).tolist(),
            "paired_ci95_bp_by_date_block": bootstrap_gain(gain, dates, weights,
                {"bootstrap_seed": 23, "bootstrap_replicates": 2000})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--alignment", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--code-commit", default="uncommitted")
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if a.alignment:
        fed_data.ALIGNMENT = a.alignment
    fed, fed_audit = fed_data.load_endpoint_cohorts(a.prices)
    train5, val5, audit5 = load_dense_cohorts(a.prices, 5)
    # GRID15 is a subset of GRID5, with unchanged arrays and scaling.
    minute = pd.DatetimeIndex(train5.times).hour * 60 + pd.DatetimeIndex(train5.times).minute
    train15_mask = np.asarray((minute - 570) % 15 == 0)
    audit = {"grid5": audit5, "grid15_train_rows": int(train15_mask.sum()),
             "fed": fed_audit, "test_loaded": False,
             "selection": "Lowest 20-minute MAE on common SPX-PX-GRID5-v1 validation; Fed validation is scored only after selection"}
    # Independent check against the original Fed validation support.
    alignment = pd.read_csv(fed_data.ALIGNMENT)
    fed_val_first = pd.to_datetime(alignment.loc[alignment.event_id.isin(fed["val"].rows.event_id), "input_first_bar"]).min()
    assert pd.Timestamp(audit5["splits"]["train"]["last_used_bar"]) < fed_val_first
    audit["dense_train_before_first_fed_validation_bar"] = str(fed_val_first)
    reserved_first = pd.to_datetime(alignment.loc[alignment.split.eq("test") & alignment.main_candidate, "input_first_bar"]).min()
    assert pd.Timestamp(audit5["splits"]["val"]["last_used_bar"]) + pd.Timedelta(minutes=5) <= reserved_first
    audit["validation_before_first_reserved_test_bar"] = str(reserved_first)
    (a.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": audit}), flush=True)
    if a.audit_only:
        return
    if a.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    torch.set_num_threads(8 if a.device == "cuda" else 2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(a.device)
    input_scale = float(np.sqrt(np.mean(np.square(train5.history.astype(np.float64)))))
    target_scale = np.sqrt(np.mean(np.square(train5.returns), axis=0))
    assert input_scale > 0 and np.all(target_scale > 0)
    xtrain5 = torch.tensor(train5.history / input_scale, dtype=torch.float32, device=device)
    ytrain5 = torch.tensor(train5.returns / target_scale, dtype=torch.float32, device=device)
    xval = torch.tensor(val5.history / input_scale, dtype=torch.float32, device=device)
    fx = torch.tensor(fed["val"].history / input_scale, dtype=torch.float32, device=device)
    reference = pd.read_csv(ROOT / "results/fed-spx-paired-scorer-v1/reference-FED-PXTX-ALL-REL5-v2-val.csv")
    assert len(reference) == 149 and set(reference.split) == {"val"}
    lookup = dict(zip(fed["val"].rows.event_id, range(len(fed["val"].rows))))
    matched_ix = np.array([lookup[e] for e in reference.event_id])
    np.testing.assert_allclose(fed["val"].returns[matched_ix], reference[["target_20m_bp", "target_30m_bp"]], atol=1e-6, rtol=0)
    matched_weights = instant_weights(reference.market_et)
    results, curves = [], []
    for stride in (5, 15):
        torch.manual_seed(0)
        arm = CompactReturn(output_dim=2).to(device)
        optimizer = torch.optim.AdamW(arm.parameters(), lr=.001, weight_decay=.001)
        generator = torch.Generator().manual_seed(0)
        if stride == 5:
            x, y, truth = xtrain5, ytrain5, train5.returns
        else:
            ix = torch.as_tensor(np.flatnonzero(train15_mask), device=device)
            x, y, truth = xtrain5[ix], ytrain5[ix], train5.returns[train15_mask]
        best, step, step_rows, epoch_rows = float("inf"), 0, [], []
        directory = a.out / f"grid{stride}"
        directory.mkdir()
        for epoch in range(a.epochs + 1):
            epoch_start = time.perf_counter()
            if epoch:
                arm.train()
                for ix in torch.randperm(len(x), generator=generator).split(64):
                    ix = ix.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.smooth_l1_loss(arm(x[ix]), y[ix])
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(arm.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    step_rows.append({"step": step, "epoch": epoch, "minibatch_loss": loss.item(), "gradient_norm": float(norm)})
            _, tm = evaluate(arm, x, truth, target_scale)
            vp, vm = evaluate(arm, xval, val5.returns, target_scale)
            row = {"stride_minutes": stride, "epoch": epoch, "updates": step,
                   "train_smooth_l1": tm["smooth_l1"], "val_smooth_l1": vm["smooth_l1"],
                   "train_mae20_bp": tm["20"]["mae_bp"], "val_mae20_bp": vm["20"]["mae_bp"],
                   "train_mae30_bp": tm["30"]["mae_bp"], "val_mae30_bp": vm["30"]["mae_bp"],
                   "seconds": time.perf_counter() - epoch_start}
            epoch_rows.append(row)
            print(json.dumps(row), flush=True)
            if row["val_mae20_bp"] < best:
                best, chosen_epoch = row["val_mae20_bp"], epoch
                checkpoint = {k: v.detach().cpu().clone() for k, v in arm.state_dict().items()}
            pd.DataFrame(epoch_rows).to_csv(directory / "learning-curves.csv", index=False)
        arm.load_state_dict(checkpoint)
        forecast, fm = evaluate(arm, fx, fed["val"].returns, target_scale, fed["val"].weights)
        _, grid_vm = evaluate(arm, xval, val5.returns, target_scale)
        mw = matched_weights
        record = {"architecture": "none+CNN~none~none", "train_cohort": f"SPX-PX-GRID{stride}-v1",
                  "selection_cohort": "SPX-PX-GRID5-v1 val", "train_rows": len(x),
                  "total_updates": step, "selected_epoch": chosen_epoch,
                  "grid5_validation": grid_vm, "fed_validation": fm,
                  "matched_fed_validation": endpoint_metrics(fed["val"].returns[matched_ix], forecast[matched_ix], mw),
                  "matched_fed_paired": paired(fed["val"].returns[matched_ix], forecast[matched_ix], mw, reference.market_et.str[:10].to_numpy()),
                  "final_training_row": epoch_rows[-1], "checkpoint_sha256": None}
        checkpoint_path = directory / "price.pt"
        torch.save({"state_dict": checkpoint, "input_scale": input_scale,
                    "target_scale": target_scale.tolist(), "selected_epoch": chosen_epoch,
                    "train_cohort": record["train_cohort"]}, checkpoint_path)
        record["checkpoint_sha256"] = sha(checkpoint_path)
        pd.DataFrame(step_rows).to_csv(directory / "optimizer-steps.csv", index=False)
        np.savez_compressed(directory / "fed-validation.npz", predictions=forecast, targets=fed["val"].returns,
                            event_ids=fed["val"].rows.event_id.to_numpy(str))
        results.append(record); curves.extend(epoch_rows)
        (directory / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    result = {"id": "dense-price-grid-v1", "architecture": "none+CNN~none~none", "status": "completed",
              "test_loaded": False, "code_commit": a.code_commit, "script_sha256": sha(__file__),
              "dataset_script_sha256": sha(ROOT / "data/dense_price_endpoints.py"), "audit": audit,
              "input_scale": input_scale, "target_scale": target_scale.tolist(),
              "configuration": {"epochs": a.epochs, "batch_size": 64, "learning_rate": .001, "weight_decay": .001,
                                "gradient_clip": 1, "selection": audit["selection"]},
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                              "device": str(device), "gpu": torch.cuda.get_device_name(device) if a.device == "cuda" else None},
              "runtime_seconds": time.perf_counter() - started, "runs": results,
              "limitations": ["Exploratory validation; no reserved test scoring.",
                              "Training-population package: dense prices and dense-training RMS scaling replace event-only fitting.",
                              "Adjacent rolling windows overlap and are not independent observations.",
                              "Price clock and bar-start convention remain inherited assumptions.",
                              "No text encoder or fusion arm was trained in this experiment."]}
    (a.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(curves).to_csv(a.out / "learning-curves.csv", index=False)
    print("DENSE_PRICE_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
