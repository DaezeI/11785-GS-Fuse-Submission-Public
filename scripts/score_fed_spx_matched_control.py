#!/usr/bin/env python3
"""Freeze the price-only control and score it with baselines on the text-matched cohort.

The control is the three-seed compact CNN from ``fed-spx-endpoints-price-v1``, trained on
all 978 price-eligible train events. This script

1. checks each checkpoint hash against that run's metrics and replays it on the rebuilt
   validation inputs, requiring the saved forecasts to be reproduced exactly;
2. refits a ridge on the same 978 train events and the 20/30-minute target;
3. restricts validation to the text-verified cohort, reweights each release instant to unit
   weight within it, and scores the control, ridge and zero return with paired date-block
   intervals;
4. writes the per-event control forecasts in the paired format the text arm must also emit.

Only validation is scored. Test is not loaded.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from data.fed_spx_endpoints import HORIZONS, load_endpoint_cohorts  # noqa: E402
from model.compact_return import CompactReturn  # noqa: E402
from model.dlinear import DLinearBaseline  # noqa: E402
from scripts.build_fed_price_alignment import sha256_file  # noqa: E402
from scripts.run_fed_price_baselines import fit_ridge  # noqa: E402
from scripts.run_fed_spx_endpoints_price import bootstrap_gain, endpoint_metrics  # noqa: E402

RUN = ROOT / "downloads/fed-spx-endpoints-price-v1"
DLINEAR_CONFIG = ROOT / "experiments/fed-spx-matched-control-v1/dlinear.config.json"
COHORT = ROOT / "results/fed-spx-news-cohort-v1/eligible_cohort.csv"
INFERRED = {
    "speech_release_line": {"eligible_document_release_line_unmodified_file"},
    "board_release_day_capture": {"eligible_document_release_line_release_day_capture"},
    "minutes_release_record": {"eligible_official_release_record_unmodified_file"},
}


def instant_weights(rows):
    return 1.0 / rows.groupby("market_et")["event_id"].transform("size").to_numpy(dtype=float)


def replay(checkpoint, history, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CompactReturn(output_dim=len(HORIZONS)).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    scale = np.float64(state["input_scale"])
    with torch.inference_mode():
        x = torch.as_tensor(history / scale, dtype=torch.float32, device=device)
        out = model(x).cpu().numpy().astype(np.float64)
    return out * np.asarray(state["target_scale"], dtype=np.float64), state


def ridge_predict(x, model):
    features = (x.reshape(len(x), -1) - model["x_mean"]) / model["x_scale"]
    return features @ model["coefficients"] + model["intercept"]


def fit_dlinear(train, val_history, config, seed, device):
    """Channel-independent DLinear on the 35-bar history, read from the Close channel.

    Same settings as the historical Fed/SPX DLinear run, with a two-step output
    [20 min, 30 min] instead of the 140-bar path. Targets use train-only RMS scales
    per horizon. The last epoch is kept, so validation plays no part in fitting.
    """
    torch.manual_seed(seed)
    input_scale = np.sqrt(np.average(np.square(train.history.astype(np.float64)).mean(axis=(1, 2)), weights=train.weights))
    target_scale = np.sqrt(np.average(np.square(train.returns), weights=train.weights, axis=0))
    x = torch.as_tensor(train.history / input_scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(train.returns / target_scale, dtype=torch.float32, device=device)
    w = torch.as_tensor(train.weights / train.weights.mean(), dtype=torch.float32, device=device)
    model = DLinearBaseline(x.shape[-1], len(HORIZONS), kernel_size=config["kernel_size"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                          eta_min=config["min_learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    for _ in range(config["epochs"]):
        for idx in torch.randperm(len(x), generator=generator).split(config["batch_size"]):
            idx = idx.to(device)
            optimizer.zero_grad(set_to_none=True)
            residual = model(x[idx])[:, 3, :] - y[idx]
            loss = ((residual.square() + config["mae_weight"] * residual.abs()).mean(dim=1) * w[idx]).mean()
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite DLinear loss for seed {seed}")
            loss.backward()
            optimizer.step()
        schedule.step()
    model.eval()
    with torch.inference_mode():
        out = model(torch.as_tensor(val_history / input_scale, dtype=torch.float32, device=device))[:, 3, :]
    return out.cpu().numpy().astype(np.float64) * target_scale, {key: v.cpu() for key, v in model.state_dict().items()}


def score(name_to_forecast, truth, weights, dates, config):
    zero = np.zeros_like(truth)
    out = {"zero_return": endpoint_metrics(truth, zero, weights)}
    zero_loss = np.abs(truth)
    for name, forecast in name_to_forecast.items():
        loss = np.abs(forecast - truth).mean(axis=0) if forecast.ndim == 3 else np.abs(forecast - truth)
        gain = zero_loss - loss
        per_seed = ([endpoint_metrics(truth, f, weights) for f in forecast] if forecast.ndim == 3 else None)
        mean_forecast = forecast.mean(axis=0) if forecast.ndim == 3 else forecast
        out[name] = {
            "metrics_seed_average_forecast": endpoint_metrics(truth, mean_forecast, weights),
            "mae_bp_mean_over_seeds": ({str(h): float(np.mean([s[str(h)]["mae_bp"] for s in per_seed]))
                                        for h in HORIZONS} if per_seed else None),
            "direction_accuracy_mean_over_seeds": ({str(h): float(np.mean([s[str(h)]["direction_accuracy"]
                                                                          for s in per_seed])) for h in HORIZONS}
                                                   if per_seed else None),
            "gain_over_zero_bp_20_30": np.average(gain, weights=weights, axis=0).tolist(),
            "paired_gain_ci95_bp_by_block_20_30": bootstrap_gain(gain, dates, weights, config),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="the control was fit on CUDA; exact replay needs the same device")
    parser.add_argument("--forecast-dir", type=Path, required=True, help="ignored location for per-event forecasts")
    parser.add_argument("--cohort", type=Path, default=COHORT,
                        help="eligible cohort CSV (v1 default; results/fed-spx-news-cohort-v2/eligible_cohort.csv for v2)")
    args = parser.parse_args()
    cohort_path = args.cohort if args.cohort.is_absolute() else ROOT / args.cohort
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    run = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    config = run["config"]
    cohorts, audit = load_endpoint_cohorts(args.price_csv)
    train, val = cohorts["train"], cohorts["val"]
    if (len(train.rows), len(val.rows)) != (978, 207):
        raise ValueError("Unexpected price-eligible cohort size")
    saved = np.load(RUN / "val-paired.npz")
    if list(saved["event_ids"]) != val.rows.event_id.tolist() or not np.array_equal(saved["targets"], val.returns):
        raise ValueError("Saved validation forecasts do not match the rebuilt cohort")

    forecasts, frozen = [], []
    for record in run["price_only_seeds"]:
        path = RUN / f"compact-price-seed-{record['seed']}.pt"
        digest = sha256_file(path)
        if digest != record["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash changed for seed {record['seed']}")
        forecast, state = replay(path, val.history, device)
        if state["epoch"] != record["selected_epoch"]:
            raise ValueError(f"Selected epoch mismatch for seed {record['seed']}")
        forecasts.append(forecast)
        frozen.append({"seed": record["seed"], "selected_epoch": record["selected_epoch"], "checkpoint_sha256": digest})
    forecasts = np.stack(forecasts)
    replay_max_abs_diff = float(np.max(np.abs(forecasts - saved["forecasts"])))
    if not np.array_equal(forecasts, saved["forecasts"]):
        raise ValueError(f"Replay differs from the saved forecasts by up to {replay_max_abs_diff} bp")

    ridge = fit_ridge(train.history, train.returns, train.weights)
    ridge_val = ridge_predict(val.history, ridge)
    dlinear_config = json.loads(DLINEAR_CONFIG.read_text(encoding="utf-8"))
    dlinear_val, dlinear_states = [], {}
    for seed in dlinear_config["seeds"]:
        forecast, state = fit_dlinear(train, val.history, dlinear_config, seed, device)
        dlinear_val.append(forecast)
        dlinear_states[f"seed{seed}"] = state
    dlinear_val = np.stack(dlinear_val)

    cohort = pd.read_csv(cohort_path)
    matched = cohort.loc[cohort.split == "val"]
    keep = val.rows.event_id.isin(matched.event_id).to_numpy()
    if keep.sum() != len(matched):
        raise ValueError("Some text-verified validation events are missing from the price cohort")
    rows = val.rows.loc[keep].reset_index(drop=True)
    evidence = dict(zip(matched.event_id, matched.evidence_class))
    rows["evidence_class"] = rows.event_id.map(evidence)
    truth, dates = val.returns[keep], rows.market_et.str[:10].to_numpy(dtype=str)

    subsets = {"all_text_verified": np.ones(len(rows), bool)}
    subsets["without_any_inferred_class"] = ~rows.evidence_class.isin(set().union(*INFERRED.values())).to_numpy()
    results = {}
    for name, mask in subsets.items():
        sub = rows.loc[mask]
        weights = instant_weights(sub)
        results[name] = {
            "rows": int(mask.sum()), "release_instants": int(sub.market_et.nunique()),
            "dates": int(sub.market_et.str[:10].nunique()),
            "scores": score({"price_cnn_control": forecasts[:, keep][:, mask], "ridge": ridge_val[keep][mask],
                             "dlinear": dlinear_val[:, keep][:, mask]},
                            truth[mask], weights, dates[mask], config),
        }
    full_weights = val.weights
    results["price_eligible_reference_207"] = {
        "rows": len(val.rows),
        "scores": score({"price_cnn_control": forecasts, "ridge": ridge_val, "dlinear": dlinear_val}, val.returns, full_weights,
                        val.rows.market_et.str[:10].to_numpy(dtype=str), config),
    }

    args.forecast_dir.mkdir(parents=True, exist_ok=True)
    weights = instant_weights(rows)
    paired = pd.DataFrame({
        "event_id": rows.event_id, "event_type": rows.event_type, "evidence_class": rows.evidence_class,
        "market_et": rows.market_et, "origin_et": rows.origin_et, "weight": weights,
        "target_20m_bp": truth[:, 0], "target_30m_bp": truth[:, 1],
        **{f"control_seed{s}_{h}m_bp": forecasts[i, keep][:, j]
           for i, s in enumerate(config["seeds"]) for j, h in enumerate(HORIZONS)},
        **{f"ridge_{h}m_bp": ridge_val[keep][:, j] for j, h in enumerate(HORIZONS)},
        **{f"dlinear_seed{s}_{h}m_bp": dlinear_val[i, keep][:, j]
           for i, s in enumerate(dlinear_config["seeds"]) for j, h in enumerate(HORIZONS)},
    })
    paired_path = args.forecast_dir / "val-matched-control-forecasts.csv"
    paired.to_csv(paired_path, index=False, float_format="%.10g", lineterminator="\n")
    np.savez_compressed(args.forecast_dir / "ridge-endpoint.npz", **ridge)
    torch.save(dlinear_states, args.forecast_dir / "dlinear-endpoint.pt")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "split": "val", "test_loaded": False, "horizons_minutes": list(HORIZONS),
        "control": {"run_id": "fed-spx-endpoints-price-v1", "train_rows": len(train.rows),
                    "code_commit_of_fit": run["code_commit"], "seeds": frozen,
                    "input_scale_bp": run["input_scale_bp"], "target_scales_bp": run["target_scales_bp"],
                    "replay_exact": True, "replay_max_abs_diff_bp": replay_max_abs_diff,
                    "selection_note": "checkpoint epochs were chosen on the 207 price-eligible validation rows, "
                                      "which contain the 123 matched rows"},
        "dlinear": {"config": dlinear_config, "config_sha256": sha256_file(DLINEAR_CONFIG),
                    "train_rows": len(train.rows), "output": "Close channel, [20 min, 30 min]"},
        "ridge": {"alpha": 10.0, "train_rows": len(train.rows), "features": "35x4 flattened, train-standardized"},
        "cohort_file": cohort_path.relative_to(ROOT).as_posix(), "cohort_sha256": sha256_file(cohort_path),
        "paired_forecasts_sha256": hashlib.sha256(paired_path.read_bytes()).hexdigest(),
        "price_audit": audit, "results": results,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, block in results.items():
        s = block["scores"]
        print(f"{name}: {block['rows']} rows")
        print(f"  zero_return        MAE20 {s['zero_return']['20']['mae_bp']:.3f}  MAE30 {s['zero_return']['30']['mae_bp']:.3f}")
        for k in ("price_cnn_control", "ridge", "dlinear"):
            m = s[k]["metrics_seed_average_forecast"]
            ci = s[k]["paired_gain_ci95_bp_by_block_20_30"]["10"]
            print(f"  {k:18s} MAE20 {m['20']['mae_bp']:.3f}  MAE30 {m['30']['mae_bp']:.3f}  "
                  f"gain {np.round(s[k]['gain_over_zero_bp_20_30'], 3).tolist()}  "
                  f"CI10 {np.round(ci, 3).tolist() if ci is not None else 'too few dates'}")

if __name__ == "__main__":
    main()
