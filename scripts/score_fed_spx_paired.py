#!/usr/bin/env python3
"""Paired endpoint scorer for Fed/SPX 20/30-minute forecasts (issue #54).

Every forecaster, baseline and control is scored on the same releases from one reference file,
with unit weight per release instant, weighted MAE and direction accuracy, and paired
date-block bootstrap intervals on the per-release loss difference. ``Persistence`` (zero return
from the decision-time Close) is built in.

Subcommands:

``score``                 score long-format forecast files against a reference file.
``from-text-run``         convert ``val-paired.npz`` from ``scripts/run_fed_spx_text.py``.
``from-matched-control``  convert ``val-matched-control-forecasts.csv`` from
                          ``scripts/score_fed_spx_matched_control.py`` into a reference file and
                          a baseline forecast file.

The file formats are specified in ``results/fed-spx-paired-scorer-v1/README.md``. This module
needs only numpy and pandas: no torch, no price CSV and no test split.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HORIZONS = (20, 30)
PERSISTENCE = "Persistence"
BOOTSTRAP = {"bootstrap_seed": 23, "bootstrap_replicates": 2000}
# Evidence classes inferred rather than read from an official record or a pre-cutoff archive copy.
# Rows in these classes are dropped from the STRICT subset.
INFERRED_CLASSES = frozenset({
    "eligible_document_release_line_unmodified_file",
    "eligible_document_release_line_release_day_capture",
    "eligible_official_release_record_unmodified_file",
})
REFERENCE_COLUMNS = ("event_id", "split", "evidence_class", "market_et", "target_20m_bp", "target_30m_bp")
FORECAST_COLUMNS = ("event_id", "label", "seed", "forecast_20m_bp", "forecast_30m_bp")
TEXT_ARM = "Llama-3.1-8B(frozen)+CNN(frozen)~Pool~Add"
TEXT_RUN_SEEDS = (0, 1, 2)
TEXT_RUN_DONOR_SEEDS = (11, 23, 37)
LABEL = re.compile(r"^[^\s,\"]+$")


def endpoint_metrics(truth, forecast, weights):
    if truth.shape != forecast.shape or truth.shape != (len(weights), len(HORIZONS)):
        raise ValueError("Expected matched [event, horizon] arrays")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
        raise ValueError("Non-finite endpoint value")
    result = {}
    for col, horizon in enumerate(HORIZONS):
        actual, predicted = truth[:, col], forecast[:, col]
        eligible = actual != 0
        hits = np.sign(actual[eligible]) == np.sign(predicted[eligible])
        result[str(horizon)] = {
            "mae_bp": float(np.average(np.abs(actual - predicted), weights=weights)),
            "direction_accuracy": float(np.average(hits, weights=weights[eligible])) if eligible.any() else None,
            "direction_eligible_rows": int(eligible.sum()),
            "predicted_flat_fraction": float(np.average(predicted == 0, weights=weights)),
        }
    return result


def bootstrap_gain(gain, dates, weights, config):
    days, idx = np.unique(dates, return_inverse=True)
    day_weights = np.bincount(idx, weights=weights)
    day_gain = np.stack([np.bincount(idx, weights=weights * col) for col in gain.T], axis=1)
    intervals = {}
    for block in (5, 10):
        if len(days) < 3 * block:
            intervals[str(block)] = None
            continue
        rng = np.random.default_rng(config["bootstrap_seed"])
        start = rng.integers(len(days), size=(config["bootstrap_replicates"], math.ceil(len(days) / block)))
        chosen = ((start[..., None] + np.arange(block)) % len(days)).reshape(len(start), -1)[:, :len(days)]
        draws = day_gain[chosen].sum(axis=1) / day_weights[chosen].sum(axis=1, keepdims=True)
        intervals[str(block)] = np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()
    return intervals


def instant_weights(market_et):
    """Each release instant gets total weight 1, shared equally by its releases."""
    series = pd.Series(np.asarray(market_et, dtype=str))
    return 1.0 / series.map(series.value_counts()).to_numpy(dtype=float)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference(path, allow_test=False):
    ref = pd.read_csv(path, dtype={"event_id": str, "split": str, "evidence_class": str, "market_et": str})
    missing = [c for c in REFERENCE_COLUMNS if c not in ref.columns]
    if missing:
        raise ValueError(f"Reference file lacks columns {missing}")
    if ref.event_id.duplicated().any():
        raise ValueError("Duplicate event_id in reference file")
    splits = set(ref.split)
    if len(splits) != 1:
        raise ValueError(f"Reference file mixes splits {sorted(splits)}")
    if "test" in splits and not allow_test:
        raise ValueError("Reference file is the test split; pass --allow-test only after the team freezes the test run")
    if ref[["target_20m_bp", "target_30m_bp"]].isna().any().any():
        raise ValueError("Missing target in reference file")
    return ref.reset_index(drop=True)


def load_forecasts(paths, reference):
    known = set(reference.event_id)
    frames, owner = [], {}
    for path in paths:
        frame = pd.read_csv(path, dtype={"event_id": str, "label": str})
        missing = [c for c in FORECAST_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        frame = frame[list(FORECAST_COLUMNS)].copy()
        if frame.seed.isna().any() or not np.array_equal(frame.seed, frame.seed.astype(np.int64)):
            raise ValueError(f"{path}: seed must be an integer on every row")
        frame["seed"] = frame.seed.astype(np.int64)
        values = frame[["forecast_20m_bp", "forecast_30m_bp"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite forecast")
        for label in frame.label.unique():
            if not isinstance(label, str) or not LABEL.match(label):
                raise ValueError(f"{path}: invalid label {label!r}")
            if label == PERSISTENCE:
                raise ValueError(f"{path}: {PERSISTENCE} is built in; do not supply it")
            if label in owner:
                raise ValueError(f"Label {label} appears in both {owner[label]} and {path}")
            owner[label] = str(path)
        unknown = set(frame.event_id) - known
        if unknown:
            raise ValueError(f"{path}: {len(unknown)} event_id(s) not in the reference, e.g. {sorted(unknown)[:3]}")
        if frame.duplicated(["event_id", "label", "seed"]).any():
            raise ValueError(f"{path}: duplicate (event_id, label, seed) rows")
        frames.append(frame)
    forecasts = pd.concat(frames, ignore_index=True)
    for (label, seed), group in forecasts.groupby(["label", "seed"]):
        first = forecasts.loc[forecasts.label == label, "seed"].iloc[0]
        expected = set(forecasts.loc[(forecasts.label == label) & (forecasts.seed == first), "event_id"])
        if set(group.event_id) != expected:
            raise ValueError(f"Seeds of {label} cover different releases; each seed must cover the same set")
    return forecasts


def seed_matrix(forecasts, label, event_ids):
    """Return [seed, event, horizon] forecasts for ``label`` on ``event_ids`` and the seed list."""
    rows = forecasts.loc[forecasts.label == label]
    seeds = sorted(rows.seed.unique())
    out = []
    for seed in seeds:
        indexed = rows.loc[rows.seed == seed].set_index("event_id")
        out.append(indexed.loc[event_ids, ["forecast_20m_bp", "forecast_30m_bp"]].to_numpy(dtype=float))
    return np.stack(out), [int(s) for s in seeds]


def coverage(forecasts, label):
    if label == PERSISTENCE:
        return None
    return set(forecasts.loc[forecasts.label == label, "event_id"])


def subset_rows(reference, mask, *labels, forecasts):
    keep = mask.copy()
    for label in labels:
        covered = coverage(forecasts, label)
        if covered is not None:
            keep &= reference.event_id.isin(covered).to_numpy()
    return reference.loc[keep].reset_index(drop=True)


def forecast_for(forecasts, label, rows):
    if label == PERSISTENCE:
        return np.zeros((1, len(rows), len(HORIZONS))), []
    return seed_matrix(forecasts, label, rows.event_id.tolist())


def describe(rows):
    return {"rows": len(rows), "release_instants": int(rows.market_et.nunique()),
            "dates": int(rows.market_et.str[:10].nunique())}


def score_label(forecasts, reference, mask, label):
    rows = subset_rows(reference, mask, label, forecasts=forecasts)
    if rows.empty:
        return None
    truth = rows[["target_20m_bp", "target_30m_bp"]].to_numpy(dtype=float)
    weights = instant_weights(rows.market_et)
    matrix, seeds = forecast_for(forecasts, label, rows)
    per_seed = [endpoint_metrics(truth, m, weights) for m in matrix]
    return {
        **describe(rows), "seeds": seeds,
        "metrics_seed_average_forecast": endpoint_metrics(truth, matrix.mean(axis=0), weights),
        "mae_bp_mean_over_seeds": {str(h): float(np.mean([s[str(h)]["mae_bp"] for s in per_seed])) for h in HORIZONS},
        "direction_accuracy_mean_over_seeds": {
            str(h): (None if any(s[str(h)]["direction_accuracy"] is None for s in per_seed)
                     else float(np.mean([s[str(h)]["direction_accuracy"] for s in per_seed]))) for h in HORIZONS},
    }


def compare(forecasts, reference, mask, label, other, config=BOOTSTRAP):
    """Paired gain of ``label`` over ``other``: loss(other) - loss(label). Positive favours ``label``."""
    rows = subset_rows(reference, mask, label, other, forecasts=forecasts)
    if rows.empty:
        return None
    truth = rows[["target_20m_bp", "target_30m_bp"]].to_numpy(dtype=float)
    weights = instant_weights(rows.market_et)
    dates = rows.market_et.str[:10].to_numpy(dtype=str)
    mine = forecast_for(forecasts, label, rows)[0]
    theirs = forecast_for(forecasts, other, rows)[0]
    # Primary: loss of the seed-average forecast (as in scripts/run_fed_spx_text.py).
    gain = np.abs(theirs.mean(axis=0) - truth) - np.abs(mine.mean(axis=0) - truth)
    # Secondary: per-release loss averaged over seeds (as in scripts/score_fed_spx_matched_control.py).
    seed_gain = np.abs(theirs - truth).mean(axis=0) - np.abs(mine - truth).mean(axis=0)
    return {
        "label": label, "other": other, **describe(rows),
        "gain_bp_20_30": np.average(gain, weights=weights, axis=0).tolist(),
        "paired_gain_ci95_bp_by_block_20_30": bootstrap_gain(gain, dates, weights, config),
        "seed_mean_loss_gain_bp_20_30": np.average(seed_gain, weights=weights, axis=0).tolist(),
        "seed_mean_loss_paired_gain_ci95_bp_by_block_20_30": bootstrap_gain(seed_gain, dates, weights, config),
    }


def default_comparisons(labels, text_arms):
    pairs = [(label, PERSISTENCE) for label in labels]
    for arm in text_arms:
        if arm not in labels:
            raise ValueError(f"--text-arm {arm} has no forecasts")
        if "none+CNN~none~none" in labels:
            pairs.append((arm, "none+CNN~none~none"))
        pairs += [(arm, label) for label in labels if label.startswith(arm + "[")]
    return pairs


def score(reference, forecasts, comparisons):
    labels = sorted(forecasts.label.unique())
    strict = ~reference.evidence_class.isin(INFERRED_CLASSES).to_numpy()
    subsets = {"ALL": np.ones(len(reference), bool), "STRICT": strict}
    results = {}
    for name, mask in subsets.items():
        results[name] = {
            "reference": describe(reference.loc[mask]),
            "forecasters": {label: score_label(forecasts, reference, mask, label)
                            for label in [PERSISTENCE, *labels]},
            "comparisons": [compare(forecasts, reference, mask, a, b) for a, b in comparisons],
        }
    return results


def print_summary(results):
    for name, block in results.items():
        ref = block["reference"]
        print(f"{name}: {ref['rows']} releases, {ref['release_instants']} instants, {ref['dates']} dates")
        for label, s in block["forecasters"].items():
            if s is None:
                continue
            m = s["metrics_seed_average_forecast"]
            print(f"  {label:60s} n={s['rows']:4d}  MAE20 {m['20']['mae_bp']:8.3f}  MAE30 {m['30']['mae_bp']:8.3f}")
        for c in block["comparisons"]:
            if c is None:
                continue
            ci = c["paired_gain_ci95_bp_by_block_20_30"]["10"]
            ci_text = ("[" + ", ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in ci) + "]") if ci else "too few dates"
            gain = ", ".join(f"{g:+.3f}" for g in c["gain_bp_20_30"])
            print(f"  {c['label']} vs {c['other']}: n={c['rows']} gain [{gain}] bp, CI95 10-date blocks {ci_text}")


def cmd_score(args):
    reference = load_reference(args.reference, allow_test=args.allow_test)
    forecasts = load_forecasts(args.forecasts, reference)
    labels = sorted(forecasts.label.unique())
    comparisons = default_comparisons(labels, args.text_arm)
    for a, b in args.compare or []:
        for label in (a, b):
            if label != PERSISTENCE and label not in labels:
                raise ValueError(f"--compare names unknown label {label}")
        comparisons.append((a, b))
    comparisons = list(dict.fromkeys(comparisons))
    results = score(reference, forecasts, comparisons)
    report = {
        "scorer": "scripts/score_fed_spx_paired.py", "scorer_sha256": sha256_file(__file__),
        "split": reference.split.iloc[0], "horizons_minutes": list(HORIZONS),
        "weighting": "unit weight per release instant within each scored row set",
        "gain_sign": "loss(other) - loss(label); positive favours label",
        "seed_handling": ("primary metrics and gains use the seed-average forecast; seed_mean_loss_* gains "
                          "average each release's loss over seeds instead"),
        "strict_subset_excludes": sorted(INFERRED_CLASSES), "bootstrap": BOOTSTRAP,
        "reference": {"path": Path(args.reference).as_posix(), "sha256": sha256_file(args.reference)},
        "forecast_files": [{"path": Path(p).as_posix(), "sha256": sha256_file(p)} for p in args.forecasts],
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_summary(results)


def write_forecasts(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def long_rows(label, seed, event_ids, values):
    return pd.DataFrame({"event_id": event_ids, "label": label, "seed": seed,
                         "forecast_20m_bp": values[:, 0], "forecast_30m_bp": values[:, 1]})


def cmd_from_text_run(args):
    reference = load_reference(args.reference)
    data = np.load(args.npz)
    event_ids = data["event_ids"].astype(str)
    n = len(event_ids)
    expected = {"real": (3, n, 2), "price": (3, n, 2), "text_zero": (3, n, 2), "text_shuffled": (3, 3, n, 2)}
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {shape}")
    ref = reference.set_index("event_id")
    if not set(event_ids) <= set(ref.index):
        raise ValueError("Text-run events are missing from the reference")
    truth = ref.loc[event_ids, ["target_20m_bp", "target_30m_bp"]].to_numpy(dtype=float)
    max_truth_diff = float(np.max(np.abs(truth - data["truth"])))
    if max_truth_diff > 1e-6:
        raise ValueError(f"Text-run targets differ from the reference by up to {max_truth_diff} bp")
    donor = data["donor_eligible"].astype(bool)
    frames = []
    for i, seed in enumerate(TEXT_RUN_SEEDS):
        frames.append(long_rows(args.label, seed, event_ids, data["real"][i]))
        frames.append(long_rows(f"{args.label}[text=zero]", seed, event_ids, data["text_zero"][i]))
        for j, draw in enumerate(TEXT_RUN_DONOR_SEEDS):
            frames.append(long_rows(f"{args.label}[text=shuffled-{draw}]", seed, event_ids[donor],
                                    data["text_shuffled"][i, j][donor]))
    write_forecasts(pd.concat(frames, ignore_index=True), args.out)
    print(f"Wrote {args.out}: {n} releases, {int(donor.sum())} with a past donor; targets match to {max_truth_diff:.2g} bp")
    if args.baselines:
        base = load_forecasts([args.baselines], reference)
        cnn = seed_matrix(base, "none+CNN~none~none", list(event_ids))[0]
        diff = float(np.max(np.abs(cnn - data["price"])))
        print(f"Text-run price-only output vs packaged none+CNN~none~none forecasts: max |diff| {diff:.3g} bp")


def cmd_from_matched_control(args):
    source = pd.read_csv(args.csv, dtype={"event_id": str, "evidence_class": str, "market_et": str})
    weights = instant_weights(source.market_et)
    if not np.allclose(weights, source.weight.to_numpy(dtype=float), rtol=0, atol=1e-9):
        raise ValueError("Source weights are not unit-per-instant weights")
    reference = source[["event_id", "event_type", "evidence_class", "market_et", "origin_et",
                        "target_20m_bp", "target_30m_bp"]].copy()
    reference.insert(1, "split", args.split)
    Path(args.reference_out).parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(args.reference_out, index=False, float_format="%.10g", lineterminator="\n")
    ids = source.event_id.to_numpy(str)
    frames = []
    for column, label in (("control", "none+CNN~none~none"), ("dlinear", "none+DLinear~none~none")):
        for seed in (0, 1, 2):
            values = source[[f"{column}_seed{seed}_20m_bp", f"{column}_seed{seed}_30m_bp"]].to_numpy(dtype=float)
            frames.append(long_rows(label, seed, ids, values))
    frames.append(long_rows("none+Ridge~none~none", 0, ids, source[["ridge_20m_bp", "ridge_30m_bp"]].to_numpy(float)))
    write_forecasts(pd.concat(frames, ignore_index=True), args.forecasts_out)
    print(f"Wrote {args.reference_out} ({len(reference)} releases) and {args.forecasts_out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score", help="score forecast files against a reference file")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--forecasts", type=Path, nargs="+", required=True)
    p.add_argument("--text-arm", action="append", default=[],
                   help="architecture ID; adds comparisons with none+CNN~none~none and its [..] controls")
    p.add_argument("--compare", nargs=2, action="append", metavar=("LABEL", "OTHER"),
                   help="extra paired comparison; positive gain favours LABEL")
    p.add_argument("--out", type=Path, help="metrics JSON path")
    p.add_argument("--allow-test", action="store_true", help="permit a test-split reference (team decision only)")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("from-text-run", help="convert run_fed_spx_text.py val-paired.npz")
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", default=TEXT_ARM)
    p.add_argument("--baselines", type=Path, help="baseline forecast file, to check the text run's price-only output")
    p.set_defaults(func=cmd_from_text_run)
    p = sub.add_parser("from-matched-control", help="convert score_fed_spx_matched_control.py forecasts")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--split", default="val", choices=("val",))
    p.add_argument("--reference-out", type=Path, required=True)
    p.add_argument("--forecasts-out", type=Path, required=True)
    p.set_defaults(func=cmd_from_matched_control)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
