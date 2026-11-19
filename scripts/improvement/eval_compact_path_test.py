#!/usr/bin/env python3
"""Score the six already-selected compact-path checkpoints on the canonical C1 test split."""
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.improvement.compact_path_metrics import CHANNELS, FORECAST_BARS, HORIZONS, metrics

TASK = "c1-ohlc-return-path-v1"
SNAPSHOT = "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf"
MANIFEST = "1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1"
TIMING = "category4_1400_strict_before"
MODES, SEEDS = ("price", "residual"), (0, 1, 2)
VAL_COUNT, TEST_COUNT, TEST_DATES, WIDTH = 2302, 2236, 845, 32
R3 = {
    "path_mae_bp": 80.61504705986441,
    "zero_path_mae_bp": 80.48726445009157,
    "source": "https://github.com/DaezeI/11785-GS-Fuse-Submission-Public/blob/main/"
              "results/adaeze-screened-r3-h200-v2/metrics.json",
}
# Freeze the six choices already reported from the RTX 4060 validation run.
VALIDATION_LOCK = {
    ("price", 0): (1, 82.51069734456678),
    ("price", 1): (2, 82.31276913381045),
    ("price", 2): (1, 83.05439274865526),
    ("residual", 0): (3, 82.34442801793188),
    ("residual", 1): (0, 82.31276913381045),
    ("residual", 2): (2, 82.55060333880964),
}
SOURCE_FILES = (
    "model/compact_return.py", "model/compact_raw_text.py", "model/compact_news_vocab.py",
    "scripts/improvement/train_compact_path.py", "scripts/improvement/compact_path_metrics.py",
    "scripts/improvement/train_compact_raw.py", "scripts/improvement/prepare_raw_inputs.py",
    "scripts/freeze_c1_inputs.py", "data/c1_protocol.py", "data/forecast_task.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_same(actual, expected, message):
    for key, value in expected.items():
        check(actual.get(key) == value, f"{message}: {key}")


def check_close(actual, expected, message):
    check(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
          and np.allclose(actual, expected, rtol=1e-5, atol=0.002), message)


def check_ids(actual, expected, count, split):
    check(actual.shape == (count,) and actual.dtype.kind in "US" and len(set(actual)) == count
          and np.array_equal(actual, expected), f"{split} event IDs or order differ from canonical C1")


def read_runs(validation_root, sources):
    """Read all six checkpoint/validation byte pairs and lock their hashes."""
    records = {}
    for seed in SEEDS:
        for mode in MODES:
            folder = validation_root / mode / f"seed-{seed}"
            summary_blob = (folder / "resolved_config.json").read_bytes()
            summary = json.loads(summary_blob)
            label = f"{mode}/seed-{seed}"
            check_same(summary, dict(task=TASK, forecast_bars=FORECAST_BARS,
                       channels=["open", "high", "low", "close"], width=WIDTH,
                       seq_len=35, horizons=list(HORIZONS), timing_policy=TIMING,
                       snapshot_content_sha256=SNAPSHOT, manifest_sha256=MANIFEST,
                       mode=mode, seed=seed, epochs=30, batch_size=32,
                       input_class="canonical_bytes_timing_unverified", source_sha256=sources,
                       release_timestamps_verified=False, test_scored=False, scale_fit="training_only",
                       selection="minimum validation whole-path OHLC return MAE (epoch 0 eligible), patience 5"),
                       f"Unexpected validation run {label}")
            check(summary["vali"]["events"] == VAL_COUNT and 0 <= summary["best_epoch"] <= summary["epochs_run"] <= 30,
                  f"Invalid validation count or checkpoint selection: {label}")
            locked_epoch, locked_score = VALIDATION_LOCK[mode, seed]
            check(summary["best_epoch"] == locked_epoch and math.isclose(
                summary["vali"]["matched"]["path_mae_bp"], locked_score, rel_tol=0, abs_tol=0.000001),
                f"Validation record does not match the previously reported checkpoint selection: {label}")
            checkpoint = (folder / "best.pt").read_bytes()
            val_blob = (folder / "vali.npz").read_bytes()
            if mode == "residual":
                check(summary["price_checkpoint_sha256"] == records["price", seed]["lock"]["checkpoint_sha256"],
                      f"Residual does not use its same-seed price checkpoint: {label}")
                check(summary["encoder_gradient_observed"] is True, f"Invalid encoder run: {label}")
            records[mode, seed] = dict(summary=summary, checkpoint=checkpoint, validation=val_blob,
                lock=dict(mode=mode, seed=seed, best_epoch=summary["best_epoch"],
                          validation_path_mae_bp=summary["vali"]["matched"]["path_mae_bp"],
                          checkpoint_sha256=digest(checkpoint), validation_sha256=digest(val_blob),
                          config_sha256=digest(summary_blob)))
    return records


def load_model(record, mode, torch, device):
    from model.compact_raw_text import RawTextEncoder, TrainVocabulary
    from scripts.improvement.train_compact_path import CompactPath, PathForecast
    from scripts.improvement.train_compact_raw import TEXT_DIM

    payload = torch.load(BytesIO(record["checkpoint"]), map_location="cpu", weights_only=True)
    config, summary = payload["configuration"], record["summary"]
    mandatory = {"task", "forecast_bars", "channels", "snapshot_content_sha256", "manifest_sha256",
                 "scale_bp", "seed", "width", "seq_len", "horizons", "timing_policy", "epochs",
                 "batch_size", "source_commit", "source_dirty", "source_sha256", "mode"}
    check_same(config, {key: summary[key] for key in mandatory}, f"Checkpoint config mismatch: {record['lock']}")
    if mode == "price":
        check(payload["tokenizer"] is None, "Unexpected price-only checkpoint tokenizer")
        model, vocab = PathForecast(CompactPath(WIDTH)), None
    else:
        vocab = TrainVocabulary.from_dict(payload["tokenizer"])
        check(2 <= len(vocab) <= 16_000 and vocab.max_tokens == 512, "Unexpected saved vocabulary dimensions")
        model = PathForecast(CompactPath(WIDTH, TEXT_DIM), RawTextEncoder(len(vocab), TEXT_DIM, 64))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.requires_grad_(False).to(device).eval(), vocab


def validate_replay(record, val_ids, truth, matched, price):
    with np.load(BytesIO(record["validation"]), allow_pickle=False) as val:
        check_ids(val["event_ids"], val_ids, VAL_COUNT, "Validation")
        check_close(val["targets_bp"], truth, "Validation targets differ from the canonical snapshot")
        check_close(val["predictions_bp"], matched, "Checkpoint does not replay its saved validation predictions")
        check_close(val["price_reference_bp"], price, "Same-seed price checkpoint does not replay its saved reference")
        for key, arr in (("matched", val["predictions_bp"]), ("price_reference", val["price_reference_bp"])):
            expected = record["summary"]["vali"][key]["path_mae_bp"]
            check(math.isclose(metrics(val["targets_bp"], arr)["path_mae_bp"], expected, abs_tol=1e-7),
                  f"Incorrect recorded validation MAE: {key}")


def make_scores(truth, price, residual):
    rows = []
    for i, seed in enumerate(SEEDS):
        p, r = metrics(truth, price[i]), metrics(truth, residual[i])
        rows.append(dict(seed=seed, price=p, price_plus_news=r,
                         improvement_bp=p["path_mae_bp"] - r["path_mae_bp"]))
    pmean = float(np.mean([row["price"]["path_mae_bp"] for row in rows]))
    rmean = float(np.mean([row["price_plus_news"]["path_mae_bp"] for row in rows]))
    return rows, dict(price_path_mae_bp=pmean, price_plus_news_path_mae_bp=rmean,
                      improvement_bp=pmean - rmean)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    out, snapshot, val_root = (p.resolve() for p in (args.out, args.snapshot_root, args.validation_root))
    for source in (snapshot, val_root):
        check(out != source and source not in out.parents, "Use a new output directory outside both input trees")
    check(not out.exists(), "Use a new output directory")
    sources = {key: digest((ROOT / key).read_bytes().replace(b"\r\n", b"\n")) for key in SOURCE_FILES}
    records = read_runs(val_root, sources)

    import torch
    from scripts.improvement.train_compact_raw import articles, load_c1, predict
    from scripts.improvement.train_compact_path import price_tensors

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this Python environment")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    data, snap_sha, manifest_sha = load_c1(snapshot, seed=0, exploratory=False)
    check((snap_sha, manifest_sha) == (SNAPSHOT, MANIFEST), "Canonical C1 inputs required")
    scale = float(data.task_config.scale_bp)
    check(math.isfinite(scale) and scale > 0, "Invalid train-only return scale")
    role_ids = {role: np.asarray(getattr(data, f"{role}_set").event_ids) for role in ("vali", "test")}
    # load_c1 verifies the exact count and order separately against the checked-in manifest.
    for role, count in (("vali", VAL_COUNT), ("test", TEST_COUNT)):
        check_ids(role_ids[role], role_ids[role], count, role.capitalize())
    test_dates = len({event_id.split("/")[-1][:8] for event_id in role_ids["test"]})
    check(test_dates == TEST_DATES, f"Expected {TEST_DATES} canonical test dates; found {test_dates}")
    xval, vy = price_tensors(data.vali_set, scale)
    vtruth = vy.numpy() * scale
    vtexts = articles(data.vali_set)

    # No test inference until all six predeclared checkpoints replay their validation results.
    loaded = {}
    with torch.inference_mode():
        for seed in SEEDS:
            for mode in MODES:
                key = mode, seed
                record = records[key]
                check(math.isclose(record["summary"]["scale_bp"], scale, abs_tol=1e-7),
                      "Checkpoint scale differs from the canonical train-only scale")
                model, vocab = load_model(record, mode, torch, args.device)
                tokens = (torch.tensor([vocab.encode(t) for t in vtexts], dtype=torch.long)
                          if vocab is not None else None)
                matched = predict(model, xval, args.device, tokens).numpy() * scale
                if mode == "price":
                    base = matched
                    price_weights = model.compact.state_dict()
                else:
                    residual_weights = model.compact.state_dict()
                    check(all(torch.equal(v, residual_weights[k]) for k, v in price_weights.items()),
                          "Residual price weights differ from the same-seed frozen price checkpoint")
                    check_close(predict(model, xval, args.device).numpy() * scale, base,
                                "Residual no-text behavior differs from same-seed price model")
                validate_replay(record, role_ids["vali"], vtruth, matched, base)
                loaded[key] = model, vocab
        lock = dict(schema="gs-fuse-compact-path-test-lock-v1", task=TASK, seeds=list(SEEDS),
                    modes=list(MODES), snapshot_content_sha256=snap_sha, manifest_sha256=manifest_sha,
                    timing_policy=TIMING, release_timestamps_verified=False, scale_bp=scale,
                    test_events=TEST_COUNT, test_dates=test_dates,
                    test_ids_sha256=digest("\n".join(role_ids["test"]).encode("utf-8")),
                    evaluator_sha256=digest(Path(__file__).read_bytes()),
                    train_source_sha256=sources, runs=[r["lock"] for r in records.values()])
        out.mkdir(parents=True, exist_ok=False)
        lock_path = out / "checkpoint-lock.json"
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        print("All six validation checkpoints verified. Scoring canonical C1 test (no training).", flush=True)

        xtest, ty = price_tensors(data.test_set, scale)
        truth = ty.numpy() * scale
        texts = articles(data.test_set)
        prices, residuals = [], []
        for seed in SEEDS:
            for mode, collection in (("price", prices), ("residual", residuals)):
                model, vocab = loaded[mode, seed]
                tokens = (torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
                          if vocab is not None else None)
                forecast = predict(model, xtest, args.device, tokens).numpy() * scale
                check_close(forecast, forecast, f"Non-finite test output: {mode}/{seed}")
                collection.append(forecast)
    price, residual = np.stack(prices), np.stack(residuals)
    rows, means = make_scores(truth, price, residual)
    zero = metrics(truth, np.zeros_like(truth))
    zero_delta = zero["path_mae_bp"] - R3["zero_path_mae_bp"]
    comparison = dict(**R3, local_minus_r3_zero_mae_bp=zero_delta, zero_matches=abs(zero_delta) <= 0.02,
                      caveat="Same published C1 snapshot, manifest, timing and scoring metric; "
                             "R3 per-event predictions are not available here for an independent ID check.")
    if not comparison["zero_matches"]:
        print("Warning: test zero baseline differs from published R3 by over 0.02 bp.", flush=True)
    summary = dict(schema="gs-fuse-compact-path-test-summary-v1", task=TASK, split="test", test_scored=True,
                   test_events=TEST_COUNT, test_dates=test_dates, timing_policy=TIMING, release_timestamps_verified=False,
                   snapshot_content_sha256=snap_sha, manifest_sha256=manifest_sha,
                   test_ids_sha256=lock["test_ids_sha256"], lock_sha256=digest(lock_path.read_bytes()),
                   training=False, aggregation="mean absolute error over events, 140 bars and OHLC",
                   reporting_units="basis-point log return relative to last observed close",
                   seeds=list(SEEDS), device=args.device, torch_version=str(torch.__version__),
                   per_seed=rows, means=means, zero=zero, published_r3=comparison)
    np.savez_compressed(out / "test-predictions.npz", event_ids=role_ids["test"], seed_order=np.asarray(SEEDS),
                        targets_bp=truth, price_bp=price, price_plus_news_bp=residual)
    (out / "test-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(means, indent=2), flush=True)
    print(f"Saved {out / 'test-summary.json'}", flush=True)


if __name__ == "__main__":
    main()
