#!/usr/bin/env python3
"""Fit frozen-C1 price, metadata and article ridge readouts on identical rows."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from data.c1_protocol import ROLES, build_c1_dataloaders, sha256_file
from model.text_cache import TextEmbeddingCache, cache_signature
from model.text_encoders import build_text_encoder
from scripts.analyze_text_predictions import block_bootstrap_improvement


HORIZON = 35
TIMING_POLICY = "category4_1400_strict_before"
EXPECTED_COUNTS = {"train": 6167, "vali": 2302, "test": 2236}
DONOR_SEEDS = (23, 24, 25)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
JACCARD_CUTOFF = 0.80


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def shingles(text):
    words = re.findall(r"[a-z0-9]+", text.casefold())
    normalized = hashlib.sha256(" ".join(words).encode()).digest()
    grams = frozenset(" ".join(words[index:index + 5]) for index in range(len(words) - 4))
    return normalized, grams


def admissible_donors(categories, dates, signatures, grams):
    groups = defaultdict(list)
    for row, (category, date) in enumerate(zip(categories, dates)):
        groups[category, date[:4]].append(row)
    donors = []
    for row, (category, date) in enumerate(zip(categories, dates)):
        accepted = []
        for other in groups[category, date[:4]]:
            if dates[other] >= date or signatures[other] == signatures[row]:
                continue
            shared = len(grams[row] & grams[other])
            total = len(grams[row]) + len(grams[other]) - shared
            if total and shared / total >= JACCARD_CUTOFF:
                continue
            accepted.append(other)
        donors.append(accepted)
    eligible = np.array([row for row, pool in enumerate(donors) if pool], dtype=int)
    draws = []
    for seed in DONOR_SEEDS:
        rng = np.random.default_rng(seed)
        draws.append(np.array([rng.choice(donors[row]) for row in eligible], dtype=int))
    return eligible, draws


def extract_split(dataset, cache, fill_embedding):
    price, calendar, categories, dates, target, raw_texts = [], [], [], [], [], []
    signatures, grams = [], []
    for event_id, item in zip(dataset.event_ids, dataset.list, strict=True):
        path, _, _, _, _, category, past_bp, targets_bp = item
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        signature, tokens = shingles(text)
        signatures.append(signature)
        grams.append(tokens)
        raw_texts.append(text)
        price.append(past_bp.numpy().ravel())
        date = event_id.split("/", maxsplit=1)[1].removesuffix(".txt")
        parsed = datetime.strptime(date, "%Y%m%d")
        phases = 2 * np.pi * np.array([parsed.weekday() / 7, (parsed.month - 1) / 12])
        calendar.append(np.r_[np.sin(phases), np.cos(phases)])
        categories.append(category)
        dates.append(date)
        target.append(float(targets_bp[HORIZON - 1, 3]))

    eligible, draws = admissible_donors(categories, dates, signatures, grams)
    if not len(eligible):
        raise ValueError("No events have an eligible past-date article donor")
    used = np.unique(np.concatenate([eligible, *draws]))
    positions = np.full(len(dates), -1, dtype=int)
    positions[used] = np.arange(len(used))
    vectors = []
    hits = 0
    for index in used:
        text = raw_texts[index]
        vector = cache.get(text)
        if vector is None:
            vector = fill_embedding(text)
            cache.put(text, vector)
        else:
            hits += 1
        vectors.append(vector.numpy())
    matching = positions[eligible]
    excluded = Counter(categories[row] for row in set(range(len(dates))) - set(eligible))
    return {
        "price": np.asarray(price)[eligible], "calendar": np.asarray(calendar)[eligible],
        "category": np.asarray(categories)[eligible], "date": np.asarray(dates)[eligible],
        "target": np.asarray(target)[eligible], "vectors": np.stack(vectors),
        "matched": matching, "donors": [positions[draw] for draw in draws],
        "ids": np.asarray(dataset.event_ids)[eligible],
        "donor_ids": np.asarray(dataset.event_ids)[np.stack(draws)],
        "coverage": {"input_events": len(dates), "eligible_events": len(eligible),
                     "excluded_without_donor_by_type": dict(sorted(excluded.items())),
                     "article_rows_used": len(used), "cache_hits": hits},
    }


def choose_ridge(train_features, train_y, vali_features, vali_y):
    scaler = StandardScaler().fit(train_features)
    train_x, vali_x = scaler.transform(train_features), scaler.transform(vali_features)
    error, best_alpha, best_model = min(
        (float(np.abs(model.predict(vali_x) - vali_y).mean()), alpha, model)
        for alpha in RIDGE_ALPHAS
        for model in [Ridge(alpha=alpha).fit(train_x, train_y)]
    )
    return best_model, scaler, {"alpha": best_alpha, "validation_mae_bp": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--text-model", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True)

    print("LOAD_FROZEN_C1", flush=True)
    c1 = build_c1_dataloaders(args.snapshot / "manifest.json", args.snapshot / "event",
                              args.snapshot / "series/SP500.csv", timing_policy=TIMING_POLICY,
                              task_mode="return_bp", seed=0)
    if c1.audit["emitted_counts"] != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected frozen event counts: {c1.audit['emitted_counts']}")
    write_json(args.out / "c1_data_audit.json", c1.audit)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cache = TextEmbeddingCache.load(args.cache, cache_signature(args.text_model, 500, 400, True))
    encoder = None
    misses = 0

    def fill_embedding(text):
        nonlocal encoder, misses
        if encoder is None:
            if not torch.cuda.is_available():
                raise RuntimeError("Cache misses need the frozen Llama encoder on a GPU")
            encoder = build_text_encoder("llama", str(args.text_model))
        misses += 1
        if misses == 1 or misses % 200 == 0:
            print(f"EMBEDDING_CACHE_MISSES {misses}", flush=True)
        return encoder.encode_instance(text, 500, 400, torch.device("cuda")).detach().float().cpu()

    splits = {}
    for role in ROLES:
        splits[role] = extract_split(getattr(c1, role + "_set"), cache, fill_embedding)
        print(f"SPLIT_READY {role} {splits[role]['coverage']}", flush=True)
    train, vali, test = (splits[role] for role in ROLES)
    category_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train["category"][:, None])
    pca = PCA(n_components=min(64, len(train["target"]) - 1, train["vectors"].shape[1]),
              svd_solver="randomized", random_state=0).fit(train["vectors"][train["matched"]])
    for split in splits.values():
        one_hot = category_encoder.transform(split["category"][:, None])
        split["metadata"] = np.column_stack([split["price"], split["calendar"], one_hot])
        projected = pca.transform(split["vectors"])
        split["article"] = np.column_stack([split["metadata"], projected[split["matched"]]])
        split["reassigned"] = [np.column_stack([split["metadata"], projected[draw]]) for draw in split["donors"]]

    print("FIT_READOUTS", flush=True)
    predictions = {"zero_return": np.zeros_like(test["target"])}
    validation = {}
    for arm in ("price", "metadata", "article", "reassigned"):
        def features(split):
            return split[arm][0] if arm == "reassigned" else split[arm]

        model, scaler, validation[arm] = choose_ridge(
            features(train), train["target"], features(vali), vali["target"],
        )
        predictions[arm] = model.predict(scaler.transform(features(test)))
        if arm == "article":
            predictions["article_with_reassigned_test"] = np.stack([
                model.predict(scaler.transform(features)) for features in test["reassigned"]
            ])

    counts = {role: split["coverage"] for role, split in splits.items()}
    losses = {}
    for name, prediction in predictions.items():
        error = np.abs(prediction - test["target"])
        losses[name] = error.mean(axis=0) if error.ndim == 2 else error
    mae = {name: float(error.mean()) for name, error in losses.items()}
    comparisons = {}
    for name in ("zero_return", "price", "metadata", "reassigned", "article_with_reassigned_test"):
        mask = np.ones(len(test["target"]), dtype=bool)
        comparisons[name] = {
            "mae_improvement_bp": mae[name] - mae["article"],
            "mae_improvement_pct": 100 * (1 - mae["article"] / mae[name]) if mae[name] > 0 else None,
            "relative_date_block_intervals": {
                str(length): block_bootstrap_improvement(losses["article"], losses[name], test["date"],
                                                         mask, block=length, reps=2000)
                for length in (10, 20, 40)
            },
        }
    write_json(args.out / "metrics.json", {
        "target": "35-bar endpoint close log return in basis points",
        "eligible_test_dates": len(set(test["date"])), "coverage": counts,
        "test_mae_bp": mae, "article_vs_controls": comparisons, "validation_selection": validation,
    })
    write_json(args.out / "provenance.json", {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_audit_sha256": sha256_file(args.out / "c1_data_audit.json"),
        "embedding_cache_sha256": sha256_file(args.cache), "new_unique_embeddings": misses,
        "embedding_cache_signature": cache.signature, "embedding_pca_dimensions": pca.n_components_,
        "ridge_alphas": RIDGE_ALPHAS, "donor_seeds": DONOR_SEEDS,
        "donor_rule": "same split, category and calendar year; strictly earlier recorded date",
        "duplicate_rule": "normalized exact match or word five-shingle Jaccard at least 0.80",
        "release_timestamps_verified": False,
    })
    np.savez_compressed(args.out / "paired_readout.npz", event_ids=test["ids"], dates=test["date"],
                        donor_ids=test["donor_ids"], target_bp=test["target"], **predictions)
    print(f"READOUT_COMPLETE {mae}", flush=True)


if __name__ == "__main__":
    main()
