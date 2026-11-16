#!/usr/bin/env python3
"""Replay structured return metrics from a completed smoke checkpoint.

This entrypoint performs inference only. It reconstructs the exact screened-C1
task and model settings from the completed run's resolved configuration, loads
the selected checkpoint, and evaluates the same evenly spaced validation and
test subsets used by ``smoke_train.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.forecast_task import ForecastTaskConfig, RETURN_TASK
from interpret import evenly_spaced
import model.gs_fuse as gs_fuse_mod
from model.gs_fuse import (
    GSFuse,
    evaluate_return_loader,
    return_metrics_for_tracking,
    write_return_metrics,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--series-dir", type=Path, required=True)
    parser.add_argument("--moment-path", type=Path, required=True)
    parser.add_argument("--text-model-path", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; use a fresh replay directory")
    return args


def load_run_contract(source_run: Path):
    resolved_path = source_run / "resolved_config.json"
    audit_path = source_run / "c1_data_audit.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    task = ForecastTaskConfig(**resolved["forecast_task_config"])
    if task.task != RETURN_TASK or resolved.get("forecast_task") != RETURN_TASK:
        raise ValueError("Source run is not a return-task run")
    if resolved.get("run_kind") != "smoke_subset":
        raise ValueError("Metrics replay requires a smoke-subset source run")
    required = {
        "source_commit", "seed", "n_event_eval", "seq_len", "pred_len",
        "batch_size", "sw_stride", "window", "stride", "decoder_layers",
        "decoder_heads", "use_ts_memory", "max_token_num", "fusion_type",
        "gate_type", "num_experts", "moe_top_k", "num_queries",
        "qformer_layers", "use_raw_ts_residual", "use_learnable_last_value",
        "c1_timing_policy", "c1_skip_missing", "no_4bit", "tf32",
    }
    missing = sorted(required.difference(resolved))
    if missing:
        raise ValueError(f"Source run configuration is missing: {missing}")
    if not isinstance(resolved["source_commit"], str) or len(resolved["source_commit"]) != 40:
        raise ValueError("Source run does not record a full source commit")
    return resolved, audit, task, resolved_path, audit_path


def subset_loader(dataset, count: int, batch_size: int, seed: int):
    indices = evenly_spaced(len(dataset), count)
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    ), [int(index) for index in indices]


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        text=True,
    ).strip()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Metrics replay requires a CUDA GPU")
    for path in (args.checkpoint, args.manifest, args.text_cache):
        if not path.is_file():
            raise FileNotFoundError(path)

    resolved, prior_audit, task, resolved_path, audit_path = load_run_contract(args.source_run)
    if file_hash(args.manifest) != prior_audit["manifest_sha256"]:
        raise ValueError("Manifest hash differs from the completed smoke audit")

    seed = int(resolved["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gs_fuse_mod.USE_4BIT = not bool(resolved["no_4bit"])
    gs_fuse_mod.TEXT_CACHE_PATH = str(args.text_cache)
    torch.backends.cuda.matmul.allow_tf32 = bool(resolved["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(resolved["tf32"])

    from data.c1_protocol import build_c1_dataloaders
    events = build_c1_dataloaders(
        args.manifest, args.event_dir, args.series_dir / "SP500.csv",
        seq_len=resolved["seq_len"], pred_len=resolved["pred_len"],
        batch_size=resolved["batch_size"], sw_stride=resolved["sw_stride"],
        timing_policy=resolved["c1_timing_policy"],
        skip_missing=resolved["c1_skip_missing"], task_mode=RETURN_TASK,
        return_scale_bp=task.scale_bp, huber_delta_bp=task.huber_delta_bp,
        direction_tau_bp=task.direction_tau_bp,
        direction_deadband_bp=task.direction_deadband_bp,
        direction_weight=task.direction_weight,
        direction_horizons=task.direction_horizons, seed=seed,
    )
    if events.task_config.to_dict() != task.to_dict():
        raise ValueError("Reconstructed task configuration differs from training")
    audit_keys = ("manifest_sha256", "price_sha256", "return_scale_population_sha256")
    for key in audit_keys:
        if events.audit[key] != prior_audit[key]:
            raise ValueError(f"Reconstructed data audit differs at {key}")

    provenance = {
        "manifest_sha256": events.audit["manifest_sha256"],
        "price_sha256": events.audit["price_sha256"],
        "scale_population_sha256": events.audit["return_scale_population_sha256"],
        "training_seed": seed,
        "source_commit": resolved["source_commit"],
    }
    model_args = SimpleNamespace(
        text_model_path=str(args.text_model_path), ts_backbone="moment",
        text_backbone="llama", moment_path=str(args.moment_path),
        kronos_model_path=None, kronos_tokenizer_path=None,
        seq_len=resolved["seq_len"], pred_len=resolved["pred_len"], d=events.d,
        window=resolved["window"], stride=resolved["stride"],
        batch_size=resolved["batch_size"], decoder_layers=resolved["decoder_layers"],
        decoder_heads=resolved["decoder_heads"], use_ts_memory=resolved["use_ts_memory"],
        max_token_num=resolved["max_token_num"], fusion_type=resolved["fusion_type"],
        gate_type=resolved["gate_type"], num_experts=resolved["num_experts"],
        moe_top_k=resolved["moe_top_k"], num_queries=resolved["num_queries"],
        qformer_layers=resolved["qformer_layers"], forecast_task_config=task,
        use_raw_ts_residual=resolved["use_raw_ts_residual"],
        use_learnable_last_value=resolved["use_learnable_last_value"],
    )
    model = GSFuse(**vars(model_args))
    model.run_provenance = provenance
    model.load_model_combined(str(args.checkpoint), strict=True)
    model.eval()

    validation_loader, validation_indices = subset_loader(
        events.vali_set, resolved["n_event_eval"], resolved["batch_size"], seed + 1)
    test_loader, test_indices = subset_loader(
        events.test_set, resolved["n_event_eval"], resolved["batch_size"], seed + 2)

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha256 = file_hash(args.checkpoint)
    common = {
        "checkpoint_sha256": checkpoint_sha256,
        "source_run": str(args.source_run),
        "source_resolved_config_sha256": file_hash(resolved_path),
        "source_data_audit_sha256": file_hash(audit_path),
        "replay_code_commit": source_commit(),
        "inference_only": True,
    }
    results = {}
    for split, loader, indices in (
        ("validation", validation_loader, validation_indices),
        ("test", test_loader, test_indices),
    ):
        result = evaluate_return_loader(model, loader)
        results[split] = result
        write_return_metrics(
            args.output / f"{split}_return_metrics.json", split, result, model,
            args.checkpoint, subset_indices=indices, **common,
        )
        print(json.dumps(return_metrics_for_tracking(split, result), indent=2))

    summary = {
        "schema_version": 1,
        **common,
        "source_checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "forecast_task": task.to_dict(),
        "subsets": {"validation": validation_indices, "test": test_indices},
        "metrics": results,
    }
    (args.output / "metrics_replay_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[done] Structured metrics written to {args.output}")


if __name__ == "__main__":
    main()
