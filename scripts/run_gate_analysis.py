#!/usr/bin/env python3
"""Paired learned-vs-50/50 gate evaluation using Nicole's audited C1 loader."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_audit(recorded, actual):
    for key in ("manifest_sha256", "price_sha256", "timing_policy", "emitted_counts",
                "scaler_mean", "scaler_scale", "scaler_fit_unique_bars"):
        if key not in recorded or recorded[key] != actual[key]:
            raise ValueError(f"C1 training/evaluation audit mismatch: {key}")
    if recorded.get("excluded_counts") or actual.get("excluded_counts"):
        raise ValueError("Resolve C1 exclusions before this experiment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=("vali", "test"), required=True)
    parser.add_argument("--max-events", type=int, default=16,
                        help="16 for debugging; 0 for all events (no model-quality claim for subsets)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--probe-gradients", action="store_true",
                        help="Additional gate derivative check on the first C1 TRAIN batch")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_events < 0 or config["batch_size"] < 1:
        parser.error("max-events must be nonnegative and batch_size positive")
    # Fail on repeated outputs before loading any backbone or checkpoint.
    args.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from data.c1_protocol import build_c1_dataloaders
    import model.gs_fuse as module
    from model.gate_analysis import measure_gate_batch, gate_gradient_probe
    from model.no_text_control import predict_without_article

    checkpoint = Path(config["checkpoint"])
    training_log = checkpoint.parent / "full_training_log.txt"
    if checkpoint.name != "best_model.pth" or not checkpoint.is_file():
        raise ValueError("A completed Stage-3 best_model.pth is required")
    if "THREE-STAGE TRAINING COMPLETED" not in training_log.read_text(encoding="utf-8"):
        raise ValueError("Training log does not establish three-stage completion")
    training_audit = json.loads((checkpoint.parent / "c1_data_audit.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Full-model evaluation requires a Delta CUDA GPU")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    module.USE_4BIT = False
    module.FINETUNE_LLM_LAST_LAYER = False
    module.FINETUNE_TS_LAST_BLOCK = False
    module.TEXT_CACHE_PATH = config["text_cache"]
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    events = build_c1_dataloaders(config["manifest"], config["event_dir"], config["price"],
                                batch_size=config["batch_size"], sw_stride=10,
                                timing_policy=config["timing_policy"])
    for key, expected in config["expected_data"].items():
        if events.audit.get(key) != expected:
            raise ValueError(f"Input differs from Nicole's recorded full-run plan: {key}")
    verify_audit(training_audit, events.audit)
    checkpoint_hash = sha256(checkpoint)
    model = module.GSFuse(
        text_model_path=config["text_model_path"], moment_path=config["moment_path"],
        ts_backbone="moment", text_backbone="llama", seq_len=35, pred_len=140,
        d=events.d, window=500, stride=400, batch_size=config["batch_size"],
        decoder_layers=3, decoder_heads=8, max_token_num=1024, use_ts_memory=False,
        fusion_type="crossattn", gate_type="softmax")
    # Validate every saved component against the instantiated model, since the
    # legacy loader uses strict=False for some individual modules internally.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = saved["metadata"]
    for key, value in {"seq_len": 35, "pred_len": 140, "fusion_type": "crossattn",
                       "gate_type": "softmax", "use_4bit": False}.items():
        if meta.get(key) != value:
            raise ValueError(f"Incompatible checkpoint metadata: {key}")
    current_metadata = model._arch_metadata()
    for key in ("use_raw_ts_residual", "use_learnable_last_value", "use_multi_scale_ts",
                "llm_hidden", "text_backbone", "ts_backbone", "d", "decoder_layers", "decoder_heads"):
        if meta.get(key) != current_metadata.get(key):
            raise ValueError(f"Checkpoint/model architecture differs: {key}")
    states = saved["state_dicts"]
    required = [name for name, _ in model.named_children() if name in states]
    for name in ("ts_encoder", "text_encoder", "llm_proj", "ts_feat_to_fuse", "ts_inst_to_fuse",
                 "residual_project", "cross_attn_text_to_ts", "cross_attn_ts_to_text",
                 "bi_gate_layer", "decoder", "token_align_module"):
        if name not in states:
            raise ValueError(f"Missing checkpoint module: {name}")
    if module.USE_MULTI_SCALE_TS and "multi_scale_extractor" not in states:
        raise ValueError("Missing multi-scale module")
    if module.USE_LEARNABLE_LAST_VALUE and "last_value_module" not in states:
        raise ValueError("Missing last-value module")
    for name in required:
        getattr(model, name).load_state_dict(states[name], strict=True)
    del saved, states
    model.load_model_combined(str(checkpoint), strict=True)
    model.eval()
    cache = args.out / "text_token_cache" / checkpoint_hash[:12]
    cache.mkdir(parents=True)
    model.text_encoder.cache_dir = str(cache)

    def batch_items(ds, indices):
        rows = [ds.list[int(i)] for i in indices]
        texts = [Path(row[0]).read_text(encoding="utf-8", errors="ignore") for row in rows]
        seq = torch.stack([row[-2].T for row in rows]).float()
        targets = torch.stack([row[-1].T for row in rows]).float()
        return texts, seq, targets

    if args.probe_gradients:
        ds_train = events.train_set
        train_ids = list(range(min(len(ds_train), config["batch_size"])))
        measured = measure_gate_batch(model, *batch_items(ds_train, train_ids), module.TrainingConfig)
        probe = gate_gradient_probe(model, measured)
        probe["train_event_ids"] = [ds_train.event_ids[i] for i in train_ids]
        (args.out / "gradient_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
        del measured
    ds = events.vali_set if args.split == "vali" else events.test_set
    indices = np.arange(len(ds))
    if args.max_events and args.max_events < len(ds):
        indices = np.linspace(0, len(ds) - 1, args.max_events, dtype=int)
    chunks = {name: [] for name in ("learned", "fixed_050", "zeroed_article_tokens", "targets",
                                  "gate_text", "fixed_gate_text", "recomputed_gate_target",
                                  "legacy_reference_delta")}
    canonical_max_difference = None
    for start in range(0, len(indices), config["batch_size"]):
        batch_ids = indices[start:start + config["batch_size"]]
        texts, seq, targets = batch_items(ds, batch_ids)
        measured = measure_gate_batch(model, texts, seq, targets, module.TrainingConfig)
        if start == 0:
            with torch.inference_mode():
                canonical, *_ = model.predict_single_case(texts, seq)
            torch.testing.assert_close(measured["learned"], canonical, rtol=1e-4, atol=1e-5)
            canonical_max_difference = float((measured["learned"] - canonical).abs().max())
        measured["zeroed_article_tokens"] = predict_without_article(model, seq, "zeroed_article_tokens")
        for name in chunks:
            value = measured[name].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite output: {name}")
            chunks[name].append(value)
        print(f"Gate comparison: {start + len(batch_ids)}/{len(indices)}", flush=True)
    source_files = ["scripts/run_gate_analysis.py", "scripts/analyze_gate_analysis.py",
                    "model/gate_analysis.py", "model/gs_fuse.py", "model/no_text_control.py",
                    "data/c1_protocol.py", "model/text_encoders.py", "model/text_cache.py",
                    "model/decoder.py", "model/ts_encoders.py"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation; outputs will not be published")
    metadata = {"status": "completed", "owner": "Adaeze", "config": config,
                "checkpoint_sha256": checkpoint_hash, "training_log_sha256": sha256(training_log),
                "c1_data_audit": events.audit, "split": args.split,
                "subset": len(indices) != len(ds), "event_count": len(indices),
                "git_commit": revision.stdout.strip() if revision.returncode == 0 else "PENDING",
                "source_sha256": {p: sha256(ROOT / p) for p in source_files},
                "config_sha256": sha256(args.config),
                "gate_target_settings": {k: getattr(module.TrainingConfig, k) for k in
                                         ("GC_USE_ADAPTIVE_SCALE", "GC_MIN_SCALE", "GC_SIGMOID_SCALE",
                                          "GC_GAIN", "GC_LOGIT_CLAMP")},
                "constant": 0.5, "constant_selected_using_test": False,
                "canonical_first_batch_max_abs_difference": canonical_max_difference,
                "intervention": "Only normalized aligned fusion weights replaced; not article removal",
                "target_note": "Current formula recomputed in evaluation batches using text-conditioned legacy reference; not historical targets",
                "timing_note": "C1 timing sensitivity; article release times remain unverified"}
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    arrays["event_ids"] = np.array([ds.event_ids[int(i)] for i in indices])
    arrays["event_dates"] = np.array([ds.event_ids[int(i)].split("/")[1][:8] for i in indices])
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(args.out / "predictions.npz", **arrays)
    (args.out / "run_record.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved paired predictions; keep prediction/target files on Delta.")


if __name__ == "__main__":
    main()
