#!/usr/bin/env python3
"""Three-stage 16/8/8 smoke using the full frozen population's preprocessing.

Invoke with train.py flags. Adapters train only in Stage 3. Output is execution
evidence, never a full-population benchmark. No model downloads are mocked here.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from torch.utils.data import DataLoader, Subset
    import data.c1_protocol as protocol
    import model.gs_fuse as module
    import train

    original_builder = protocol.build_c1_dataloaders
    original_train = module.train
    evidence = dict(kind="16/8/8 three-stage smoke", lora_checked=False)

    def small_loaders(*args, **kwargs):
        data = original_builder(*args, **kwargs)
        batch_size = kwargs.get("batch_size", 2)
        for name, limit in (("train_loader", 16), ("vali_loader", 8), ("test_loader", 8),
                            ("sw_train_loader", 16), ("sw_vali_loader", 8)):
            dataset = getattr(data, name).dataset
            setattr(data, name, DataLoader(Subset(dataset, range(min(limit, len(dataset)))),
                                           batch_size=batch_size, shuffle=False))
        data.audit["smoke_only"] = True
        data.audit["smoke_loader_counts"] = {name: len(getattr(data, name).dataset)
                                              for name in ("train_loader", "vali_loader", "test_loader")}
        return data

    def checked_stage3(model, *args, **kwargs):
        encoder = model.text_encoder
        adapted = getattr(encoder, "use_lora", False)
        before = {n: p.detach().cpu().clone() for n, p in encoder.named_parameters()
                  if n.endswith(("lora_A", "lora_B"))} if adapted else {}
        base = next((p for n, p in encoder.named_parameters()
                     if n.endswith("base_layer.weight")), None) if adapted else None
        base_before = base.detach().cpu().clone() if base is not None else None
        result = original_train(model, *args, **kwargs)
        if adapted:
            changed = [n for n, p in encoder.named_parameters() if n in before
                       and not torch.equal(before[n], p.detach().cpu())]
            if not changed:
                raise RuntimeError("LoRA smoke failed: no Stage-3 adapter weight changed")
            if base is None or not torch.equal(base_before, base.detach().cpu()):
                raise RuntimeError("LoRA smoke failed: sampled frozen base changed or is missing")
            evidence.update(lora_checked=True, changed_adapter_tensors=changed,
                            sampled_base_unchanged=True, cache_allowed=encoder.cache_allowed)
            if encoder.cache_allowed:
                raise RuntimeError("LoRA smoke failed: static text cache enabled")
        return result

    protocol.build_c1_dataloaders = small_loaders
    module.train = checked_stage3
    argv = sys.argv[1:]
    # Last occurrence wins in argparse, independently of full-training budget.
    sys.argv = ["train.py", *argv, "--stage1-epochs", "1", "--stage2-epochs", "1", "--num-epochs", "1"]
    train.main()
    output = Path(argv[argv.index("--output-path") + 1])
    # Use the exact saved task/provenance; the normal evaluator checks the reload.
    config = json.loads((output / "resolved_config.json").read_text())
    from types import SimpleNamespace
    from data.forecast_task import ForecastTaskConfig
    from test_pretrained_model import load_pretrained_model
    config["model_path"] = str(output / "best_model.pth")
    config["forecast_task_config"] = ForecastTaskConfig(**config["forecast_task_config"])
    audit = json.loads((output / "c1_data_audit.json").read_text())
    config["run_provenance"] = dict(manifest_sha256=audit["manifest_sha256"],
        price_sha256=audit["price_sha256"], scale_population_sha256=audit["return_scale_population_sha256"],
        training_seed=config["seed"], source_commit=config["source_commit"])
    if config.get("snapshot_content_sha256"):
        config["run_provenance"]["snapshot_content_sha256"] = config["snapshot_content_sha256"]
    reloaded = load_pretrained_model(SimpleNamespace(**config), d=4)
    if config.get("use_lora") and reloaded.text_encoder.cache_allowed:
        raise RuntimeError("Reload incorrectly enabled adapted-text cache")
    evidence["checkpoint_reload"] = "passed"
    (output / "smoke-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
