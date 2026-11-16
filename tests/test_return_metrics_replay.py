import json

import pytest

from scripts.run_return_metrics_replay import load_run_contract


def test_load_run_contract_requires_return_smoke(tmp_path):
    resolved = {
        "forecast_task": "return_bp",
        "forecast_task_config": {
            "task": "return_bp", "anchor": "last_observed_close",
            "units": "basis_point_log_return",
            "scale_bp": 10.0, "huber_delta_bp": 10.0,
            "direction_tau_bp": 10.0, "direction_deadband_bp": 0.0,
            "direction_weight": 1.0, "direction_horizons": [35, 70, 140],
            "close_channel": 3,
        },
        "run_kind": "smoke_subset", "source_commit": "a" * 40,
        "seed": 0, "n_event_eval": 8, "seq_len": 35, "pred_len": 140,
        "batch_size": 2, "sw_stride": 1, "window": 500, "stride": 400,
        "decoder_layers": 3, "decoder_heads": 8, "use_ts_memory": False,
        "max_token_num": 1024, "fusion_type": "crossattn", "gate_type": "softmax",
        "num_experts": 4, "moe_top_k": 2, "num_queries": 16,
        "qformer_layers": 2, "use_raw_ts_residual": True,
        "use_learnable_last_value": True,
        "c1_timing_policy": "category4_1400_strict_before",
        "c1_skip_missing": False, "no_4bit": True, "tf32": True,
    }
    (tmp_path / "resolved_config.json").write_text(json.dumps(resolved))
    (tmp_path / "c1_data_audit.json").write_text(json.dumps({"manifest_sha256": "abc"}))

    loaded, audit, task, *_ = load_run_contract(tmp_path)
    assert loaded["n_event_eval"] == 8
    assert audit["manifest_sha256"] == "abc"
    assert task.task == "return_bp"

    resolved["run_kind"] = "full"
    (tmp_path / "resolved_config.json").write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="smoke-subset"):
        load_run_contract(tmp_path)
