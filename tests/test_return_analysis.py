import json

import numpy as np
import pytest

from scripts.analyze_return_predictions import analyze, compare_runs
from scripts.run_text_controls import resolved_return_run


def test_resolved_run_rejects_price_config(tmp_path):
    path = tmp_path / "resolved_config.json"
    path.write_text(json.dumps({"forecast_task": "price"}))
    with pytest.raises(ValueError, match="not a return run"):
        resolved_return_run(path)


def test_aligned_return_report_uses_close_endpoints_and_eligible_counts():
    n = 24
    target = np.zeros((n, 140, 4), dtype=np.float32)
    target[:, 34, 3] = np.tile([10., -10., 0.], 8)
    target[:, 69, 3] = 5.
    target[:, 139, 3] = -5.
    real = target.copy()
    wrong = -target
    config = {"task": "return_bp", "close_channel": 3,
              "direction_horizons": [35, 70, 140], "direction_deadband_bp": 0.0}
    blob = {"metadata_json": np.array(json.dumps({
                "forecast_task_config": config,
                "training_majority_direction": {"35": 1, "70": 1, "140": -1}})),
            "targets": target, "real_text": real, "zero_text": wrong,
            "fixed_gate": wrong, "strict_ts": wrong,
            "shuffled_text": np.stack([wrong, wrong]),
            "seqs": np.zeros((n, 35, 4)), "anchor_close": np.full(n, 100.),
            "event_ids": np.array([f"1/{i}" for i in range(n)]),
            "event_dates": np.array([f"202001{i + 1:02}" for i in range(n)]),
            "event_types": np.full(n, "1"),
            "gate_text_weight": np.linspace(.2, .8, n),
            "full_task_loss": np.zeros(n), "strict_ts_task_loss": np.ones(n),
            "gate_target": np.full(n, .7)}
    result = analyze(blob, reps=20, block=3)
    assert result["close_endpoint_error"]["35"]["mae_bp"]["real_text"] == 0
    blob["seqs"] = np.zeros((n, 140, 4))
    assert analyze(blob, reps=5, block=3)["events"] == n
    assert result["all_channel"]["all"]["conditions"]["real_text"]["mae_bp"] == 0
    assert result["close_endpoint_dhr"]["35"]["all"]["eligible_events"] == 16
    assert result["close_endpoint_dhr"]["35"]["all"]["hit_rate"]["real_text"] == 1
    assert result["close_endpoint_dhr"]["35"]["all"]["hit_rate"]["zero_text"] == 0
    assert result["close_endpoint_dhr"]["140"]["all"]["eligible_events"] == 24
    assert result["comparisons"]["strict_ts"]["mae_gain_pct"] == 100
    blob["anchor_close"][0] = 0
    with pytest.raises(ValueError, match="anchors"):
        analyze(blob, reps=20, block=3)


def test_compare_runs_rejects_changed_population_and_snapshot():
    truth = np.ones((2, 1, 4))
    metadata = dict(evaluation_split="c1_test", no_text_definition="zeroed_article_tokens",
                    reference_source_commit="commit", seed=23, batch_size=8,
                    tf32=True, four_bit=False, article_content_sha256=["one", "two"],
                    checkpoint_id="reference",
                    forecast_task_config=dict(direction_horizons=[1],
                                              direction_deadband_bp=0, close_channel=3),
                    c1_data_audit=dict(snapshot_content_sha256="frozen"))
    reference = dict(targets=truth, real_text=-truth, event_ids=np.array(["1", "2"]),
                     event_dates=np.array(["20200101", "20200102"]),
                     anchor_close=np.ones(2), shuffle_donor_ids=np.array([["2", "1"]]),
                     metadata_json=np.array(json.dumps(metadata)))
    candidate = dict(reference, real_text=truth)
    result = compare_runs(candidate, reference)
    assert result["errors"]["mae_bp"]["candidate_minus_reference"] == -2
    assert result["close_dhr"]["1"]["candidate_minus_reference"] == 1
    with pytest.raises(ValueError, match="event_ids"):
        compare_runs(dict(candidate, event_ids=np.array(["2", "1"])), reference)
    metadata["c1_data_audit"]["snapshot_content_sha256"] = "different"
    with pytest.raises(ValueError, match="same frozen snapshot"):
        compare_runs(dict(candidate, metadata_json=np.array(json.dumps(metadata))), reference)
