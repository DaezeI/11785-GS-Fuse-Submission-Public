import json

import pytest
import torch

from data.forecast_task import ForecastTaskConfig, return_task_loss
from model.gs_fuse import (
    evaluate_return_loader,
    return_metrics_for_tracking,
    write_return_metrics,
)


def test_return_validation_weights_events_and_excludes_flat_close():
    config = ForecastTaskConfig(task="return_bp", units="basis_point_log_return",
                                scale_bp=10, huber_delta_bp=10, direction_tau_bp=10,
                                direction_horizons=(1, 2, 3))
    y1 = torch.zeros(2, 4, 3)
    y1[:, 3, 0] = torch.tensor([5., -5.])
    y2 = torch.zeros(1, 4, 3)
    y2[:, 3, 0] = 5.
    p1 = y1.clone()
    p2 = -y2

    class Stub:
        forecast_task_config = config
        run_provenance = {"source_commit": "test-commit"}
        def __init__(self): self.predictions = [p1, p2]
        def eval(self): return self
        def predict_single_case(self, texts, seq):
            return (self.predictions.pop(0),)

    def batch(target):
        return (["article"] * len(target), None, None, None, None,
                torch.zeros(len(target), 4, 35), target)

    result = evaluate_return_loader(Stub(), [batch(y1), batch(y2)])
    expected = (return_task_loss(p1, y1, config, reduction="sum") +
                return_task_loss(p2, y2, config, reduction="sum")) / 3
    _, c1 = return_task_loss(p1, y1, config, reduction="none", return_components=True)
    _, c2 = return_task_loss(p2, y2, config, reduction="none", return_components=True)
    assert result["events"] == 3
    assert result["task_loss"] == pytest.approx(float(expected))
    assert result["magnitude_loss"] == pytest.approx(float(
        (c1["magnitude"].sum() + c2["magnitude"].sum()) / 3
    ))
    assert result["direction_loss"] == pytest.approx(float(
        (c1["direction"].sum() + c2["direction"].sum()) / 3
    ))
    assert result["weighted_direction_loss"] == pytest.approx(result["direction_loss"])
    assert result["task_loss"] == pytest.approx(
        result["magnitude_loss"] + result["direction_loss"]
    )
    assert result["close_dhr"]["1"] == {"rate": pytest.approx(2 / 3),
                                         "eligible_events": 3}
    assert result["close_dhr"]["2"] == {"rate": None, "eligible_events": 0}

    tracked = return_metrics_for_tracking("vali", result)
    assert tracked["vali/close_dhr_h1"] == pytest.approx(2 / 3)
    assert tracked["vali/close_dhr_h1_eligible"] == 3
    assert "vali/close_dhr_h2" not in tracked
    assert tracked["vali/close_dhr_h2_eligible"] == 0


def test_structured_return_metrics_are_atomic_and_provenanced(tmp_path):
    config = ForecastTaskConfig(task="return_bp", units="basis_point_log_return",
                                scale_bp=10, huber_delta_bp=10, direction_tau_bp=10,
                                direction_horizons=(1,))

    class Stub:
        forecast_task_config = config
        run_provenance = {"source_commit": "abc123", "training_seed": 0}

    result = {
        "events": 2, "task_loss": 1.0, "magnitude_loss": 0.25,
        "direction_loss": 0.75, "weighted_direction_loss": 0.75,
        "return_mae_bp": 2.0,
        "return_mse_bp2": 5.0,
        "close_dhr": {"1": {"rate": 0.5, "eligible_events": 2}},
    }
    path = tmp_path / "return_metrics.json"
    write_return_metrics(path, "validation", result, Stub(), "best_model.pth",
                         epoch=3, global_step=12)

    payload = json.loads(path.read_text())
    assert payload["split"] == "validation"
    assert payload["checkpoint"] == "best_model.pth"
    assert payload["run_provenance"]["source_commit"] == "abc123"
    assert payload["metrics"] == result
    assert not (tmp_path / "return_metrics.json.tmp").exists()
