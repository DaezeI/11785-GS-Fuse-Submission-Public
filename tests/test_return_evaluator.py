import pytest
import torch

from data.forecast_task import ForecastTaskConfig, return_task_loss
from model.gs_fuse import evaluate_return_loader


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
    assert result["events"] == 3
    assert result["task_loss"] == pytest.approx(float(expected))
    assert result["close_dhr"]["1"] == {"rate": pytest.approx(2 / 3),
                                         "eligible_events": 3}
    assert result["close_dhr"]["2"] == {"rate": None, "eligible_events": 0}
