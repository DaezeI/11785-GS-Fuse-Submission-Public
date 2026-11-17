import pytest
import torch

from data.forecast_task import ForecastTaskConfig, return_task_loss
from model.gs_fuse import gate_supervision_loss_per_event, gate_task_loss_per_event


def test_gate_target_switch_changes_only_supervision_loss():
    config = ForecastTaskConfig(task="return_bp", units="basis_point_log_return",
                                scale_bp=10, huber_delta_bp=10, direction_tau_bp=10,
                                direction_horizons=(1, 2, 3))
    target = torch.zeros(2, 4, 3)
    target[:, 3, :] = 5
    prediction = torch.zeros_like(target)
    prediction[:, 3, :] = -5

    class Model:
        forecast_task_config = config
        gate_target_objective = "magnitude"

    model = Model()
    full, components = return_task_loss(prediction, target, config, reduction="none",
                                        return_components=True)
    assert torch.allclose(gate_supervision_loss_per_event(model, prediction, target),
                          components["magnitude"])
    assert torch.allclose(gate_task_loss_per_event(model, prediction, target), full)
    model.gate_target_objective = "task"
    assert torch.allclose(gate_supervision_loss_per_event(model, prediction, target), full)
    assert torch.all(full > components["magnitude"])
