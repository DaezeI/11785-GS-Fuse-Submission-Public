import math

import numpy as np
import pytest
import torch

from data.forecast_task import (
    ForecastTaskConfig,
    RETURN_TASK,
    gate_supervision_target,
    pre_event_close_volatility_bp,
    prices_to_log_returns_bp,
    reconstruct_prices,
    return_metrics,
    return_task_loss,
)


def return_config(**kwargs):
    values = dict(task=RETURN_TASK, units="basis_point_log_return", scale_bp=10.0,
                  huber_delta_bp=10.0, direction_tau_bp=10.0,
                  direction_horizons=(1, 2, 3))
    values.update(kwargs)
    return ForecastTaskConfig(**values)


def test_return_transform_uses_one_close_anchor_and_round_trips():
    prices = np.array([[99.0, 102.0, 98.0, 100.0],
                       [101.0, 103.0, 100.0, 102.0]], dtype=np.float32)
    transformed = prices_to_log_returns_bp(prices, 100.0)
    assert transformed[0, 3] == pytest.approx(0.0)
    assert transformed[1, 0] == pytest.approx(10_000 * math.log(1.01), rel=1e-6)
    assert np.allclose(reconstruct_prices(transformed, 100.0), prices, rtol=1e-6)


def test_return_task_loss_handles_positive_negative_and_flat_endpoints():
    target = torch.zeros(1, 4, 3)
    target[0, 3] = torch.tensor([5.0, -5.0, 0.0])
    good = target.clone().requires_grad_()
    bad = (-target).clone().requires_grad_()
    good_loss, good_parts = return_task_loss(good, target, return_config(),
                                              return_components=True)
    bad_loss = return_task_loss(bad, target, return_config())
    assert good_parts["eligible_direction_count"].item() == 2
    assert good_loss < bad_loss
    good_loss.backward()
    assert torch.isfinite(good.grad).all()


def test_all_flat_direction_has_zero_penalty_and_finite_gradient():
    prediction = torch.zeros(2, 4, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss, parts = return_task_loss(prediction, target, return_config(),
                                   return_components=True)
    assert parts["direction"].tolist() == [0.0, 0.0]
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_gate_target_prefers_full_branch_and_is_detached():
    strict_ts = torch.tensor([2.0, 1.0], requires_grad=True)
    full = torch.tensor([1.0, 2.0], requires_grad=True)
    target, difference = gate_supervision_target(strict_ts, full)
    assert target[0] > 0.5
    assert target[1] < 0.5
    assert not target.requires_grad
    assert not difference.requires_grad


def test_pre_event_volatility_and_metrics_are_reporting_helpers():
    past = np.ones((3, 4), dtype=np.float32) * 100
    assert pre_event_close_volatility_bp(past) == pytest.approx(0.0)
    target = torch.zeros(2, 4, 3)
    target[:, 3] = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])
    metrics = return_metrics(target.clone(), target, horizons=(1, 2, 3))
    assert metrics["return_mse"] == 0.0
    assert metrics["return_mae"] == 0.0
    assert metrics["close_dhr_h1"] == 1.0
    assert metrics["close_dhr_h1_n"] == 2
