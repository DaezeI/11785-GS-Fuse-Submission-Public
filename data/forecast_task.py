"""Pure transformations, losses, and metrics for GS-FUSE forecast tasks."""

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


BP_PER_LOG_UNIT = 10_000.0
RETURN_TASK = "return_bp"
PRICE_TASK = "price"


@dataclass(frozen=True)
class ForecastTaskConfig:
    """Resolved forecast objective stored with every return-task run."""

    task: str = PRICE_TASK
    anchor: str = "last_observed_close"
    units: str = "standardized_price"
    scale_bp: float = 1.0
    huber_delta_bp: float = 1.0
    direction_tau_bp: float = 1.0
    direction_deadband_bp: float = 0.0
    direction_weight: float = 1.0
    direction_horizons: tuple = (35, 70, 140)
    close_channel: int = 3

    def __post_init__(self):
        if self.task not in (PRICE_TASK, RETURN_TASK):
            raise ValueError(f"Unknown forecast task: {self.task}")
        if self.task == RETURN_TASK and (self.anchor != "last_observed_close" or
                                         self.units != "basis_point_log_return"):
            raise ValueError("Return task requires one last-observed-close anchor and basis-point log returns")
        if self.scale_bp <= 0 or self.huber_delta_bp <= 0 or self.direction_tau_bp <= 0:
            raise ValueError("Return loss scales must be positive")
        if self.direction_deadband_bp < 0 or self.direction_weight < 0:
            raise ValueError("Direction deadband and weight must be non-negative")
        if not self.direction_horizons or any(int(h) <= 0 for h in self.direction_horizons):
            raise ValueError("Direction horizons must be positive one-based endpoints")

    def to_dict(self):
        result = asdict(self)
        result["direction_horizons"] = list(self.direction_horizons)
        return result


def _check_prices(prices):
    if torch.is_tensor(prices):
        if not torch.isfinite(prices).all() or torch.any(prices <= 0):
            raise ValueError("Prices must be finite and strictly positive")
    else:
        values = np.asarray(prices)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Prices must be finite and strictly positive")


def prices_to_log_returns_bp(prices, anchor_close):
    """Convert any OHLC window to basis-point log returns around one close anchor."""
    _check_prices(prices)
    _check_prices(anchor_close)
    if torch.is_tensor(prices):
        anchor = torch.as_tensor(anchor_close, dtype=prices.dtype, device=prices.device)
        return BP_PER_LOG_UNIT * torch.log(prices / anchor)
    values = np.asarray(prices)
    anchor = np.asarray(anchor_close, dtype=values.dtype)
    return (BP_PER_LOG_UNIT * np.log(values / anchor)).astype(values.dtype, copy=False)


def reconstruct_prices(log_returns_bp, anchor_close):
    """Invert :func:`prices_to_log_returns_bp`."""
    if torch.is_tensor(log_returns_bp):
        anchor = torch.as_tensor(anchor_close, dtype=log_returns_bp.dtype,
                                 device=log_returns_bp.device)
        return anchor * torch.exp(log_returns_bp / BP_PER_LOG_UNIT)
    values = np.asarray(log_returns_bp)
    anchor = np.asarray(anchor_close, dtype=values.dtype)
    return (anchor * np.exp(values / BP_PER_LOG_UNIT)).astype(values.dtype, copy=False)


def pre_event_close_volatility_bp(past_ohlc, close_channel=3):
    """Population std. dev. of consecutive pre-event close log changes, in bp."""
    _check_prices(past_ohlc)
    if torch.is_tensor(past_ohlc):
        close = past_ohlc[..., :, close_channel]
        changes = BP_PER_LOG_UNIT * torch.diff(torch.log(close), dim=-1)
        return changes.std(dim=-1, correction=0)
    values = np.asarray(past_ohlc)
    close = values[..., :, close_channel]
    changes = BP_PER_LOG_UNIT * np.diff(np.log(close), axis=-1)
    return np.std(changes, axis=-1, ddof=0)


def resolve_scale_bp(targets, horizons=(35, 70, 140), close_channel=3):
    """Training-only median absolute close endpoint return, floored at one bp."""
    values = np.asarray(targets)
    if values.ndim != 3:
        raise ValueError("targets must have shape [events, horizon, channels]")
    indices = [int(h) - 1 for h in horizons]
    if min(indices) < 0 or max(indices) >= values.shape[1]:
        raise ValueError("Direction horizon is outside the target window")
    scale = float(np.median(np.abs(values[:, indices, close_channel])))
    return max(scale, 1.0)


def _endpoint_indices(config: ForecastTaskConfig, pred_len: int, device):
    indices = [int(h) - 1 for h in config.direction_horizons]
    if min(indices) < 0 or max(indices) >= pred_len:
        raise ValueError("Direction horizon is outside the prediction window")
    return torch.tensor(indices, dtype=torch.long, device=device)


def return_task_loss(prediction, target, config: ForecastTaskConfig, reduction="mean",
                     return_components=False):
    """Huber magnitude plus smooth close-direction loss.

    Inputs use the model convention ``[batch, channels, horizon]``. With
    ``reduction='none'``, one complete task loss is returned per event; this is
    the quantity used to supervise the gate.
    """
    if config.task != RETURN_TASK:
        raise ValueError("return_task_loss requires task='return_bp'")
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must match [batch, channels, horizon]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")

    element_huber = F.huber_loss(
        prediction, target, reduction="none", delta=float(config.huber_delta_bp)
    )
    magnitude = element_huber.mean(dim=(1, 2)) / float(config.scale_bp) ** 2

    indices = _endpoint_indices(config, prediction.shape[-1], prediction.device)
    actual = target[:, config.close_channel, :].index_select(-1, indices)
    predicted = prediction[:, config.close_channel, :].index_select(-1, indices)
    eligible = (actual != 0) & (actual.abs() > float(config.direction_deadband_bp))
    penalties = F.softplus(-torch.sign(actual) * predicted / float(config.direction_tau_bp))
    eligible_count = eligible.sum(dim=-1)
    direction = (penalties * eligible).sum(dim=-1) / eligible_count.clamp_min(1)
    direction = torch.where(eligible_count > 0, direction, torch.zeros_like(direction))
    total = magnitude + float(config.direction_weight) * direction

    if reduction == "mean":
        reduced = total.mean()
    elif reduction == "sum":
        reduced = total.sum()
    elif reduction == "none":
        reduced = total
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    if return_components:
        return reduced, {"magnitude": magnitude, "direction": direction,
                         "eligible_direction_count": eligible_count}
    return reduced


def forecast_training_loss(prediction, target, config: ForecastTaskConfig, reduction="mean"):
    """Dispatch the training objective without changing price-task behavior."""
    if config.task == RETURN_TASK:
        return return_task_loss(prediction, target, config, reduction=reduction)
    per_event = (0.7 * (prediction - target).square().mean(dim=(1, 2)) +
                 0.3 * (prediction - target).abs().mean(dim=(1, 2)))
    if reduction == "none":
        return per_event
    if reduction == "sum":
        return per_event.sum()
    if reduction == "mean":
        return per_event.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def gate_supervision_target(ts_task_loss, full_task_loss, adaptive=True,
                            fixed_scale=4e-5, min_scale=1e-6,
                            gain=1.0, logit_clamp=6.0):
    """Detached text-gate target from strict-TS minus full forecasting utility."""
    if ts_task_loss.shape != full_task_loss.shape:
        raise ValueError("Branch task losses must have matching per-event shapes")
    difference = ts_task_loss - full_task_loss
    if adaptive:
        scale = difference.detach().abs().mean().clamp_min(float(min_scale))
    else:
        scale = torch.as_tensor(fixed_scale, dtype=difference.dtype, device=difference.device)
    logits = (float(gain) * difference / scale).clamp(-float(logit_clamp), float(logit_clamp))
    return torch.sigmoid(logits).detach(), difference.detach()


def return_metrics(prediction, target, horizons: Sequence[int] = (35, 70, 140),
                   close_channel=3, deadband_bp=0.0):
    """Evaluation-only return MSE/MAE and close endpoint directional hit rates."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must match [batch, channels, horizon]")
    result = {
        "return_mse": float((prediction - target).square().mean().detach().cpu()),
        "return_mae": float((prediction - target).abs().mean().detach().cpu()),
    }
    for horizon in horizons:
        index = int(horizon) - 1
        actual = target[:, close_channel, index]
        predicted = prediction[:, close_channel, index]
        eligible = (actual != 0) & (actual.abs() > float(deadband_bp))
        hits = (torch.sign(actual) == torch.sign(predicted)) & eligible
        count = int(eligible.sum().detach().cpu())
        result[f"close_dhr_h{horizon}"] = (
            float(hits.sum().detach().cpu()) / count if count else float("nan")
        )
        result[f"close_dhr_h{horizon}_n"] = count
    return result
