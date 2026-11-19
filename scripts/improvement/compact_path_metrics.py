"""Unscaled full-path and close-endpoint scoring for canonical C1."""
import numpy as np

FORECAST_BARS, CHANNELS = 140, 4
HORIZONS = (35, 70, 140)


def metrics(truth, pred):
    if truth.shape != pred.shape or truth.ndim != 3 or truth.shape[1:] != (FORECAST_BARS, CHANNELS):
        raise ValueError("Expected matching arrays [events, 140 bars, 4 OHLC channels]")
    error = np.abs(truth.astype(np.float64) - pred.astype(np.float64))
    endpoints = [h - 1 for h in HORIZONS]
    target_close = truth[:, endpoints, 3]
    pred_close = pred[:, endpoints, 3]
    directions = []
    for i in range(len(endpoints)):
        eligible = target_close[:, i] != 0
        directions.append(float((np.sign(target_close[eligible, i]) == np.sign(pred_close[eligible, i])).mean())
                          if eligible.any() else None)
    return dict(path_mae_bp=float(error.mean()), close_endpoint_mae_bp=error[:, endpoints, 3].mean(0).tolist(),
                close_endpoint_direction_hit_rate=directions)
