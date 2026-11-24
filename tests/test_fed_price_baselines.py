"""Check the baseline's concurrent-event weights and forecast metrics."""

import numpy as np
import pandas as pd
import pytest

from scripts.run_fed_price_baselines import fit_ridge, instant_weights, predict_ridge, score_paths


def test_duplicate_announcement_instant_leaves_ridge_predictions_unchanged():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 35, 4))
    x[:, -1, 3] = 0  # The anchor close is constant after conversion to returns.
    y = rng.normal(size=(3, 140, 4))
    rows = pd.DataFrame({"event_id": ["a", "b", "c", "d"], "available_et": [1, 2, 3, 3]})
    duplicate_rows = [0, 1, 2, 2]
    weights = instant_weights(rows)
    np.testing.assert_array_equal(weights, [1, 1, 0.5, 0.5])
    baseline = fit_ridge(x, y, np.ones(3))
    duplicate_fit = fit_ridge(x[duplicate_rows], y[duplicate_rows], weights)
    np.testing.assert_allclose(predict_ridge(x, baseline), predict_ridge(x, duplicate_fit), atol=1e-12)


def test_path_prefix_and_endpoint_metrics_respect_row_weights_and_flat_predictions():
    target = np.zeros((3, 140, 4))
    target[0, :35] = 2.0
    target[1:, :35] = -4.0
    target[:, 69, 3] = 5.0
    metrics = score_paths(np.zeros_like(target), target, np.array([1, 0.5, 0.5]))
    assert metrics["35"]["path_mae_bp"] == pytest.approx(3.0)
    assert metrics["35"]["path_rmse_bp"] == pytest.approx(np.sqrt(10))
    assert metrics["35"]["direction_accuracy_nonflat"] == 0.0
    assert metrics["70"]["endpoint_close_mae_bp"] == 5.0
    assert metrics["70"]["predicted_flat_fraction"] == 1.0
    assert metrics["140"]["direction_eligible_rows"] == 0
    assert metrics["140"]["direction_accuracy_nonflat"] is None
