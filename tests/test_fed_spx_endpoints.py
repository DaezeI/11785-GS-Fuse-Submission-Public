"""Guard the shared forecast anchor, parallel endpoints and paired scoring."""

import numpy as np
import pytest

from data.fed_spx_endpoints import endpoint_returns
from scripts.run_fed_spx_endpoints_price import bootstrap_gain, endpoint_metrics


def test_returns_use_origin_and_fourth_and_sixth_future_closes():
    closes = [101, 990, 2, 800, 102, 700, 99]
    expected = 10_000 * np.log(np.array([102, 99]) / 101)
    np.testing.assert_allclose(endpoint_returns(closes), expected)
    with pytest.raises(ValueError, match="positive"):
        endpoint_returns([0, *closes[1:]])


def test_weighted_endpoint_metrics_exclude_only_realized_flat_directions():
    truth = np.array([[4.0, -4.0], [4.0, -4.0], [-8.0, 0.0]])
    prediction = np.array([[2.0, -3.0], [2.0, 1.0], [-6.0, 2.0]])
    scores = endpoint_metrics(truth, prediction, np.array([0.5, 0.5, 1]))
    assert scores["20"]["mae_bp"] == 2.0
    assert scores["30"]["mae_bp"] == 2.5
    assert scores["30"]["direction_accuracy"] == 0.5
    assert scores["30"]["direction_eligible_rows"] == 2


def test_paired_intervals_report_absolute_bp_in_endpoint_order():
    days = np.arange(45)
    delta = np.tile([2.0, -1.5], (len(days), 1))
    intervals = bootstrap_gain(delta, days, np.ones(len(days)),
                               {"bootstrap_replicates": 100, "bootstrap_seed": 23})
    assert intervals["10"] == [[2.0, 2.0], [-1.5, -1.5]]
