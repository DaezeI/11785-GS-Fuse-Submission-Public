"""Clock, purging and stride checks independent of private price data."""
import numpy as np
import pandas as pd

from data.dense_price_endpoints import anchor_indices, make_cohort


def session():
    t = pd.date_range("2020-01-02 09:30", "2020-01-02 16:00", freq="5min").to_numpy()
    op = np.full(len(t), np.datetime64("2020-01-02T09:30"), dtype="datetime64[ns]")
    cl = np.full(len(t), np.datetime64("2020-01-02T16:00"), dtype="datetime64[ns]")
    return t, op, cl


def test_no_anchor_or_future_price_enters_history():
    t, op, cl = session()
    j = anchor_indices(t, op, cl, t[0], np.datetime64("2020-01-02T16:00"), 5)
    prices = np.exp(np.arange(len(t))[:, None] * .001) * np.ones((len(t), 4))
    c = make_cohort(prices, t, j)
    assert c.history.shape == (len(j), 4, 35)
    np.testing.assert_allclose(c.history[:, :, -1], 0, atol=1e-6)
    np.testing.assert_allclose(c.returns, np.tile([40, 60], (len(j), 1)), atol=1e-8)
    assert t[j[-1]] == np.datetime64("2020-01-02T15:25")


def test_split_cut_purges_entire_history_and_target():
    t, op, cl = session()
    t = np.concatenate([t, t + np.timedelta64(1, "D"), t + np.timedelta64(4, "D")])
    day = t.astype("datetime64[D]")
    op, cl = day + np.timedelta64(570, "m"), day + np.timedelta64(960, "m")
    cut = np.datetime64("2020-01-03T13:00")
    a = anchor_indices(t, op, cl, t[0], cut, 5)
    b = anchor_indices(t, op, cl, cut, np.datetime64("2020-01-06T16:00"), 5)
    assert len(a) and len(b)
    used_a = {int(i + d) for i in a for d in range(-35, 7)}
    used_b = {int(i + d) for i in b for d in range(-35, 7)}
    assert not used_a & used_b
    assert np.all(t[a + 6] + np.timedelta64(5, "m") <= cut)


def test_fifteen_minute_grid_is_subset_and_early_close_applies():
    t, op, cl = session()
    cl[:] = np.datetime64("2020-01-02T13:00")
    a = anchor_indices(t, op, cl, t[0], np.datetime64("2020-01-02T16:00"), 5)
    b = anchor_indices(t, op, cl, t[0], np.datetime64("2020-01-02T16:00"), 15)
    assert set(b) <= set(a)
    assert np.all(t[a] + np.timedelta64(35, "m") <= cl[a])
    assert np.all(pd.DatetimeIndex(t[b]).minute % 15 == 0)


def test_missing_intraday_bar_excludes_affected_windows():
    t, op, cl = session()
    t, op, cl = (np.delete(a, 40) for a in (t, op, cl))
    j = anchor_indices(t, op, cl, t[0], np.datetime64("2020-01-02T16:00"), 5)
    for i in j:
        np.testing.assert_array_equal(np.diff(t[i - 35:i + 7]), np.full(41, np.timedelta64(5, "m")))


def test_six_supervised_horizons_preserve_existing_endpoints():
    t, op, cl = session()
    j = anchor_indices(t, op, cl, t[0], np.datetime64("2020-01-02T16:00"), 5)
    prices = np.exp(np.arange(len(t))[:, None] * .001) * np.ones((len(t), 4))
    old = make_cohort(prices, t, j)
    new = make_cohort(prices, t, j, target_steps=range(1, 7), include_time_offsets=True)
    np.testing.assert_allclose(new.returns, np.tile(np.arange(1, 7) * 10, (len(j), 1)), atol=1e-8)
    np.testing.assert_array_equal(old.returns, new.returns[:, [3, 5]])
    np.testing.assert_array_equal(new.past_minutes[0], np.arange(-175, 0, 5))


def test_overnight_time_offsets_account_for_dst():
    friday = pd.date_range("2020-03-06 09:30", "2020-03-06 16:00", freq="5min").to_numpy()
    monday = pd.date_range("2020-03-09 09:30", "2020-03-09 16:00", freq="5min").to_numpy()
    t = np.concatenate((friday, monday))
    c = make_cohort(np.ones((len(t), 4)), t, np.array([len(friday)]),
                    target_steps=range(1, 7), include_time_offsets=True)
    assert c.past_minutes[0, -1] == -3870
