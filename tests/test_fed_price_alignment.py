"""Offline tests for scripts/build_fed_price_alignment.py (synthetic prices, no real data)."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import build_fed_price_alignment as fpa

ET = "America/New_York"


def make_prices(start="2008-02-25", end="2008-04-15", drop_days=(), drop_bars=(), late_open_days=(),
                with_filler=True):
    """Weekday bars 09:30..16:00 (79 per day), distinct non-flat prices, plus flat filler after 16:00."""
    rows, k = [], 0
    for d in pd.bdate_range(start, end):
        if str(d.date()) in drop_days:
            continue
        for m in range(79):
            t = d + pd.Timedelta(hours=9, minutes=30 + 5 * m)
            k += 1
            if str(d.date()) in late_open_days and t < d + pd.Timedelta(hours=10):
                continue
            if str(t) in drop_bars:
                continue
            rows.append((t, 100 + k, 100.5 + k, 99.5 + k, 100.2 + k))
        if with_filler:
            for m in (5, 10):  # 16:05, 16:10
                rows.append((d + pd.Timedelta(hours=16, minutes=m), 500.0, 500.0, 500.0, 500.0))
    return pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close"])


def write_prices(tmp_path, df, name="SP500.csv"):
    p = tmp_path / name
    df.assign(date_int=0).to_csv(p, index=False)
    return p


def events_csv(tmp_path, items, name="events.csv", lag_min=5, market_text=None):
    """items: (event_id, MARKET time as ET string, usable[, text_chars]).
    available = market time + lag_min, as in Person 1's file."""
    rows = []
    for it in items:
        eid, et, usable = it[:3]
        chars = it[3] if len(it) > 3 else 1500
        mk = pd.Timestamp(et)
        utc = (mk + pd.Timedelta(minutes=lag_min)).tz_localize(ET).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        h12 = mk.hour % 12 or 12
        text = f"{mk:%Y-%m-%d} {h12}:{mk:%M} {'a.m.' if mk.hour < 12 else 'p.m.'}"
        rows.append({"event_id": eid, "event_type": "fed_speech", "available_utc": utc,
                     "usable_intraday": usable, "text_chars": chars, "processing_lag_min": lag_min,
                     "market_time_et": market_text or text})
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def run(tmp_path, items, prices=None, **kw):
    prices = prices if prices is not None else make_prices()
    bars, rep = fpa.load_prices(write_prices(tmp_path, prices))
    ev = fpa.load_events(events_csv(tmp_path, items))
    return fpa.align(ev, bars, **kw).set_index("event_id"), rep


# ---------------------------------------------------------------- price cleaning
def test_duplicates_removed_and_sorted(tmp_path):
    p = make_prices(end="2008-02-29")
    day = p[p.date.dt.date == pd.Timestamp("2008-02-27").date()]
    dirty = pd.concat([p, day]).sample(frac=1, random_state=0)  # duplicated day, shuffled
    bars, rep = fpa.load_prices(write_prices(tmp_path, dirty))
    assert rep["duplicate_rows_removed"] == len(day)
    assert rep["days_with_duplicates"] == ["2008-02-27"]
    assert bars.date.is_monotonic_increasing and not bars.date.duplicated().any()


def test_conflicting_duplicates_raise(tmp_path):
    p = make_prices(end="2008-02-27")
    bad = p.iloc[[0]].copy()
    bad["Close"] = bad["Close"] + 1
    with pytest.raises(ValueError, match="different prices"):
        fpa.load_prices(write_prices(tmp_path, pd.concat([p, bad])))


def test_post_close_filler_dropped(tmp_path):
    bars, rep = fpa.load_prices(write_prices(tmp_path, make_prices(end="2008-02-27")))
    assert bars.date.dt.strftime("%H:%M").max() == "16:00"
    assert rep["bars_after_session_dropped"] == 6  # 16:05 and 16:10 on three days
    assert rep["share_flat_in_dropped"] == 1.0 and rep["share_flat_kept"] == 0.0
    assert rep["full_days"] == rep["trading_days"] == 3


def test_missing_weekday_and_late_open_reported(tmp_path):
    p = make_prices(end="2008-03-07", drop_days=("2008-03-04",), late_open_days=("2008-03-05",))
    _, rep = fpa.load_prices(write_prices(tmp_path, p))
    assert rep["weekdays_without_bars"]["dates"] == ["2008-03-04"]
    assert rep["days_not_starting_0930"] == {"2008-03-05": "10:00"}
    assert "2008-03-05" in rep["short_or_incomplete_days"]


# ---------------------------------------------------------------- alignment and leakage
def test_off_grid_event_input_stops_before_the_announcement(tmp_path):
    out, _ = run(tmp_path, [("a", "2008-03-13 10:23:00", True)])      # public 10:23, available 10:28
    r = out.loc["a"]
    assert r.market_et == pd.Timestamp("2008-03-13 10:23")
    assert r.available_et == pd.Timestamp("2008-03-13 10:28")
    assert r.target_first_bar == pd.Timestamp("2008-03-13 10:30")      # first start >= available
    assert r.input_last_bar == pd.Timestamp("2008-03-13 10:15")        # ends 10:20 <= market 10:23
    assert r.input_last_bar_end == pd.Timestamp("2008-03-13 10:20")
    assert r.embargo_bars == 2                                          # 10:20 and 10:25 bars
    assert r.reaction_lag_min == 2 and r.input_staleness_min == 3
    assert r.session_at_market == "regular" and r.main_candidate


def test_grid_aligned_statement_keeps_the_reaction_bar_out_of_the_input(tmp_path):
    """2:00 p.m. statement: input ends 13:55-14:00, the 14:00-14:05 bar is in neither window."""
    out, _ = run(tmp_path, [("a", "2008-03-13 14:00:00", True)])
    r = out.loc["a"]
    assert r.input_last_bar == pd.Timestamp("2008-03-13 13:55")
    assert r.input_last_bar_end == pd.Timestamp("2008-03-13 14:00")
    assert r.target_first_bar == pd.Timestamp("2008-03-13 14:05")
    assert r.embargo_bars == 1 and r.reaction_lag_min == 0 and r.input_staleness_min == 0


def test_older_available_time_rule_is_still_selectable(tmp_path):
    out, _ = run(tmp_path, [("a", "2008-03-13 14:00:00", True)], input_cutoff="available")
    r = out.loc["a"]
    assert r.input_last_bar == pd.Timestamp("2008-03-13 14:00")        # the reaction bar is input here
    assert r.embargo_bars == 0
    with pytest.raises(ValueError):
        run(tmp_path, [("a", "2008-03-13 14:00:00", True)], input_cutoff="bogus")


def test_window_lengths(tmp_path):
    bars, _ = fpa.load_prices(write_prices(tmp_path, make_prices()))
    ev = fpa.load_events(events_csv(tmp_path, [("a", "2008-03-13 14:05:00", True)]))
    r = fpa.align(ev, bars, seq_len=35, pred_len=140).iloc[0]
    n_in = ((bars.date >= r.input_first_bar) & (bars.date <= r.input_last_bar)).sum()
    n_out = ((bars.date >= r.target_first_bar) & (bars.date <= r.target_last_bar)).sum()
    assert (n_in, n_out) == (35, 140)
    assert r.target_crosses_day  # 140 bars is more than one 79-bar session


def test_pre_market_event_waits_for_open_and_input_is_stale(tmp_path):
    out, _ = run(tmp_path, [("a", "2008-03-13 08:30:00", True)])
    r = out.loc["a"]
    assert r.session_at_market == "pre_market"
    assert r.target_first_bar == pd.Timestamp("2008-03-13 09:30")
    assert r.input_last_bar == pd.Timestamp("2008-03-12 16:00")
    assert r.reaction_lag_min == 55
    assert r.input_staleness_min == 16 * 60 + 25  # last input bar ends 16:05; announcement 08:30 next day
    assert not r.main_candidate


def test_after_hours_and_weekend_events(tmp_path):
    out, _ = run(tmp_path, [("ah", "2008-03-13 17:25:00", True), ("we", "2008-03-15 12:00:00", True)])
    assert out.loc["ah"].session_at_market == "after_hours"
    assert out.loc["ah"].target_first_bar == pd.Timestamp("2008-03-14 09:30")
    assert out.loc["we"].session_at_market == "weekend"
    assert out.loc["we"].target_first_bar == pd.Timestamp("2008-03-17 09:30")
    assert not out.loc["ah"].main_candidate and not out.loc["we"].main_candidate


def test_closed_weekday_is_flagged_without_calendar(tmp_path):
    out, _ = run(tmp_path, [("a", "2008-03-20 11:00:00", True)],
                 prices=make_prices(drop_days=("2008-03-20",)))
    assert out.loc["a"].session_at_market == "closed_day_or_data_gap"
    assert out.loc["a"].target_first_bar == pd.Timestamp("2008-03-21 09:30")


def test_dst_conversion_uses_new_york_wall_time(tmp_path):
    # 14:05 New York is 19:05Z in winter (EST) and 18:05Z after DST starts on 2008-03-09 (EDT).
    out, _ = run(tmp_path, [("w", "2008-02-28 14:00:00", True), ("s", "2008-04-01 14:00:00", True)])
    for eid, d in (("w", "2008-02-28"), ("s", "2008-04-01")):
        assert out.loc[eid].target_first_bar == pd.Timestamp(f"{d} 14:05")
    assert out.loc["w"].available_utc.startswith("2008-02-28T19:05")
    assert out.loc["s"].available_utc.startswith("2008-04-01T18:05")


def test_coverage_limits_give_explicit_reasons(tmp_path):
    items = [("early", "2008-02-25 09:40:00", True),      # nothing before it
             ("few", "2008-02-26 09:40:00", True),         # needs 35 input bars: only 24 + prior day exist? 79+2 exist
             ("late", "2008-04-14 15:00:00", True),        # fewer than 140 target bars left
             ("after", "2008-05-01 10:00:00", True),       # beyond the data
             ("nope", "2008-03-13 10:00:00", False)]
    out, _ = run(tmp_path, items)
    assert out.loc["early"].exclude_reason == "insufficient_input_bars"
    assert out.loc["few"].exclude_reason == ""
    assert out.loc["late"].exclude_reason == "insufficient_target_bars"
    assert out.loc["after"].exclude_reason == "after_price_coverage"
    assert out.loc["nope"].exclude_reason == "not_usable_intraday_in_events_file"


def test_missing_bar_inside_window_is_flagged(tmp_path):
    p = make_prices(drop_bars=("2008-03-13 11:00:00",))
    out, _ = run(tmp_path, [("gap", "2008-03-13 12:03:00", True), ("clean", "2008-03-20 12:03:00", True)], prices=p)
    assert out.loc["gap"].input_has_gap and not out.loc["gap"].main_candidate
    assert not out.loc["clean"].input_has_gap


def test_late_open_day_is_flagged(tmp_path):
    p = make_prices(late_open_days=("2008-03-12",))
    out, _ = run(tmp_path, [("a", "2008-03-12 14:00:00", True), ("b", "2008-03-26 14:00:00", True)], prices=p)
    assert out.loc["a"].input_has_gap and not out.loc["a"].main_candidate
    assert not out.loc["b"].input_has_gap


def test_concurrent_events_share_a_group(tmp_path):
    out, _ = run(tmp_path, [("a", "2008-03-13 14:00:00", True), ("b", "2008-03-13 14:00:00", True),
                            ("c", "2008-03-13 15:00:00", True, 50)])
    assert out.loc["a"].concurrent_group_id == out.loc["b"].concurrent_group_id
    assert out.loc["a"].n_concurrent_usable == 2 and out.loc["c"].n_concurrent_usable == 1
    assert out.loc["c"].short_text and not out.loc["a"].short_text


def test_leakage_invariant_holds_for_many_random_times(tmp_path):
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2008-03-03")
    items = []
    for i in range(300):
        t = base + pd.Timedelta(minutes=int(rng.integers(0, 60 * 24 * 21)))
        items.append((f"e{i}", str(t.floor("s")), True))
    for cutoff in ("market", "available"):
        out, _ = run(tmp_path, items, input_cutoff=cutoff)
        ok = out[out.exclude_reason == ""]
        assert len(ok) > 200
        assert (ok.input_last_bar_end <= ok.available_et).all()
        assert (ok.market_et <= ok.available_et).all()
        assert (ok.available_et <= ok.target_first_bar).all()
        assert (ok.target_first_bar - ok.available_et <= pd.Timedelta(days=4)).all()
        if cutoff == "market":
            assert (ok.input_last_bar_end <= ok.market_et).all()


# ---------------------------------------------------------------- CLI
def test_cli_writes_outputs_and_audit(tmp_path):
    prices = write_prices(tmp_path, make_prices(start="2008-02-25", end="2008-09-30"))
    days = pd.bdate_range("2008-03-10", "2008-08-01")[::3]
    items = [(f"e{i}", f"{d:%Y-%m-%d} 14:00:00", True) for i, d in enumerate(days)]
    items += [("weekend", "2008-03-15 12:00:00", True), ("x", "2008-03-13 10:00:00", False)]
    events = events_csv(tmp_path, items)
    out_dir = tmp_path / "out"
    args = ["--events", str(events), "--prices", str(prices), "--out-dir", str(out_dir),
            "--expect-price-sha256", ""]
    fpa.main(args)
    audit = json.loads((out_dir / "price_alignment_audit_v1.json").read_text())
    n_main = len(days)
    assert audit["events_total"] == n_main + 2 and audit["aligned"] == n_main + 1
    assert audit["main_candidates"] == n_main
    assert audit["leakage_check"]["violations"] == 0
    assert audit["rows_by_outcome"]["not_usable_intraday_in_events_file"] == 1
    assert audit["meta"]["prices_sha256"] == fpa.sha256_file(prices)
    assert audit["params"]["input_cutoff"] == "market" and audit["market_time_text_mismatches"] == 0
    assert audit["splits"]["windows_disjoint"] and sum(audit["splits"]["rows_main"].values()) == n_main
    assert (out_dir / "fed_price_alignment_v1.csv").exists()
    with pytest.raises(SystemExit, match="exist"):
        fpa.main(args)
    fpa.main(args + ["--force"])


def test_cli_rejects_a_different_price_file(tmp_path):
    prices = write_prices(tmp_path, make_prices(end="2008-02-29"))
    events = events_csv(tmp_path, [("a", "2008-02-28 14:00:00", True)])
    with pytest.raises(SystemExit, match="frozen"):
        fpa.main(["--events", str(events), "--prices", str(prices), "--out-dir", str(tmp_path / "o")])


def test_market_time_text_disagreeing_with_lag_is_counted(tmp_path):
    ev = fpa.load_events(events_csv(tmp_path, [("a", "2008-03-13 14:00:00", True)],
                                    market_text="2008-03-13 2:30 p.m."))
    assert int(ev.market_time_mismatch.sum()) == 1
    ok = fpa.load_events(events_csv(tmp_path, [("a", "2008-03-13 00:00:00", True)], name="e2.csv"))
    assert int(ok.market_time_mismatch.sum()) == 0   # "12:00 a.m." parses as midnight
