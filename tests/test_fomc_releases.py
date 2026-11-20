"""Offline checks for the Person 1 FOMC release list (no network needed)."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts.build_fomc_releases import (
    OUT_CSV, available_before, build_rows, derive_row, integrity_problems,
    load_evidence, market_session, parse_clock, read_csv, write_csv,
)

UTC = ZoneInfo("UTC")


@pytest.fixture(scope="module")
def rows():
    return read_csv(OUT_CSV)


def test_committed_csv_matches_evidence(rows):
    rebuilt = build_rows(load_evidence())
    assert [r["event_id"] for r in rebuilt] == [r["event_id"] for r in rows]
    for a, b in zip(rebuilt, rows):
        assert str(a["available_utc"]) == b["available_utc"]
        assert a["text_sha256"] == b["text_sha256"]


def test_integrity(rows):
    assert integrity_problems(rows) == []


def test_every_verified_time_has_official_evidence(rows):
    for r in rows:
        assert r["source_url"].startswith("https://www.federalreserve.gov/")
        if r["time_verified"] == "True":
            assert r["time_evidence"] and r["time_evidence_url"].startswith("https://www.federalreserve.gov/")
            assert r["release_time_et"] in r["time_evidence"] or r["release_time_et"].replace(" ", "") in r["time_evidence"].replace(" ", "")


def test_minutes_use_release_date_not_meeting_date(rows):
    m = {r["event_id"]: r for r in rows}
    jan = m["min_20080130"]
    assert jan["meeting_date"] == "20080130" and jan["release_date"] == "20080220"
    assert all(r["release_date"] > r["meeting_date"] for r in rows if r["event_type"] == "fomc_minutes")


def test_known_times_and_dst(rows):
    m = {r["event_id"]: r for r in rows}
    assert m["stmt_20080122b"]["release_utc"] == "2008-01-22T13:30:00Z"   # 8:30 a.m. EST
    assert m["stmt_20200315a"]["release_utc"] == "2020-03-15T21:00:00Z"   # 5:00 p.m. EDT, Sunday
    assert m["stmt_20200315a"]["market_session_et"] == "weekend"
    assert m["min_20130320"]["release_utc"] == "2013-04-10T13:00:00Z"     # 9:00 a.m. EDT
    assert m["stmt_20110427a"]["release_time_et"] == "12:30 p.m."


def test_unverified_time_is_not_usable_intraday(rows):
    r = {x["event_id"]: x for x in rows}["stmt_20080311a"]
    assert r["time_verified"] == "False" and r["usable_intraday"] == "False"
    assert r["available_utc"] == "2008-03-12T04:00:00Z"  # end of the ET day


def test_available_before_prediction(rows):
    r = {x["event_id"]: x for x in rows}["stmt_20200315a"]
    assert available_before(r, datetime(2020, 3, 16, 13, 30, tzinfo=UTC))
    assert not available_before(r, datetime(2020, 3, 15, 21, 0, tzinfo=UTC))   # inside lag
    assert not available_before(r, datetime(2020, 3, 15, 21, 5, tzinfo=UTC))   # strict


def test_helpers():
    assert parse_clock("2:15 p.m.").hour == 14
    assert parse_clock("12:30 p.m.").hour == 12
    et = ZoneInfo("America/New_York")
    assert market_session(datetime(2024, 1, 31, 14, 0, tzinfo=et)) == "regular"
    assert market_session(datetime(2024, 1, 31, 16, 0, tzinfo=et)) == "after_hours"


def test_tz_mismatch_is_rejected():
    ev = {"event_id": "x", "event_type": "fomc_statement", "release_date": "20200701",
          "release_time_et": "2:00 p.m.", "time_evidence": "For release at 2:00 p.m. EST"}
    with pytest.raises(ValueError):
        derive_row(ev)


def test_never_overwrites(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("old")
    with pytest.raises(FileExistsError):
        write_csv([], p)
    assert p.read_text() == "old"
