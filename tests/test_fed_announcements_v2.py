"""Offline checks for Person 1 v2 (all Fed announcements). No network needed."""

from datetime import date, datetime, time
import json

import pytest

from scripts.build_fed_announcements_v2 import (
    EVIDENCE, OUT_CSV, V1_CSV, board_subcategory, build_v2_rows, calendar_index,
    calendar_match, feed_dt, speaker_role, v2_problems, write_new_csv,
)
from scripts.build_fomc_releases import UTC, available_before, read_csv


@pytest.fixture(scope="module")
def rows():
    return read_csv(OUT_CSV)


def test_csv_rebuilds_from_evidence(rows):
    rebuilt = build_v2_rows(json.loads(EVIDENCE.read_text(encoding="utf-8")), read_csv(V1_CSV))
    assert [r["event_id"] for r in rebuilt] == [r["event_id"] for r in rows]
    assert [str(r["available_utc"]) for r in rebuilt] == [r["available_utc"] for r in rows]


def test_integrity(rows):
    assert v2_problems(rows) == []


def test_v1_rows_carried_over_unchanged(rows):
    v1 = {r["event_id"]: r for r in read_csv(V1_CSV)}
    carried = [r for r in rows if r["source_version"] == "v1"]
    assert len(carried) == len(v1)
    for r in carried:
        assert r["available_utc"] == v1[r["event_id"]]["available_utc"]
        assert r["text_sha256"] == v1[r["event_id"]]["text_sha256"]


def test_date_only_rows_are_not_usable(rows):
    for r in rows:
        if r["evidence_level"].startswith("C_"):
            assert r["usable_intraday"] == "False"


def test_press_conference_counts_after_the_meeting_day(rows):
    r = {x["event_id"]: x for x in rows}["presconf_20240918"]
    # Live on Sep 18 afternoon; the full transcript is used only after that day ends.
    assert r["available_utc"].startswith("2024-09-19T04:05")
    assert not available_before(r, datetime(2024, 9, 18, 20, 0, tzinfo=UTC))
    assert available_before(r, datetime(2024, 9, 19, 13, 30, tzinfo=UTC))


def test_testimony_uses_the_earlier_public_time(rows):
    # Semiannual testimony: prepared text 8:30 a.m., hearing 10:00 a.m. -> market knew at 8:30.
    r = {x["event_id"]: x for x in rows}["testimony_powell20250624a"]
    assert r["calendar_check"] == "differs" and r["market_time_et"].endswith("8:30 a.m.")


def test_conflicting_speech_times_are_not_usable(rows):
    for r in rows:
        if r["event_type"] == "fed_speech" and r["calendar_check"] in ("differs", "ambiguous_multiple_events"):
            assert r["usable_intraday"] == "False"


def test_feed_and_calendar_helpers():
    assert feed_dt("11/20/2026 12:15:00 PM") == (date(2026, 11, 20), time(12, 15))
    assert feed_dt("3/14/2008") == (date(2008, 3, 14), None)
    idx = calendar_index([{"m": "2026-11", "dd": "20", "ty": "Speeches",
                           "ti": "Speech - Governor Lisa D. Cook", "tm": "12:15 p.m."}])
    assert calendar_match(idx, date(2026, 11, 20), "Governor Lisa D. Cook", time(12, 15)) == (time(12, 15), "agrees")
    assert calendar_match(idx, date(2026, 11, 20), "Governor Lisa D. Cook", time(9, 0)) == (time(12, 15), "differs")
    assert calendar_match(idx, date(2026, 11, 21), "Governor Lisa D. Cook", time(9, 0))[1] == "not_in_calendar"


def test_labels():
    assert speaker_role("Chair Jerome H. Powell") == "chair"
    assert speaker_role("Vice Chair for Supervision Michael S. Barr") == "vice_chair_supervision"
    assert speaker_role("Governor Lisa D. Cook") == "governor"
    assert board_subcategory("Federal Reserve announces results of auction of $5 billion") == "auction_or_operation"
    assert board_subcategory("Minutes of the Board's discount rate meetings") == "discount_rate"


def test_never_overwrites(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("old")
    with pytest.raises(FileExistsError):
        write_new_csv([], p)
    assert p.read_text() == "old"


def test_fetch_text_saves_by_event_id_and_checks_hash(tmp_path, monkeypatch):
    import hashlib

    import scripts.build_fed_announcements_v2 as v2

    html = ('<html><body><div id="article"><div class="col-sm-8 heading">Head</div>'
            '<div class="col-sm-8"><p>First  paragraph.</p><p>Second<br>line.</p>'
            '<p>For media inquiries, call 202</p></div></div></body></html>')
    text = "First paragraph.\nSecond line."

    class Resp:
        content = html.encode()

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, timeout):
            return Resp()

    monkeypatch.setattr(v2, "_session", lambda: Session())
    row = {"event_id": "board_test", "usable_intraday": "True", "text_chars": str(len(text)),
           "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text_format": "html",
           "text_selector": "#article > div.col-sm-8:not(.heading)",
           "source_url": "https://www.federalreserve.gov/x.htm"}
    report = v2.fetch_texts([row], out_dir=tmp_path)
    assert report[0]["hash_ok"] is True and report[0]["error"] == ""
    assert (tmp_path / "board_test.txt").read_text() == text
