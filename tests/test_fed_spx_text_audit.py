"""A lookup only clears the timing gate when the exact replay matches."""

import hashlib
from threading import Event
from types import SimpleNamespace

from scripts import audit_fed_spx_text as audit


def event():
    return SimpleNamespace(event_id="release-a", split="val", event_type="fomc_statement",
                           source_url="https://www.federalreserve.gov/release.htm",
                           text_cutoff_utc="2021-07-28T18:00:00+00:00", text_format="html",
                           text_selector="#article", text_sha256=hashlib.sha256(b"Prepared text.").hexdigest())


def response(url, content=b"", data=None):
    return SimpleNamespace(url=url, content=content, json=lambda: data, raise_for_status=lambda: None)


def test_exact_timely_replay_can_be_used(monkeypatch, tmp_path):
    row = event()
    match = "20210728175900"
    archived_url = f"{audit.ARCHIVE}/web/{match}id_/{row.source_url}"
    replies = iter([response(audit.ARCHIVE, b"[]", [["timestamp", "original"], [match, row.source_url]]),
                    response(archived_url, b'<div id="article"><p>Prepared text.</p></div>')])
    monkeypatch.setattr(audit.requests, "get", lambda *args, **kwargs: next(replies))
    result = audit.audit_event(row, tmp_path, Event(), "cdx")
    assert result["status"] == "verified_exact_version"
    assert (tmp_path / "release-a.txt").read_bytes() == b"Prepared text."


def test_a_later_nearest_capture_is_excluded(monkeypatch, tmp_path):
    late = {"archived_snapshots": {"closest": {"status": "200", "available": True, "timestamp": "20210728180107"}}}
    monkeypatch.setattr(audit.requests, "get", lambda *args, **kwargs: response("https://archive.org", data=late))
    result = audit.audit_event(event(), tmp_path, Event(), "availability")
    assert result["status"] == "nearest_capture_after_cutoff"
    assert not list(tmp_path.iterdir())


def test_replay_redirect_to_a_new_timestamp_is_excluded(monkeypatch, tmp_path):
    row = event()
    entries = [["timestamp", "original"], ["20210728175900", row.source_url]]
    redirected = f"{audit.ARCHIVE}/web/20210728180107id_/{row.source_url}"
    replies = iter([response(audit.ARCHIVE, b"[]", entries),
                    response(redirected, b'<div id="article"><p>Prepared text.</p></div>')])
    monkeypatch.setattr(audit.requests, "get", lambda *args, **kwargs: next(replies))
    assert audit.audit_event(row, tmp_path, Event(), "cdx")["status"] == "replay_redirected_to_different_version"
    assert not list(tmp_path.iterdir())
