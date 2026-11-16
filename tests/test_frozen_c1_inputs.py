import json

import pytest

from scripts.freeze_c1_inputs import freeze, verify


def test_frozen_inputs_are_copied_and_tampering_is_detected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train_events": ["data/event\\1\\20200101.txt"],
                                    "vali_events": [], "test_events": []}))
    event = tmp_path / "source" / "1" / "20200101.txt"
    event.mkdir(parents=True)
    (event / "20200101.txt_full_summary.txt").write_text("original article")
    series = tmp_path / "SP500.csv"
    series.write_text("date,date_int,Open,High,Low,Close\n")
    destination = tmp_path / "snapshot-v1"
    record = freeze(manifest, tmp_path / "source", series, destination)
    assert verify(destination)["content_sha256"] == record["content_sha256"]
    assert record["event_folder_count"] == 1
    with pytest.raises(FileExistsError):
        freeze(manifest, tmp_path / "source", series, destination)
    (event / "20200101.txt_full_summary.txt").write_text("changed source")
    assert verify(destination)["content_sha256"] == record["content_sha256"]
    copied = destination / "event" / "1" / "20200101.txt" / "20200101.txt_full_summary.txt"
    copied.chmod(0o640)
    copied.write_text("tampered")
    for folder in (destination, destination / "event", destination / "event" / "1", copied.parent, destination / "series"):
        folder.chmod(0o750)
    with pytest.raises(ValueError, match="fingerprint"):
        verify(destination)
