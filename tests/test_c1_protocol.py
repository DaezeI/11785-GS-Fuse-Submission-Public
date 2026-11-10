"""Focused C1 checks using synthetic prices and article files only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.c1_protocol import build_c1_dataloaders, event_key, load_price_timeline


def fixture_inputs(tmp_path):
    event_dir = tmp_path / "event"
    series = tmp_path / "SP500.csv"
    dates = pd.date_range("2020-01-01", periods=13)
    rows = []
    for day_no, date in enumerate(dates):
        for bar in range(30):
            stamp = date + pd.Timedelta(hours=9, minutes=30 + 5 * bar)
            value = float(100 + day_no * 100 + bar)
            rows.append([stamp.strftime("%Y-%m-%d %H:%M:%S"),
                         int(stamp.strftime("%Y%m%d%H%M%S")),
                         value, value + 1, value - 1, value + .5])
    pd.DataFrame(rows, columns=["date", "date_int", "Open", "High", "Low", "Close"]).to_csv(series, index=False)
    manifest = {}
    for role, day_numbers in (("train", (1, 2)), ("vali", (5, 6)), ("test", (9, 10))):
        manifest[role + "_events"] = []
        for day_no in day_numbers:
            date = dates[day_no].strftime("%Y%m%d")
            for category in range(1, 7):
                folder = event_dir / str(category) / f"{date}.txt"
                folder.mkdir(parents=True)
                (folder / f"{date}.txt_full_summary.txt").write_text(f"article {role} {category} {date}")
                for n in range(10):
                    (folder / f"{date}.txt_report_sent{n}.txt").write_text(f"sentence {n}")
                manifest[role + "_events"].append(f"data/event\\{category}\\{date}.txt")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, event_dir, series


def test_c1_windows_scaler_negatives_and_batch_contract(tmp_path):
    manifest, event_dir, series = fixture_inputs(tmp_path)
    data = build_c1_dataloaders(manifest, event_dir, series, seq_len=2, pred_len=2,
                                batch_size=3, sw_stride=1)
    assert data.audit["emitted_counts"] == {"train": 12, "vali": 12, "test": 12}
    assert data.audit["scaler_fit_unique_bars"] == 12
    assert data.scaler.mean_[0] < 400  # held-out high-price days are not fitted
    assert len(data.sw_train_set.indices) >= 2
    assert len(data.sw_vali_set.indices) >= 2
    assert len(data.train_set.event_ids) == len(set(data.train_set.event_ids))
    for ds, days in ((data.train_set, {"20200102", "20200103"}),
                     (data.vali_set, {"20200106", "20200107"}),
                     (data.test_set, {"20200110", "20200111"})):
        for item, event_id in zip(ds.list, ds.event_ids):
            assert event_id.split("/")[1][:8] in days
            assert len(item[1]) == 10 and len(item[2]) == 5
            assert all(Path(p).parent.name[:8] in days for p in item[2])
        sample = ds[0]
        assert len(sample) == 7 and sample[3].shape == (4, 2)
    assert next(iter(data.train_loader))[3].shape == (3, 4, 2)
    assert all(data.sw_train_set.data[i:i + 4].shape == (4, 4)
               for i in data.sw_train_set.indices)


def test_missing_articles_are_reported_and_opt_in_filtered(tmp_path):
    manifest, event_dir, series = fixture_inputs(tmp_path)
    missing = event_dir / "1" / "20200102.txt" / "20200102.txt_full_summary.txt"
    missing.unlink()
    with pytest.raises(ValueError, match="missing_summary"):
        build_c1_dataloaders(manifest, event_dir, series, seq_len=2, pred_len=2)
    data = build_c1_dataloaders(manifest, event_dir, series, seq_len=2, pred_len=2,
                                skip_missing=True)
    assert data.audit["excluded_counts"] == {"train:missing_summary": 1}
    assert data.audit["emitted_counts"]["train"] == 11


def test_price_deduplication_requires_identical_ohlc(tmp_path):
    _, _, series = fixture_inputs(tmp_path)
    table = pd.read_csv(series)
    pd.concat([table, table.iloc[[0]]]).to_csv(series, index=False)
    _, _, audit = load_price_timeline(series)
    assert audit["exact_duplicate_rows"] == 1
    table.iloc[0, table.columns.get_loc("Close")] += 1
    pd.concat([table, pd.read_csv(series).iloc[[0]]]).to_csv(series, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_price_timeline(series)


def test_manifest_key_is_stable_across_path_separators():
    assert event_key(r"data/event\4\20200102.txt") == ("4", "20200102", "4/20200102.txt")
