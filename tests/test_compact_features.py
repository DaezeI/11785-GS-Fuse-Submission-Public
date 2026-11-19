import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.compact_return import CompactReturn
from scripts.improvement.compact_features import load_features, past_donors, load_numeric_pack, SNAPSHOT_SHA
from scripts.improvement.train_compact import fit, predictions, evaluate_controls


def test_feature_pack_requires_checksum_snapshot_and_complete_id_mapping(tmp_path):
    path = tmp_path / "features.npz"
    np.savez_compressed(path, event_ids=np.array(["1/20200102.txt", "1/20200101.txt"]),
        embeddings=np.array([[2, 3], [0, 1]], dtype=np.float16),
        normalized_article_sha256=np.array(["hash2", "hash1"]),
        metadata_json=np.array(json.dumps(dict(snapshot_sha256="snapshot", cache_sha256="cache"))))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    vectors, hashes, _ = load_features(path, sha, ["1/20200101.txt", "1/20200102.txt"], "snapshot")
    assert vectors.tolist() == [[0, 1], [2, 3]] and hashes.tolist() == ["hash1", "hash2"]
    with pytest.raises(ValueError, match="checksum"):
        load_features(path, "wrong", [], "snapshot")
    with pytest.raises(ValueError, match="snapshot"):
        load_features(path, sha, [], "different")
    with pytest.raises(ValueError, match="missing required"):
        load_features(path, sha, ["1/20200201.txt"], "snapshot")


def test_donors_never_cross_category_year_or_date_and_exclude_duplicate_content():
    ids = ["1/20200101.txt", "1/20200102.txt", "1/20200103.txt", "2/20200102.txt", "1/20210101.txt"]
    hashes = np.array(["a", "b", "a", "c", "d"])
    draw = past_donors(ids, hashes, 23)
    assert draw.tolist() == [-1, 0, 1, -1, -1]
    assert np.array_equal(draw, past_donors(ids, hashes, 23))


def test_residual_fit_accepts_pooled_features_without_updating_price_model():
    torch.manual_seed(3)
    torch.set_num_threads(2)
    model = CompactReturn(width=16, text_dim=4)
    model.freeze_price()
    history, text = torch.randn(12, 4, 35), torch.randn(12, 4)
    target = text[:, :3]
    before = predictions(model, history, "cpu").clone()
    args = SimpleNamespace(device="cpu", seed=0, batch_size=4, epochs=2)
    assert fit(model, history, target, history, target, args, text, text) == 2
    assert torch.equal(before, predictions(model, history, "cpu"))
    assert predictions(model, history, "cpu", features=text).shape == (12, 3)


def test_multimodal_controls_share_eligible_targets_and_save_past_donors(tmp_path):
    model = CompactReturn(width=16, text_dim=4)
    history, features = torch.randn(4, 4, 35), torch.randn(4, 4)
    target = torch.ones(4, 3)
    ids = [f"1/2020010{i}.txt" for i in (1, 2, 3, 4)]
    class Dataset:
        event_ids = ids
        def __len__(self): return 4
    args = SimpleNamespace(device="cpu", out=tmp_path)
    record = evaluate_controls(model, history, target, features, np.array(list("abcd")), Dataset(), args, 1, "vali")
    assert record["events"] == 3 and record["excluded_no_donor"] == 1
    with np.load(tmp_path / "vali-text-controls.npz", allow_pickle=False) as saved:
        assert saved["targets_bp"].shape == saved["real_text_bp"].shape == (3, 3)
        for event_id, donors in zip(saved["event_ids"], saved["donor_event_ids"].T):
            assert all(donor < event_id for donor in donors)


def test_portable_pack_rechecks_window_separation_training_scale_and_checksum(tmp_path):
    path = tmp_path / "portable.npz"
    ids = np.array(["1/20080101.txt", "1/20080102.txt", "1/20180101.txt", "1/20220101.txt"])
    roles = np.array(["train", "train", "vali", "test"])
    counts = dict(train=2, vali=1, test=1)
    windows = np.array([[0, 34, 175], [200, 234, 375], [500, 534, 675], [800, 834, 975]])
    def write(scale=1., changed_windows=None):
        np.savez_compressed(path, event_ids=ids, roles=roles, history_bp=np.zeros((4, 4, 35)),
            targets_bp=np.zeros((4, 3)), bar_windows=windows if changed_windows is None else changed_windows,
            metadata_json=np.array(json.dumps(dict(snapshot_sha256=SNAPSHOT_SHA,
                numeric=dict(seq_len=35, scale_bp=scale)))))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    sha = write()
    data = load_numeric_pack(path, sha, 35, expected_counts=counts)
    assert len(data.train_set) == 2 and data.vali_set.history_bp.shape == (1, 4, 35)
    with pytest.raises(ValueError, match="checksum"):
        load_numeric_pack(path, "wrong", 35, expected_counts=counts)
    with pytest.raises(ValueError, match="lookback"):
        load_numeric_pack(path, sha, 140, expected_counts=counts)
    sha = write(scale=2)
    with pytest.raises(ValueError, match="training-only"):
        load_numeric_pack(path, sha, 35, expected_counts=counts)
    overlap = windows.copy()
    overlap[2] = [100, 134, 275]
    sha = write(changed_windows=overlap)
    with pytest.raises(ValueError, match="overlap"):
        load_numeric_pack(path, sha, 35, expected_counts=counts)
