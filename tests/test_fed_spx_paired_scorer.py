"""Guard the paired Fed/SPX scorer: formats, weighting, gain sign, coverage and packaged files."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import score_fed_spx_paired as scorer

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/fed-spx-paired-scorer-v1"
ARM = "Text(frozen)+CNN(frozen)~Pool~Add"


def reference(split="val"):
    return pd.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "split": split,
        "evidence_class": ["eligible_official_record", "eligible_document_release_line_unmodified_file",
                           "eligible_official_record", "eligible_official_record"],
        "market_et": ["2020-01-01 14:00:00", "2020-01-01 14:00:00", "2020-01-02 14:00:00", "2020-01-03 14:00:00"],
        "target_20m_bp": [4.0, -2.0, 1.0, -3.0],
        "target_30m_bp": [5.0, -1.0, 2.0, -4.0],
    })


def forecast_file(tmp_path, name, rows):
    path = tmp_path / name
    pd.DataFrame(rows, columns=scorer.FORECAST_COLUMNS).to_csv(path, index=False)
    return path


def test_horizons_match_the_cohort_builder():
    tree = ast.parse((ROOT / "data/fed_spx_endpoints.py").read_text(encoding="utf-8"))
    value = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "HORIZONS" for t in node.targets))
    assert value == scorer.HORIZONS


def test_instant_weights_share_one_unit_per_instant():
    np.testing.assert_allclose(scorer.instant_weights(reference().market_et), [0.5, 0.5, 1, 1])


def test_reference_refuses_test_split_without_flag(tmp_path):
    path = tmp_path / "ref.csv"
    reference("test").to_csv(path, index=False)
    with pytest.raises(ValueError, match="test split"):
        scorer.load_reference(path)
    assert len(scorer.load_reference(path, allow_test=True)) == 4


@pytest.mark.parametrize("rows, message", [
    ([("zz", "X", 0, 1.0, 1.0)], "not in the reference"),
    ([("a", "Persistence", 0, 1.0, 1.0)], "built in"),
    ([("a", "X", 0, 1.0, 1.0), ("a", "X", 0, 2.0, 2.0)], "duplicate"),
    ([("a", "X", 0, float("nan"), 1.0)], "non-finite"),
    ([("a", "X", 0, 1.0, 1.0), ("a", "X", 1, 1.0, 1.0), ("b", "X", 1, 1.0, 1.0)], "same set"),
    ([("a", "has space", 0, 1.0, 1.0)], "invalid label"),
])
def test_forecast_files_are_validated(tmp_path, rows, message):
    path = forecast_file(tmp_path, "f.csv", rows)
    with pytest.raises(ValueError, match=message):
        scorer.load_forecasts([path], reference())


def test_label_may_not_appear_in_two_files(tmp_path):
    first = forecast_file(tmp_path, "1.csv", [("a", "X", 0, 1.0, 1.0)])
    second = forecast_file(tmp_path, "2.csv", [("b", "X", 0, 1.0, 1.0)])
    with pytest.raises(ValueError, match="both"):
        scorer.load_forecasts([first, second], reference())


def test_gain_sign_seed_average_and_persistence(tmp_path):
    ref = reference()
    truth = ref[["target_20m_bp", "target_30m_bp"]].to_numpy()
    rows = []
    for seed, shift in ((0, 1.0), (1, -1.0)):  # seed average equals the truth exactly
        rows += [(e, "Perfect", seed, *(t + shift)) for e, t in zip(ref.event_id, truth)]
    forecasts = scorer.load_forecasts([forecast_file(tmp_path, "f.csv", rows)], ref)
    mask = np.ones(len(ref), bool)
    score = scorer.score_label(forecasts, ref, mask, "Perfect")
    assert score["metrics_seed_average_forecast"]["20"]["mae_bp"] == 0
    assert score["mae_bp_mean_over_seeds"]["20"] == 1
    persistence = scorer.score_label(forecasts, ref, mask, scorer.PERSISTENCE)
    weights = np.array([0.5, 0.5, 1, 1])
    assert persistence["metrics_seed_average_forecast"]["20"]["mae_bp"] == pytest.approx(
        np.average(np.abs(truth[:, 0]), weights=weights))
    comparison = scorer.compare(forecasts, ref, mask, "Perfect", scorer.PERSISTENCE)
    expected = np.average(np.abs(truth), weights=weights, axis=0)
    np.testing.assert_allclose(comparison["gain_bp_20_30"], expected)
    np.testing.assert_allclose(comparison["seed_mean_loss_gain_bp_20_30"], expected - 1)


def test_comparisons_use_common_rows_and_reweight_instants(tmp_path):
    ref = reference()
    rows = [(e, ARM, 0, 1.0, 1.0) for e in ref.event_id]
    rows += [(e, f"{ARM}[text=shuffled-11]", 0, 0.0, 0.0) for e in ("b", "c")]
    forecasts = scorer.load_forecasts([forecast_file(tmp_path, "f.csv", rows)], ref)
    comparison = scorer.compare(forecasts, ref, np.ones(4, bool), ARM, f"{ARM}[text=shuffled-11]")
    assert comparison["rows"] == 2 and comparison["release_instants"] == 2
    # b: |0-(-2)| - |1-(-2)| = -1 ; c: |0-1| - |1-1| = 1 ; both instants now weigh 1.
    assert comparison["gain_bp_20_30"][0] == pytest.approx(0.0)


def test_strict_subset_drops_inferred_classes_and_text_arm_pairs(tmp_path):
    ref = reference()
    rows = [(e, ARM, 0, 1.0, 1.0) for e in ref.event_id]
    rows += [(e, f"{ARM}[text=zero]", 0, 0.0, 0.0) for e in ref.event_id]
    rows += [(e, "none+CNN~none~none", 0, 0.5, 0.5) for e in ref.event_id]
    forecasts = scorer.load_forecasts([forecast_file(tmp_path, "f.csv", rows)], ref)
    pairs = scorer.default_comparisons(sorted(forecasts.label.unique()), [ARM])
    assert (ARM, "none+CNN~none~none") in pairs and (ARM, f"{ARM}[text=zero]") in pairs
    results = scorer.score(ref, forecasts, pairs)
    assert results["STRICT"]["reference"]["rows"] == 3
    assert results["STRICT"]["forecasters"][ARM]["rows"] == 3


def test_text_run_conversion_round_trip(tmp_path):
    ref = reference()
    ref_path = tmp_path / "ref.csv"
    ref.to_csv(ref_path, index=False)
    n = len(ref)
    rng = np.random.default_rng(0)
    truth = ref[["target_20m_bp", "target_30m_bp"]].to_numpy()
    real = rng.normal(size=(3, n, 2))
    shuffled = rng.normal(size=(3, 3, n, 2))
    donor = np.array([False, True, True, True])
    np.savez(tmp_path / "val-paired.npz", event_ids=ref.event_id.to_numpy(str), truth=truth, real=real,
             price=rng.normal(size=(3, n, 2)), text_zero=rng.normal(size=(3, n, 2)), text_shuffled=shuffled,
             donor_eligible=donor, weights=scorer.instant_weights(ref.market_et))
    out = tmp_path / "text.csv"
    scorer.main(["from-text-run", "--npz", str(tmp_path / "val-paired.npz"), "--reference", str(ref_path),
                 "--out", str(out), "--label", ARM])
    forecasts = scorer.load_forecasts([out], ref)
    np.testing.assert_allclose(scorer.seed_matrix(forecasts, ARM, list(ref.event_id))[0], real, rtol=1e-9)
    shuffled_ids = list(ref.event_id[donor])
    np.testing.assert_allclose(scorer.seed_matrix(forecasts, f"{ARM}[text=shuffled-23]", shuffled_ids)[0],
                               shuffled[:, 1][:, donor], rtol=1e-9)
    assert scorer.coverage(forecasts, f"{ARM}[text=shuffled-37]") == set(shuffled_ids)

    bad = dict(np.load(tmp_path / "val-paired.npz"))
    bad["truth"] = truth + 1
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="targets differ"):
        scorer.main(["from-text-run", "--npz", str(tmp_path / "bad.npz"), "--reference", str(ref_path),
                     "--out", str(tmp_path / "bad.csv")])


def test_packaged_baselines_reproduce_matched_control_v2(tmp_path):
    out = tmp_path / "metrics.json"
    scorer.main(["score", "--reference", str(PACKAGE / "reference-FED-PXTX-ALL-REL5-v2-val.csv"),
                 "--forecasts", str(PACKAGE / "baselines-FED-PXTX-ALL-REL5-v2-val.csv"), "--out", str(out)])
    new = json.loads(out.read_text(encoding="utf-8"))["results"]
    old = json.loads((ROOT / "results/fed-spx-matched-control-v2/metrics.json").read_text(encoding="utf-8"))["results"]
    labels = {"price_cnn_control": "none+CNN~none~none", "dlinear": "none+DLinear~none~none",
              "ridge": "none+Ridge~none~none"}
    for old_name, new_name in (("all_text_verified", "ALL"), ("without_any_inferred_class", "STRICT")):
        assert new[new_name]["reference"]["rows"] == old[old_name]["rows"]
        for key, label in labels.items():
            before, after = old[old_name]["scores"][key], new[new_name]["forecasters"][label]
            for h in ("20", "30"):
                assert after["metrics_seed_average_forecast"][h]["mae_bp"] == pytest.approx(
                    before["metrics_seed_average_forecast"][h]["mae_bp"], abs=1e-7)
            paired = next(c for c in new[new_name]["comparisons"]
                          if c["label"] == label and c["other"] == scorer.PERSISTENCE)
            np.testing.assert_allclose(paired["seed_mean_loss_gain_bp_20_30"], before["gain_over_zero_bp_20_30"],
                                       atol=1e-7)
            ci = before["paired_gain_ci95_bp_by_block_20_30"]["10"]
            if ci is not None:
                np.testing.assert_allclose(paired["seed_mean_loss_paired_gain_ci95_bp_by_block_20_30"]["10"], ci,
                                           atol=1e-7)
