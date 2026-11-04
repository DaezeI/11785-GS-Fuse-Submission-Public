import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("paired", Path(__file__).resolve().parents[1] / "scripts/analyze_text_predictions.py")
paired = importlib.util.module_from_spec(spec); spec.loader.exec_module(paired)


class PairedAnalysisTests(unittest.TestCase):
    def blob(self):
        n = 24
        y = np.ones((n, 140, 4))
        return {"targets": y, "real_text": y.copy(), "zero_text": y + 1,
                "shuffled_text": np.stack([y + 0.5, y + 0.6]),
                "seqs": np.ones((n, 35, 4)), "raw_seqs": np.full((n, 35, 4), 100.),
                "event_ids": np.array([f"1/{i}" for i in range(n)]),
                "event_dates": np.array([f"202001{i+1:02}" for i in range(n)]),
                "event_types": np.full(n, "1"), "train_volatility": np.array([0.001, 0.002, 0.003]),
                "metadata_json": np.array(json.dumps({"checkpoint_id": "synthetic_test_only"}))}

    def test_known_signal_has_expected_gain(self):
        result = paired.analyze(self.blob(), reps=20, block=3)
        row = result["horizons"]["140"]["all"]
        self.assertEqual(row["comparisons"]["zero_text"]["real_text_mse_gain_pct"], 100.)
        self.assertEqual(row["comparisons"]["zero_text"]["ci95_pct"], [100., 100.])
        self.assertIn("training_split", result["missing_provenance"])

    def test_duplicate_ids_rejected(self):
        blob = self.blob(); blob["event_ids"][:] = "same"
        with self.assertRaisesRegex(ValueError, "unique"):
            paired.validate(blob)

    def test_wrong_array_layout_rejected(self):
        blob = self.blob(); blob["targets"] = blob["targets"].transpose(0, 2, 1)
        with self.assertRaisesRegex(ValueError, "channel-last"):
            paired.validate(blob)

    def test_normalized_negative_prices_cannot_define_volatility(self):
        with self.assertRaisesRegex(ValueError, "unscaled"):
            paired.pre_event_volatility(np.full((2, 35, 4), -1.))

    def test_single_block_cannot_report_false_certainty(self):
        dates = np.array([f"202001{i+1:02}" for i in range(20)])
        result = paired.block_bootstrap_improvement(np.arange(20.), np.full(20, 20.), dates,
            np.ones(20, dtype=bool), reps=20, block=20)
        self.assertIsNone(result["ci95_pct"])
        self.assertIn("three blocks", result["reason"])

    def test_perfect_control_has_undefined_relative_interval(self):
        dates = np.array([f"202001{i+1:02}" for i in range(24)])
        result = paired.block_bootstrap_improvement(np.ones(24), np.zeros(24), dates,
            np.ones(24, dtype=bool), reps=20, block=3)
        self.assertIsNone(result["ci95_pct"])
        self.assertIn("zero", result["reason"])

    def test_invalid_calendar_dates_rejected(self):
        blob = self.blob(); blob["event_dates"][0] = '20200231'
        with self.assertRaises(ValueError):
            paired.validate(blob)

    def test_unverified_training_split_remains_unresolved(self):
        blob = self.blob()
        blob["metadata_json"] = np.array(json.dumps({"checkpoint_id": "test", "training_split": "unverified"}))
        result = paired.analyze(blob, reps=20, block=3)
        self.assertIn("training_split", result["missing_provenance"])


if __name__ == "__main__":
    unittest.main()
