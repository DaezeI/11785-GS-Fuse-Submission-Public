import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("summary", Path(__file__).resolve().parents[1] / "scripts/summarize_text_controls.py")
summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_matching_and_retained_components_sum_to_total(self):
        d = summary.decompose(real=8., shuffled=9., zero=10.)
        self.assertEqual(d["zero_minus_real_mse"], d["zero_minus_shuffled_mse"] + d["shuffled_minus_real_mse"])
        self.assertEqual(d["benefit_retained_with_shuffled_pct"], 50.)
        self.assertEqual(d["matching_gap_share_pct"], 50.)

    def test_shuffle_better_than_real_does_not_get_clipped(self):
        d = summary.decompose(real=8., shuffled=7., zero=10.)
        self.assertEqual(d["matching_gap_share_pct"], -50.)
        self.assertEqual(d["benefit_retained_with_shuffled_pct"], 150.)

    def test_no_positive_total_benefit_has_no_benefit_share(self):
        d = summary.decompose(real=10., shuffled=9., zero=10.)
        self.assertIsNone(d["matching_gap_share_pct"])

    def test_perfect_real_control_does_not_divide_by_zero(self):
        self.assertIsNone(summary.decompose(0., 1., 2.)["shuffle_vs_real_error_increase_pct"])

    def test_nonfinite_mse_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            summary.decompose(float("nan"), 1., 2.)


if __name__ == "__main__": unittest.main()
