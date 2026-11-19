import unittest
import numpy as np

from scripts.improvement import train_compact_context_ablation as ablation


def window(event_id, anchor):
    return dict(event_id=event_id, anchor=anchor, short_start=anchor - 34, stop=anchor + 141)


class PriceContextAblationTests(unittest.TestCase):
    def setUp(self):
        self.canonical = {
            "train": [window("train_edge", 80), window("train_keep", 360), window("train_test_overlap", 2060)],
            "vali": [window("vali_keep", 700), window("vali_train_overlap", 555)],
            "test": [window("test_protected", 1850)],
        }

    def test_preserves_order_and_uses_only_paired_non_overlapping_windows(self):
        selected, excluded, baseline, paired, coverage = ablation.eligible_rows(self.canonical, 2400)
        self.assertEqual([r["event_id"] for r in selected["train"]], ["train_keep"])
        self.assertEqual([r["event_id"] for r in selected["vali"]], ["vali_keep"])
        self.assertEqual(excluded["train"], [
            dict(event_id="train_edge", reasons=["insufficient_price_history"]),
            dict(event_id="train_test_overlap", reasons=["touches_canonical_test_bars"]),
        ])
        self.assertEqual(excluded["vali"], [
            dict(event_id="vali_train_overlap", reasons=["touches_canonical_train_bars"]),
        ])
        self.assertTrue(all(v == 0 for v in baseline.values()))
        self.assertTrue(all(v == 0 for r in paired.values() for v in r.values()))
        self.assertEqual(coverage["L35"], {"train": 175, "vali": 175, "test": 175})
        self.assertEqual(coverage["L140"], {"train": 280, "vali": 280, "test": 175})

    def test_canonical_baseline_collision_fails_closed(self):
        self.canonical["test"].append(window("bad_test", 362))
        with self.assertRaisesRegex(ValueError, "Canonical L35 windows already overlap"):
            ablation.eligible_rows(self.canonical, 2400)

    def test_both_lengths_use_identical_anchor_and_full_ohlc_target(self):
        row = window("x", 180) | {"long_start": 41}
        values = np.arange(1, 1601, dtype=np.float32).reshape(-1, 4)
        to_returns = lambda prices, close: np.log(prices / close) * 10000
        x, y = ablation.make_arrays(values, [row], to_returns)
        self.assertEqual(x.shape, (1, 4, 140))
        self.assertEqual(y.shape, (1, 140, 4))
        self.assertEqual(x[0, 3, -1], 0)
        close = values[180, 3]
        np.testing.assert_array_equal(x[0, :, -35:], to_returns(values[146:181], close).T)
        np.testing.assert_array_equal(y[0], to_returns(values[181:321], close))
        with self.assertRaisesRegex(ValueError, "inconsistent geometry"):
            ablation.make_arrays(values, [row | {"short_start": 1}], to_returns)


if __name__ == "__main__":
    unittest.main()
