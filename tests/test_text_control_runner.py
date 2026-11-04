import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("runner", Path(__file__).resolve().parents[1] / "scripts/run_text_controls.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class ShuffleTests(unittest.TestCase):
    def test_shuffle_is_deterministic_and_has_no_fixed_indices(self):
        texts = [str(i) for i in range(20)]; dates = np.array(texts)
        a, meta = runner.shuffle_donors(texts, dates, np.full(20, "1"), 23)
        b, _ = runner.shuffle_donors(texts, dates, np.full(20, "1"), 23)
        np.testing.assert_equal(a, b)
        self.assertEqual(set(a), set(range(20)))
        self.assertEqual(meta["fixed_index_fraction"], 0.)
        self.assertEqual(meta["identical_content_fraction"], 0.)

    def test_duplicate_content_is_reported_not_hidden(self):
        _, meta = runner.shuffle_donors(["same", "same"], ["20200101", "20200102"], ["1", "1"], 23)
        self.assertEqual(meta["identical_content_fraction"], 1.)
        self.assertEqual(meta["fixed_index_fraction"], 0.)

    def test_singleton_type_is_reported_as_unshuffled(self):
        d, meta = runner.shuffle_donors(["a", "b", "c"], ["1", "2", "3"], ["1", "2", "2"], 23, True)
        self.assertEqual(d[0], 0)
        self.assertAlmostEqual(meta["fixed_index_fraction"], 1 / 3)


if __name__ == "__main__": unittest.main()
