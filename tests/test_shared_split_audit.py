"""Small synthetic checks for errors that would invalidate the audit."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("audit", Path(__file__).resolve().parents[1] / "scripts/audit_shared_splits.py")
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_distinct_event_dates_can_still_share_future_bars(self):
        a = [{"valid": True, "start": 0, "anchor": 2, "end": 6, "event_date": "20200101"}]
        b = [{"valid": True, "start": 5, "anchor": 7, "end": 9, "event_date": "20200102"}]
        result = audit.pair_overlap(a, b, 10)
        self.assertEqual(result["shared_dates"], 0)
        self.assertEqual(result["shared_full_window_bars"], 2)
        self.assertEqual(result["a_events_touching_b"], 1)

    def test_strict_before_policy_does_not_use_release_time_candle(self):
        t = np.arange(20200101110000, 20200101150000, 100)
        legacy = audit.window_for_event("data/event\\4\\20200101.txt", t, "category4_1400_strict_before", 2, 2)
        self.assertLess(legacy["input_end"], 20200101140000)
        self.assertEqual(legacy["target_start"], 20200101140000)

    def test_full_horizon_is_required(self):
        t = np.array([20200101093000, 20200101093500, 20200101094000])
        w = audit.window_for_event("data/event/1/20200101.txt", t, "legacy_all0830_floor", 2, 2)
        self.assertFalse(w["valid"])

    def test_exact_duplicate_prices_are_removed_and_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            p.write_text("date,date_int,Open,High,Low,Close\n2020-01-01 09:35:00,20200101093500,2,2,2,2\n2020-01-01 09:30:00,20200101093000,1,1,1,1\n2020-01-01 09:30:00,20200101093000,1,1,1,1\n")
            t, x, q = audit.load_prices(p)
            self.assertEqual(q["extra_duplicate_rows"], 1)
            self.assertEqual(len(q["backward_jumps"]), 1)
            self.assertTrue(np.all(np.diff(t) > 0))

    def test_conflicting_prices_stop_the_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "p.csv"
            p.write_text("date,date_int,Open,High,Low,Close\n2020-01-01 09:30:00,20200101093000,1,1,1,1\n2020-01-01 09:30:00,20200101093000,2,2,2,2\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                audit.load_prices(p)


if __name__ == "__main__":
    unittest.main()
