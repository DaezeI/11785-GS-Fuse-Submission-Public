import unittest

from scripts.run_text_controls import selected_event_indices


class CheckedEventSelectionTest(unittest.TestCase):
    def test_exact_order_and_missing_id_failure(self):
        ids = ["2/20231208.txt", "5/20231212.txt", "2/20240105.txt"]
        self.assertEqual(selected_event_indices(ids, [ids[2], ids[0]]).tolist(), [2, 0])
        with self.assertRaisesRegex(ValueError, "absent"):
            selected_event_indices(ids, [ids[0], "5/20990101.txt"])
        with self.assertRaisesRegex(ValueError, "unique"):
            selected_event_indices(ids, [ids[0], ids[0]])


if __name__ == "__main__":
    unittest.main()
