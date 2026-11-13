"""Check the committed date-exclusion ledger against the exact C1 manifests."""

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ArticleDatePolicyTest(unittest.TestCase):
    def test_policy_ids_and_evidence_levels_match_screen(self):
        original = json.loads((ROOT / "splits/manifests/purged_split_c1_macro_eras.json").read_text())
        screened_path = ROOT / "splits/manifests/purged_split_c1_date_screen_v1.json"
        screened = json.loads(screened_path.read_text())
        with (ROOT / "analysis/text_value/c1_article_date_policy_v1.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        ids = {r["original_event_id"] for r in rows}
        self.assertEqual(len(rows), len(ids))
        self.assertEqual(ids, set(screened["article_date_screen"]["excluded_event_ids"]))
        self.assertEqual(len(rows), 883)
        self.assertEqual(sum(r["evidence_status"] == "source_verified_date_only" for r in rows), 5)
        self.assertEqual(sum(r["evidence_status"] == "unverified_reuters_dateline_candidate" for r in rows), 878)
        self.assertTrue(all(not r["verified_publication_time"] and not r["forecast_anchor"] for r in rows))
        self.assertTrue(all(r["decision"] == "exclude_until_time_and_anchor_verified" for r in rows))
        for role in ("train", "vali", "test"):
            original_ids = {"/".join(x.replace("\\", "/").split("/")[-2:]) for x in original[role + "_events"]}
            retained_ids = {"/".join(x.replace("\\", "/").split("/")[-2:]) for x in screened[role + "_events"]}
            self.assertEqual(retained_ids, original_ids - ids)
        summary = json.loads((ROOT / "analysis/text_value/c1_article_date_policy_v1_summary.json").read_text())
        self.assertEqual(summary["screened_manifest_sha256"], hashlib.sha256(screened_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
