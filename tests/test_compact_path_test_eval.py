import copy
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.improvement import eval_compact_path_test as evaluator


class LockedPathTest(unittest.TestCase):
    def validation_config(self, mode, seed, sources):
        epoch, mae = evaluator.VALIDATION_LOCK[mode, seed]
        return dict(task=evaluator.TASK, forecast_bars=140, channels=["open", "high", "low", "close"],
                    width=32, seq_len=35, horizons=[35, 70, 140], timing_policy=evaluator.TIMING,
                    snapshot_content_sha256=evaluator.SNAPSHOT, manifest_sha256=evaluator.MANIFEST,
                    mode=mode, seed=seed, epochs=30, batch_size=32, source_sha256=sources,
                    input_class="canonical_bytes_timing_unverified", release_timestamps_verified=False,
                    test_scored=False, scale_fit="training_only", best_epoch=epoch, epochs_run=7,
                    encoder_gradient_observed=mode == "residual",
                    selection="minimum validation whole-path OHLC return MAE (epoch 0 eligible), patience 5",
                    vali=dict(events=2302, matched=dict(path_mae_bp=mae)))

    def make_locks(self, root):
        sources = {"some_code.py": "known hash"}
        for seed in evaluator.SEEDS:
            for mode in evaluator.MODES:
                folder = root / mode / f"seed-{seed}"
                folder.mkdir(parents=True)
                record = self.validation_config(mode, seed, sources)
                contents = str(seed).encode()
                if mode == "residual":
                    record["price_checkpoint_sha256"] = evaluator.digest(contents)
                (folder / "best.pt").write_bytes(contents)
                (folder / "vali.npz").write_bytes(b"fake this test checks hashes")
                (folder / "resolved_config.json").write_text(json.dumps(record), encoding="utf-8")
        return sources

    def test_six_artifact_locks_require_same_seed_price_and_known_selection(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            sources = self.make_locks(root)
            locked = evaluator.read_runs(root, sources)
            self.assertEqual(len(locked), 6)
            self.assertEqual(locked["residual", 1]["lock"]["best_epoch"], 0)
            p = root / "price/seed-2/best.pt"
            p.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "same-seed price checkpoint"):
                evaluator.read_runs(root, sources)
            p.write_bytes(b"2")
            c = root / "residual/seed-1/resolved_config.json"
            rec = json.loads(c.read_text())
            rec["best_epoch"] = 1
            c.write_text(json.dumps(rec))
            with self.assertRaisesRegex(ValueError, "previously reported checkpoint"):
                evaluator.read_runs(root, sources)

    @patch.object(evaluator, "VAL_COUNT", 2)
    def test_validation_replay_requires_canonical_order_and_predictions(self):
        truth = np.zeros((2, 140, 4))
        pred = np.ones_like(truth) * 2
        price = pred + 1
        ids = np.array(["1/20080101", "1/20080102"])
        b = BytesIO()
        np.savez(b, event_ids=ids, targets_bp=truth, predictions_bp=pred, price_reference_bp=price)
        record = dict(validation=b.getvalue(), summary=dict(vali=dict(
            matched=dict(path_mae_bp=2), price_reference=dict(path_mae_bp=3))))
        evaluator.validate_replay(record, ids, truth, pred, price)
        with self.assertRaisesRegex(ValueError, "event IDs or order"):
            evaluator.validate_replay(record, ids[::-1], truth, pred, price)
        with self.assertRaisesRegex(ValueError, "replay its saved validation"):
            evaluator.validate_replay(record, ids, truth, pred + 0.1, price)
        bad = copy.deepcopy(record)
        bad["summary"]["vali"]["matched"]["path_mae_bp"] += 0.01
        with self.assertRaisesRegex(ValueError, "Incorrect recorded validation MAE"):
            evaluator.validate_replay(bad, ids, truth, pred, price)

    def test_scores_every_bar_and_channel_and_preserves_seed_order(self):
        truth = np.zeros((2, 140, 4))
        prices = np.stack([truth + v for v in (2, 3, 4)])
        news = prices.copy()
        news[2, :, 0, 0] -= 1
        rows, means = evaluator.make_scores(truth, prices, news)
        self.assertEqual([r["seed"] for r in rows], [0, 1, 2])
        self.assertAlmostEqual(rows[2]["improvement_bp"], 1 / 560)
        self.assertAlmostEqual(means["price_path_mae_bp"], 3)
        self.assertAlmostEqual(means["improvement_bp"], 1 / (3 * 560))


if __name__ == "__main__":
    unittest.main()
