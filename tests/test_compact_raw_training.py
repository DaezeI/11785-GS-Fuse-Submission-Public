"""Synthetic checks for the screened raw-news trainer (no local archive required)."""
import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as err:
    if err.name != "torch":
        raise
    torch = None
else:
    import numpy as np

    from model.compact_raw_text import RawTextEncoder, TrainVocabulary
    from scripts.improvement import train_compact_raw as raw

    class CompactFixture(torch.nn.Module):
        """Fast tiny compact-model interface with the legacy zero-init text head."""

        def __init__(self, text_dim=0):
            super().__init__()
            self.price_encoder = torch.nn.Linear(4, 4)
            self.price_head = torch.nn.Linear(4, 3)
            self.text_head = torch.nn.Linear(text_dim, 3) if text_dim else None
            if self.text_head is not None:
                torch.nn.init.zeros_(self.text_head.weight)
                torch.nn.init.zeros_(self.text_head.bias)

        def freeze_price(self):
            self.price_encoder.requires_grad_(False)
            self.price_head.requires_grad_(False)

        def forward(self, x):
            return self.price_head(self.price_encoder(x.mean(-1)))


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class RawTrainingTest(unittest.TestCase):
    def _run_frozen(self):
        torch.manual_seed(42)
        vocab = TrainVocabulary.fit(["alpha reports growth", "beta rates fall"], max_tokens=5)
        reference = raw.RawForecast(CompactFixture()).eval().requires_grad_(False)
        model = raw.RawForecast(CompactFixture(4), RawTextEncoder(len(vocab), 4, 5))
        model.compact.price_encoder.load_state_dict(reference.compact.price_encoder.state_dict())
        model.compact.price_head.load_state_dict(reference.compact.price_head.state_dict())
        model.compact.freeze_price()
        x = torch.randn(8, 4, 35)
        texts = ["alpha reports growth", "beta rates fall"] * 4
        tokens = torch.tensor([vocab.encode(t) for t in texts])
        y = raw.predict(reference, x, "cpu") + 2.0
        before = raw.state_cpu(model)
        before_price = raw.predict(reference, x, "cpu")
        args = SimpleNamespace(device="cpu", seed=0, batch_size=2, epochs=2, mode="residual")
        with contextlib.redirect_stdout(io.StringIO()):
            epochs, best_epoch, saw_grad = raw.fit(model, (x, y), (x, y), tokens, tokens, args)
        self.assertEqual(epochs, 2)
        self.assertGreater(best_epoch, 0)
        self.assertTrue(saw_grad)
        self.assertFalse(torch.equal(before["encoder.embedding.weight"], model.encoder.embedding.weight))
        for name, param in model.named_parameters():
            if name.startswith(("compact.price_encoder.", "compact.price_head.")):
                self.assertFalse(param.requires_grad)
                self.assertIsNone(param.grad)
                self.assertTrue(torch.equal(before[name], param))
        self.assertTrue(torch.equal(raw.predict(model, x, "cpu"), before_price))
        blank = torch.zeros_like(tokens)
        self.assertTrue(torch.equal(raw.predict(model, x, "cpu", blank), before_price))
        learned = raw.predict(model, x, "cpu", tokens)
        self.assertFalse(torch.equal(learned, before_price))
        return learned

    def test_reproducible_residual_gradients_and_exact_frozen_price(self):
        self.assertTrue(torch.equal(self._run_frozen(), self._run_frozen()))

    def test_empty_text_masks_even_nonzero_head_bias(self):
        model = raw.RawForecast(CompactFixture(2), RawTextEncoder(4, 2, 3))
        with torch.no_grad():
            model.compact.text_head.bias.fill_(5)
        x = torch.zeros(2, 4, 35)
        ids = torch.tensor([[0, 0], [2, 0]])
        price = model(x)
        output = model(x, ids)
        self.assertTrue(torch.equal(output[0], price[0]))
        torch.testing.assert_close(output[1], price[1] + 5)

    def test_close_return_endpoints_and_training_scale(self):
        history = torch.arange(140).reshape(35, 4)
        future = torch.arange(560).reshape(140, 4)

        class Dataset:
            list = [(None,) * 6 + (history, future)]
            def __len__(self):
                return len(self.list)

        x, y = raw.price_tensors(Dataset(), 2.0)
        self.assertEqual(tuple(x.shape), (1, 4, 35))
        torch.testing.assert_close(x[0], history.T.float() / 2)
        torch.testing.assert_close(y, torch.tensor([[139.0, 279.0, 559.0]]) / 2)

    def test_c1_snapshot_rejects_noncanonical_without_override_and_never_changes_split(self):
        ids = {"train": ["A/20200201.txt"], "vali": ["A/20210201.txt"], "test": ["B/20220201.txt"]}
        manifest = {role + "_events": values for role, values in ids.items()}
        counts = {key: 1 for key in ids}
        protocol = ModuleType("data.c1_protocol")
        freezer = ModuleType("scripts.freeze_c1_inputs")
        preparer = ModuleType("scripts.improvement.prepare_raw_inputs")
        preparer.MANIFEST_SHA256 = "manifest-fixture"
        loaded = SimpleNamespace(audit={"emitted_counts": counts})
        for role, keys in ids.items():
            setattr(loaded, role + "_set", SimpleNamespace(event_ids=keys))
        protocol.event_key = lambda path: (None, None, path)
        passed = []

        def loader(*paths, **kwargs):
            passed.append((paths, kwargs))
            return loaded

        protocol.build_c1_dataloaders = loader
        freezer.verify = lambda _: {"content_sha256": "not-canonical",
                                    "manifest_sha256": preparer.MANIFEST_SHA256}
        with TemporaryDirectory() as directory:
            base = Path(directory)
            fixed = base / "fixed.json"
            fixed.write_text(json.dumps(manifest), encoding="utf-8")
            (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.dict(sys.modules, {
                "data.c1_protocol": protocol,
                "scripts.freeze_c1_inputs": freezer,
                "scripts.improvement.prepare_raw_inputs": preparer,
            }), \
                    patch.object(raw, "MANIFEST", fixed), patch.object(raw, "COUNTS", counts):
                with self.assertRaisesRegex(ValueError, "Snapshot differs"):
                    raw.load_c1(base, 0, False)
                self.assertFalse(passed)
                data, sha, _ = raw.load_c1(base, 0, True)
                self.assertIs(data, loaded)
                self.assertEqual(sha, "not-canonical")
                self.assertEqual(passed[-1][1]["timing_policy"], raw.TIMING)
                self.assertEqual(passed[-1][1]["task_mode"], "return_bp")
                self.assertFalse(passed[-1][1]["skip_missing"])
                loaded.vali_set.event_ids = ["A/20210202.txt"]
                with self.assertRaisesRegex(ValueError, r"article IDs|population|ordering"):
                    raw.load_c1(base, 0, True)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class DonorControlTest(unittest.TestCase):
    def test_donor_pool_excludes_same_date_duplicates_and_other_year_category(self):
        events = [
            "A/20191231-first.txt", "B/20200101-first.txt", "A/20200101-early.txt",
            "A/20200102-same.txt", "A/20200102-other.txt", "A/20200103-last.txt",
        ]
        texts = ["prev year", "other category", "First Article", " first  ARTICLE ", "different", "last"]
        hashes = [raw.article_hash(t) for t in texts]
        samples = raw.past_donors(events, hashes, 23)
        self.assertEqual(samples[:5].tolist(), [-1, -1, -1, -1, 2])
        self.assertIn(int(samples[5]), [2, 3, 4])
        np.testing.assert_array_equal(samples, raw.past_donors(events, hashes, 23))

    def test_report_uses_matched_donor_subset_and_counts_coverage(self):
        events = ["A/20200101-a.txt", "A/20200102-b.txt", "A/20200103-c.txt"]
        texts = ["start", "NEXT", " start "]
        ids = torch.tensor([[2], [3], [2]])
        model = raw.RawForecast(CompactFixture(4), RawTextEncoder(4, 4))
        x = torch.zeros(3, 4, 35)
        zeros = np.zeros((3, 3), dtype=np.float32)
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), device="cpu")
            outcome = raw.controls(model, x, zeros, zeros, zeros, ids, texts, events, args, 1.0)
            self.assertEqual(outcome["events"], 2)
            self.assertEqual(outcome["excluded_no_donor"], 1)
            self.assertAlmostEqual(outcome["coverage"], 2 / 3)
            with np.load(args.out / "vali-text-controls.npz") as archive:
                self.assertEqual(archive["event_ids"].tolist(), events[1:])
                self.assertEqual(archive["donor_event_ids"].tolist(), [[events[0], events[1]]] * 3)


if __name__ == "__main__":
    unittest.main()
