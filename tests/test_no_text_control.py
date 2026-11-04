"""CPU torch tests: no-text predictions cannot depend on real article features."""
import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for the no-text CPU tests")
class NoTextControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("control", Path(__file__).resolve().parents[1] / "model/no_text_control.py")
        cls.control = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.control)

    def model(self):
        class PriceModel:
            device = torch.device("cpu")
            # Simulates a cached real-text analysis that must never be used.
            real_text_features = torch.ones(2, 4)

            def eval(self): return self
            def _ensure_cf(self, x): return x
            def get_ts_tokens_full(self, x):
                tok = x.transpose(1, 2)
                return tok, torch.zeros(tok.shape[:2], dtype=torch.bool), tok.mean(dim=1)
            def get_text_tokens_all(self, text):
                raise AssertionError("No-text control invoked the text encoder")
            def _fuse_from_sequences(self, text, ts, text_mask, ts_mask, ts_inst=None, **kwargs):
                self.last_text, self.last_mask = text, text_mask
                return ts_inst + text.mean(dim=1), {}
            def decoder(self, fused, x_cf=None):
                return fused[:, :, None].expand(-1, -1, 140)
            def _apply_output_refinement(self, core, fused, x_cf): return core, None
        return PriceModel()

    def test_zero_control_ignores_changed_real_text_state(self):
        model = self.model(); x = torch.arange(2 * 4 * 35).reshape(2, 4, 35).float()
        a = self.control.predict_without_article(model, x)
        model.real_text_features = torch.full((2, 4), 9999.)
        b = self.control.predict_without_article(model, x)
        torch.testing.assert_close(a, b)
        self.assertFalse(model.last_mask.any())
        self.assertEqual(tuple(model.last_text.shape), (2, 1, 4))
        self.assertFalse(model.last_text.any())
        self.assertFalse(a.requires_grad)

    def test_no_text_prediction_still_responds_to_price(self):
        model = self.model()
        a = self.control.predict_without_article(model, torch.ones(2, 4, 35))
        b = self.control.predict_without_article(model, torch.full((2, 4, 35), 2.))
        self.assertFalse(torch.equal(a, b))

    def test_bypass_mode_does_not_invoke_fusion(self):
        model = self.model()
        def forbidden(*args, **kwargs):raise AssertionError("bypass invoked fusion")
        model._fuse_from_sequences = forbidden
        pred = self.control.predict_without_article(model, torch.ones(2, 4, 35), "fusion_bypass_ts_inst")
        self.assertEqual(tuple(pred.shape), (2, 4, 140))

    def test_unknown_mode_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.control.predict_without_article(self.model(), torch.ones(2, 4, 35), "silent-fallback")


if __name__ == "__main__":
    unittest.main()
