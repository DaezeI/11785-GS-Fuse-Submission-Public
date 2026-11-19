"""Synthetic output, target, metric and gradient checks for compact OHLC paths."""
import unittest

import numpy as np

from scripts.improvement.compact_path_metrics import metrics

try:
    import torch
except ModuleNotFoundError as err:
    if err.name != "torch":
        raise
    torch = None
else:
    from model.compact_raw_text import RawTextEncoder
    from scripts.improvement import train_compact_path as path


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class CompactPathTest(unittest.TestCase):
    def test_price_and_news_paths_are_zero_initialized_with_ohlc_shape(self):
        torch.manual_seed(0)
        model = path.PathForecast(path.CompactPath(width=4, text_dim=4), RawTextEncoder(6, 4, 4))
        x = torch.randn(2, 4, 35)
        ids = torch.tensor([[0, 0], [2, 3]])
        forecasts = (model(x), model(x, ids))
        for prediction in forecasts:
            self.assertEqual(tuple(prediction.shape), (2, 140, 4))
            torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=0, rtol=0)

    def test_frozen_price_matches_reference_and_news_embedding_learns(self):
        torch.manual_seed(23)
        baseline = path.PathForecast(path.CompactPath(width=4)).eval().requires_grad_(False)
        model = path.PathForecast(path.CompactPath(width=4, text_dim=4), RawTextEncoder(6, 4, 4))
        model.compact.price_encoder.load_state_dict(baseline.compact.price_encoder.state_dict())
        model.compact.price_head.load_state_dict(baseline.compact.price_head.state_dict())
        model.compact.freeze_price()
        x = torch.randn(3, 4, 35)
        ids = torch.tensor([[2, 3, 0], [3, 4, 0], [0, 0, 0]])
        expected = baseline(x)
        target = expected + 1.0
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

        # The zero-initialized legacy residual head delays encoder gradients one step.
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.smooth_l1_loss(model(x, ids), target).backward()
            optimizer.step()
        gradient = model.encoder.embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient[2:5].abs().sum().item(), 0)
        self.assertFalse(torch.equal(before["encoder.embedding.weight"], model.encoder.embedding.weight))
        for name, value in model.named_parameters():
            if name.startswith(("compact.price_encoder.", "compact.price_head.")):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad)
                self.assertTrue(torch.equal(before[name], value))

        self.assertTrue(torch.equal(model(x), expected))
        self.assertTrue(torch.equal(model(x, torch.zeros_like(ids)), expected))
        learned = model(x, ids)
        self.assertTrue(torch.equal(learned[2], expected[2]))
        self.assertFalse(torch.equal(learned[:2], expected[:2]))
        self.assertTrue(torch.equal(learned, model(x, ids)))

    def test_price_tensors_keep_all_rows_channels_and_train_scale(self):
        history = torch.arange(35 * 4).reshape(35, 4)
        future = torch.arange(140 * 4).reshape(140, 4) * 3

        class Dataset:
            list = [(None,) * 6 + (history, future), (None,) * 6 + (history + 1, future + 1)]

            def __len__(self):
                return len(self.list)

        x, y = path.price_tensors(Dataset(), 2)
        self.assertEqual(tuple(x.shape), (2, 4, 35))
        self.assertEqual(tuple(y.shape), (2, 140, 4))
        torch.testing.assert_close(x[0], history.T.float() / 2)
        torch.testing.assert_close(y[0], future.float() / 2)
        torch.testing.assert_close(y[1], (future + 1).float() / 2)

class CompactPathMetricsTest(unittest.TestCase):
    def test_path_mae_uses_nonclose_channels_and_interior_timesteps(self):
        truth = np.zeros((2, 140, 4), dtype=np.float32)
        forecasts = truth.copy()
        forecasts[0, 0, 0] = 28
        forecasts[0, 61, 1] = -56
        forecasts[1, 96, 2] = 84
        forecasts[:, 34, 3] = (4, -2)
        forecasts[:, 69, 3] = (5, -3)
        forecasts[:, 139, 3] = (7, -5)
        report = metrics(truth, forecasts)
        self.assertAlmostEqual(report["path_mae_bp"], (28 + 56 + 84 + 6 + 8 + 12) / (2 * 140 * 4))
        np.testing.assert_allclose(report["close_endpoint_mae_bp"], [3, 4, 6])

    def test_direction_is_scored_on_nonzero_targets_and_shape_is_checked(self):
        truth = np.zeros((2, 140, 4), dtype=np.float32)
        forecasts = truth.copy()
        truth[:, 34, 3] = (0, 2)
        forecasts[:, 34, 3] = (0, 0)
        truth[:, 69, 3] = (4, -2)
        forecasts[:, 69, 3] = (4, 1)
        self.assertEqual(metrics(truth, forecasts)["close_endpoint_direction_hit_rate"], [0.0, 0.5, None])
        with self.assertRaises(ValueError):
            metrics(truth, forecasts[:, :, 3])


if __name__ == "__main__":
    unittest.main()
