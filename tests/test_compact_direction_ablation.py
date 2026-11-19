import unittest
import numpy as np

from scripts.improvement import train_compact_direction_ablation as ablation

try:
    import torch
except ImportError:
    torch = None


class PairedScoresTest(unittest.TestCase):
    def test_full_path_mae_and_nonzero_target_direction_reported(self):
        truth = np.zeros((2, 140, 4), dtype=np.float32)
        truth[0, 34, 3] = -2
        truth[1, 34, 3] = 2
        baseline = truth.copy()
        baseline[0, 34, 3] = 1
        changed = truth.copy()
        changed[0, 0, 0] = 4
        by_seed, mean = ablation.compare(truth, {
            ablation.MODES[0]: [baseline],
            ablation.MODES[1]: [changed],
        }, [2])
        self.assertEqual(by_seed[0]["seed"], 2)
        self.assertEqual(by_seed[0]["direction_eligible"], [2, 0, 0])
        self.assertAlmostEqual(mean["improvement_bp"], (3 - 4) / (2 * 140 * 4))
        self.assertEqual(mean["direction_hit_rate_change_pp"], [50.0, None, None])


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class DirectionLossTest(unittest.TestCase):
    def test_penalty_prefers_correct_sign_and_excludes_zero_targets(self):
        target = torch.zeros(2, 140, 4)
        target[0, 34, 3] = 3
        target[0, 69, 3] = -3
        good = torch.zeros_like(target)
        good[0, 34, 3] = 2
        good[0, 69, 3] = -2
        good[0, 139, 3] = 200  # Target is zero here; this logit cannot influence the penalty.
        good[0, 34, 0] = -200  # Open channel is outside the direction term.
        bad = -good
        self.assertLess(ablation.direction_penalty(good, target).item(),
                        ablation.direction_penalty(bad, target).item())
        good.requires_grad_(True)
        ablation.direction_penalty(good, target).backward()
        self.assertEqual(torch.count_nonzero(good.grad).item(), 2)
        self.assertLess(good.grad[0, 34, 3].item(), 0)
        self.assertGreater(good.grad[0, 69, 3].item(), 0)
        self.assertEqual(ablation.direction_penalty(bad, torch.zeros_like(target)).item(), 0.0)

    def test_augmented_objective_only_adds_fixed_direction_term(self):
        target = torch.full((1, 140, 4), 2.0)
        pred = torch.full_like(target, 1.0)
        base = ablation.objective(pred, target, ablation.MODES[0])
        augmented = ablation.objective(pred, target, ablation.MODES[1])
        self.assertAlmostEqual(base.item(), 0.5)
        self.assertAlmostEqual((augmented - base).item(), ablation.direction_penalty(pred, target).item(),
                               places=6)


if __name__ == "__main__":
    unittest.main()
