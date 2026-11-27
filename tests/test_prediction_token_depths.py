import numpy as np
import pytest
import torch

from model.prediction_token_transformer import PredictionTokenTransformer
from scripts.compare_prediction_token_depths import validate_pairing


@pytest.mark.parametrize("depth", [4, 8, 12])
def test_all_depths_preserve_shared_query_and_backpropagate(depth):
    model = PredictionTokenTransformer(depth=depth, dropout=0,
                                      activation_checkpointing=True, attention_batch_chunk=1)
    x = torch.randn(2, 4, 35)
    t = torch.arange(-175., 0., 5.).expand(2, -1)
    prediction = model(x, t)
    assert prediction.shape == (2, 6)
    assert model.prediction_token.numel() == 128
    prediction.square().mean().backward()
    assert model.prediction_token.grad.abs().sum() > 0
    assert all(layer.self_attn.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)


def pair():
    return {"truth": np.zeros((40548, 6)), "forecast": np.zeros((40548, 6)),
            "anchor_times": np.arange(40548)}


def test_depth_comparison_accepts_identical_rows():
    validate_pairing(pair(), pair())


@pytest.mark.parametrize("key", ["truth", "anchor_times"])
def test_depth_comparison_rejects_changed_targets_or_times(key):
    first, second = pair(), pair()
    second[key].flat[-1] += 1
    with pytest.raises(ValueError, match="mismatched"):
        validate_pairing(first, second)
