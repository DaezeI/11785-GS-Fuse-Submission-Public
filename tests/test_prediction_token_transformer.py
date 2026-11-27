"""Shared-query, clock and gradient contracts for direct multi-horizon output."""
import torch
from model.prediction_token_transformer import PredictionTokenTransformer


def inputs(batch=2):
    return torch.randn(batch, 4, 35), torch.arange(-175., 0., 5.).expand(batch, -1)


def test_one_shared_query_supports_six_or_other_future_positions():
    model = PredictionTokenTransformer(dropout=0).eval()
    x, t = inputs()
    assert model.prediction_token.shape == (1, 1, 128)
    assert model.prediction_token.numel() == 128
    assert model(x, t).shape == (2, 6)
    assert model(x, t, torch.tensor([20., 30.])).shape == (2, 2)
    assert model.readout.in_features == 128 and model.readout.out_features == 1


def test_prediction_loss_reaches_shared_query_and_price_embedding():
    model = PredictionTokenTransformer(dropout=0)
    x, t = inputs()
    (model(x, t) - torch.arange(6.).expand(2, -1)).square().mean().backward()
    assert model.prediction_token.grad.abs().sum() > 0
    assert model.input_embedding.weight.grad.abs().sum() > 0
    assert all(layer.self_attn.in_proj_weight.grad.abs().sum() > 0 for layer in model.layers)


def test_history_states_do_not_depend_on_future_queries():
    model = PredictionTokenTransformer(dropout=0).eval()
    x, t = inputs()
    before = model.encode(x, t)[:, :35]
    after = model.encode(x, t, torch.tensor([100., 500., 1000.]))[:, :35]
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)


def test_checkpointing_and_batch_tiling_preserve_outputs_and_gradients():
    normal = PredictionTokenTransformer(dropout=0).train()
    memory = PredictionTokenTransformer(dropout=0, activation_checkpointing=True,
                                        attention_batch_chunk=1).train()
    memory.load_state_dict(normal.state_dict())
    x, t = inputs()
    a, b = normal(x, t), memory(x, t)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    a.square().mean().backward();b.square().mean().backward()
    for left, right in zip(normal.parameters(), memory.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)
