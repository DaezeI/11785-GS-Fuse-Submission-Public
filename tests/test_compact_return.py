import torch
from model.compact_return import CompactReturn, CausalBlock


def test_compact_model_has_small_state_and_zero_initial_forecast():
    model = CompactReturn()
    assert sum(p.numel() for p in model.parameters()) < 100_000
    assert torch.equal(model(torch.randn(3, 4, 140)), torch.zeros(3, 3))


def test_causal_block_does_not_look_at_later_input_positions():
    block = CausalBlock(4, 2)
    before = torch.randn(2, 4, 35)
    after = before.clone()
    after[:, :, 20:] += 100
    assert torch.equal(block(before)[:, :, :20], block(after)[:, :, :20])


def test_text_correction_starts_at_zero_and_cannot_update_frozen_price_baseline():
    model = CompactReturn(text_dim=384)
    model.freeze_price()
    x, text = torch.randn(4, 4, 35), torch.randn(4, 384)
    assert torch.equal(model(x), model(x, text))
    before = {n: p.clone() for n, p in model.named_parameters() if not n.startswith("text_head")}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    (model(x, text) - 1).square().mean().backward()
    optimizer.step()
    assert not torch.equal(model(x), model(x, text))
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
