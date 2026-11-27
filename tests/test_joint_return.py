"""Joint price/text interaction and compatibility checks; no private inputs."""
import pytest
import torch
from model.compact_return import CompactReturn, JointResidualHead


def test_shared_decoder_can_make_text_effect_depend_on_price():
    head = JointResidualHead(1, 1, 1, text_width=1, hidden_width=1)
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head.text_projector[0].weight.fill_(1)
        head.joint_mlp[0].weight.fill_(1)
        head.joint_mlp[-1].weight.fill_(1)
    zero, one = torch.zeros(1, 1), torch.ones(1, 1)
    change_at_zero = head(zero, one) - head(zero, zero)
    change_at_one = head(one, one) - head(one, zero)
    assert abs((change_at_one - change_at_zero).item()) > .01


@pytest.mark.parametrize("depth", [1, 2])
def test_joint_learning_preserves_frozen_baseline_and_updates_both_parts(depth):
    base = CompactReturn(output_dim=2)
    with torch.no_grad():
        base.price_head[-1].weight.fill_(.01)
    arm = CompactReturn(text_dim=8, output_dim=2, fusion="joint_mlp", joint_hidden_layers=depth)
    missing, unexpected = arm.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("text_head.") for k in missing)
    arm.freeze_price()
    x = torch.linspace(-1, 1, 5 * 4 * 35).reshape(5, 4, 35)
    z = torch.linspace(-2, 2, 5 * 8).reshape(5, 8)
    expected = base(x).detach()
    assert torch.equal(arm(x, z), expected)
    original = {k: v.clone() for k, v in arm.state_dict().items() if not k.startswith("text_head.")}
    optimizer = torch.optim.AdamW(arm.text_head.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        (arm(x, z) - 1).square().mean().backward()
        optimizer.step()
    assert arm.text_head.text_projector[0].weight.grad.abs().sum() > 0
    assert arm.text_head.joint_mlp[0].weight.grad.abs().sum() > 0
    assert torch.equal(arm(x), expected)
    assert all(torch.equal(arm.state_dict()[k], v) for k, v in original.items())
    assert all(p.grad is None for p in arm.price_encoder.parameters())
    assert not torch.equal(arm(x, z), expected)


def test_existing_additive_checkpoint_loads_without_changes():
    # Parameter schema from CompactReturn before joint decoding was added.
    shapes = {"price_encoder.0.weight": (32, 4, 1), "price_encoder.0.bias": (32,),
              "price_head.0.weight": (64, 64), "price_head.0.bias": (64,),
              "price_head.2.weight": (2, 64), "price_head.2.bias": (2,),
              "text_head.0.weight": (32, 8), "text_head.0.bias": (32,),
              "text_head.2.weight": (2, 32), "text_head.2.bias": (2,)}
    for block in (2, 3, 4, 5):
        shapes[f"price_encoder.{block}.conv.weight"] = (32, 32, 3)
        shapes[f"price_encoder.{block}.conv.bias"] = (32,)
    state = {name: torch.full(shape, .01) for name, shape in shapes.items()}
    legacy = CompactReturn(text_dim=8, output_dim=2)
    legacy.load_state_dict(state, strict=True)
    x0, x1 = torch.zeros(3, 4, 35), torch.ones(3, 4, 35)
    t0, t1 = torch.zeros(3, 8), torch.ones(3, 8)
    delta0, delta1 = legacy(x0, t1) - legacy(x0, t0), legacy(x1, t1) - legacy(x1, t0)
    assert torch.allclose(delta0, delta1, atol=1e-6, rtol=1e-5)


def test_invalid_joint_configuration_is_rejected():
    with pytest.raises(ValueError):
        CompactReturn(fusion="joint_mlp")
    with pytest.raises(ValueError):
        CompactReturn(text_dim=8, fusion="joint_mlp", joint_hidden_layers=0)


def test_runner_factory_selects_joint_decoder_without_changing_price(tmp_path, monkeypatch):
    from scripts.run_fed_spx_text import frozen_arm
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    base = CompactReturn(output_dim=2)
    with torch.no_grad():
        base.price_head[-1].weight.fill_(.02)
    checkpoint = tmp_path / "price.pt"
    torch.save({"state_dict": base.state_dict()}, checkpoint)
    reference, arm, _ = frozen_arm(checkpoint, 8, fusion="joint_mlp", joint_hidden_layers=2)
    assert isinstance(arm.text_head, JointResidualHead)
    x, z = torch.ones(2, 4, 35), torch.ones(2, 8)
    assert torch.equal(reference(x), arm(x, z))
    assert all(not p.requires_grad for p in arm.price_encoder.parameters())
    assert all(p.requires_grad for p in arm.text_head.parameters())
