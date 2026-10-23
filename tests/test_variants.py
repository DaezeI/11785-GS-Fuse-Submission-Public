import torch

from model.variants import MoEModalityGate, QFormerFusion


def test_moe_gate_is_a_modality_softmax_and_router_gets_gradients():
    torch.manual_seed(0)
    B, D, K = 6, 32, 4
    gate_mod = MoEModalityGate(D, num_experts=K, top_k=2)
    t, s = torch.randn(B, D), torch.randn(B, D)

    gate, info = gate_mod(t, s)

    assert gate.shape == (B, 2, D)
    assert torch.allclose(gate.sum(dim=1), torch.ones(B, D), atol=1e-5)
    assert info["moe_router_probs"].shape == (B, K)
    assert torch.allclose(info["moe_dispatch"].sum(dim=1), torch.ones(B), atol=1e-5)
    assert ((info["moe_dispatch"] > 0).sum(dim=1) == 2).all()
    assert info["moe_expert_text_open"].shape == (B, K)

    (gate[:, 0].mean() + info["moe_aux_loss"]).backward()
    assert gate_mod.router[-1].weight.grad.abs().sum() > 0


def test_qformer_shapes_mask_and_gradients():
    torch.manual_seed(0)
    B, Lt, D, Nq = 3, 20, 32, 5
    qf = QFormerFusion(D, num_queries=Nq, num_layers=2, num_heads=4, dropout=0.0)
    text = torch.randn(B, Lt, D, requires_grad=True)
    mask = torch.zeros(B, Lt, dtype=torch.bool)
    mask[:, 15:] = True

    out, w = qf(text, mask, cond=torch.randn(B, D), need_weights=True)

    assert out.shape == (B, Nq, D)
    assert w.shape == (B, Nq, Lt)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B, Nq), atol=1e-5)
    assert w[:, :, 15:].abs().max() < 1e-6

    out.sum().backward()
    assert qf.queries.grad.abs().sum() > 0
    assert text.grad[:, :15].abs().sum() > 0
