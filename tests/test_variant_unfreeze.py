import inspect

import torch.nn as nn

import model.gs_fuse as gs_fuse_mod
from model.gs_fuse import GSFuse, unfreeze_variant_modules
from model.variants import MoEModalityGate, QFormerFusion


def _shell(qformer=None, moe_gate=None):
    model = GSFuse.__new__(GSFuse)
    nn.Module.__init__(model)
    model.qformer = qformer
    model.moe_gate = moe_gate
    return model


def test_unfreeze_enables_qformer_and_moe_after_global_freeze():
    model = _shell(
        qformer=QFormerFusion(16, num_queries=4, num_layers=1, num_heads=2),
        moe_gate=MoEModalityGate(16, num_experts=2, top_k=1),
    )
    for p in model.parameters():          # what train() does first
        p.requires_grad = False
    unfreeze_variant_modules(model)
    assert all(p.requires_grad for p in model.qformer.parameters())
    assert all(p.requires_grad for p in model.moe_gate.parameters())


def test_unfreeze_is_a_noop_without_variants():
    model = _shell()
    unfreeze_variant_modules(model)       # baseline/R3: nothing to unfreeze, no error


def test_stage3_train_calls_unfreeze():
    assert "unfreeze_variant_modules(model)" in inspect.getsource(gs_fuse_mod.train)