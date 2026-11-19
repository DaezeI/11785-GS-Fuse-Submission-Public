import inspect
import unittest

import torch.nn as nn

import model.gs_fuse as gs_fuse_mod
from model.gs_fuse import GSFuse, unfreeze_variant_modules
from model.variants import MoEModalityGate, QFormerFusion
from model.text_encoders import LoRALinear


def _shell(qformer=None, moe_gate=None):
    model = GSFuse.__new__(GSFuse)
    nn.Module.__init__(model)
    model.qformer = qformer
    model.moe_gate = moe_gate
    return model


class TestVariantUnfreeze(unittest.TestCase):
    def test_unfreeze_enables_qformer_and_moe_after_global_freeze(self):
        model = _shell(
            qformer=QFormerFusion(16, num_queries=4, num_layers=1, num_heads=2),
            moe_gate=MoEModalityGate(16, num_experts=2, top_k=1),
        )
        for p in model.parameters():          # what train() does first
            p.requires_grad = False
        unfreeze_variant_modules(model)
        self.assertTrue(all(p.requires_grad for p in model.qformer.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.moe_gate.parameters()))

    def test_unfreeze_is_a_noop_without_variants(self):
        model = _shell()
        unfreeze_variant_modules(model)       # baseline/R3: nothing to unfreeze, no error

    def test_stage3_train_calls_unfreeze(self):
        self.assertIn("unfreeze_variant_modules(model)", inspect.getsource(gs_fuse_mod.train))

    def test_unfreeze_enables_lora_after_global_freeze(self):
        class DummyTextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_lora = True
                self.linear = nn.Linear(16, 16)
                self.lora_linear = LoRALinear(self.linear, rank=2, alpha=4.0)

            def unfreeze_lora(self):
                for p in self.parameters():
                    p.requires_grad = False
                self.lora_linear.lora_A.requires_grad = True
                self.lora_linear.lora_B.requires_grad = True

        model = _shell()
        model.text_encoder = DummyTextEncoder()
        for p in model.parameters():
            p.requires_grad = False
        unfreeze_variant_modules(model)
        self.assertTrue(model.text_encoder.lora_linear.lora_A.requires_grad)
        self.assertTrue(model.text_encoder.lora_linear.lora_B.requires_grad)
        self.assertFalse(model.text_encoder.lora_linear.base_layer.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()