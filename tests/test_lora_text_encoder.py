# -*- coding: utf-8 -*-
"""Unit tests for native PyTorch LoRA adapter injection and gradient flow."""

import math
import unittest
import torch
import torch.nn as nn

from model.text_encoders import LoRALinear, inject_lora_adapters


class DummyAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(x.size(-1)), dim=-1)
        return self.out_proj(attn @ v)


class TestLoRAAdapters(unittest.TestCase):
    def test_zero_initialization(self):
        """LoRA initial delta must be zero so initial model predictions are identical."""
        base = nn.Linear(32, 32)
        x = torch.randn(4, 10, 32)
        expected = base(x)

        lora_layer = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        actual = lora_layer(x)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_gradient_isolation(self):
        """Backpropagation must update LoRA parameters while base weights receive zero gradients."""
        block = DummyAttentionBlock(hidden_dim=32)
        num_adapted = inject_lora_adapters(block, rank=4, alpha=8.0, target_modules=("q_proj", "v_proj"))
        self.assertGreater(num_adapted, 0)

        # Base layer weights must be frozen
        self.assertFalse(block.q_proj.base_layer.weight.requires_grad)
        self.assertFalse(block.v_proj.base_layer.weight.requires_grad)

        # LoRA parameters must be trainable
        self.assertTrue(block.q_proj.lora_A.requires_grad)
        self.assertTrue(block.q_proj.lora_B.requires_grad)
        self.assertTrue(block.v_proj.lora_A.requires_grad)
        self.assertTrue(block.v_proj.lora_B.requires_grad)

        x = torch.randn(2, 5, 32)
        out = block(x)
        loss = out.sum()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(block.q_proj.lora_A.grad)
        self.assertIsNotNone(block.q_proj.lora_B.grad)
        self.assertIsNone(block.q_proj.base_layer.weight.grad)

    def test_state_dict_save_and_reload(self):
        """Verify that adapted weights can be saved and reloaded."""
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)

        # Simulate optimization step by shifting lora_B
        with torch.no_grad():
            lora.lora_B.add_(0.5)

        x = torch.randn(2, 16)
        out1 = lora(x)

        state = lora.state_dict()
        new_base = nn.Linear(16, 16)
        new_base.load_state_dict(base.state_dict())
        new_lora = LoRALinear(new_base, rank=2, alpha=4.0, dropout=0.0)
        new_lora.load_state_dict(state)

        out2 = new_lora(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_stage3_unfreeze_and_optimizer_step(self):
        """Verify that global freeze + unfreeze preserves base freeze and updates only LoRA adapters."""
        block = DummyAttentionBlock(hidden_dim=32)
        inject_lora_adapters(block, rank=4, alpha=8.0, dropout=0.0)

        # 1. Global freeze (what train() does at start of Stage 3)
        for p in block.parameters():
            p.requires_grad = False

        # 2. Unfreeze LoRA adapters
        for m in block.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True

        # Verify requires_grad status
        self.assertTrue(block.q_proj.lora_A.requires_grad)
        self.assertTrue(block.q_proj.lora_B.requires_grad)
        self.assertFalse(block.q_proj.base_layer.weight.requires_grad)

        # Record initial weights
        q_base_before = block.q_proj.base_layer.weight.detach().clone()
        q_lora_A_before = block.q_proj.lora_A.detach().clone()
        q_lora_B_before = block.q_proj.lora_B.detach().clone()

        # 3. Optimization step
        optimizer = torch.optim.AdamW([block.q_proj.lora_A, block.q_proj.lora_B], lr=1e-2)
        x = torch.randn(2, 5, 32)
        out = block(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # 4. Verify base layer weights are UNCHANGED while LoRA weights CHANGED
        q_base_after = block.q_proj.base_layer.weight.detach()
        q_lora_A_after = block.q_proj.lora_A.detach()
        q_lora_B_after = block.q_proj.lora_B.detach()

        base_delta = (q_base_after - q_base_before).abs().max().item()
        lora_A_delta = (q_lora_A_after - q_lora_A_before).abs().max().item()
        lora_B_delta = (q_lora_B_after - q_lora_B_before).abs().max().item()

        self.assertEqual(base_delta, 0.0, "Base linear layer weights changed during optimization!")
        self.assertGreater(lora_A_delta, 0.0, "LoRA A weights failed to update!")
        self.assertGreater(lora_B_delta, 0.0, "LoRA B weights failed to update!")


if __name__ == "__main__":
    unittest.main()
