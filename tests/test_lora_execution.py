from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from model import text_encoders
from model.gs_fuse import GSFuse


class Tokenizer:
    def __call__(self, text, **kwargs):
        length = kwargs.get("max_length", 3)
        ids = torch.zeros(1, length, dtype=torch.long)
        ids[:, :min(length, 3)] = torch.tensor([1, 2, 3])[:min(length, 3)]
        return dict(input_ids=ids, attention_mask=(ids != 0).long())


def encoder(monkeypatch, tmp_path):
    # Real transformers Llama attention with random tiny weights; no download.
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    backbone = LlamaForCausalLM(config)
    monkeypatch.setattr(text_encoders, "_load_causal_lm", lambda *a, **kw: (Tokenizer(), backbone))
    monkeypatch.chdir(tmp_path)
    result = text_encoders.LlamaTextEncoder("tiny-local", use_lora=True, lora_rank=2,
                                          lora_alpha=4, lora_dropout=0)
    result.cache_dir = str(tmp_path)
    return result


def test_adapter_unfreezing_live_projection_gradients_and_base_isolation(monkeypatch, tmp_path):
    text = encoder(monkeypatch, tmp_path)
    text.freeze()  # Reproduce the trainer's stage transition.
    assert not any(p.requires_grad for p in text.parameters())
    text.enable_lora_training()
    model = GSFuse.__new__(GSFuse)
    nn.Module.__init__(model)
    model.text_encoder = text
    model.llm_proj = nn.Linear(16, 8)
    model.window = model.stride = model.max_text_tokens = 8
    model.device = torch.device("cpu")
    model.train()
    tokens, _ = model.get_text_tokens_all("article")
    tokens.square().sum().backward()
    assert model.llm_proj.weight.grad is not None
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for n, p in text.named_parameters()
               if n.endswith("lora_B"))
    assert all(p.grad is None for n, p in text.named_parameters()
               if not n.endswith(("lora_A", "lora_B")))


def test_eval_recomputes_features_after_adapter_update(monkeypatch, tmp_path):
    text = encoder(monkeypatch, tmp_path)
    text.eval()
    before = text.encode_instance("article", 8, 8, torch.device("cpu"))
    with torch.no_grad():
        for name, parameter in text.named_parameters():
            if name.endswith("lora_B"):
                parameter.add_(0.5)
    after = text.encode_instance("article", 8, 8, torch.device("cpu"))
    assert not text.cache_allowed
    assert not torch.allclose(before, after)
    assert not list(tmp_path.glob("*_inst.pt"))


def test_instance_helper_cannot_return_memoized_frozen_features(monkeypatch, tmp_path):
    text = encoder(monkeypatch, tmp_path)
    text.train()
    model = SimpleNamespace(text_encoder=text, window=8, stride=8, device=torch.device("cpu"),
                            text_inst_cache=SimpleNamespace(get=lambda _: pytest.fail("stale instance cache read")))
    result = GSFuse.text_embedding(model, "article")
    assert result.requires_grad


def test_lora_matches_base_device_dtype_and_rejects_zero_rank():
    layer = text_encoders.LoRALinear(nn.Linear(8, 8, dtype=torch.float64), rank=2)
    assert layer.lora_A.dtype == layer.weight.dtype == torch.float64
    assert layer(torch.ones(2, 8, dtype=torch.float64)).dtype == torch.float64
    with pytest.raises(ValueError, match="rank must be positive"):
        text_encoders.LoRALinear(nn.Linear(2, 2), rank=0)
