# -*- coding: utf-8 -*-
"""Pluggable text encoders for GS-FUSE (LLaMA-family and Phi-family)."""

from __future__ import annotations

import os
import math
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


class LoRALinear(nn.Module):
    """Parameter-efficient low-rank adaptation layer wrapping a frozen linear base."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(base_layer.weight.new_empty(base_layer.in_features, rank))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros(rank, base_layer.out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Kaiming uniform init for A, zeros for B ensures Delta W = 0 at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base linear layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_term = (self.dropout(x) @ self.lora_A) @ self.lora_B * self.scaling
        return base_out + lora_term

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias


def inject_lora_adapters(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
) -> int:
    """Inject LoRALinear into named submodules of a causal LM. Returns adapted parameter count."""
    adapted_count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                sub = getattr(module, target)
                if isinstance(sub, nn.Linear) and not isinstance(sub, LoRALinear):
                    wrapped = LoRALinear(sub, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, target, wrapped)
                    adapted_count += (wrapped.lora_A.numel() + wrapped.lora_B.numel())
    return adapted_count


def _load_causal_lm(name: str, use_4bit: bool = False):
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        # GSFuse places its own modules on cuda-or-cpu; "auto" would pick MPS/disk offload on non-CUDA hosts.
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if use_4bit else torch.float32,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config if use_4bit else None,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return tok, model


class BaseTextEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def encode_tokens(self, text: str, window: int, stride: int, max_text_tokens: int, device: torch.device):
        """Return hidden token features [1, T, hidden] and mask [1, T]."""

    @abstractmethod
    def encode_instance(self, text: str, window: int, stride: int, device: torch.device) -> torch.Tensor:
        """Return pooled hidden vector [hidden]."""

    @abstractmethod
    def last_block_parameters(self):
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        ...

    @property
    @abstractmethod
    def llm(self):
        ...

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self) -> None:
        self.freeze()
        for p in self.last_block_parameters():
            p.requires_grad = True


class HuggingFaceTextEncoder(BaseTextEncoder):
    """Shared chunked hidden-state extraction for causal LM text backbones."""

    def __init__(
        self,
        model_path: str,
        family: str = "llama",
        use_4bit: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.family = family.lower()
        self._tokenizer, self.backbone = _load_causal_lm(model_path, use_4bit=use_4bit)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        cfg = getattr(self.backbone, "config", None)
        self._output_dim = int(getattr(cfg, "hidden_size", 3072))

        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        if use_lora:
            # Keep downstream shared-module initialization aligned with the
            # frozen arm at the same seed despite adapter initialization.
            with torch.random.fork_rng():
                self.adapted_params_count = inject_lora_adapters(
                    self.backbone, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
                )
            print(f"[LoRA] Injected LoRA adapters into text backbone: {self.adapted_params_count:,} trainable parameters.")
        else:
            self.adapted_params_count = 0

        # Persistent disk cache directory (RAM dictionaries omitted to avoid memory exhaustion)
        self.cache_dir = os.path.join("data", f"cache_{self.family}_emb")
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def llm(self):
        return self.backbone

    def lora_parameters(self) -> List[nn.Parameter]:
        """Return trainable LoRA adapter parameters."""
        if not self.use_lora:
            return []
        return [p for name, p in self.named_parameters()
                if name.endswith(("lora_A", "lora_B"))]

    def enable_lora_training(self):
        """Restore adapters after a stage freezes all model parameters."""
        self.freeze()
        for parameter in self.lora_parameters():
            parameter.requires_grad_(True)

    @property
    def cache_allowed(self):
        # Adapted weights change between updates/checkpoints, including eval.
        return not self.use_lora and not any(p.requires_grad for p in self.backbone.parameters())

    def _window_pad_mask(self, text, window, stride, max_text_tokens, device):
        """Padding mask aligned with the windowed hidden states. True = PAD."""
        ids = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_text_tokens,
            add_special_tokens=True,
            padding="max_length",
        )
        attn = ids["attention_mask"][0]
        pieces, start = [], 0
        while start < attn.size(0):
            pieces.append(attn[start:min(start + window, attn.size(0))])
            start += stride
        pad = torch.cat(pieces) == 0
        return pad.unsqueeze(0).to(device)

    def _forward_chunk(self, chunk_ids, chunk_mask):
        live = self.training and any(p.requires_grad for p in self.backbone.parameters())
        grad_context = torch.enable_grad() if live else torch.no_grad()
        with grad_context:
            out = self.backbone.model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            return out.hidden_states[-1][0].float()

    def encode_instance(self, text: str, window: int, stride: int, device: torch.device) -> torch.Tensor:
        h_key = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

        # 1. Check Disk cache (only if NOT training with LoRA)
        if self.cache_allowed:
            disk_path = os.path.join(self.cache_dir, f"{h_key}_inst.pt")
            if os.path.exists(disk_path):
                try:
                    emb = torch.load(disk_path, map_location=device, weights_only=True)
                    return emb.to(device)
                except Exception:
                    pass

        # 2. Compute via LLM
        ids = self._tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        input_ids = ids["input_ids"][0]
        attn_mask_full = torch.ones_like(input_ids)
        embeddings = []
        start = 0
        while start < input_ids.shape[0]:
            end = min(start + window, input_ids.shape[0])
            chunk_ids = input_ids[start:end].unsqueeze(0).to(device)
            chunk_mask = attn_mask_full[start:end].unsqueeze(0).to(device)
            h = self._forward_chunk(chunk_ids, chunk_mask)
            embeddings.append(h.mean(dim=0, keepdim=True))
            start += stride
        if not embeddings:
            res = torch.zeros(self._output_dim, device=device)
        else:
            res = torch.mean(torch.cat(embeddings, dim=0), dim=0, keepdim=False).to(device)

        # 3. Save to Disk (only if not training with LoRA)
        if self.cache_allowed:
            try:
                torch.save(res.cpu(), disk_path)
            except Exception:
                pass
        return res

    def encode_tokens(
        self,
        text: str,
        window: int,
        stride: int,
        max_text_tokens: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_key = hashlib.md5(f"{text}_{max_text_tokens}".encode("utf-8", errors="ignore")).hexdigest()

        # 1. Check Disk cache (only if NOT training with LoRA)
        if self.cache_allowed:
            disk_path = os.path.join(self.cache_dir, f"{h_key}_tok.pt")
            if os.path.exists(disk_path):
                try:
                    # Cached masks were saved as all False; rebuild the real one from the tokenizer.
                    data = torch.load(disk_path, map_location=device, weights_only=True)
                    tok_h = data["tok_h"].to(device)
                    mask = self._window_pad_mask(text, window, stride, max_text_tokens, device)
                    if mask.size(1) != tok_h.size(1):
                        raise ValueError(
                            f"Cached tokens ({tok_h.size(1)}) and rebuilt mask ({mask.size(1)}) "
                            "have different lengths; the cache may be stale."
                        )
                    return tok_h, mask
                except Exception as e:
                    print(f"[text cache] Ignoring cached tokens for {h_key}: {e}")

        # 2. Compute via LLM
        ids = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_text_tokens,
            add_special_tokens=True,
            padding="max_length",
        )
        input_ids = ids["input_ids"][0].to(device)
        attn_mask_full = ids["attention_mask"][0].to(device)

        feats = []
        start = 0
        while start < input_ids.size(0):
            end = min(start + window, input_ids.size(0))
            chunk_ids = input_ids[start:end].unsqueeze(0)
            chunk_mask = attn_mask_full[start:end].unsqueeze(0)
            feats.append(self._forward_chunk(chunk_ids, chunk_mask))
            start += stride

        if len(feats) == 0:
            tokens = torch.zeros(1, 1, self._output_dim, device=device)
            mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            return tokens, mask

        tok_h = torch.cat(feats, dim=0).unsqueeze(0)
        mask = self._window_pad_mask(text, window, stride, max_text_tokens, device)

        # 3. Save to Disk (only if not training with LoRA)
        if self.cache_allowed:
            try:
                torch.save({"tok_h": tok_h.cpu(), "mask": mask.cpu()}, disk_path)
            except Exception:
                pass

        return tok_h, mask

    def last_block_parameters(self):
        llm = self.backbone
        if llm is None:
            return []

        for path in [
            ("model", "layers"),
            ("transformer", "h"),
            ("model", "decoder", "layers"),
        ]:
            cur = llm
            ok = True
            for seg in path:
                if hasattr(cur, seg):
                    cur = getattr(cur, seg)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__len__") and len(cur) > 0:
                params = list(cur[-1].parameters())
                if params:
                    return params

        decoder_layer_types = []
        if self.family == "llama":
            try:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                decoder_layer_types.append(LlamaDecoderLayer)
            except ImportError:
                pass
        elif self.family == "phi":
            for mod_name in (
                "transformers.models.phi.modeling_phi",
                "transformers.models.phi3.modeling_phi3",
            ):
                try:
                    mod = __import__(mod_name, fromlist=["PhiDecoderLayer", "Phi3DecoderLayer"])
                    for attr in ("PhiDecoderLayer", "Phi3DecoderLayer"):
                        if hasattr(mod, attr):
                            decoder_layer_types.append(getattr(mod, attr))
                except ImportError:
                    continue

        for layer_type in decoder_layer_types:
            candidates = [m for m in llm.modules() if isinstance(m, layer_type)]
            if candidates:
                return list(candidates[-1].parameters())

        print("[GS-FUSE] Warning: Could not find text encoder last block parameters")
        return []


class LlamaTextEncoder(HuggingFaceTextEncoder):
    def __init__(
        self,
        model_path: str,
        use_4bit: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__(
            model_path,
            family="llama",
            use_4bit=use_4bit,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )


class PhiTextEncoder(HuggingFaceTextEncoder):
    def __init__(
        self,
        model_path: str,
        use_4bit: bool = False,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__(
            model_path,
            family="phi",
            use_4bit=use_4bit,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )


def build_text_encoder(
    backbone: str,
    model_path: str,
    use_4bit: bool = False,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> BaseTextEncoder:
    backbone = backbone.lower()
    if backbone == "llama":
        return LlamaTextEncoder(
            model_path,
            use_4bit=use_4bit,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    if backbone == "phi":
        return PhiTextEncoder(
            model_path,
            use_4bit=use_4bit,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    raise ValueError(f"Unknown text_backbone: {backbone}")
