# -*- coding: utf-8 -*-
"""Interpretable GS-FUSE variants.

- MoEModalityGate: mixture of gating experts (top-k routing as in Mixtral /
  Switch Transformer). Each expert is a text-vs-TS fusion policy; the router
  picks policies per sample, so routing can be inspected against event types.
- QFormerFusion: learnable query tokens (BLIP-2 Q-Former / Flamingo Perceiver
  Resampler) that read the event text, conditioned on the market state. The
  query-to-token attention shows which words the model extracts.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEModalityGate(nn.Module):
    """Drop-in replacement for GSFuse.bi_gate_layer.

    Returns a modality gate of shape [B, 2, D] that sums to 1 over the modality
    axis, so the Granger supervision and gate-entropy terms apply unchanged.
    """

    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2, tau: float = 1.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.tau = tau

        self.router = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * dim),
                nn.Linear(2 * dim, dim),
                nn.GELU(),
                nn.Linear(dim, 2 * dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, t_vec: torch.Tensor, s_vec: torch.Tensor):
        B = t_vec.size(0)
        x = torch.cat([t_vec, s_vec], dim=-1)

        router_logits = self.router(x)                                    # [B, K]
        router_probs = F.softmax(router_logits, dim=-1)
        top_p, top_idx = torch.topk(router_probs, self.top_k, dim=-1)     # [B, k]
        top_w = top_p / top_p.sum(dim=-1, keepdim=True)
        dispatch = torch.zeros_like(router_probs).scatter(1, top_idx, top_w)  # [B, K]

        expert_logits = torch.stack([e(x) for e in self.experts], dim=1)  # [B, K, 2D]
        expert_gates = F.softmax(
            expert_logits.view(B, self.num_experts, 2, self.dim) / max(self.tau, 1e-6), dim=2
        )                                                                  # [B, K, 2, D]
        gate = (dispatch[:, :, None, None] * expert_gates).sum(dim=1)     # [B, 2, D]

        # Switch-Transformer load balancing: K * sum_k f_k * P_k
        frac_routed = F.one_hot(top_idx[:, 0], self.num_experts).float().mean(dim=0)
        mean_prob = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (frac_routed * mean_prob).sum()

        info = {
            "moe_router_probs": router_probs,
            "moe_dispatch": dispatch,
            "moe_expert_text_open": expert_gates[:, :, 0, :].mean(dim=-1),  # [B, K]
            "moe_aux_loss": aux_loss,
        }
        return gate, info


class _QFormerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_cross = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self, q, kv, kv_mask, need_weights: bool):
        h = self.ln_self(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        kv_n = self.ln_kv(kv)
        attn_out, w = self.cross_attn(
            self.ln_cross(q), kv_n, kv_n,
            key_padding_mask=kv_mask, need_weights=need_weights, average_attn_weights=True,
        )
        q = q + attn_out
        q = q + self.ffn(self.ln_ffn(q))
        return q, w


class QFormerFusion(nn.Module):
    """Compress variable-length text tokens into a fixed set of query tokens."""

    def __init__(self, dim: int, num_queries: int = 16, num_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.cond_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        self.layers = nn.ModuleList([_QFormerLayer(dim, num_heads, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        text_tokens: [B, Lt, D]; text_mask: [B, Lt] True = PAD; cond: [B, D] market-state vector.
        Returns query tokens [B, Nq, D] and last-layer query->text attention [B, Nq, Lt].
        """
        B = text_tokens.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        if cond is not None:
            q = q + self.cond_proj(cond).unsqueeze(1)
        w = None
        for layer in self.layers:
            q, w = layer(q, text_tokens, text_mask, need_weights)
        return self.out_norm(q), w
