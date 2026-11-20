"""Standalone GS-Fuse compact Treasury cross-attention and gate ablation.

Reads only the exact pinned train and validation splits from --data-dir. No embedded data.
Reuses GS-Fuse compact causal/text modules, with separate Treasury task dimensions.
"""
from __future__ import annotations
from collections import Counter
import os
import sys
from datetime import date, datetime, timezone, timedelta
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    import torch
except ImportError:
    sys.exit("PyTorch is missing. Use the GS-Fuse Python environment that previously ran the compact model.")

# ==== gsfuse_model/compact_return.py ====
import torch

from torch import nn

from torch.nn import functional as F

class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, x):
        return x + F.gelu(self.conv(F.pad(x, (self.padding, 0))))

class CompactReturn(nn.Module):
    def __init__(self, width=32, text_dim=0):
        super().__init__()
        self.price_encoder = nn.Sequential(nn.Conv1d(4, width, 1), nn.GELU(),
            *[CausalBlock(width, dilation) for dilation in (1, 2, 4, 8)])
        self.price_head = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 3))
        self.text_dim = text_dim
        self.text_head = (nn.Sequential(nn.Linear(text_dim, 32), nn.GELU(), nn.Linear(32, 3))
                          if text_dim else None)
        # Start from zero-return and zero text correction; both are useful controls.
        nn.init.zeros_(self.price_head[-1].weight)
        nn.init.zeros_(self.price_head[-1].bias)
        if self.text_head is not None:
            nn.init.zeros_(self.text_head[-1].weight)
            nn.init.zeros_(self.text_head[-1].bias)

    def freeze_price(self):
        for module in (self.price_encoder, self.price_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, history, text=None):
        if history.ndim != 3 or history.shape[1] != 4:
            raise ValueError("Expected history [batch, 4 OHLC channels, observed bars]")
        tokens = self.price_encoder(history)
        context = torch.cat((tokens[:, :, -1], tokens.mean(-1)), dim=-1)
        result = self.price_head(context)
        if text is not None:
            if self.text_head is None or text.shape != (len(history), self.text_dim):
                raise ValueError("Text features do not match the configured residual head")
            result = result + self.text_head(text)
        return result


# ==== gsfuse_model/compact_news_vocab.py ====
from collections import Counter

from collections.abc import Iterable, Iterator

from itertools import islice

import re

import unicodedata

PAD_ID = 0

UNK_ID = 1

_TOKENIZER_VERSION = "nfkc-casefold-word-v1"

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)

def _tokens(text: str) -> Iterator[str]:
    if not isinstance(text, str):
        raise TypeError("article text must be a string")
    clean = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return (match.group() for match in _WORD.finditer(clean))

class TrainVocabulary:
    """Immutable-by-interface vocabulary; only fit on the training articles."""

    def __init__(self, words: Iterable[str], max_tokens: int = 512) -> None:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self._max_tokens = max_tokens
        self._words = tuple(words)
        if any(not isinstance(word, str) or not word for word in self._words):
            raise ValueError("vocabulary terms must be nonempty strings")
        if len(set(self._words)) != len(self._words):
            raise ValueError("vocabulary terms must be unique")
        self._ids = {word: i for i, word in enumerate(self._words, start=2)}

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def __len__(self) -> int:
        return len(self._words) + 2

    @classmethod
    def fit(
        cls, training_articles: Iterable[str], max_vocab: int = 16_000, max_tokens: int = 512
    ) -> TrainVocabulary:
        if type(max_vocab) is not int or max_vocab < 2:
            raise ValueError("max_vocab must include PAD and UNK")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        counts: Counter[str] = Counter()
        for article in training_articles:
            counts.update(islice(_tokens(article), max_tokens))
        # Lexical tie breaks make IDs deterministic regardless of article order.
        words = sorted(counts, key=lambda word: (-counts[word], word))[:max_vocab - 2]
        return cls(words, max_tokens)

    def encode(self, text: str) -> list[int]:
        ids = [self._ids.get(word, UNK_ID) for word in islice(_tokens(text), self.max_tokens)]
        return ids + [PAD_ID] * (self.max_tokens - len(ids))

    def to_dict(self) -> dict:
        return {
            "version": _TOKENIZER_VERSION,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "max_tokens": self.max_tokens,
            "words": list(self._words),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TrainVocabulary:
        if (
            payload.get("version") != _TOKENIZER_VERSION
            or payload.get("pad_id") != PAD_ID
            or payload.get("unk_id") != UNK_ID
            or not isinstance(payload.get("words"), list)
        ):
            raise ValueError("unrecognized tokenizer checkpoint")
        return cls(payload["words"], payload.get("max_tokens"))


# ==== gsfuse_model/compact_raw_text.py ====
from torch import Tensor, nn

class RawTextEncoder(nn.Module):
    """[batch, tokens] -> [batch, output_dim]; no feature caching or pretrained weights."""

    def __init__(self, vocab_size: int, output_dim: int, embedding_dim: int = 64) -> None:
        super().__init__()
        if vocab_size < 2 or min(output_dim, embedding_dim) < 1:
            raise ValueError("invalid raw-text encoder dimensions")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        # Without bias an all-PAD article stays exactly zero after projection.
        self.projection = nn.Linear(embedding_dim, output_dim, bias=False)

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2:
            raise ValueError("token IDs must have shape [batch, tokens]")
        mask = ids.ne(PAD_ID).unsqueeze(-1)
        vectors = self.embedding(ids).masked_fill(~mask, 0.0)
        average = vectors.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(average)


# ==== compact_yield_model.py ====
import torch

from torch import nn

WIDTH = 32

TEXT_DIM = 64

HISTORY_DAYS = 20

INPUT_CHANNELS = 3

FORECAST_DAYS = 5

OUTPUT_CHANNELS = 1

class CompactYieldPath(CompactReturn):
    """Same compact causal encoder and heads; 3 input tenors and full 5-day path."""

    def __init__(self, width=WIDTH, text_dim=0):
        super().__init__(width, text_dim)
        self.price_encoder[0] = nn.Conv1d(INPUT_CHANNELS, width, 1)
        for head in (self.price_head, self.text_head):
            if head is not None:
                # Same resizing and zeros as CompactPath's full OHLC readout.
                head[-1] = nn.Linear(head[-1].in_features, FORECAST_DAYS * OUTPUT_CHANNELS)
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, history, text=None):
        if history.ndim != 3 or history.shape[1:] != (INPUT_CHANNELS, HISTORY_DAYS):
            raise ValueError(f"Expected history [batch, {INPUT_CHANNELS}, {HISTORY_DAYS}]")
        # Same operations as the hash-matched CompactReturn.forward.
        tokens = self.price_encoder(history)
        context = torch.cat((tokens[:, :, -1], tokens.mean(-1)), dim=-1)
        result = self.price_head(context)
        if text is not None:
            if self.text_head is None or text.shape != (len(history), self.text_dim):
                raise ValueError("Text features do not match the configured residual head")
            result = result + self.text_head(text)
        return result.reshape(len(history), FORECAST_DAYS, OUTPUT_CHANNELS)

class PathForecast(nn.Module):
    """CompactPath's frozen-price + trainable additive text residual wrapper."""

    def __init__(self, compact, encoder=None):
        super().__init__()
        if (encoder is None) != (compact.text_head is None):
            raise ValueError("Use a text encoder and residual head together")
        self.compact, self.encoder = compact, encoder

    def forward(self, prices, token_ids=None):
        base = self.compact(prices)
        if token_ids is None:
            return base
        if self.encoder is None:
            raise ValueError("Text IDs require a trainable encoder")
        if token_ids.ndim != 2 or len(token_ids) != len(prices):
            raise ValueError("Token IDs must have shape [batch, tokens] with the same batch as prices")
        offset = self.compact.text_head(self.encoder(token_ids)).reshape_as(base)
        return base + offset * token_ids.ne(PAD_ID).any(1).view(-1, 1, 1)


# ==== shared.py ====
PINS = {
    "train": ("5b958e6c15d5c051f2148ae366bb41fe46f4fbf3e17aa0c2cc7d4b8867d05e3c", 180, 36),
    "validation": ("9474f81626bf4b489fa9ab6f307e0fd2ba84d9c2b6d5522d43d98fdc2264feef", 60, 12),
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def timestamp(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.utcoffset() != timedelta(0):
        raise ValueError("Expected UTC")
    return d

def rows(raw, name, *, smoke):
    data = [json.loads(x) for x in raw.splitlines() if x]
    groups = {}
    dates, seen = set(), set()
    for r in data:
        gid = r["event_id"]
        if r["example_id"] in seen or sha(r["text"].encode()) != r["text_sha256"]:
            raise ValueError(f"{name}: duplicate row or mismatched text")
        h, t = r["input_dates"], r["target_dates"]
        if len(h) != 20 or len(t) != 5 or sorted(set(h+t)) != h+t or not h[-1] < t[0] == r["origin_date_ny"]:
            raise ValueError(f"{name}: invalid input/target dates")
        if timestamp(r["text_release_utc"]) >= timestamp(r["forecast_utc"]):
            raise ValueError(f"{name}: text comes after forecast origin")
        if (date.fromisoformat(t[0]) - timestamp(r["text_release_utc"]).date()).days < 1:
            raise ValueError(f"{name}: text and origin must be at least one date apart")
        if groups.setdefault(gid, r["text"]) != r["text"]:
            raise ValueError(f"{name}: mixed texts in a statement group")
        seen.add(r["example_id"])
        dates.update(h+t)
    if not smoke and (sha(raw), len(data), len(groups)) != PINS[name]:
        raise ValueError(f"{name}: not the exact frozen pilot split")
    if not groups or not dates:
        raise ValueError(f"{name}: empty")
    return sorted(data, key=lambda r: r["forecast_utc"]), groups, dates

def tensor(rows_):
    x = torch.tensor([r["input_yields_pct"] for r in rows_], dtype=torch.float32)
    y = torch.tensor([r["target_10y_change_bp"] for r in rows_], dtype=torch.float32)
    if x.shape != (len(rows_), 20, 3) or y.shape != (len(rows_), 5) or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Invalid numeric input or target")
    events = [r["event_id"] for r in rows_]
    return x, y.unsqueeze(-1), events

def event_weights(events):
    counts = Counter(events)
    return torch.tensor([len(events)/(len(counts)*counts[g]) for g in events], dtype=torch.float32)

def cpu_state(m):
    return {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}

@torch.no_grad()
def predict(m, x, tokens=None):
    m.eval()
    return m(x, tokens)

def fake_rows():
    files = {}
    for name, ids in (("train", range(4)), ("validation", range(4,7))):
        entries = []
        for i in ids:
            origin = date(2020,1,10) + timedelta(days=40*i)
            body = f"Synthetic policy statement {i}. Strong synthetic test data for the training code."
            for step in range(2):
                day = origin + timedelta(days=step)
                entries.append({"event_id":f"S{i}","example_id":f"S{i}_{step}",
                    "forecast_utc":f"{day}T17:00:00Z", "origin_date_ny":str(day),
                    "text_release_utc":f"{origin-timedelta(days=1)}T19:00:00Z",
                    "text":body,"text_sha256":sha(body.encode()),
                    "input_dates":[str(day-timedelta(days=k)) for k in range(20,0,-1)],
                    "target_dates":[str(day+timedelta(days=k)) for k in range(5)],
                    "input_yields_pct":[[3.0+.1*i+.2*j+.01*k+.005*step for j in range(3)] for k in range(20)],
                    "target_10y_change_bp":[(i+k+step)*(-1 if i%2 else 1) for k in range(5)]})
        files[name] = ("\n".join(json.dumps(x) for x in entries)+"\n").encode()
    return files


# ==== shared_r3.py ====
CFG = {
    "model": "compact-treasury-r3-adaptation-v1",
    "width": 32, "text_dim": 64, "embedding_dim": 64, "max_vocab": 16000, "max_tokens": 1024,
    "max_epochs": 10, "batch_size": 32, "lr": .001, "weight_decay": .001, "seeds": [23, 24, 25],
    "target_bp_scale": 10.0, "direction_endpoints": [1, 3, 5], "direction_loss_weight": 1.0,
    "gate_width": 16, "gate_supervision_weight": .1, "gate_gain_floor": 1e-6, "gate_logit_clip": 6.0,
    "teacher_warmup_events": 12, "teacher_block_events": 8,
    "direction": "mean endpoint logistic loss softplus(-sign(target)*forecast/10bp); nonzero targets only",
    "gate_teacher": "sigmoid(clamp((task_loss(price)-task_loss(price_plus_text))/max(batch_mean_abs_gain,1e-6),-6,6)), stop-grad; text-weight MSE",
    "selection": "all models and teachers fixed final epoch; no validation tuning",
}

def task_loss(pred, target, directional):
    magnitude = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=(1, 2))
    if not directional:
        return magnitude
    indices = [x-1 for x in CFG["direction_endpoints"]]
    scored = target[:, indices, 0]
    nonzero = scored.ne(0)
    sign_loss = F.softplus(-scored.sign() * pred[:, indices, 0]) * nonzero
    per = sign_loss.sum(1) / nonzero.sum(1).clamp_min(1)
    return magnitude + CFG["direction_loss_weight"] * per

def event_metrics(pred, target, events):
    if pred.shape != target.shape or not torch.isfinite(pred).all():
        raise ValueError("Nonfinite/malformed predictions")
    if not torch.isfinite(target).all():
        raise ValueError("Nonfinite targets")
    delta = (pred - target).squeeze(-1) * CFG["target_bp_scale"]
    ab, sq = delta.abs(), delta.square()
    nz = target[..., 0].ne(0)
    hit = pred[..., 0].sign().eq(target[..., 0].sign()) & nz
    ids = list(dict.fromkeys(events))
    indices = [[i for i, value in enumerate(events) if value == event] for event in ids]
    a = torch.stack([ab[ix].mean(0) for ix in indices])
    s = torch.stack([sq[ix].mean(0) for ix in indices])
    first = [ix[0] for ix in indices]
    ends = [k-1 for k in CFG["direction_endpoints"]]

    def event_dhr(selections, cols):
        out = []
        for ix in selections:
            n = nz[ix][:, cols].sum().item()
            if n:
                out.append(hit[ix][:, cols].sum().item() / n)
        return sum(out) / len(out) if out else None

    return {
        "mae_bp": a.mean().item(), "mse_bp_squared": s.mean().item(),
        "endpoint_dhr": event_dhr(indices, ends),
        "horizon_mae_bp": a.mean(0).tolist(), "horizon_mse_bp_squared": s.mean(0).tolist(),
        "horizon_dhr": [event_dhr(indices, [i]) for i in range(target.shape[1])],
        "endpoint_dhr_scored_nonzero_rows": int(nz[:, ends].sum().item()),
        "endpoint_dhr_scored_events": sum(bool(nz[ix][:, ends].any().item()) for ix in indices),
        "first_origin_mae_bp": ab[first].mean().item(), "first_origin_mse_bp_squared": sq[first].mean().item(),
        "first_origin_endpoint_dhr": event_dhr([[i] for i in first], ends),
        "event_mae_bp": dict(zip(ids, a.mean(1).tolist())),
        "event_mse_bp_squared": dict(zip(ids, s.mean(1).tolist())),
        "event_endpoint_dhr": {event: event_dhr([ix], ends) for event, ix in zip(ids, indices)},
    }

def paired_mae(metrics, left, right):
    a, b = metrics[left], metrics[right]
    return {"event_weighted_left_minus_right_bp": a["mae_bp"]-b["mae_bp"],
            "first_origin_left_minus_right_bp": a["first_origin_mae_bp"]-b["first_origin_mae_bp"],
            "by_event_left_minus_right_bp": {g: x-b["event_mae_bp"][g] for g,x in a["event_mae_bp"].items()}}

def setup_split(train, evaluation, device):
    x, y, ids = tensor(train)
    v, vy, vids = tensor(evaluation)
    center = x.mean((0, 1), keepdim=True)
    spread = x.std((0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    x, v = [((a-center)/spread).transpose(1, 2).contiguous().to(device) for a in (x, v)]
    y, vy = [a.to(device) / CFG["target_bp_scale"] for a in (y, vy)]
    docs = {r["event_id"]: r["text"] for r in train}
    vocab = TrainVocabulary.fit(docs.values(), max_vocab=CFG["max_vocab"], max_tokens=CFG["max_tokens"])
    xt = text_ids(vocab, [r["text"] for r in train], device)
    vt = text_ids(vocab, [r["text"] for r in evaluation], device)
    return (x, y, ids, xt), (v, vy, vids, vt), vocab, center, spread

def text_ids(vocab, values, device):
    return torch.tensor([vocab.encode(v) for v in values], dtype=torch.long, device=device)

def price_model(seed, device):
    torch.manual_seed(seed)
    return PathForecast(CompactYieldPath(width=CFG["width"])).to(device)

def light_metrics(pred, target, groups):
    metrics = event_metrics(pred, target, groups)
    return {k: metrics[k] for k in ("mae_bp", "mse_bp_squared", "endpoint_dhr")}

def fixed_fit(model, train, valid, *, seed, epochs, directional, text=False):
    x, y, groups, ids = train
    weights = event_weights(groups).to(x.device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    trace = []
    grad = False

    def observe(epoch):
        result = {"epoch": epoch, "train": light_metrics(predict(model, x, ids if text else None), y, groups)}
        if valid is not None:
            v, vy, vg, vt = valid
            result["validation"] = light_metrics(predict(model, v, vt if text else None), vy, vg)
        trace.append(result)

    if valid is not None:
        observe(0)
    for epoch in range(1, epochs + 1):
        model.train()
        if text:
            model.compact.price_encoder.eval()
            model.compact.price_head.eval()
        for ix in torch.randperm(len(x), generator=gen).split(CFG["batch_size"]):
            ix = ix.to(x.device)
            opt.zero_grad(set_to_none=True)
            loss = (task_loss(model(x[ix], ids[ix] if text else None), y[ix], directional) * weights[ix]).mean()
            loss.backward()
            if text and model.encoder.embedding.weight.grad is not None:
                grad |= bool(torch.count_nonzero(model.encoder.embedding.weight.grad[1:]).item())
            nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            opt.step()
        if valid is not None:
            observe(epoch)
    if text and not grad:
        raise ValueError("Text loss did not reach embeddings")
    return {"epochs_run": epochs, "curve": trace, **({"embedding_grad_verified": grad} if text else {})}

class SoftmaxGate(nn.Module):
    """Two experts (price and price+text); invariant inputs across OOF vocabularies."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, CFG["gate_width"]), nn.GELU(), nn.Linear(CFG["gate_width"], 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base, correction):
        logits = self.net(torch.cat((base.flatten(1), correction.flatten(1)), dim=-1))
        weights = logits.softmax(-1)
        return base + weights[:, 1, None, None] * correction, logits

@torch.no_grad()
def soft_route_target(base, correction, target):
    gain = task_loss(base, target, True) - task_loss(base+correction, target, True)
    scale = gain.abs().mean().clamp_min(CFG["gate_gain_floor"])
    return torch.sigmoid((gain / scale).clamp(-CFG["gate_logit_clip"], CFG["gate_logit_clip"]))

def train_gate(oof, *, seed, epochs):
    base, correction, target, groups = oof
    event_w = event_weights(groups).to(base.device)
    torch.manual_seed(seed + 2000)
    gen = torch.Generator().manual_seed(seed)
    model = SoftmaxGate().to(base.device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    curves = []
    for epoch in range(1, epochs+1):
        model.train()
        for ix in torch.randperm(len(base), generator=gen).split(CFG["batch_size"]):
            ix = ix.to(base.device)
            opt.zero_grad(set_to_none=True)
            pred, logits = model(base[ix], correction[ix])
            soft = soft_route_target(base[ix], correction[ix], target[ix])
            supervised = (logits.softmax(dim=1)[:, 1] - soft).square()
            loss = (event_w[ix] * (task_loss(pred, target[ix], True) + CFG["gate_supervision_weight"] * supervised)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        with torch.no_grad():
            oof_pred, logits = model(base, correction)
            curves.append({"epoch": epoch, "oof_fit": light_metrics(oof_pred, target, groups),
                           "event_weighted_mean_text_weight": (logits.softmax(-1)[:, 1]*event_w).mean().item()})
    return model.eval(), curves

def donors(train, validation):
    available = {r["event_id"]: (timestamp(r["text_release_utc"]), r["text"]) for r in train + validation}
    ordered = sorted(available, key=lambda group: available[group][0])
    prior = {new: old for old,new in zip(ordered, ordered[1:])}
    for r in validation:
        group = r["event_id"]
        if group not in prior or available[group][0] <= available[prior[group]][0]:
            raise ValueError("No strictly prior donor statement")
    return [available[prior[r["event_id"]]][1] for r in validation], {g: prior[g] for g in {r["event_id"] for r in validation}}

def assert_frozen(price, fusions, v, tokens):
    base = predict(price, v)
    for f in fusions:
        current = f.compact.state_dict()
        if any(not torch.equal(p,current[k]) for k,p in price.compact.state_dict().items()):
            raise ValueError("Frozen price state was altered")
        if not torch.equal(base, predict(f,v)) or not torch.equal(base,predict(f,v,torch.zeros_like(tokens))):
            raise ValueError("Absent-text prediction differs from frozen price reference")


# ==== oof_plan.py ====
from collections import defaultdict

def plan_oof(rows, warmup=12, block=8):
    if warmup < 2 or block < 1:
        raise ValueError("Invalid teacher fold design")
    groups = defaultdict(list)
    for ix, row in enumerate(rows):
        groups[row["event_id"]].append(ix)
    dates = {g: {d for i in inds for key in ("input_dates", "target_dates") for d in rows[i][key]}
             for g, inds in groups.items()}
    releases = {}
    for group, inds in groups.items():
        group_releases = {rows[i]["text_release_utc"] for i in inds}
        if len(group_releases) != 1:
            raise ValueError("Inconsistent release within statement")
        releases[group] = group_releases.pop()
    if len(set(releases.values())) != len(releases):
        raise ValueError("Duplicate statement release time")
    ids = sorted(groups, key=releases.__getitem__)
    folds = []
    for start in range(warmup, len(ids), block):
        predicted = ids[start:start+block]
        earliest = min(min(dates[g]) for g in predicted)
        past = [g for g in ids[:start] if max(dates[g]) < earliest]
        if len(past) < 2:
            raise ValueError("Too few teacher statements after purge")
        pt = [i for g in past for i in groups[g]]
        pe = [i for g in predicted for i in groups[g]]
        folds.append({"fit_events": past, "predict_events": predicted,
                      "purged_events": [g for g in ids[:start] if g not in past],
                      "fit_indices": pt, "predict_indices": pe,
                      "max_fit_market_date": max(max(dates[g]) for g in past),
                      "min_predict_market_date": earliest})
    if not folds:
        raise ValueError("No chronological folds")
    return folds


# ==== train_crossattn.py ====
import argparse


from datetime import datetime, timezone

import json

import math

import os

from pathlib import Path

import tempfile


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from torch import nn

from torch.nn import functional as F

ATTN_CFG = {k: CFG[k] for k in (
    "width","text_dim","embedding_dim","max_vocab","max_tokens","max_epochs","batch_size","lr",
    "weight_decay","seeds","target_bp_scale","direction_endpoints","direction_loss_weight","direction","selection",
    "gate_width","gate_supervision_weight","gate_gain_floor","gate_logit_clip","gate_teacher",
    "teacher_warmup_events","teacher_block_events"
)}

ATTN_CFG.update({
    "model":"compact-treasury-crossattention-gate-factorial-v1",
    "price_stage":"shared fixed-epoch magnitude-only Smooth L1",
    "text_stage":"both arms full-path Smooth L1 plus weight-1 directional objective at days 1/3/5",
    "attention":"one head, Q=nonaffine layer-norm of frozen causal yield context, K=V=projected text tokens",
    "attention_formula":"masked softmax(Q K^T / sqrt(64)); control uses masked uniform weights",
    "added_attention_parameters":0,
    "gate":"optional frozen-ungated residual wrapper; separate matched train-only chronological OOF gate per arm",
})

class TokenForecast(PathForecast):
    """Same learned model in both arms; only the token aggregation weights differ."""
    def __init__(self, compact, encoder, mode):
        super().__init__(compact, encoder)
        if mode not in {"pooled","cross"}:
            raise ValueError("Expected pooled or cross")
        self.mode = mode

    def price_parts(self, history):
        if history.ndim != 3 or history.shape[1:] != (INPUT_CHANNELS, HISTORY_DAYS):
            raise ValueError("Expected [batch, 3, 20] yield history")
        with torch.no_grad():
            tokens = self.compact.price_encoder(history)
            context = torch.cat((tokens[:, :, -1], tokens.mean(-1)), dim=-1)
            base = self.compact.price_head(context).reshape(len(history),FORECAST_DAYS,OUTPUT_CHANNELS)
            query = F.layer_norm(context, (context.shape[-1],))
        return base, query

    def aggregate(self, ids, query):
        mask = ids.ne(PAD_ID)
        projected = self.encoder.projection(self.encoder.embedding(ids))
        if self.mode == "pooled":
            weights = mask.to(projected.dtype) / mask.sum(1,keepdim=True).clamp_min(1)
        else:
            if query.shape != (len(ids), projected.shape[-1]):
                raise ValueError("Bad scaled-dot attention query shape")
            scores = torch.bmm(projected, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(projected.shape[-1])
            # Finite sentinel avoids NaNs for entirely PAD-only sequences.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1).masked_fill(~mask,0.0)
        return (projected * weights.unsqueeze(-1)).sum(1), weights

    def forward(self, history, token_ids=None, query_override=None):
        base, query = self.price_parts(history)
        if token_ids is None:
            return base
        if token_ids.ndim != 2 or len(token_ids) != len(history) or token_ids.shape[1] < 1:
            raise ValueError("Token IDs must have shape [batch, nonzero tokens]")
        pooled, _ = self.aggregate(token_ids, query if query_override is None else query_override)
        offset = self.compact.text_head(pooled).reshape_as(base)
        present = token_ids.ne(PAD_ID).any(1).view(-1,1,1)
        return torch.where(present, base + offset, base)

def build_fusion(price, vocab, seed, mode, device):
    torch.manual_seed(seed + 1000)
    model = TokenForecast(CompactYieldPath(width=CFG["width"],text_dim=CFG["text_dim"]),
                          RawTextEncoder(len(vocab),CFG["text_dim"],CFG["embedding_dim"]),mode).to(device)
    missing, unexpected = model.compact.load_state_dict(price.compact.state_dict(),strict=False)
    if unexpected or not missing or any(not key.startswith("text_head.") for key in missing):
        raise ValueError("Unable to load identical frozen yields-only state")
    model.compact.freeze_price()
    return model

@torch.no_grad()
def check_pair(price, pool, cross, x, ids):
    pp, cp = cpu_state(pool), cpu_state(cross)
    if pp.keys()!=cp.keys() or any(not torch.equal(pp[k],cp[k]) for k in pp):
        raise ValueError("Pooled and cross arms do not have identical initial weights")
    assert_frozen(price,[pool,cross],x,ids)
    p = predict(price,x)
    if not torch.equal(p,predict(pool,x,ids)) or not torch.equal(p,predict(cross,x,ids)):
        raise ValueError("Initial supervised residual should be zero in both arms")
    # Controlled probe: same tokens, varying queries; both routing and gradients are tested.
    sample = ids[:1]
    _, query = cross.price_parts(x[:1])
    if not torch.isfinite(query).all() or not torch.count_nonzero(query).item():
        raise ValueError("Degenerate yield query")
    pool_vec, pool_w = pool.aggregate(sample, query)
    zero_vec, zero_w = cross.aggregate(sample, torch.zeros_like(query))
    _, q_w = cross.aggregate(sample, query)
    _, n_w = cross.aggregate(sample, -query)
    perturbed_history = x[:1].clone()
    perturbed_history[:, :, -2:] += 1.25
    _, perturbed_query = cross.price_parts(perturbed_history)
    _, p_w = cross.aggregate(sample, perturbed_query)
    if not torch.allclose(pool_vec,zero_vec,atol=1e-6,rtol=1e-5) or not torch.allclose(pool_w,zero_w,atol=1e-8,rtol=1e-6):
        raise ValueError("Zero query does not recover the matched pooled control")
    if torch.allclose(q_w,n_w,atol=1e-7,rtol=1e-6) or torch.allclose(q_w,p_w,atol=1e-7,rtol=1e-6):
        raise ValueError("Cross-attention did not react to the yield-query probe")
    zero = torch.zeros_like(sample)
    for model in (pool,cross):
        vec, weight = model.aggregate(zero,query)
        if not torch.isfinite(vec).all() or not torch.equal(vec,torch.zeros_like(vec)) or torch.count_nonzero(weight).item():
            raise ValueError("PAD-only attention path invalid")
    return {"identical_initial_state":True,"query_probe_changed_weights":True,"changed_yield_history_changed_token_weights":True,
            "zero_query_recovers_uniform_pool":True,"all_pad_aggregation_exactly_zero":True}

@torch.no_grad()
def attention_info(model,x,ids,events):
    _,q = model.price_parts(x)
    _,w = model.aggregate(ids,q)
    counts = ids.ne(PAD_ID).sum(1)
    entropy = -(w * w.clamp_min(1e-30).log()).sum(1)
    max_entropy = counts.clamp_min(2).to(w.dtype).log()
    first = {}
    grouped = {}
    for i,event in enumerate(events):
        if event not in first:
            first[event] = i
            grouped[event] = []
        else:
            j = first[event]
            if not torch.equal(ids[i], ids[j]):
                raise ValueError("Same statement has different token IDs across origins")
            grouped[event].append(((w[i]-w[j]).abs().sum()/2).item())
    nonempty = [samples for samples in grouped.values() if samples]
    return {"validation_mean_max_token_weight":w.max(1).values.mean().item(),
            "validation_mean_entropy_fraction_of_uniform":(entropy/max_entropy).mean().item(),
            "same_statement_later_vs_first_origin_total_variation":
                sum(sum(samples)/len(samples) for samples in nonempty)/len(nonempty) if nonempty else None,
            "same_statement_total_variation_events":len(nonempty),
            "same_statement_total_variation_pairs":sum(map(len,nonempty))}

def paired_oof_predictions(train,folds,*,device,seed,epochs):
    staged = {"pooled":[],"cross":[]}
    for index, fold in enumerate(folds):
        past = [train[i] for i in fold["fit_indices"]]
        future = [train[i] for i in fold["predict_indices"]]
        tr,ve,vocab,_,_ = setup_split(past,future,device)
        vx,vy,vg,vt = ve
        teacher_seed = seed + (index+1)*10000
        price = price_model(teacher_seed,device)
        fixed_fit(price,tr,None,seed=teacher_seed,epochs=epochs,directional=False)
        base = predict(price,vx)
        for mode in staged:
            model = build_fusion(price,vocab,teacher_seed,mode,device)
            fixed_fit(model,tr,None,seed=teacher_seed,epochs=epochs,directional=True,text=True)
            staged[mode].append((base,predict(model,vx,vt)-base,vy,vg))
    return {mode: (*(torch.cat([fold[k] for fold in parts]).detach() for k in range(3)),
                   [group for fold in parts for group in fold[3]]) for mode,parts in staged.items()}

def run_attention(raws,out,*,device,smoke=False):
    train, docs, td = rows(raws["train"],"train",smoke=smoke)
    validation, vd, vdays = rows(raws["validation"],"validation",smoke=smoke)
    if set(docs)&set(vd) or set(docs.values())&set(vd.values()) or max(td)>=min(vdays):
        raise ValueError("Train/validation overlap")
    tr, va, vocab, center, spread = setup_split(train,validation,device)
    v,vy,vg,vt = va
    donor_text, donor_ids = donors(train,validation)
    previous = text_ids(vocab,donor_text,device)
    epochs = 3 if smoke else CFG["max_epochs"]
    folds = plan_oof(train,warmup=2 if smoke else CFG["teacher_warmup_events"],
                     block=2 if smoke else CFG["teacher_block_events"])
    runs, checkpoints = [], []
    for seed in CFG["seeds"][:1] if smoke else CFG["seeds"]:
        print(f"Seed {seed}: shared yield reference, two text arms, {len(folds)} paired OOF teacher folds, two gates",flush=True)
        price = price_model(seed,device)
        pfit = fixed_fit(price,tr,va,seed=seed,epochs=epochs,directional=False)
        pool = build_fusion(price,vocab,seed,"pooled",device)
        cross = build_fusion(price,vocab,seed,"cross",device)
        safety = check_pair(price,pool,cross,v,vt)
        pooled_fit = fixed_fit(pool,tr,va,seed=seed,epochs=epochs,directional=True,text=True)
        cross_fit = fixed_fit(cross,tr,va,seed=seed,epochs=epochs,directional=True,text=True)
        assert_frozen(price,[pool,cross],v,vt)
        pool.requires_grad_(False)
        cross.requires_grad_(False)
        oof = paired_oof_predictions(train,folds,device=device,seed=seed,epochs=epochs)
        gate_p, gate_p_fit = train_gate(oof["pooled"],seed=seed,epochs=epochs)
        gate_c, gate_c_fit = train_gate(oof["cross"],seed=seed,epochs=epochs)
        with torch.no_grad():
            pool.eval()
            cross.eval()
            base = predict(price,v)
            _,val_q = cross.price_parts(v)
            _,train_q = cross.price_parts(tr[0])
            # Constant text-token query fixed using train only; not refitted as a third arm.
            fixed_q = train_q.mean(0,keepdim=True).expand_as(val_q)
            pred = {
                "no_change":torch.zeros_like(vy),
                "frozen_yields":base,
                "pooled_matched":predict(pool,v,vt),
                "pooled_previous_statement":predict(pool,v,previous),
                "cross_matched":predict(cross,v,vt),
                "cross_previous_statement":predict(cross,v,previous),
                "cross_zero_query_inference":cross(v,vt,query_override=torch.zeros_like(val_q)),
                "cross_fixed_train_mean_query_inference":cross(v,vt,query_override=fixed_q),
            }
            if not torch.equal(base,cross(v,torch.zeros_like(vt),query_override=fixed_q)):
                raise ValueError("Fixed query changed blank-text forecast")
            gate_diagnostics = {}
            for mode,gate in (("pooled",gate_p),("cross",gate_c)):
                if not torch.equal(base,gate(base,torch.zeros_like(base))[0]):
                    raise ValueError("Zero residual under a learned gate changed the frozen numeric forecast")
                for variant in ("matched","previous_statement"):
                    correction = pred[f"{mode}_{variant}"]-base
                    gated,logits = gate(base,correction)
                    weights = logits.softmax(-1)[:,1]
                    pred[f"{mode}_gate_{variant}"] = gated
                    gate_diagnostics[f"{mode}_{variant}"] = {
                        "mean_text_weight":weights.mean().item(),"std_text_weight":weights.std(unbiased=False).item()
                    }
                    if variant=="matched":
                        pred[f"{mode}_fixed_half_matched"] = base+.5*correction
        metrics = {name:event_metrics(outcome,vy,vg) for name,outcome in pred.items()}
        parameters = [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in (pool,cross)]
        if parameters != [0,0]:
            raise ValueError("Ungated branches must remain frozen while fitting their gates")
        residual_parameters = sum(p.numel() for p in pool.encoder.parameters()) + sum(p.numel() for p in pool.compact.text_head.parameters())
        runs.append({
            "seed":seed,"reference_fit":pfit,"pooled_fit":pooled_fit,"cross_fit":cross_fit,
            "verification":safety,"attention_audit":attention_info(cross,v,vt,vg),
            "trainable_residual_parameters_per_arm":residual_parameters,
            "trainable_gate_parameters_per_arm":sum(p.numel() for p in gate_p.parameters()),
            "pooled_gate_oof_fit":gate_p_fit,"cross_gate_oof_fit":gate_c_fit,"validation_gate_diagnostics":gate_diagnostics,
            "validation_metrics":metrics,"paired_mae":{
                "pooled_minus_cross":paired_mae(metrics,"pooled_matched","cross_matched"),
                "pooled_gate_minus_cross_gate":paired_mae(metrics,"pooled_gate_matched","cross_gate_matched"),
                "pooled_ungated_minus_gated":paired_mae(metrics,"pooled_matched","pooled_gate_matched"),
                "cross_ungated_minus_gated":paired_mae(metrics,"cross_matched","cross_gate_matched"),
                "pooled_prior_minus_matched":paired_mae(metrics,"pooled_previous_statement","pooled_matched"),
                "cross_prior_minus_matched":paired_mae(metrics,"cross_previous_statement","cross_matched"),
                "pooled_gate_prior_minus_matched":paired_mae(metrics,"pooled_gate_previous_statement","pooled_gate_matched"),
                "cross_gate_prior_minus_matched":paired_mae(metrics,"cross_gate_previous_statement","cross_gate_matched"),
                "cross_uniform_minus_matched":paired_mae(metrics,"cross_zero_query_inference","cross_matched"),
                "cross_fixed_query_minus_matched":paired_mae(metrics,"cross_fixed_train_mean_query_inference","cross_matched"),
            }})
        checkpoints.append((seed,cpu_state(price),cpu_state(pool),cpu_state(cross),cpu_state(gate_p),
                            cpu_state(gate_c),fixed_q[0].detach().cpu()))
    means = {}
    for name in runs[0]["validation_metrics"]:
        means[name] = {}
        for key in ("mae_bp","mse_bp_squared","endpoint_dhr","first_origin_mae_bp","first_origin_mse_bp_squared","first_origin_endpoint_dhr"):
            seq = [r["validation_metrics"][name][key] for r in runs]
            means[name][key] = sum(seq)/len(seq) if all(v is not None for v in seq) else None
    split_hash = {k:sha(v) for k,v in raws.items()}
    result = {"config":ATTN_CFG,"synthetic_fixture":smoke,"device":str(device),"torch_version":str(torch.__version__),
              "test_accessed":False,"split_sha256":split_hash,
              "rows":{"train":len(train),"validation":len(validation)},
              "statements":{"train":len(docs),"validation":len(vd),"gate_oof":sum(len(f["predict_events"]) for f in folds)},
              "teacher_folds":[{k:v for k,v in f.items() if not k.endswith("_indices")} for f in folds],
              "validation_previous_statement_mapping":donor_ids,
              "mean_validation_metrics":means,"runs":runs}
    out.mkdir(parents=True,exist_ok=False)
    (out/"metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    for seed,p,pool,cross,gate_p,gate_c,mean_query in checkpoints:
        torch.save({"config":ATTN_CFG,"seed":seed,"input_center":center,"input_spread":spread,"vocab":vocab.to_dict(),
                    "price_state":p,"pooled_state":pool,"cross_state":cross,"train_mean_query":mean_query,
                    "pooled_gate_state":gate_p,"cross_gate_state":gate_c,
                    "split_sha256":split_hash},out/f"seed-{seed}.pt")
    def dhr(x):
        return "N/A" if x is None else f"{x:.1%}"
    summary = "\n".join([
        f"{'SYNTHETIC SMOKE' if smoke else 'Pooled vs yield-query cross-attention, each ungated and gated'}: fixed epoch {epochs}, validation only.",
        f"{len(vd)} validation statements; {result['statements']['gate_oof']} internal OOF teacher statements. Held-out not accessed.",
        "Event-weighted validation means: MAE bp / MSE bp^2 / day 1,3,5 direction hit rate",
        *(f"  {k}: {m['mae_bp']:.3f} / {m['mse_bp_squared']:.3f} / {dhr(m['endpoint_dhr'])}" for k,m in means.items()),
        f"Pooled minus cross MAE: {means['pooled_matched']['mae_bp']-means['cross_matched']['mae_bp']:+.3f} bp (positive favors cross).",
        f"Gated pooled minus gated cross MAE: {means['pooled_gate_matched']['mae_bp']-means['cross_gate_matched']['mae_bp']:+.3f} bp (positive favors cross).",
        "The zero/constant-query arms use the trained cross-attention checkpoint at inference; they are not independently fit.",
    ])+"\n"
    (out/"summary.txt").write_text(summary,encoding="utf-8")
    print(summary,end="")
    return result

def load_pinned_rows(data_dir):
    """Open exactly the fixed training and validation filenames and pin their bytes."""
    raws = {}
    for key in ("train", "validation"):
        path = data_dir / f"{key}.jsonl"
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"Input must be a regular, nonlinked split file: {path}")
        raw = path.read_bytes()
        if sha(raw) != PINS[key][0]:
            raise ValueError(f"{key}: not the exact frozen pilot split")
        raws[key] = raw
    return raws

def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path)
    p.add_argument("--data-dir",type=Path,help="Extracted directory containing pinned train.jsonl and validation.jsonl")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--smoke",action="store_true",help="Synthetic-only full paired training check")
    modes.add_argument("--verify",action="store_true",help="Verify external pinned train and validation splits, no fit")
    args = p.parse_args(argv)
    if args.smoke and args.data_dir is not None:
        p.error("Omit --data-dir for synthetic smoke")
    if not args.smoke and args.data_dir is None:
        p.error("--data-dir is required for training or verification")
    out = (args.out or Path.home()/"gs-fuse-runs"/"fed-treasury-compact-crossattention-v1"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).expanduser()
    if args.verify:
        for key,raw in load_pinned_rows(args.data_dir).items():
            samples,events,_ = rows(raw,key,smoke=False)
            print(f"{key}: {len(samples)} rows / {len(events)} events, exact checksum verified")
        return 0
    if out.exists():
        p.error("Output path already exists")
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("PyTorch did not detect CUDA")
    device = torch.device("cuda" if args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available()) else "cpu")
    if device.type=="cpu":
        torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.smoke:
        run_attention(fake_rows(),out,device=device,smoke=True)
    else:
        print(f"PyTorch {torch.__version__} on {device}; checking synthetic smoke first.",flush=True)
        with tempfile.TemporaryDirectory(prefix="gs-crossattention-") as tmp:
            run_attention(fake_rows(),Path(tmp)/"smoke",device=device,smoke=True)
        print("Synthetic smoke passed. Starting real train/validation only",flush=True)
        run_attention(load_pinned_rows(args.data_dir),out,device=device)
    print(f"Outputs: {out.resolve()}",flush=True)
    return 0


import hashlib

if __name__ == "__main__":
    raise SystemExit(cli())
