#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interpretability dump for GS-FUSE and its MoE-gate / Q-Former variants.

For every analysed event it records the text-gate openness and, depending on the
variant, the MoE routing weights or the Q-Former query->word attention. Outputs
`interpretability.json` plus PNG plots in --out-dir.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


_STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those there their they them he she his her we our you your i not no so than then too very can will
would should could may might must do does did has have had into over under about above below after before
during between through up down out off again further once more most other some such only own same s t
""".split())


def _is_content_word(word: str) -> bool:
    w = word.strip().lower()
    return len(w) > 1 and any(c.isalpha() for c in w) and w not in _STOPWORDS


def _text_positions(model, text):
    """Token ids / attention mask aligned with the chunked hidden states of encode_tokens()."""
    ids = model.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=model.max_text_tokens,
        add_special_tokens=True, padding="max_length",
    )
    input_ids, attn = ids["input_ids"][0], ids["attention_mask"][0]
    pos, start, L = [], 0, input_ids.size(0)
    while start < L:
        pos.extend(range(start, min(start + model.window, L)))
        start += model.stride
    pos = torch.tensor(pos)
    return input_ids[pos], attn[pos]


@torch.no_grad()
def collect(model, dataset, indices, batch_size=4):
    """dataset: data.dataloader.data_set_new (needs .list for event type / file path)."""
    model.eval()
    special_ids = set(model.tokenizer.all_special_ids)
    records = []
    for b0 in range(0, len(indices), batch_size):
        chunk = indices[b0:b0 + batch_size]
        items = [dataset[i] for i in chunk]
        texts = [it[0] for it in items]
        seq = torch.stack([it[5] for it in items]).float().to(model.device)

        tl, ml = zip(*[model.get_text_tokens_all(t) for t in texts])
        text_tok, text_mask = model._pad_and_stack(list(tl), list(ml))
        x_cf = model._ensure_cf(seq)
        ts_tok, ts_mask, ts_inst = model.get_ts_tokens_full(x_cf)
        _, an = model._fuse_from_sequences(text_tok, ts_tok, text_mask, ts_mask,
                                           ts_inst=ts_inst, return_analysis=True)

        for b, idx in enumerate(chunk):
            raw = dataset.list[idx]
            rec = {
                "index": int(idx),
                "event_type": str(raw[5]),
                "date": os.path.basename(raw[0])[:8],
                "gate_text": float(an["gate_softmax"][b, 0].mean()),
            }
            if "moe_router_probs" in an:
                rec["router_probs"] = an["moe_router_probs"][b].cpu().tolist()
                rec["dispatch"] = an["moe_dispatch"][b].cpu().tolist()
                rec["expert_text_open"] = an["moe_expert_text_open"][b].cpu().tolist()
            if an.get("qformer_attn") is not None:
                tok_ids, valid = _text_positions(model, texts[b])
                w = an["qformer_attn"][b, :, :tok_ids.size(0)].cpu()        # [Nq, Lt]
                keep = valid.bool() & ~torch.isin(tok_ids, torch.tensor(sorted(special_ids)))
                rec["pad_attention_mass"] = float(w[:, ~valid.bool()].sum(dim=1).mean())
                word_score = defaultdict(float)
                for tid, s in zip(tok_ids[keep].tolist(), w.mean(dim=0)[keep].tolist()):
                    word_score[tid] += s
                ranked = [(model.tokenizer.decode([t]).strip(), s)
                          for t, s in sorted(word_score.items(), key=lambda kv: -kv[1])]
                rec["top_words"] = [[w_, round(s, 5)] for w_, s in ranked[:15]]
                rec["top_content_words"] = [[w_, round(s, 5)] for w_, s in ranked if _is_content_word(w_)][:15]
                rec["per_query_top_words"] = []
                for q in range(w.size(0)):
                    qs = w[q][keep]
                    k = min(5, qs.numel())
                    if k == 0:
                        rec["per_query_top_words"].append([])
                        continue
                    top_pos = torch.topk(qs, k).indices
                    rec["per_query_top_words"].append(
                        [model.tokenizer.decode([int(tok_ids[keep][p])]).strip() for p in top_pos]
                    )
                if b0 == 0 and b == 0:
                    rec["_example_attn"] = {
                        "tokens": [model.tokenizer.decode([int(t)]) for t in tok_ids[keep][:60]],
                        "weights": w[:, keep][:, :60].tolist(),
                    }
            records.append(rec)
    return records


def summarize_and_plot(records, out_dir, variant_name):
    os.makedirs(out_dir, exist_ok=True)
    summary = {"variant": variant_name, "n_samples": len(records)}

    by_type = defaultdict(list)
    for r in records:
        by_type[r["event_type"]].append(r["gate_text"])
    types = sorted(by_type, key=lambda t: int(t) if t.isdigit() else t)
    summary["gate_text_by_event_type"] = {t: float(np.mean(by_type[t])) for t in types}

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(types, [summary["gate_text_by_event_type"][t] for t in types], color="#4C72B0")
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("event type")
    ax.set_ylabel("mean text-gate openness")
    ax.set_title(f"{variant_name}: text gate by event type")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_text_by_event_type.png"), dpi=150)
    plt.close(fig)

    if records and "dispatch" in records[0]:
        K = len(records[0]["dispatch"])
        mat = np.array([np.mean([r["dispatch"] for r in records if r["event_type"] == t], axis=0) for t in types])
        summary["moe_dispatch_by_event_type"] = {t: mat[i].round(4).tolist() for i, t in enumerate(types)}
        summary["moe_expert_load"] = np.mean([r["dispatch"] for r in records], axis=0).round(4).tolist()
        summary["moe_expert_text_open"] = np.mean([r["expert_text_open"] for r in records], axis=0).round(4).tolist()

        fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(types) + 2.2),
                                 gridspec_kw={"width_ratios": [3, 2]})
        im = axes[0].imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(1e-6, mat.max()))
        axes[0].set_xticks(range(K), [f"E{k}" for k in range(K)])
        axes[0].set_yticks(range(len(types)), types)
        axes[0].set_xlabel("expert")
        axes[0].set_ylabel("event type")
        axes[0].set_title("mean routing weight")
        for i in range(mat.shape[0]):
            for j in range(K):
                axes[0].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=axes[0])
        axes[1].bar([f"E{k}" for k in range(K)], summary["moe_expert_text_open"], color="#DD8452")
        axes[1].axhline(0.5, color="gray", lw=0.8, ls="--")
        axes[1].set_title("text openness per expert")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "moe_routing_by_event_type.png"), dpi=150)
        plt.close(fig)

    if records and "top_words" in records[0]:
        def _aggregate(key, filt):
            agg = defaultdict(float)
            for r in records:
                for w, s in r.get(key, r["top_words"]):
                    if filt(w):
                        agg[w] += s
            return sorted(agg.items(), key=lambda kv: -kv[1])[:25]

        summary["qformer_top_words"] = [[w, round(s, 4)] for w, s in _aggregate("top_words", lambda w: True)]
        top = _aggregate("top_content_words", _is_content_word)
        summary["qformer_top_content_words"] = [[w, round(s, 4)] for w, s in top]
        summary["qformer_mean_pad_attention_mass"] = float(np.mean([r["pad_attention_mass"] for r in records]))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh([w for w, _ in top][::-1], [s for _, s in top][::-1], color="#55A868")
        ax.set_xlabel("summed attention (all queries, all samples)")
        ax.set_title(f"{variant_name}: most-attended content words")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "qformer_top_words.png"), dpi=150)
        plt.close(fig)

        ex = next((r["_example_attn"] for r in records if "_example_attn" in r), None)
        if ex is not None:
            w = np.array(ex["weights"])
            fig, ax = plt.subplots(figsize=(min(24, 0.3 * w.shape[1] + 2), 0.35 * w.shape[0] + 2))
            ax.imshow(w, aspect="auto", cmap="magma")
            ax.set_xticks(range(len(ex["tokens"])), ex["tokens"], rotation=90, fontsize=7)
            ax.set_yticks(range(w.shape[0]), [f"q{q}" for q in range(w.shape[0])], fontsize=7)
            ax.set_title("Q-Former query -> word attention (first 60 words of one event)")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "qformer_attention_example.png"), dpi=150)
            plt.close(fig)

    with open(os.path.join(out_dir, "interpretability.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=1)
    return summary


def evenly_spaced(n_total, n):
    n = min(n, n_total)
    return sorted({int(i) for i in np.linspace(0, n_total - 1, n)})


def run_and_log(model, dataset, n_samples, out_dir, variant_name, batch_size=4):
    """collect + summarize_and_plot, then push the plots and scalar summary to W&B if active."""
    from model import tracking

    records = collect(model, dataset, evenly_spaced(len(dataset), n_samples), batch_size)
    summary = summarize_and_plot(records, out_dir, variant_name)
    tracking.log_images([os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith(".png")])
    flat = {f"interpret/gate_text_type{t}": v for t, v in summary["gate_text_by_event_type"].items()}
    if "moe_expert_load" in summary:
        flat.update({f"interpret/moe_expert{k}_load": v for k, v in enumerate(summary["moe_expert_load"])})
        flat.update({f"interpret/moe_expert{k}_text_open": v
                     for k, v in enumerate(summary["moe_expert_text_open"])})
    if "qformer_mean_pad_attention_mass" in summary:
        flat["interpret/qformer_pad_attention_mass"] = summary["qformer_mean_pad_attention_mass"]
    tracking.summary(flat)
    return summary


def main():
    from data.dataloader import event_set
    import model.gs_fuse as gs_fuse_mod
    from model.gs_fuse import GSFuse

    p = argparse.ArgumentParser(description="Interpretability analysis for a trained GS-FUSE checkpoint")
    p.add_argument("--model-path", required=True)
    p.add_argument("--moment-path", required=True)
    p.add_argument("--text-model-path", required=True)
    p.add_argument("--text-backbone", choices=["llama", "phi"], default="llama")
    p.add_argument("--event-dir", default="data/event")
    p.add_argument("--series-dir", default="data/series")
    p.add_argument("--series-id", default="SP500")
    p.add_argument("--seq-len", type=int, default=35)
    p.add_argument("--pred-len", type=int, default=140)
    p.add_argument("--max-token-num", type=int, default=1024)
    p.add_argument("--fusion-type", choices=["crossattn", "qformer"], default="crossattn")
    p.add_argument("--gate-type", choices=["softmax", "moe"], default="softmax")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--moe-top-k", type=int, default=2)
    p.add_argument("--num-queries", type=int, default=16)
    p.add_argument("--qformer-layers", type=int, default=2)
    p.add_argument("--split", choices=["test", "vali"], default="test")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--text-cache", type=str, default=None)
    p.add_argument("--tf32", action="store_true", help="Allow TF32 matmuls (must match the text cache)")
    p.add_argument("--no-4bit", action="store_true", help="Load the text backbone in full precision instead of 4-bit")
    args = p.parse_args()

    gs_fuse_mod.TEXT_CACHE_PATH = args.text_cache
    if args.no_4bit:
        gs_fuse_mod.USE_4BIT = False
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    events = event_set(args.seq_len, args.pred_len, series_id=args.series_id, batch_size=4,
                       event_dir=args.event_dir, series_dir=args.series_dir)
    model = GSFuse(
        text_model_path=args.text_model_path, ts_backbone="moment", text_backbone=args.text_backbone,
        moment_path=args.moment_path, seq_len=args.seq_len, pred_len=args.pred_len, d=events.d,
        use_ts_memory=False, max_token_num=args.max_token_num,
        fusion_type=args.fusion_type, gate_type=args.gate_type, num_experts=args.num_experts,
        moe_top_k=args.moe_top_k, num_queries=args.num_queries, qformer_layers=args.qformer_layers,
    )
    model.load_model_combined(args.model_path, strict=False)
    ds = events.test_set if args.split == "test" else events.vali_set
    records = collect(model, ds, evenly_spaced(len(ds), args.n_samples))
    name = f"fusion={args.fusion_type}, gate={args.gate_type}"
    print(json.dumps(summarize_and_plot(records, args.out_dir, name), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
