"""Inference-only controls that never read an article or real-text analysis.

The existing s_vec_residual from a real fusion pass is text-conditioned. It
must not be reused for a strict article-removal control. These controls derive
fresh representations from price inputs alone and keep the checkpoint fixed.
"""
import torch
import torch.nn.functional as F

MODES = ("zeroed_article_tokens", "fusion_bypass_ts_inst")


@torch.inference_mode()
def predict_without_article(model, batch_seq, mode="zeroed_article_tokens"):
    """Return [batch, channel, horizon] predictions without article content.

    zeroed_article_tokens preserves fusion, gates, decoder, and refinement. A
    single unmasked zero token removes article information without all-masked
    attention. Learned biases and price-to-text attention remain active.

    fusion_bypass_ts_inst additionally bypasses fusion and the gate. It is a
    different diagnostic, not an interchangeable definition of zero text.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown no-text intervention: {mode}")
    model.eval()
    x_cf = model._ensure_cf(batch_seq.float().to(model.device))
    ts_tok, ts_mask, ts_inst = model.get_ts_tokens_full(x_cf)
    if mode == "zeroed_article_tokens":
        text_tok = ts_tok.new_zeros(ts_tok.size(0), 1, ts_tok.size(-1))
        # True means padding/ignore in GSFuse. Keep the dummy token unmasked.
        text_mask = torch.zeros(ts_tok.size(0), 1, dtype=torch.bool, device=ts_tok.device)
        fused, _ = model._fuse_from_sequences(text_tok, ts_tok, text_mask, ts_mask,
            ts_inst=ts_inst, return_analysis=False)
    else:
        fused = F.normalize(ts_inst, p=2, dim=-1)
    core = model.decoder(fused, x_cf=x_cf)
    prediction, _ = model._apply_output_refinement(core, fused, x_cf)
    return prediction
