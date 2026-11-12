"""Measured, inference-only fusion controls for the original softmax gate.

A zero text fusion weight is NOT article removal: both aligned representations
may already contain text. Use no_text_control for explicit article removal.
"""
import torch
import torch.nn.functional as F


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"Non-finite {name}")


def _decode(model, fused, x_cf):
    core = model.decoder(fused, x_cf=x_cf)
    pred, _ = model._apply_output_refinement(core, fused, x_cf)
    _finite(pred, "predictions")
    return pred


def gate_target(err_reference, err_full, config):
    """Reconstruct the CURRENT training formula, including batch-dependent scale.

    The legacy reference is text-conditioned. This is not a causal text-benefit
    estimate, nor a reconstruction of targets encountered during training.
    """
    diff = (err_reference - err_full).detach()
    if config.GC_USE_ADAPTIVE_SCALE:
        scale = diff.abs().mean().clamp_min(config.GC_MIN_SCALE)
    else:
        scale = diff.new_tensor(config.GC_SIGMOID_SCALE)
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid supervision scale")
    logits = (config.GC_GAIN * diff / scale).clamp(
        -config.GC_LOGIT_CLAMP, config.GC_LOGIT_CLAMP)
    return logits.sigmoid(), diff


@torch.no_grad()
def measure_gate_batch(model, texts, seq, targets, config):
    """Return actual [B,C,H] predictions and [B,D] gate values for fixed inputs."""
    if model.gate_type != "softmax" or model.fusion_type != "crossattn":
        raise ValueError("This experiment requires crossattn + softmax")
    if model.training:
        raise ValueError("Call model.eval() before gate comparison")
    if len(texts) != len(seq) or len(seq) != len(targets):
        raise ValueError("Article/price/target batch lengths differ")
    text_pairs = [model.get_text_tokens_all(t) for t in texts]
    text, text_mask = model._pad_and_stack(
        [p[0] for p in text_pairs], [p[1] for p in text_pairs])
    x_cf = model._ensure_cf(seq.float().to(model.device))
    targets = targets.float().to(model.device)
    ts, ts_mask, ts_inst = model.get_ts_tokens_full(x_cf)
    fused, analysis = model._fuse_from_sequences(
        text, ts, text_mask, ts_mask, ts_inst=ts_inst, return_analysis=True)
    gate = analysis["gate_softmax"]
    text_vec = F.normalize(analysis["t_vec_residual"], p=2, dim=-1)
    price_vec = F.normalize(analysis["s_vec_residual"], p=2, dim=-1)
    expected_shape = (len(seq), 2, text_vec.shape[-1])
    if tuple(gate.shape) != expected_shape:
        raise ValueError(f"Unexpected gate shape: {tuple(gate.shape)}")
    _finite(gate, "learned gate")
    torch.testing.assert_close(gate.sum(dim=1), torch.ones_like(text_vec))
    reconstructed = gate[:, 0] * text_vec + gate[:, 1] * price_vec
    torch.testing.assert_close(fused, reconstructed)
    learned = _decode(model, fused, x_cf)
    # Only the fusion weights change. Encoded/aligned features are shared.
    fixed_gate = torch.full_like(gate, 0.5)
    fixed = _decode(model, fixed_gate[:, 0] * text_vec + fixed_gate[:, 1] * price_vec, x_cf)
    if learned.shape != targets.shape or fixed.shape != targets.shape:
        raise ValueError("Prediction/target shape mismatch")
    _finite(targets, "targets")
    reference = model.predict_ts_only_full_decoder(ts, ts_mask,
                                                  analysis["s_vec_residual"], x_cf)
    _finite(reference, "legacy supervision reference")
    err_full = (learned - targets).square().mean(dim=(1, 2))
    err_reference = (reference - targets).square().mean(dim=(1, 2))
    target, diff = gate_target(err_reference, err_full, config)
    return {"learned": learned, "fixed_050": fixed,
            "gate_text": gate[:, 0], "fixed_gate_text": fixed_gate[:, 0],
            "recomputed_gate_target": target, "legacy_reference_delta": diff,
            "text_vec": text_vec, "price_vec": price_vec, "x_cf": x_cf,
            "targets": targets}


def gate_gradient_probe(model, measured):
    """Gate-only derivative check in eval mode on ONE TRAIN batch; no updates.

    Not the full training loss or a measurement of historical training gradients.
    Encoded features and supervision targets are detached; autograd.grad does
    not accumulate into parameter .grad buffers or step an optimizer.
    """
    named = list(model.bi_gate_layer.named_parameters())
    flags = [p.requires_grad for _, p in named]
    try:
        for _, p in named:
            p.requires_grad_(True)
        with torch.enable_grad():
            t, s = measured["text_vec"].detach(), measured["price_vec"].detach()
            logits = model.bi_gate_layer(torch.cat([t, s], dim=-1)).reshape(len(t), 2, -1)
            gate = F.softmax(logits / max(model.gate_tau, 1e-6), dim=1)
            pred = _decode(model, gate[:, 0] * t + gate[:, 1] * s, measured["x_cf"].detach())
            losses = {
                "forecast_mse": (pred - measured["targets"].detach()).square().mean(),
                "gate_supervision_mse": F.mse_loss(gate[:, 0].mean(dim=-1),
                                                  measured["recomputed_gate_target"].detach())}
            output = {}
            for loss_name, loss in losses.items():
                _finite(loss, loss_name)
                grads = torch.autograd.grad(loss, [p for _, p in named],
                                            retain_graph=True, allow_unused=True)
                rows = {}
                for (name, _), grad in zip(named, grads):
                    if grad is not None:
                        _finite(grad, f"{loss_name}/{name} gradient")
                    rows[name] = None if grad is None else {
                        "l2": float(grad.norm()), "rms": float(grad.square().mean().sqrt())}
                output[loss_name] = {"loss": float(loss.detach()), "parameters": rows}
            return {"mode": "eval_gate_only_no_optimizer_step", "losses": output}
    finally:
        for (_, p), flag in zip(named, flags):
            p.requires_grad_(flag)
