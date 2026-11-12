"""CPU-only checks for real intervention semantics and measured paired metrics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load("gate_control", "model/gate_analysis.py")
analyzer = load("gate_analyzer", "scripts/analyze_gate_analysis.py")
runner = load("gate_runner", "scripts/run_gate_analysis.py")
CONFIG = SimpleNamespace(GC_USE_ADAPTIVE_SCALE=True, GC_MIN_SCALE=1e-6,
                         GC_SIGMOID_SCALE=4e-5, GC_GAIN=1., GC_LOGIT_CLAMP=6.)


class TinyModel(nn.Module):
    device = torch.device("cpu")
    gate_type, fusion_type, gate_tau = "softmax", "crossattn", 1.

    def __init__(self):
        super().__init__()
        self.bi_gate_layer = nn.Linear(8, 8)
        nn.init.zeros_(self.bi_gate_layer.weight)
        with torch.no_grad():
            self.bi_gate_layer.bias[:4].fill_(2.)
            self.bi_gate_layer.bias[4:].fill_(-2.)

    def get_text_tokens_all(self, text):
        x = torch.tensor([float(text), 2., 3., 1.]).reshape(1, 1, 4)
        return x, torch.zeros(1, 1, dtype=torch.bool)

    def _pad_and_stack(self, values, masks):
        return torch.cat(values), torch.cat(masks)

    def _ensure_cf(self, x):
        return x

    def get_ts_tokens_full(self, x):
        return x.transpose(1, 2), torch.zeros(x.shape[0], 35, dtype=torch.bool), x.mean(dim=-1)

    def _fuse_from_sequences(self, text, ts, text_mask, ts_mask, ts_inst=None, **kwargs):
        t, s = F.normalize(text.mean(dim=1), dim=-1), F.normalize(ts_inst, dim=-1)
        gate = self.bi_gate_layer(torch.cat([t, s], dim=-1)).reshape(-1, 2, 4).softmax(dim=1)
        return gate[:, 0] * t + gate[:, 1] * s, {
            "gate_softmax": gate, "t_vec_residual": text.mean(dim=1), "s_vec_residual": ts_inst}

    def decoder(self, fused, x_cf=None):
        return fused.unsqueeze(-1).expand(-1, -1, 140)

    def _apply_output_refinement(self, pred, fused, x_cf):
        return pred, None

    def predict_ts_only_full_decoder(self, ts, mask, s, x):
        return self.decoder(F.normalize(s, dim=-1), x)


def measured():
    model = TinyModel().eval()
    seq = torch.tensor([1., 4., 2., 5.]).reshape(1, 4, 1).expand(3, 4, 35)
    data = control.measure_gate_batch(model, ["1", "2", "4"], seq, torch.zeros(3, 4, 140), CONFIG)
    return model, data


def test_actual_gate_intervention_and_unchanged_parameters():
    model = TinyModel().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    seq = torch.tensor([1., 4., 2., 5.]).reshape(1, 4, 1).expand(3, 4, 35)
    data = control.measure_gate_batch(model, ["1", "2", "4"], seq, torch.zeros(3, 4, 140), CONFIG)
    expected = ((data["text_vec"] + data["price_vec"]) / 2).unsqueeze(-1).expand(-1, -1, 140)
    torch.testing.assert_close(data["fixed_050"], expected)
    assert torch.all(data["fixed_gate_text"] == .5)
    assert not torch.allclose(data["learned"], data["fixed_050"])
    assert not data["learned"].requires_grad
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_gradient_probe_restores_flags_parameters_and_existing_gradients():
    model, data = measured()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for p in model.bi_gate_layer.parameters():
        p.requires_grad_(False)
        p.grad = torch.full_like(p, 7.)
    result = control.gate_gradient_probe(model, data)
    assert result["losses"]["forecast_mse"]["parameters"]["bias"]["l2"] > 0
    assert result["losses"]["gate_supervision_mse"]["parameters"]["bias"]["l2"] > 0
    for p in model.bi_gate_layer.parameters():
        assert not p.requires_grad
        assert torch.all(p.grad == 7.)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_half_gate_equivalence_if_learned_logits_are_equal():
    model = TinyModel().eval()
    with torch.no_grad():
        model.bi_gate_layer.bias.zero_()
    seq = torch.ones(2, 4, 35)
    data = control.measure_gate_batch(model, ["1", "2"], seq, torch.zeros(2, 4, 140), CONFIG)
    torch.testing.assert_close(data["learned"], data["fixed_050"])


def test_finite_guard_and_variant_guard():
    model = TinyModel().eval()
    with pytest.raises(ValueError, match="Non-finite"):
        control.measure_gate_batch(model, ["1"], torch.ones(1, 4, 35),
                                   torch.full((1, 4, 140), float("nan")), CONFIG)
    model.gate_type = "moe"
    with pytest.raises(ValueError, match="crossattn"):
        control.measure_gate_batch(model, ["1"], torch.ones(1, 4, 35), torch.zeros(1, 4, 140), CONFIG)


def test_training_target_uses_actual_adaptive_scale():
    target, diff = control.gate_target(torch.tensor([1., 3.]), torch.tensor([1., 1.]), CONFIG)
    torch.testing.assert_close(diff, torch.tensor([0., 2.]))
    torch.testing.assert_close(target, torch.tensor([0., 2.]).sigmoid())


def test_analyzer_computes_measured_losses_and_detects_bad_override():
    _, data = measured()
    blob = {k: v.numpy() for k, v in data.items() if k not in ("text_vec", "price_vec", "x_cf")}
    blob["zeroed_article_tokens"] = blob["learned"] + 1
    blob["event_ids"] = np.array(["1/20200101.txt", "2/20200102.txt", "3/20200103.txt"])
    blob["event_dates"] = np.array(["20200101", "20200102", "20200103"])
    blob["metadata_json"] = np.array(json.dumps({"subset": True}))
    summary, rows = analyzer.summarize(blob)
    assert len(rows) == 9
    assert summary["horizons"]["140"]["mse"]["learned"] == pytest.approx(float((blob["learned"] ** 2).mean()))
    blob["fixed_gate_text"] = np.zeros_like(blob["gate_text"])
    with pytest.raises(ValueError, match="exactly 0.5"):
        analyzer.summarize(blob)


def test_audit_rejects_changed_scaler_and_exclusions():
    audit = {k: 1 for k in ("manifest_sha256", "price_sha256", "timing_policy", "emitted_counts",
                            "scaler_mean", "scaler_scale", "scaler_fit_unique_bars")}
    runner.verify_audit(audit, dict(audit))
    with pytest.raises(ValueError, match="scaler_mean"):
        runner.verify_audit(audit, {**audit, "scaler_mean": 2})
    with pytest.raises(ValueError, match="exclusions"):
        runner.verify_audit({**audit, "excluded_counts": {"missing": 1}}, audit)


def test_analysis_cli_reads_saved_predictions_and_refuses_overwrite(tmp_path):
    _, data = measured()
    blob = {k: v.numpy() for k, v in data.items() if k not in ("text_vec", "price_vec", "x_cf")}
    blob.update(zeroed_article_tokens=blob["learned"] + 1,
                event_ids=np.array(["1/20200101.txt", "2/20200102.txt", "3/20200103.txt"]),
                event_dates=np.array(["20200101", "20200102", "20200103"]),
                metadata_json=np.array(json.dumps({"subset": True})))
    source, output = tmp_path / "predictions.npz", tmp_path / "analysis"
    np.savez_compressed(source, **blob)
    command = [sys.executable, str(ROOT / "scripts/analyze_gate_analysis.py"),
               "--predictions", str(source), "--out", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    saved = json.loads((output / "summary.json").read_text())
    assert saved["distinct_dates"] == 3
    assert len((output / "event_losses.csv").read_text().splitlines()) == 10
    before = (output / "summary.json").read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert (output / "summary.json").read_bytes() == before
