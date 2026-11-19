from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.gs_fuse import prepare_pretraining_stage, stage2_prediction, contrastive_series_anchor
from model.gs_fuse import atomic_torch_save
from scripts.improvement.resources import classify_log, disk_record


def test_fresh_and_resumed_stage_both_load_validation_selected_weights(tmp_path):
    path = tmp_path / "selected.pth"
    events = []
    model = SimpleNamespace(load_model_combined=lambda **kwargs: events.append(("load", kwargs)))
    def stage(model, **kwargs):
        events.append(("train", kwargs))
        path.write_text("validation-selected weights")
    prepare_pretraining_stage(model, str(path), stage, num_epochs=2)
    prepare_pretraining_stage(model, str(path), stage, num_epochs=2)
    assert [event[0] for event in events] == ["train", "load", "load"]


def test_missing_selected_stage_checkpoint_stops_pipeline(tmp_path):
    with pytest.raises(FileNotFoundError, match="selected checkpoint"):
        prepare_pretraining_stage(None, str(tmp_path / "missing"), lambda model, **kw: None)


def test_strict_text_warmup_is_invariant_to_all_price_inputs():
    calls = []
    def decoder(text, x_cf=None):
        calls.append(x_cf)
        result = text[:, :1].unsqueeze(-1).expand(-1, 4, 2)
        return result if x_cf is None else result + x_cf[:, :, -1:]
    def last_value(core, last, text):
        return core + last.expand_as(core), None
    model = SimpleNamespace(stage2_price_context=False, decoder=decoder,
                            use_learnable_last_value=True, last_value_module=last_value)
    text = torch.ones(2, 8)
    first = stage2_prediction(model, text, torch.zeros(2, 4, 35))
    second = stage2_prediction(model, text, torch.ones(2, 4, 35) * 100)
    assert torch.equal(first, second) and all(x is None for x in calls)
    model.stage2_price_context = True
    assert not torch.equal(first, stage2_prediction(model, text, torch.ones(2, 4, 35)))


def test_clean_contrastive_anchor_cannot_silently_fall_back_to_fused_features():
    clean, fused = torch.ones(2, 4), torch.zeros(2, 4)
    model = SimpleNamespace(contrastive_anchor="pre_attention")
    assert contrastive_series_anchor(model, dict(ts_inst=clean, s_vec_residual=fused), fused) is clean
    with pytest.raises(ValueError, match="requires pre-attention"):
        contrastive_series_anchor(model, dict(s_vec_residual=fused), fused)


@pytest.mark.parametrize("message,expected", [
    ("OSError: No space left on device", "disk_space"),
    ("Disk quota exceeded", "disk_quota"),
    ("Storage quota exceeded", "disk_quota"),
    ("CUDA out of memory", "gpu_memory"),
    ("walltime exceeded", "walltime"),
])
def test_failure_triage_keeps_distinct_resource_failures_separate(message, expected):
    assert classify_log(message) == [expected]


def test_disk_inventory_is_read_only_and_does_not_certify_quota(tmp_path):
    record = disk_record(tmp_path / "not-created")
    assert record["free_bytes"] > 0 and record["quota_verified"] is False
    assert not (tmp_path / "not-created").exists()


def test_quota_failure_preserves_previous_checkpoint_and_removes_only_own_partial(tmp_path, monkeypatch):
    path = tmp_path / "best.pth"
    atomic_torch_save({"weight": torch.ones(2)}, path)
    before = path.read_bytes()
    def quota_failure(payload, target):
        Path(target).write_bytes(b"partial checkpoint")
        raise OSError(122, "Disk quota exceeded")
    monkeypatch.setattr(torch, "save", quota_failure)
    with pytest.raises(OSError, match="quota exceeded"):
        atomic_torch_save({"weight": torch.zeros(2)}, path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pth"]


def test_frozen_qformer_still_attends_and_padding_mask_changes_attention():
    from model.variants import QFormerFusion
    model = QFormerFusion(16, num_queries=4, num_layers=1, num_heads=2, dropout=0).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokens = torch.randn(1, 10, 16)
    mask = torch.tensor([[False] * 4 + [True] * 6])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    _, unmasked = model(tokens, need_weights=True)
    _, masked = model(tokens, text_mask=mask, need_weights=True)
    assert unmasked[:, :, 4:].sum() > 0
    assert masked[:, :, 4:].sum() == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
