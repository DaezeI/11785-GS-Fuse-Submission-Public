from types import SimpleNamespace

import torch
import torch.nn as nn

from model.gs_fuse import GSFuse


class _TokenizerStub:
    def save_pretrained(self, _path):
        pass


class _TextEncoderStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self._tokenizer = _TokenizerStub()

    @property
    def llm(self):
        return self.backbone

    @property
    def tokenizer(self):
        return self._tokenizer


_MODULE_NAMES = (
    "ts_encoder", "llm_proj", "embed_project", "ts_feat_to_fuse",
    "ts_inst_to_fuse", "residual_project", "cross_attn_text_to_ts",
    "cross_attn_ts_to_text", "bi_gate_layer", "decoder", "mem_type_embed",
    "mem_encoder", "mem_gate", "mem_gate_scalar", "ts_tokenizer",
    "align_t2s", "align_s2t", "ts_only_head", "text_only_head",
    "token_align_module", "multi_scale_extractor",
)


def _checkpoint_model():
    model = GSFuse.__new__(GSFuse)
    nn.Module.__init__(model)
    model.text_encoder = _TextEncoderStub()
    for name in _MODULE_NAMES:
        setattr(model, name, nn.Linear(2, 2))
    model.moe_gate = None
    model.qformer = None
    model.use_learnable_last_value = False
    model.run_provenance = {}
    model.forecast_task_config = SimpleNamespace(task="price")
    model.device = torch.device("cpu")
    model._arch_metadata = lambda: {"forecast_task": {"task": "price"}}
    return model


def test_checkpoint_saves_text_backbone_once_and_still_loads(tmp_path):
    source = _checkpoint_model()
    path = tmp_path / "checkpoint.pth"
    source.save_model_combined(path)

    checkpoint = torch.load(path, map_location="cpu")
    states = checkpoint["state_dicts"]
    assert "text_encoder" in states
    assert "llm" not in states
    assert "backbone.weight" in states["text_encoder"]

    target = _checkpoint_model()
    target.load_model_combined(path, strict=False)
    assert torch.equal(
        target.text_encoder.backbone.weight,
        source.text_encoder.backbone.weight,
    )


def test_legacy_llm_entry_remains_loadable(tmp_path):
    source = _checkpoint_model()
    path = tmp_path / "checkpoint.pth"
    source.save_model_combined(path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["state_dicts"]["llm"] = source.text_encoder.llm.state_dict()
    torch.save(checkpoint, path)

    target = _checkpoint_model()
    target.load_model_combined(path, strict=False)
    assert torch.equal(
        target.text_encoder.backbone.bias,
        source.text_encoder.backbone.bias,
    )
