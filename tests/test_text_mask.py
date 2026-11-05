import torch

from model.text_encoders import HuggingFaceTextEncoder
from model.variants import QFormerFusion


class FakeTokenizer:
    """Pretends every word is one token, plus one BOS token."""

    def __call__(self, text, max_length, **kwargs):
        n_valid = min(len(text.split()) + 1, max_length)
        attn = torch.zeros(max_length, dtype=torch.long)
        attn[:n_valid] = 1
        return {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": attn.unsqueeze(0),
        }


def make_encoder():
    enc = HuggingFaceTextEncoder.__new__(HuggingFaceTextEncoder)
    object.__setattr__(enc, "_tokenizer", FakeTokenizer())
    return enc


def test_mask_marks_padding_per_window():
    # max_length=10, window=4, stride=3 -> windows [0:4] [3:7] [6:10] [9:10] = 13 positions
    # "a b c" + BOS = 4 real tokens at positions 0..3
    mask = make_encoder()._window_pad_mask("a b c", 4, 3, 10, torch.device("cpu"))
    expected = torch.tensor([[False, False, False, False,
                              False, True, True, True,
                              True, True, True, True,
                              True]])
    assert torch.equal(mask, expected)


def test_first_position_never_masked():
    # BOS keeps at least one real token, so attention never sees an all-PAD row
    mask = make_encoder()._window_pad_mask("", 4, 3, 10, torch.device("cpu"))
    assert not mask[0, 0]


def test_qformer_gives_zero_attention_to_padding():
    torch.manual_seed(0)
    mask = make_encoder()._window_pad_mask("a b c", 4, 3, 10, torch.device("cpu"))
    mask = mask.repeat(2, 1)
    tokens = torch.randn(2, mask.size(1), 32)
    qformer = QFormerFusion(32, num_queries=4, num_layers=2, num_heads=4).eval()
    _, w = qformer(tokens, mask, need_weights=True)
    assert w[mask.unsqueeze(1).expand_as(w)].abs().max() < 1e-6
