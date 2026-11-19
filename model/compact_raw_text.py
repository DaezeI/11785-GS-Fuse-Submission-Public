"""A small trainable masked-mean text encoder for compact GS-Fuse."""
from __future__ import annotations

from torch import Tensor, nn

from .compact_news_vocab import PAD_ID, UNK_ID, TrainVocabulary

__all__ = ["PAD_ID", "UNK_ID", "TrainVocabulary", "RawTextEncoder"]


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
