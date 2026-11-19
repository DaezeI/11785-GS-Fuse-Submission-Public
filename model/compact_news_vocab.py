"""Deterministic checkpoint-ready word vocabulary fitted on training text only."""
from __future__ import annotations

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
