import pytest
import torch

from model.text_cache import TextEmbeddingCache, cache_signature


def test_roundtrip_and_signature_guard(tmp_path):
    sig = cache_signature("pretrained/Llama-3.2-1B/", 500, 400, tf32=False)
    cache = TextEmbeddingCache(sig)
    a, b = torch.randn(8), torch.randn(8)
    cache.put("doc a", a)
    cache.put("doc b ünïcode", b)
    assert "doc a" in cache and "doc c" not in cache

    path = tmp_path / "cache.pt"
    cache.save(path)
    loaded = TextEmbeddingCache.load(path, cache_signature("pretrained/Llama-3.2-1B", 500, 400, tf32=False))
    assert len(loaded) == 2
    assert torch.equal(loaded.get("doc a"), a)
    assert torch.equal(loaded.get("doc b ünïcode"), b)

    with pytest.raises(ValueError):
        TextEmbeddingCache.load(path, cache_signature("pretrained/Llama-3.2-1B", 500, 400, tf32=True))
    with pytest.raises(ValueError):
        TextEmbeddingCache.load(path, cache_signature("pretrained/Llama-3.2-1B", 500, 400, tf32=False,
                                                      use_4bit=True))
