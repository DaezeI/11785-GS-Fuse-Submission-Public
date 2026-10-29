# -*- coding: utf-8 -*-
"""Content-keyed cache of pooled text embeddings from the frozen text backbone.

The text backbone runs under no_grad, so encode_instance(text) is a pure function of
(text, backbone weights, window, stride, matmul precision). The signature pins everything
except the text; loading a cache built under a different signature is an error.
"""

import hashlib
import os

import torch


def cache_signature(text_model_path, window, stride, tf32, use_4bit=False):
    sig = {
        "model": os.path.basename(os.path.normpath(text_model_path)),
        "window": int(window),
        "stride": int(stride),
        "tf32": bool(tf32),
    }
    # Only added when set, so caches built before 4-bit support keep their signature.
    if use_4bit:
        sig["quant"] = "nf4"
    return sig


class TextEmbeddingCache:
    def __init__(self, signature):
        self.signature = dict(signature)
        self._store = {}

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf-8", "surrogatepass")).digest()

    def __len__(self):
        return len(self._store)

    def __contains__(self, text):
        return self._key(text) in self._store

    def get(self, text):
        return self._store.get(self._key(text))

    def put(self, text, vec):
        self._store[self._key(text)] = vec.detach().float().cpu()

    def save(self, path):
        keys = list(self._store)
        values = torch.stack([self._store[k] for k in keys]) if keys else torch.empty(0)
        tmp = f"{path}.tmp"
        torch.save({"signature": self.signature, "keys": keys, "values": values}, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, signature):
        blob = torch.load(path, map_location="cpu")
        if blob["signature"] != dict(signature):
            raise ValueError(
                f"Text cache {path} was built with {blob['signature']}, but this run uses {dict(signature)}"
            )
        cache = cls(signature)
        cache._store = dict(zip(blob["keys"], blob["values"]))
        return cache
