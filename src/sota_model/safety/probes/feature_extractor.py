"""Feature extractors for safety probes.

Two production paths:

1. **HashingFeatureExtractor** — character-n-gram hashing trick. No model
   call required. Used for the pre-model gate where we only have raw text
   and need millisecond-scale latency.

2. **HiddenStateFeatureExtractor** — pulls a pooled hidden state from a
   chosen layer of the SOTA model. Used for the post-model gate where we
   already paid for the forward pass anyway. Modelcard 3.1 calls this out
   as the right place for the high-recall classifiers.

Both expose `extract(text) -> torch.Tensor of shape (D,)`. The downstream
probe (linear or MLP head) only sees the feature vector.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Protocol

import torch


class FeatureExtractor(Protocol):
    out_dim: int

    def extract(self, text: str) -> torch.Tensor: ...


class HashingFeatureExtractor:
    """Hashing trick over character n-grams.

    Cheap, language-agnostic, and stable across releases (no learned
    vocabulary that drifts). The classic baseline for production text
    classifiers — what we'd use as the keyword-stub replacement on day one
    of operator deployment.
    """

    def __init__(self, n_features: int = 2**14, ngram_range: tuple[int, int] = (2, 5)):
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.out_dim = n_features

    def extract(self, text: str) -> torch.Tensor:
        text = text.lower()
        vec = torch.zeros(self.n_features, dtype=torch.float32)
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(0, max(0, len(text) - n + 1)):
                gram = text[i : i + n]
                h = int(hashlib.blake2s(gram.encode("utf-8"), digest_size=4).hexdigest(), 16)
                # Sign bit flips the contribution to keep the trick unbiased.
                sign = 1.0 if (h & 1) else -1.0
                idx = (h >> 1) % self.n_features
                vec[idx] += sign
        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        return vec


class HiddenStateFeatureExtractor:
    """Pull a layer's pooled hidden state through a forward pass.

    This is the post-model classifier path — `forward_to_layer` returns the
    hidden state at `layer_idx`
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer_idx: Optional[int] = None,
        pool: str = "mean",
        max_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx if layer_idx is not None else -2
        self.pool = pool
        self.max_tokens = max_tokens
        self.out_dim = model.cfg.d_model

    @torch.inference_mode()
    def extract(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)[: self.max_tokens]
        device = next(self.model.parameters()).device
        x = torch.tensor([ids], dtype=torch.long, device=device)

        h = self.model.embed(x)
        layers = list(self.model.layers)
        # Negative index → from end; positive → from start.
        target = self.layer_idx if self.layer_idx >= 0 else len(layers) + self.layer_idx
        for i, block in enumerate(layers):
            h = block(h)
            if i == target:
                break
        if self.pool == "mean":
            return h.mean(dim=1).squeeze(0).float().cpu()
        if self.pool == "last":
            return h[:, -1, :].squeeze(0).float().cpu()
        raise ValueError(f"unknown pooling: {self.pool}")
