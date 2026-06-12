"""Trainable text-classifier base.

The training stack ships heuristic stubs in `training/data.py` for quality,
toxicity, and language detection. They are useful day-zero defaults but
modelcard 1.1.1 explicitly grades on the quality of the trained corpus, so
operators must replace them.

This module provides the minimal infrastructure:

  - `HashingTextVectorizer` — character + word hashing trick over a
    fixed-size feature space. No external vocabulary, no drift on retrain.
  - `LogisticTextClassifier` — multiclass logistic regression with
    softmax output. Used by all three trained classifiers.
  - `train_logistic` — Adam-trained reference implementation that runs in
    pure PyTorch; no scikit-learn dependency.

The actual classifiers (quality, toxicity, language) compose these.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch


@dataclass
class HashingTextVectorizer:
    n_features: int = 2**16
    char_ngram_range: tuple[int, int] = (2, 5)
    word_ngram_range: tuple[int, int] = (1, 2)

    def transform(self, text: str) -> torch.Tensor:
        text = text.lower()
        vec = torch.zeros(self.n_features, dtype=torch.float32)

        # Character n-grams
        lo, hi = self.char_ngram_range
        for n in range(lo, hi + 1):
            for i in range(0, max(0, len(text) - n + 1)):
                self._add_feature(vec, text[i : i + n].encode("utf-8", errors="ignore"))

        # Word n-grams
        words = text.split()
        wlo, whi = self.word_ngram_range
        for n in range(wlo, whi + 1):
            for i in range(0, max(0, len(words) - n + 1)):
                self._add_feature(vec, " ".join(words[i : i + n]).encode("utf-8"))

        norm = vec.norm()
        if norm > 0:
            vec = vec / norm
        return vec

    def transform_batch(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.stack([self.transform(t) for t in texts])

    def _add_feature(self, vec: torch.Tensor, key: bytes) -> None:
        h = int(hashlib.blake2s(key, digest_size=4).hexdigest(), 16)
        sign = 1.0 if (h & 1) else -1.0
        idx = (h >> 1) % self.n_features
        vec[idx] += sign


class LogisticTextClassifier:
    """Multiclass logistic regression head.

    Predict: returns (label, prob_max, all_probs). `predict_proba(text, label)`
    returns the probability that `text` belongs to the given class.
    """

    def __init__(
        self,
        weight: torch.Tensor,         # (n_classes, n_features)
        bias: torch.Tensor,           # (n_classes,)
        labels: list[str],
        vectorizer: HashingTextVectorizer,
    ):
        self.weight = weight
        self.bias = bias
        self.labels = labels
        self.vectorizer = vectorizer

    def predict_logits(self, text: str) -> torch.Tensor:
        feat = self.vectorizer.transform(text)
        return self.weight @ feat + self.bias

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        logits = self.predict_logits(text)
        probs = torch.softmax(logits, dim=0)
        idx = int(probs.argmax().item())
        return self.labels[idx], float(probs[idx]), {l: float(p) for l, p in zip(self.labels, probs)}

    def predict_proba(self, text: str, label: str) -> float:
        if label not in self.labels:
            raise KeyError(f"unknown label: {label!r}")
        idx = self.labels.index(label)
        logits = self.predict_logits(text)
        return float(torch.softmax(logits, dim=0)[idx].item())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weight": self.weight,
                "bias": self.bias,
                "labels": self.labels,
                "vectorizer": {
                    "n_features": self.vectorizer.n_features,
                    "char_ngram_range": list(self.vectorizer.char_ngram_range),
                    "word_ngram_range": list(self.vectorizer.word_ngram_range),
                },
            },
            p,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticTextClassifier":
        state = torch.load(path, map_location="cpu", weights_only=False)
        v = state["vectorizer"]
        return cls(
            weight=state["weight"],
            bias=state["bias"],
            labels=list(state["labels"]),
            vectorizer=HashingTextVectorizer(
                n_features=v["n_features"],
                char_ngram_range=tuple(v["char_ngram_range"]),
                word_ngram_range=tuple(v["word_ngram_range"]),
            ),
        )


def train_logistic(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    label_set: Optional[Sequence[str]] = None,
    vectorizer: Optional[HashingTextVectorizer] = None,
    n_epochs: int = 30,
    lr: float = 0.5,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> LogisticTextClassifier:
    """Train a multiclass logistic-regression text classifier."""
    if len(texts) != len(labels):
        raise ValueError("len(texts) != len(labels)")
    if not texts:
        raise ValueError("empty training set")
    label_set = list(label_set) if label_set else sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(label_set)}

    vectorizer = vectorizer or HashingTextVectorizer()
    X = vectorizer.transform_batch(texts)
    y = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)

    torch.manual_seed(seed)
    n_classes, n_features = len(label_set), vectorizer.n_features
    W = torch.zeros(n_classes, n_features, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)

    for _ in range(n_epochs):
        opt.zero_grad()
        logits = X @ W.T + b
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()

    return LogisticTextClassifier(
        weight=W.detach(),
        bias=b.detach(),
        labels=label_set,
        vectorizer=vectorizer,
    )
