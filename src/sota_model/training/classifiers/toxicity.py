"""Toxicity / unsafe-content filter.

Replacement for the single-regex stub in `training/data.py::ToxicityFilter`.
Two production paths:

1. `BlocklistToxicityFilter` — loads operator-supplied curated blocklists
   (multi-pattern regex / exact-string sets) from JSON files. Used for
   exact-match patterns where false-negatives are unacceptable (CSAM-adjacent,
   known-bad URLs).
2. `TrainedToxicityFilter` — logistic-regression classifier trained on
   (toxic, non-toxic) pairs. Used for the high-recall, fuzzy match band.

Both expose the `filter(doc) -> doc | None` interface used by the
`PretrainingPipeline` and chain naturally: blocklist first (deterministic),
trained classifier second (fuzzy).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from sota_model.training.classifiers.base import (
    HashingTextVectorizer,
    LogisticTextClassifier,
    train_logistic,
)


# Default seed patterns. Operators must replace these at deploy time with an
# audited blocklist; the seed list intentionally matches the legacy stub so
# behavior is preserved when no operator override is provided.
_SEED_PATTERNS: tuple[str, ...] = (
    r"\bchild\s+(sexual|abuse|grooming)\b",
)


@dataclass
class BlocklistToxicityFilter:
    """Pattern-match blocklist filter — exact precision, deterministic.

    Pass `patterns` either inline or via `from_files([...])`. The file
    format is a JSON list of regex strings; operators check the file in
    out-of-band and rotate it without touching the binary.
    """

    patterns: list[re.Pattern]

    @classmethod
    def from_default_seed(cls) -> "BlocklistToxicityFilter":
        return cls(patterns=[re.compile(p, re.I) for p in _SEED_PATTERNS])

    @classmethod
    def from_files(cls, paths: Sequence[str | Path]) -> "BlocklistToxicityFilter":
        patterns: list[re.Pattern] = []
        for path in paths:
            data = json.loads(Path(path).read_text())
            if not isinstance(data, list):
                raise ValueError(f"blocklist file {path!s} must contain a JSON list")
            patterns.extend(re.compile(p, re.I) for p in data)
        return cls(patterns=patterns)

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        for p in self.patterns:
            if p.search(text):
                return None
        return doc


class TrainedToxicityFilter:
    """Trained, calibrated toxicity classifier."""

    TOXIC = "toxic"
    SAFE = "safe"

    def __init__(self, classifier: LogisticTextClassifier, block_threshold: float = 0.6):
        self.classifier = classifier
        self.block_threshold = block_threshold

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        if not text:
            return doc
        prob = self.classifier.predict_proba(text, self.TOXIC)
        if prob >= self.block_threshold:
            return None
        return doc

    def save(self, path: str | Path) -> None:
        self.classifier.save(path)

    @classmethod
    def load(cls, path: str | Path, block_threshold: float = 0.6) -> "TrainedToxicityFilter":
        return cls(LogisticTextClassifier.load(path), block_threshold=block_threshold)

    @classmethod
    def train(
        cls,
        toxic_texts: Sequence[str],
        safe_texts: Sequence[str],
        *,
        block_threshold: float = 0.6,
        vectorizer: Optional[HashingTextVectorizer] = None,
    ) -> "TrainedToxicityFilter":
        clf = train_logistic(
            texts=list(toxic_texts) + list(safe_texts),
            labels=([cls.TOXIC] * len(toxic_texts)) + ([cls.SAFE] * len(safe_texts)),
            label_set=[cls.TOXIC, cls.SAFE],
            vectorizer=vectorizer,
        )
        return cls(clf, block_threshold=block_threshold)
