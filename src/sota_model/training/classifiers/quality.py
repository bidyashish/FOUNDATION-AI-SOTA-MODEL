"""Document-quality classifiers for pretraining filtering.

Two backends:

1. `HeuristicQualityScorer` — punctuation + line-length signals. The
   original `data.py::QualityScorer` heuristics, kept for the day-zero path.
2. `TrainedQualityScorer` — the production scorer; loads a `LogisticTextClassifier`
   trained on (high-quality, low-quality) pairs. Operators run a one-off
   labeling pass to seed a few thousand examples per class, then retrain
   monthly as the corpus grows.

Both implement `__call__(doc) -> doc | None` so they're drop-in for the
existing filter list in `training/data.py::PretrainingPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from sota_model.training.classifiers.base import (
    HashingTextVectorizer,
    LogisticTextClassifier,
    train_logistic,
)


@dataclass
class HeuristicQualityScorer:
    """Lightweight quality heuristic — punctuation density + line layout.

    The same scoring as the original `data.py::QualityScorer`, factored out
    here so the pretraining pipeline can swap in a trained scorer without
    losing the day-zero baseline.
    """

    threshold: float = 0.7

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        if not text:
            return None
        n_punct = sum(1 for c in text if c in ".,!?;:")
        n_lines = max(1, text.count("\n"))
        score = min(
            1.0,
            n_punct / (len(text) + 1) * 30 + (n_lines / max(1, len(text) / 80)) * 0.5,
        )
        return doc if score >= self.threshold else None


class TrainedQualityScorer:
    """Logistic-regression quality scorer.

    Labels: `high_quality`, `low_quality`. The threshold is on
    `predict_proba(text, "high_quality")`. Sensible defaults are 0.7
    threshold and a vectorizer of 2^16 features.
    """

    HIGH = "high_quality"
    LOW = "low_quality"

    def __init__(self, classifier: LogisticTextClassifier, threshold: float = 0.7):
        self.classifier = classifier
        self.threshold = threshold

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        if not text:
            return None
        prob = self.classifier.predict_proba(text, self.HIGH)
        if prob >= self.threshold:
            doc.setdefault("quality_score", round(prob, 3))
            return doc
        return None

    def save(self, path: str | Path) -> None:
        self.classifier.save(path)

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.7) -> "TrainedQualityScorer":
        return cls(LogisticTextClassifier.load(path), threshold=threshold)

    @classmethod
    def train(
        cls,
        high_quality_texts: Sequence[str],
        low_quality_texts: Sequence[str],
        *,
        threshold: float = 0.7,
        vectorizer: Optional[HashingTextVectorizer] = None,
    ) -> "TrainedQualityScorer":
        clf = train_logistic(
            texts=list(high_quality_texts) + list(low_quality_texts),
            labels=([cls.HIGH] * len(high_quality_texts)) + ([cls.LOW] * len(low_quality_texts)),
            label_set=[cls.HIGH, cls.LOW],
            vectorizer=vectorizer,
        )
        return cls(clf, threshold=threshold)
