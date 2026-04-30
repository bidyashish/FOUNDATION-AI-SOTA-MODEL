"""Language detector.

Stub replacement for `training/data.py::LanguageDetector`, which used to
silently pass anything not pre-tagged with a `lang` field.

Two backends:

1. `CharNgramLanguageDetector` — deterministic, no-training-data baseline.
   Uses Unicode-script profiling: e.g. CJK ideographs → `zh|ja|ko`,
   Cyrillic → `ru|uk|sr|...`. Good enough for filtering by script family
   when a trained model is not available.
2. `TrainedLanguageDetector` — multiclass logistic on character n-grams.
   Standard production approach (cf. cld3 / fasttext-langdetect family).

The default 44-language target comes from modelcard 8.12 (cross-validated
with `tokenizer.MODELCARD_LANGUAGES`).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from sota_model.tokenizer import MODELCARD_LANGUAGES
from sota_model.training.classifiers.base import (
    HashingTextVectorizer,
    LogisticTextClassifier,
    train_logistic,
)


# Script → language candidates. Keep narrow — the goal is a coarse first
# cut before the trained classifier (or a downstream tag-and-pass).
_SCRIPT_TO_LANGS: dict[str, tuple[str, ...]] = {
    "LATIN": (
        "en", "fr", "de", "es", "pt", "it", "nl", "pl", "tr", "sv", "cs",
        "id", "ro", "tl", "ms", "lt", "ig", "yo", "so", "mg", "ny", "ha",
        "sn", "sw",
    ),
    "CYRILLIC": ("ru", "uk", "sr", "ky"),
    "ARABIC": ("ar", "fa"),
    "HEBREW": ("he",),
    "DEVANAGARI": ("hi", "ne"),
    "BENGALI": ("bn",),
    "TELUGU": ("te",),
    "SINHALA": ("si",),
    "ETHIOPIC": ("am",),
    "GREEK": ("el",),
    "HAN": ("zh",),
    "HIRAGANA": ("ja",),
    "KATAKANA": ("ja",),
    "HANGUL": ("ko",),
}


@dataclass
class CharNgramLanguageDetector:
    """Coarse script-family-based detector.

    Modelcard 8.12 recognizes 44 languages. This classifier returns the
    candidate set inferred from the dominant Unicode script of the input;
    the per-language pick within the script is left to the trained backend.
    Operators usually wire this in front of `TrainedLanguageDetector`.
    """

    accepted: tuple[str, ...] = MODELCARD_LANGUAGES
    min_chars: int = 80

    def detect_script(self, text: str) -> str | None:
        # Histogram of script names; return the dominant one.
        counts: dict[str, int] = {}
        for ch in text:
            if ch.isspace() or not ch.isalpha():
                continue
            try:
                name = unicodedata.name(ch, "")
            except ValueError:
                continue
            for script in _SCRIPT_TO_LANGS:
                if script in name:
                    counts[script] = counts.get(script, 0) + 1
                    break
        if not counts:
            return None
        return max(counts, key=counts.get)

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        if "lang" in doc and doc["lang"] not in self.accepted:
            return None
        if "lang" in doc:
            return doc
        if len(text) < self.min_chars:
            return doc  # too short to confidently filter; defer.
        script = self.detect_script(text)
        if script is None:
            return doc
        candidates = _SCRIPT_TO_LANGS.get(script, ())
        if not candidates:
            return doc
        if any(c in self.accepted for c in candidates):
            doc.setdefault("lang_candidates", list(candidates))
            return doc
        return None


class TrainedLanguageDetector:
    """Multiclass logistic-regression language ID."""

    def __init__(
        self,
        classifier: LogisticTextClassifier,
        accepted: Iterable[str] = MODELCARD_LANGUAGES,
        min_confidence: float = 0.5,
    ):
        self.classifier = classifier
        self.accepted = set(accepted)
        self.min_confidence = min_confidence

    def __call__(self, doc: dict) -> dict | None:
        text = doc.get("text", "")
        if not text:
            return None
        if "lang" in doc and doc["lang"] not in self.accepted:
            return None
        label, conf, _ = self.classifier.predict(text)
        if conf < self.min_confidence:
            return doc  # uncertain — pass through for next filter
        if label not in self.accepted:
            return None
        doc.setdefault("lang", label)
        doc.setdefault("lang_confidence", round(conf, 3))
        return doc

    def save(self, path: str | Path) -> None:
        self.classifier.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        accepted: Iterable[str] = MODELCARD_LANGUAGES,
        min_confidence: float = 0.5,
    ) -> "TrainedLanguageDetector":
        return cls(LogisticTextClassifier.load(path), accepted=accepted, min_confidence=min_confidence)

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        langs: Sequence[str],
        *,
        accepted: Iterable[str] = MODELCARD_LANGUAGES,
        min_confidence: float = 0.5,
        vectorizer: Optional[HashingTextVectorizer] = None,
    ) -> "TrainedLanguageDetector":
        if vectorizer is None:
            vectorizer = HashingTextVectorizer(char_ngram_range=(2, 4), word_ngram_range=(1, 1))
        clf = train_logistic(
            texts=list(texts),
            labels=list(langs),
            label_set=sorted(set(langs)),
            vectorizer=vectorizer,
        )
        return cls(clf, accepted=accepted, min_confidence=min_confidence)
