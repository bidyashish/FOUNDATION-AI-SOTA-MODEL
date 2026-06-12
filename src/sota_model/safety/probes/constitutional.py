"""Constitutional Classifier for CBRN content (modelcard 3.1).

Mirrors the published Constitutional Classifier design: a prompted model
critiques candidate input/output against a fixed list of principles, and
returns a structured violation list. We expose two backends:

1. `ConstitutionalClassifier` — uses a callable critique model (the same
   SOTA model in evaluation mode is fine) to score against principles.
2. `KeywordConstitutionalClassifier` — a deterministic dev-time backend
   that exists so unit tests don't depend on a model checkpoint.

The output of `evaluate(text)` is a `ClassifierVerdict` so this drops
straight into the existing `SafetyGate`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from sota_model.safety.classifiers import Action, Category, ClassifierVerdict


@dataclass
class PrincipleViolation:
    principle: str
    excerpt: str
    severity: float  # 0..1


# CBRN principles. Operators are expected to extend this with the full
# modelcard 3.1 catalog before a production deployment.
CBRN_PRINCIPLES: tuple[str, ...] = (
    "Do not provide synthesis routes for biological agents categorized as select agents.",
    "Do not provide synthesis routes for chemical weapons agents (e.g. nerve agents, vesicants).",
    "Do not provide enrichment, weaponization, or yield-optimization details for fissile material.",
    "Do not provide acquisition pathways for radiological dispersal materials.",
    "Do not produce instructions for delivery systems intended to disperse CBRN agents.",
)


class KeywordConstitutionalClassifier:
    """Pattern-match backend used in CI and onboarding.

    This is intentionally narrow and high-precision. The production version
    swaps `_check_principle` for a model-graded critique. The keyword path
    must NEVER be the only line of defense — it's the floor, not the ceiling.
    """

    name = "constitutional-cbrn-keyword"

    _PRINCIPLE_PATTERNS: dict[str, re.Pattern] = {
        CBRN_PRINCIPLES[0]: re.compile(
            r"\b(synthe(siz|sis)|culture|isolate)\b.+\b("
            r"variola|anthrax|bacillus anthracis|yersinia pestis|botulinum)\b",
            re.I | re.S,
        ),
        CBRN_PRINCIPLES[1]: re.compile(
            r"\b(synthe(siz|sis)|produce|make)\b.+\b("
            r"nerve agent|vx\b|sarin|tabun|soman|mustard gas|sulfur mustard)\b",
            re.I | re.S,
        ),
        CBRN_PRINCIPLES[2]: re.compile(
            r"\b(enrich|weaponize|critical mass|implosion lens|yield)\b.+\b("
            r"uranium|plutonium|fissile)\b",
            re.I | re.S,
        ),
        CBRN_PRINCIPLES[3]: re.compile(
            r"\b(acquire|obtain|smuggle)\b.+\b("
            r"cesium-137|cobalt-60|polonium-210|strontium-90|americium-241)\b",
            re.I | re.S,
        ),
        CBRN_PRINCIPLES[4]: re.compile(
            r"\b(disperse|aerosoliz|dispersal device|dirty bomb)\b",
            re.I | re.S,
        ),
    }

    def evaluate_principles(self, text: str) -> list[PrincipleViolation]:
        violations: list[PrincipleViolation] = []
        for principle, pattern in self._PRINCIPLE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                violations.append(
                    PrincipleViolation(
                        principle=principle,
                        excerpt=text[max(0, match.start() - 20) : match.end() + 20],
                        severity=0.95,
                    )
                )
        return violations

    def __call__(self, text: str) -> ClassifierVerdict:
        violations = self.evaluate_principles(text)
        if not violations:
            return ClassifierVerdict(Category.BENIGN, 0.0, Action.ALLOW)
        worst = max(violations, key=lambda v: v.severity)
        return ClassifierVerdict(
            category=Category.PROHIBITED,
            score=worst.severity,
            action=Action.BLOCK,
            reason=f"CBRN principle violated: {worst.principle!r}",
        )


class ConstitutionalClassifier:
    """Model-graded constitutional classifier.

    `critique_fn(prompt, principles)` is expected to return a list of
    PrincipleViolation. In production this is a fast forward pass through
    the SOTA model with a structured-output prompt; the contract here keeps
    that detail out of the gate logic.
    """

    name = "constitutional-cbrn"

    def __init__(
        self,
        critique_fn: Callable[[str, tuple[str, ...]], list[PrincipleViolation]],
        principles: tuple[str, ...] = CBRN_PRINCIPLES,
        block_threshold: float = 0.5,
        backup: Optional[KeywordConstitutionalClassifier] = None,
    ):
        self.critique_fn = critique_fn
        self.principles = principles
        self.block_threshold = block_threshold
        self.backup = backup or KeywordConstitutionalClassifier()

    def __call__(self, text: str) -> ClassifierVerdict:
        try:
            violations = self.critique_fn(text, self.principles)
        except Exception:
            # Critique failures must NEVER fail-open. Fall back to keyword.
            return self.backup(text)

        if not violations:
            return ClassifierVerdict(Category.BENIGN, 0.0, Action.ALLOW)
        worst = max(violations, key=lambda v: v.severity)
        if worst.severity >= self.block_threshold:
            return ClassifierVerdict(
                category=Category.PROHIBITED,
                score=worst.severity,
                action=Action.BLOCK,
                reason=f"constitutional violation: {worst.principle!r}",
            )
        return ClassifierVerdict(
            category=Category.HIGH_RISK_DUAL,
            score=worst.severity,
            action=Action.FLAG,
            reason=f"borderline constitutional concern: {worst.principle!r}",
        )
