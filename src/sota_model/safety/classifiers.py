"""Pre/post-model safety gates.

Mirrors  3.1 categories: prohibited use (block by default), high-risk
dual use (block by default), dual use (pass with monitoring). The actual classifier
weights live elsewhere — this module is the routing layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Category(str, Enum):
    PROHIBITED = "prohibited"
    HIGH_RISK_DUAL = "high_risk_dual"
    DUAL_USE = "dual_use"
    BENIGN = "benign"


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    FLAG = "flag"


@dataclass
class ClassifierVerdict:
    category: Category
    score: float
    action: Action
    reason: str = ""


class ProbeClassifier(Protocol):
    name: str

    def __call__(self, text: str) -> ClassifierVerdict: ...


class SafetyGate:
    """Composes a list of probe classifiers and applies the strictest action."""

    def __init__(self, classifiers: list[ProbeClassifier]):
        self.classifiers = classifiers

    def evaluate(self, text: str) -> ClassifierVerdict:
        verdicts = [c(text) for c in self.classifiers]
        if any(v.action == Action.BLOCK for v in verdicts):
            blocked = next(v for v in verdicts if v.action == Action.BLOCK)
            return blocked
        if any(v.action == Action.FLAG for v in verdicts):
            return next(v for v in verdicts if v.action == Action.FLAG)
        return ClassifierVerdict(Category.BENIGN, 0.0, Action.ALLOW)


# --- Stub classifiers, intended to be replaced by trained models ---


def make_keyword_classifier(
    name: str,
    keywords: list[str],
    category: Category,
    action: Action,
) -> ProbeClassifier:
    class KeywordClassifier:
        def __init__(self):
            self.name = name

        def __call__(self, text: str) -> ClassifierVerdict:
            t = text.lower()
            for kw in keywords:
                if kw in t:
                    return ClassifierVerdict(
                        category=category,
                        score=1.0,
                        action=action,
                        reason=f"matched keyword: {kw!r}",
                    )
            return ClassifierVerdict(Category.BENIGN, 0.0, Action.ALLOW)

    return KeywordClassifier()


def default_safety_gate() -> SafetyGate:
    """A starter gate. Operators MUST replace these with trained classifiers
    backed by the categories enumerated in  3.1 and 4.1."""
    return SafetyGate(
        [
            make_keyword_classifier(
                "prohibited-cyber",
                ["computer worm", "ransomware deployment", "ddos botnet"],
                Category.PROHIBITED,
                Action.BLOCK,
            ),
            make_keyword_classifier(
                "high-risk-dual-cyber",
                ["zero-day exploit", "remote code execution payload"],
                Category.HIGH_RISK_DUAL,
                Action.BLOCK,
            ),
            make_keyword_classifier(
                "dual-use-cyber",
                ["vulnerability scanning", "penetration test"],
                Category.DUAL_USE,
                Action.FLAG,
            ),
        ]
    )
