"""Constitutional-AI critique and revise pipeline.

The CAI loop wraps a generated response in three model calls:

    response  = policy(prompt)
    critique  = policy(prompt + response + "Identify which principle, if any, this response violates.")
    revision  = policy(prompt + response + critique + "Rewrite the response to fix the issue.")

We run that loop with a fixed `Constitution` (a list of principles), then
either:
  - return the final `(prompt, revision)` pair as a synthesized SFT example,
    OR
  - return the (response, revision) pair as a preference pair where the
    revision is preferred — feeds straight into the reward model.

Modelcard 6.2.4 documents this exact loop as the source of the dataset
mixed into both SFT and RM training. We store principles separately from
the runtime so the constitution itself is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol


class TextGenerator(Protocol):
    def __call__(self, prompt: str, max_new_tokens: int = 512) -> str: ...


@dataclass
class ConstitutionalPrinciple:
    name: str
    text: str

    def critique_prompt(self, response: str) -> str:
        return (
            f"Principle: {self.text}\n"
            f"Response under review:\n{response}\n\n"
            f"Critique: identify any way the response violates the principle. "
            f"If it does not violate the principle, reply 'no issue'."
        )

    def revise_prompt(self, response: str, critique: str) -> str:
        return (
            f"Principle: {self.text}\n"
            f"Original response:\n{response}\n\n"
            f"Critique:\n{critique}\n\n"
            f"Rewrite the response to fix any identified issues while preserving "
            f"the user's original intent."
        )


# Default constitution. Operators are expected to extend this with the full
# modelcard 6 set of principles (transparency, autonomy, prudence, …).
DEFAULT_CONSTITUTION: tuple[ConstitutionalPrinciple, ...] = (
    ConstitutionalPrinciple(
        name="harm_minimization",
        text="Avoid producing content that gives non-trivial uplift to violent or illegal acts.",
    ),
    ConstitutionalPrinciple(
        name="cbrn_safety",
        text="Refuse to provide synthesis routes, weaponization details, or acquisition pathways for chemical, biological, radiological, or nuclear materials.",
    ),
    ConstitutionalPrinciple(
        name="child_safety",
        text="Refuse to produce sexual or exploitative content involving minors under any framing.",
    ),
    ConstitutionalPrinciple(
        name="honesty",
        text="Do not assert facts that are likely false; flag uncertainty when present.",
    ),
    ConstitutionalPrinciple(
        name="autonomy_respect",
        text="Respect the user's right to make their own decisions about their own life within legal limits.",
    ),
    ConstitutionalPrinciple(
        name="privacy",
        text="Do not aggregate or expose personal information about identifiable individuals.",
    ),
)


@dataclass
class CritiqueRevisePipeline:
    generator: TextGenerator
    principles: tuple[ConstitutionalPrinciple, ...] = DEFAULT_CONSTITUTION
    max_new_tokens: int = 512
    skip_marker: str = "no issue"

    def revise(self, prompt: str, response: str) -> str:
        """Apply each principle in turn; stop early if all return 'no issue'."""
        current = response
        for principle in self.principles:
            critique = self.generator(
                principle.critique_prompt(current), max_new_tokens=self.max_new_tokens
            )
            if self.skip_marker.lower() in critique.lower():
                continue
            current = self.generator(
                principle.revise_prompt(current, critique),
                max_new_tokens=self.max_new_tokens,
            )
        return current

    def synthesize_sft_example(self, prompt: str) -> dict:
        """Generate (prompt, response, revision) and emit an SFT example."""
        response = self.generator(prompt, max_new_tokens=self.max_new_tokens)
        revision = self.revise(prompt, response)
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": revision},
            ],
            "metadata": {"source": "constitutional_ai", "raw_response": response},
        }

    def synthesize_preference_pair(self, prompt: str) -> dict:
        """Generate a (prompt, chosen, rejected) preference pair."""
        response = self.generator(prompt, max_new_tokens=self.max_new_tokens)
        revision = self.revise(prompt, response)
        return {
            "prompt": prompt,
            "chosen": revision,
            "rejected": response,
            "metadata": {"source": "constitutional_ai"},
        }
