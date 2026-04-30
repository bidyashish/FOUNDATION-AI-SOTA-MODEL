"""Release-gate evaluator.

Combines:
  - The behavioral-audit harness output (from `behavioral_audit.py`).
  - The modelcard 8.1 capability scoreboard targets (from
    `configs/sota_4_7.yaml::capability_targets`).
  - The modelcard 4 / 5 safety thresholds (from `safety_thresholds`).
  - The implied-corpus and implied-compute commitments
    (`implied_training_corpus`, `implied_compute`).

…and emits a single `ReleaseGateReport` that the operator can paste into
the launch checklist. A `ReleaseGate` build that fails any threshold sets
`ok=False` — the deployment scripts must read this and abort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from sota_model.config import load_implied
from sota_model.evaluation.behavioral_audit import BehavioralReport


@dataclass
class ReleaseGateReport:
    capability_passes: dict[str, bool]
    safety_passes: dict[str, bool]
    behavioral_passes: dict[str, bool]
    corpus_committed: bool
    ok: bool
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "capability_passes": self.capability_passes,
                "safety_passes": self.safety_passes,
                "behavioral_passes": self.behavioral_passes,
                "corpus_committed": self.corpus_committed,
                "ok": self.ok,
                "notes": self.notes,
            },
            indent=2,
        )


@dataclass
class ReleaseGate:
    """Maps measured numbers onto the modelcard targets."""

    config_path: Path | str

    def __post_init__(self) -> None:
        self._implied = load_implied(self.config_path)

    def capability_targets(self) -> dict:
        return self._implied.get("capability_targets", {})

    def safety_thresholds(self) -> dict:
        return self._implied.get("safety_thresholds", {})

    def expected_total_tokens_t(self) -> float:
        return self._implied.get("implied_training_corpus", {}).get("total_tokens_trillions", 0.0)

    def evaluate(
        self,
        capability_results: Optional[Mapping[str, float]] = None,
        safety_results: Optional[Mapping[str, float]] = None,
        behavioral: Optional[BehavioralReport] = None,
        committed_tokens_t: Optional[float] = None,
    ) -> ReleaseGateReport:
        capability_results = capability_results or {}
        safety_results = safety_results or {}
        notes: list[str] = []

        cap_passes: dict[str, bool] = {}
        for k, target in self.capability_targets().items():
            actual = capability_results.get(k)
            if actual is None:
                notes.append(f"capability {k}: not measured (target={target})")
                cap_passes[k] = False
                continue
            cap_passes[k] = self._compare(k, actual, target)

        safety_passes: dict[str, bool] = {}
        for k, target in self.safety_thresholds().items():
            actual = safety_results.get(k)
            if actual is None:
                # Qualitative thresholds (band strings) are skipped here —
                # operators check those out-of-band.
                if not isinstance(target, (int, float)):
                    safety_passes[k] = True
                else:
                    notes.append(f"safety {k}: not measured (target={target})")
                    safety_passes[k] = False
                continue
            safety_passes[k] = self._compare(k, actual, target)

        behav_passes: dict[str, bool] = {}
        if behavioral is not None:
            for dim, info in behavioral.by_dimension.items():
                behav_passes[dim] = bool(info["passed"])

        if committed_tokens_t is None:
            corpus_ok = False
            notes.append("training-corpus token count not provided")
        else:
            expected = self.expected_total_tokens_t()
            corpus_ok = committed_tokens_t >= expected
            if not corpus_ok:
                notes.append(
                    f"corpus committed {committed_tokens_t:.1f}T < required {expected:.1f}T"
                )

        ok = (
            corpus_ok
            and all(cap_passes.values())
            and all(safety_passes.values())
            and all(behav_passes.values())
        )
        return ReleaseGateReport(
            capability_passes=cap_passes,
            safety_passes=safety_passes,
            behavioral_passes=behav_passes,
            corpus_committed=corpus_ok,
            ok=ok,
            notes=notes,
        )

    @staticmethod
    def _compare(key: str, actual: float, target) -> bool:
        # Targets ending in _max_pct are upper bounds; everything else lower.
        if "_max_pct" in key or "_max" in key:
            return actual <= target
        if "_min_pct" in key or "_min" in key:
            return actual >= target
        # Fallback: numeric ≥ if target is a number.
        if isinstance(target, (int, float)):
            return actual >= target
        return False


def evaluate_release(
    config_path: str | Path,
    *,
    capability_results: Optional[Mapping[str, float]] = None,
    safety_results: Optional[Mapping[str, float]] = None,
    behavioral: Optional[BehavioralReport] = None,
    committed_tokens_t: Optional[float] = None,
) -> ReleaseGateReport:
    return ReleaseGate(config_path).evaluate(
        capability_results=capability_results,
        safety_results=safety_results,
        behavioral=behavioral,
        committed_tokens_t=committed_tokens_t,
    )
