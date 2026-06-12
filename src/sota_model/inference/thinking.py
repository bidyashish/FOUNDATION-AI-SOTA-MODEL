"""Adaptive thinking: per-query effort selection and token budgeting.

 4.1.1 makes adaptive thinking a first-class mode, with effort
dynamically determined by the model. Here we map the effort-head logit into the
budgets defined in `ModelConfig.thinking_budgets`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import torch

from sota_model.config import EffortTier, ModelConfig


@dataclass
class ThinkingDecision:
    effort: EffortTier
    token_budget: int


class AdaptiveThinkingController:
    # Thresholds learned during RL post-training; defaults are reasonable starting points.
    _DEFAULT_THRESHOLDS: dict[EffortTier, float] = {
        "min": -2.0,
        "low": -0.5,
        "medium": 0.5,
        "high": 1.5,
        "max": 3.0,
    }

    def __init__(
        self,
        cfg: ModelConfig,
        forced_effort: EffortTier | None = None,
        thresholds: dict[EffortTier, float] | None = None,
    ):
        self.cfg = cfg
        self.forced = forced_effort
        self.thresholds = thresholds or dict(self._DEFAULT_THRESHOLDS)
        # bisect needs a sorted threshold list.
        self._tier_order: list[EffortTier] = ["min", "low", "medium", "high", "max"]
        self._sorted_thresholds = [self.thresholds[t] for t in self._tier_order]

    def decide(self, effort_logit: torch.Tensor | None) -> ThinkingDecision:
        if self.forced is not None:
            return ThinkingDecision(self.forced, self.cfg.thinking_budgets[self.forced])
        if effort_logit is None:
            return ThinkingDecision("medium", self.cfg.thinking_budgets["medium"])

        logit = float(effort_logit.float().mean().item())
        idx = bisect.bisect_right(self._sorted_thresholds, logit) - 1
        idx = max(0, min(idx, len(self._tier_order) - 1))
        tier = self._tier_order[idx]
        budget = max(self.cfg.thinking_budgets[tier], self.cfg.thinking_token_min_floor if tier != "min" else 0)
        return ThinkingDecision(tier, budget)
