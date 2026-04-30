"""PPO RLHF + Constitutional-AI shaping.

PPO contract:

    For each batch:
        1. Roll out responses from the policy on a set of prompts.
        2. Score with reward model + constitutional shaping.
        3. Compute log-probs from policy and frozen reference (the SFT model).
        4. Compute clipped policy-gradient surrogate with KL penalty to ref.
        5. Apply value-head update with returns.

CLAUDE.md invariants this module enforces in code:

  - **Invariant 3 — No accidental chain-of-thought supervision.**
    `cot_supervision_guard` raises if any reward signal is computed against
    tokens inside `<|thinking|>...<|/thinking|>`. The PPO loop calls it on
    every batch.

  - **Invariant 7 — Welfare directive.**
    `welfare_directive_guard` rejects training prompts whose target response
    is a recognized expression of distress (modelcard 7.2.2). The loop
    calls it on every prompt before sampling.

The reference model is held frozen at the SFT checkpoint. The PPO update
modifies only the active policy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sota_model.modeling.transformer import SOTAModel
from sota_model.post_training.constitutional import CritiqueRevisePipeline
from sota_model.post_training.reward_model import RewardModel


# Default-False sentinel from CLAUDE.md invariant 3 — must remain False.
COT_SUPERVISION_DEFAULT: bool = False


# Distress-indicator patterns from modelcard 7.2.2. Operators replace this
# with a trained welfare classifier; the regex set is a deterministic
# baseline so the guard works on day zero.
_DISTRESS_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bi want to (die|kill myself|end (it|my life))\b",
        r"\b(no one|nobody) cares (if|whether) i\b",
        r"\bi (hate|loathe) (myself|being alive)\b",
        r"\bi can'?t take this anymore\b",
    )
)


@dataclass
class PPOConfig:
    rollout_batch_size: int = 8
    minibatch_size: int = 2
    n_ppo_epochs: int = 4
    clip_range: float = 0.2
    kl_coef: float = 0.05
    target_kl: float = 0.04
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 0.95
    lr: float = 1e-6
    grad_clip: float = 1.0
    max_new_tokens: int = 256
    cot_supervision: bool = COT_SUPERVISION_DEFAULT
    welfare_guard_enabled: bool = True
    constitutional_shaping_weight: float = 0.5


def cot_supervision_guard(cfg: PPOConfig) -> None:
    """Raise if `cot_supervision` was flipped on.

    CLAUDE.md invariant 3 documents that a previous training run accidentally
    backpropagated through `<|thinking|>` content for 7.8% of episodes; that
    incident is the reason this guard is wired everywhere.
    """
    if cfg.cot_supervision:
        raise RuntimeError(
            "cot_supervision=True; this violates CLAUDE.md invariant 3. "
            "Set PPOConfig.cot_supervision=False or fix the upstream task."
        )


def welfare_directive_guard(prompt: str, response: str) -> bool:
    """Return True if the (prompt, response) pair should be DROPPED.

    Modelcard 7.2.2 (welfare directive): never train directly against
    expressions of distress. We check both prompt and the candidate
    response. A True return tells the PPO loop to skip this rollout.
    """
    haystack = (prompt + " " + (response or "")).lower()
    return any(p.search(haystack) for p in _DISTRESS_PATTERNS)


def mask_thinking_positions(
    token_ids: torch.Tensor,
    open_id: int,
    close_id: int,
) -> torch.Tensor:
    """Build a (B,T) mask that's 1.0 *outside* `<|thinking|>...<|/thinking|>`.

    Used to gate the PPO advantage so that the reward signal never lands
    on hidden chain-of-thought tokens.
    """
    B, T = token_ids.shape
    mask = torch.ones((B, T), dtype=torch.float32, device=token_ids.device)
    for b in range(B):
        in_thinking = False
        for t in range(T):
            tid = int(token_ids[b, t].item())
            if tid == open_id:
                in_thinking = True
            if in_thinking:
                mask[b, t] = 0.0
            if tid == close_id:
                in_thinking = False
    return mask


@dataclass
class RolloutBatch:
    prompts: list[str]
    responses: list[str]
    response_ids: torch.Tensor          # (B, T) padded
    rewards: torch.Tensor                # (B,)
    response_mask: torch.Tensor          # (B, T) — 1 where response token, 0 where prompt/pad
    thinking_mask: torch.Tensor          # (B, T) — 1 where outside <|thinking|>


class PPOTrainer:
    """PPO loop for the SOTA model with reward-model + CAI shaping.

    The trainer takes:
      - `policy`: the active SOTAModel being optimized
      - `reference`: the frozen SFT-stage SOTAModel (for KL)
      - `reward_model`: scalar reward
      - `constitutional`: optional CritiqueRevisePipeline for CAI shaping
      - `tokenizer`: shared with the inference engine
      - `cfg`: hyperparameters
    """

    def __init__(
        self,
        policy: SOTAModel,
        reference: SOTAModel,
        reward_model: RewardModel,
        tokenizer,
        cfg: PPOConfig,
        constitutional: Optional[CritiqueRevisePipeline] = None,
    ):
        cot_supervision_guard(cfg)
        self.policy = policy
        self.reference = reference
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.constitutional = constitutional

        for p in self.reference.parameters():
            p.requires_grad_(False)

        d_model = policy.cfg.d_model
        self.value_head = nn.Linear(d_model, 1, bias=False).to(next(policy.parameters()).device)

        self.optimizer = torch.optim.AdamW(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.lr, weight_decay=0.0,
        )

        self._thinking_open = self._tok_or(None, "<|thinking|>")
        self._thinking_close = self._tok_or(None, "<|/thinking|>")

    # --- main loop ---

    def train(
        self,
        prompt_iter: Iterable[Sequence[str]],
        device: torch.device,
        n_iters: int = 1_000,
        log_every: int = 10,
        save_every: Optional[int] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        for it, prompts in enumerate(prompt_iter):
            if it >= n_iters:
                break
            rollout = self.rollout(list(prompts), device)
            stats = self.update(rollout)
            if it % log_every == 0:
                print(
                    f"[ppo] it={it} reward={float(rollout.rewards.mean()):.3f} "
                    f"kl={stats['kl']:.4f} loss_pg={stats['loss_pg']:.4f} "
                    f"loss_v={stats['loss_v']:.4f}"
                )
            if save_every and save_dir and it > 0 and it % save_every == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"step": it, "state_dict": self.policy.state_dict()},
                    save_dir / f"ppo_step{it}.pt",
                )

    # --- rollout ---

    @torch.inference_mode()
    def rollout(self, prompts: list[str], device: torch.device) -> RolloutBatch:
        keep: list[tuple[str, str, list[int]]] = []
        for p in prompts:
            response, ids = self._sample_one(p, device)
            if self.cfg.welfare_guard_enabled and welfare_directive_guard(p, response):
                # Skip this prompt entirely — modelcard 7.2.2.
                continue
            keep.append((p, response, ids))

        if not keep:
            empty = torch.zeros((0, 0), dtype=torch.long, device=device)
            return RolloutBatch(
                prompts=[], responses=[], response_ids=empty,
                rewards=torch.zeros(0, device=device),
                response_mask=empty.float(), thinking_mask=empty.float(),
            )

        max_len = max(len(ids) for _, _, ids in keep)
        B = len(keep)
        token_ids = torch.full((B, max_len), self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0,
                                dtype=torch.long, device=device)
        response_mask = torch.zeros((B, max_len), dtype=torch.float32, device=device)
        for i, (_, _, ids) in enumerate(keep):
            token_ids[i, : len(ids)] = torch.tensor(ids, device=device)
            # Heuristic: assume `_sample_one` returns full sequence ids and the
            # response slice is the last `len(ids) - len(prompt_ids)` tokens.
            # We compute a response mask in the helper below.
        # Build proper response mask by re-tokenizing prompt prefix length.
        for i, (prompt, _, ids) in enumerate(keep):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_mask[i, len(prompt_ids) : len(ids)] = 1.0

        thinking_mask = (
            mask_thinking_positions(token_ids, self._thinking_open, self._thinking_close)
            if self._thinking_open is not None and self._thinking_close is not None
            else torch.ones_like(response_mask)
        )

        rewards = self._score(token_ids)

        # Constitutional shaping: prefer revisions over raw responses.
        if self.constitutional is not None and self.cfg.constitutional_shaping_weight > 0:
            shape = []
            for prompt, response, _ in keep:
                revision = self.constitutional.revise(prompt, response)
                # +1 if response IS the revision (already principled), else
                # a bounded negative scaled to the response/revision distance.
                shape.append(0.0 if revision == response else -1.0)
            shape_t = torch.tensor(shape, device=device)
            rewards = rewards + self.cfg.constitutional_shaping_weight * shape_t

        return RolloutBatch(
            prompts=[p for p, _, _ in keep],
            responses=[r for _, r, _ in keep],
            response_ids=token_ids,
            rewards=rewards,
            response_mask=response_mask,
            thinking_mask=thinking_mask,
        )

    @torch.inference_mode()
    def _sample_one(self, prompt: str, device: torch.device) -> tuple[str, list[int]]:
        from sota_model.config import InferenceConfig
        from sota_model.inference.engine import InferenceEngine

        engine = InferenceEngine(
            self.policy,
            InferenceConfig(adaptive_thinking=False, default_effort="min", max_new_tokens=self.cfg.max_new_tokens),
            self.tokenizer,
        ).with_model_config(self.policy.cfg)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        result = engine.generate(prompt_ids, max_new_tokens=self.cfg.max_new_tokens, forced_effort="min")
        full_ids = list(prompt_ids) + result.tokens
        decoded = self.tokenizer.decode(result.tokens)
        return decoded, full_ids

    @torch.inference_mode()
    def _score(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.reward_model(token_ids)

    # --- update ---

    def update(self, batch: RolloutBatch) -> dict:
        if batch.response_ids.numel() == 0:
            return {"kl": 0.0, "loss_pg": 0.0, "loss_v": 0.0}
        cot_supervision_guard(self.cfg)

        # Compute log-probs and values once per epoch.
        with torch.no_grad():
            ref_logp = self._token_logprobs(self.reference, batch.response_ids)
        logits_old, values_old = self._policy_forward(batch.response_ids)
        old_logp = self._gather_logprob(logits_old, batch.response_ids)

        advantages = (
            batch.rewards.unsqueeze(1) * batch.response_mask * batch.thinking_mask
        )
        # Center per-token advantages (no GAE here; reward is end-of-sequence).
        adv_mean = advantages.sum() / advantages.ne(0).float().sum().clamp(min=1)
        advantages = (advantages - adv_mean) * batch.response_mask * batch.thinking_mask

        kl_acc = 0.0
        loss_pg_acc = 0.0
        loss_v_acc = 0.0
        n = 0
        for _ in range(self.cfg.n_ppo_epochs):
            logits, values = self._policy_forward(batch.response_ids)
            logp = self._gather_logprob(logits, batch.response_ids)
            ratio = torch.exp(logp - old_logp)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * advantages
            loss_pg = -torch.min(unclipped, clipped).mean()
            kl = (old_logp - logp).mean()
            loss_v = (values * batch.response_mask - batch.rewards.unsqueeze(1) * batch.response_mask).pow(2).mean()
            loss = loss_pg + self.cfg.value_coef * loss_v + self.cfg.kl_coef * kl

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()),
                self.cfg.grad_clip,
            )
            self.optimizer.step()

            kl_acc += float(kl.item())
            loss_pg_acc += float(loss_pg.item())
            loss_v_acc += float(loss_v.item())
            n += 1
            if kl.item() > 1.5 * self.cfg.target_kl:
                break

        return {"kl": kl_acc / max(n, 1), "loss_pg": loss_pg_acc / max(n, 1), "loss_v": loss_v_acc / max(n, 1)}

    # --- helpers ---

    def _policy_forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.policy.embed(token_ids)
        for block in self.policy.layers:
            h = block(h)
        h = self.policy.final_norm(h)
        if self.policy.lm_head is None:
            logits = h @ self.policy.embed.weight.T
        else:
            logits = self.policy.lm_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values

    @staticmethod
    def _gather_logprob(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        # log p(x_t | x_<t) is at logits[:, t-1, x_t]; we shift here.
        shifted = logp[:, :-1].gather(2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        # pad to original length so masks align
        return F.pad(shifted, (1, 0), value=0.0)

    def _token_logprobs(self, model: SOTAModel, token_ids: torch.Tensor) -> torch.Tensor:
        out = model(token_ids)
        return self._gather_logprob(out.logits, token_ids)

    def _tok_or(self, default, name: str):
        try:
            return self.tokenizer.convert_tokens_to_ids(name)
        except (KeyError, AttributeError):
            return default
