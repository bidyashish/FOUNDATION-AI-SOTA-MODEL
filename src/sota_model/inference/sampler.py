"""Sampling: temperature, top-p, top-k, repetition penalty.

Defaults mirror the inference settings used in  4.1 evaluations:
temp 0.7, top_p 0.95, top_k 40, repetition_penalty 1.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    seed: int | None = None


class Sampler:
    def __init__(self, params: SamplingParams):
        self.params = params
        # Generators are created lazily per-device so a single Sampler can be
        # used with both CPU and CUDA tensors without device-mismatch errors.
        self._generators: dict[torch.device, torch.Generator] = {}

    def _generator_for(self, device: torch.device) -> torch.Generator:
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            if self.params.seed is not None:
                gen.manual_seed(self.params.seed)
            self._generators[device] = gen
        return gen

    def sample(self, logits: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        # logits: (B, V)  prev_tokens: (B, T_prev)
        logits = self._apply_repetition_penalty(logits, prev_tokens, self.params.repetition_penalty)
        logits = logits / max(self.params.temperature, 1e-6)
        logits = self._top_k_filter(logits, self.params.top_k)
        logits = self._top_p_filter(logits, self.params.top_p)
        probs = torch.softmax(logits, dim=-1)
        gen = self._generator_for(probs.device)
        return torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        prev_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        if prev_tokens.numel() == 0 or penalty == 1.0:
            return logits
        gathered = torch.gather(logits, 1, prev_tokens)
        # Standard form: positive logits divided by penalty, negative multiplied.
        gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
        return logits.scatter(1, prev_tokens, gathered)

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return logits
        top_vals, _ = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
        threshold = top_vals[:, -1:].expand_as(logits)
        return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        if p >= 1.0:
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative > p
        mask[..., 0] = False  # always keep at least the top token
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        return torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
