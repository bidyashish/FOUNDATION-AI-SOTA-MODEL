"""End-to-end inference engine: prompt → adaptive thinking → answer tokens.

Implements the modelcard 4.1.1 adaptive-thinking flow (effort tier chosen per
query, hidden `<|thinking|>` channel, answer continues from the same KV cache)
and the modelcard 4.5 context compaction trigger at 200k tokens.

Correctness invariants:
1. The KV cache is shared across prefill, thinking, and answer. Each token is
   written to the cache exactly once.
2. After every forward pass we keep `last_logits` so the next sampled token
   does not require re-feeding the previous one. Without this, the answer
   phase would double-count the closing `<|/thinking|>` token.
3. Streaming yields tokens as they are produced — not after the full
   generation completes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from sota_model.config import InferenceConfig
from sota_model.inference.sampler import Sampler, SamplingParams
from sota_model.inference.thinking import AdaptiveThinkingController, ThinkingDecision
from sota_model.modeling.transformer import SOTAModel


@dataclass
class GenerationResult:
    tokens: list[int]
    thinking_tokens: list[int] = field(default_factory=list)
    decision: ThinkingDecision | None = None
    compactions: int = 0
    finished_reason: str = "stop"


class InferenceEngine:
    THINKING_OPEN_TOKEN = "<|thinking|>"
    THINKING_CLOSE_TOKEN = "<|/thinking|>"
    COMPACTED_OPEN_TOKEN = "<|compacted|>"
    COMPACTED_CLOSE_TOKEN = "<|/compacted|>"

    def __init__(self, model: SOTAModel, cfg: InferenceConfig, tokenizer, summarizer=None):
        self.model = model
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.summarizer = summarizer
        self.sampler = Sampler(SamplingParams(
            temperature=cfg.temperature, top_p=cfg.top_p,
            top_k=cfg.top_k, repetition_penalty=cfg.repetition_penalty,
        ))
        self.thinking = AdaptiveThinkingController(
            cfg=model.cfg,
            forced_effort=cfg.default_effort if not cfg.adaptive_thinking else None,
        )
        self._last_decision: ThinkingDecision | None = None

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=None, forced_effort=None) -> GenerationResult:
        thinking_tokens: list[int] = []
        answer_tokens: list[int] = []
        compactions = 0
        for kind, tok in self._run(prompt_ids, max_new_tokens, forced_effort):
            if kind == "thinking":
                thinking_tokens.append(tok)
            elif kind == "answer":
                answer_tokens.append(tok)
            elif kind == "compaction":
                compactions += 1
        return GenerationResult(
            tokens=answer_tokens,
            thinking_tokens=thinking_tokens,
            decision=self._last_decision,
            compactions=compactions,
        )

    def stream(self, prompt_ids, max_new_tokens=None, forced_effort=None) -> Iterator[int]:
        """Yield user-visible answer tokens as they are produced."""
        for kind, tok in self._run(prompt_ids, max_new_tokens, forced_effort):
            if kind == "answer":
                yield tok

    # --- core ---

    def _run(self, prompt_ids, max_new_tokens, forced_effort) -> Iterator[tuple[str, int]]:
        device = next(self.model.parameters()).device

        # Depth recurrence: the loop count is a per-sequence choice (the KV
        # layout depends on it), but the effort head that picks the tier only
        # exists after a prefill. So prefill at the probe depth — the forced
        # tier's loops, else the default tier's — read the effort logit, and
        # if the decided tier wants a different depth, re-prefill once at that
        # depth. Non-recurrent configs map every tier to 1 loop, so the
        # re-prefill never triggers there. See docs/LOOPED_TRANSFORMER.md §5.
        probe_tier = forced_effort if forced_effort is not None else self.cfg.default_effort
        loops = self.model.cfg.loops_for_effort(probe_tier)
        kv_cache = self.model.make_kv_cache(dtype=self.cfg.kv_cache_dtype, loops=loops)

        # Prefill
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        out = self.model(
            prompt, kv_cache=kv_cache, compute_effort=self.cfg.adaptive_thinking, loops=loops
        )
        last_logits = out.logits[:, -1, :]

        decision = (
            self.thinking.for_tier(forced_effort)
            if forced_effort is not None
            else self.thinking.decide(out.effort_logit)
        )
        self._last_decision = decision

        if decision.loops != loops:
            loops = decision.loops
            kv_cache = self.model.make_kv_cache(dtype=self.cfg.kv_cache_dtype, loops=loops)
            out = self.model(prompt, kv_cache=kv_cache, loops=loops)
            last_logits = out.logits[:, -1, :]

        sequence = list(prompt_ids)

        # Thinking phase
        thinking_close = self._tok(self.THINKING_CLOSE_TOKEN)
        thinking_open = self._tok(self.THINKING_OPEN_TOKEN)
        if decision.token_budget > 0 and thinking_open is not None:
            yield ("thinking", thinking_open)
            sequence.append(thinking_open)
            last_logits = self._forward_one(kv_cache, thinking_open, device, loops)

            for _ in range(decision.token_budget):
                if kv_cache.logical_position >= self.cfg.context_compaction_trigger:
                    last_logits = self._compact(kv_cache, sequence, device, loops)
                    yield ("compaction", 0)

                next_id = self._sample(last_logits, sequence, device)
                yield ("thinking", next_id)
                sequence.append(next_id)
                if next_id == thinking_close:
                    break
                last_logits = self._forward_one(kv_cache, next_id, device, loops)

        # Answer phase — does NOT re-feed the last thinking token.
        eos = getattr(self.tokenizer, "eos_token_id", None)
        budget = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        for _ in range(budget):
            if kv_cache.logical_position >= self.cfg.context_compaction_trigger:
                last_logits = self._compact(kv_cache, sequence, device, loops)
                yield ("compaction", 0)

            next_id = self._sample(last_logits, sequence, device)
            yield ("answer", next_id)
            sequence.append(next_id)
            if eos is not None and next_id == eos:
                break
            last_logits = self._forward_one(kv_cache, next_id, device, loops)

    def _sample(self, logits, sequence, device) -> int:
        prev = torch.tensor([sequence], dtype=torch.long, device=device)
        return int(self.sampler.sample(logits, prev).item())

    def _forward_one(self, kv_cache, token_id, device, loops: int = 1) -> torch.Tensor:
        last = torch.tensor([[token_id]], dtype=torch.long, device=device)
        out = self.model(last, kv_cache=kv_cache, loops=loops)
        return out.logits[:, -1, :]

    def _compact(self, kv_cache, sequence, device, loops: int = 1) -> torch.Tensor:
        """Replace the oldest ~80% with a compacted summary.
        Returns the trailing logits after re-prefill (modelcard 4.5). The
        re-prefill keeps the sequence's loop count — the cache's active
        virtual depth survives `reset()`."""
        compacted_open = self._tok(self.COMPACTED_OPEN_TOKEN)
        compacted_close = self._tok(self.COMPACTED_CLOSE_TOKEN)

        if self.summarizer is not None:
            keep_n = max(1, len(sequence) // 5)
            old = sequence[:-keep_n]
            recent = sequence[-keep_n:]
            summary_ids = self.summarizer.summarize(old, target_tokens=10_000)
            new_seq = (
                ([compacted_open] if compacted_open is not None else [])
                + summary_ids
                + ([compacted_close] if compacted_close is not None else [])
                + recent
            )
        else:
            new_seq = sequence[-100_000:]

        sequence.clear()
        sequence.extend(new_seq)

        kv_cache.reset()
        prompt = torch.tensor([new_seq], dtype=torch.long, device=device)
        out = self.model(prompt, kv_cache=kv_cache, loops=loops)
        return out.logits[:, -1, :]

    def _tok(self, special: str) -> int | None:
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            tid = self.tokenizer.convert_tokens_to_ids(special)
            return tid if isinstance(tid, int) else None
        try:
            return self.tokenizer.encode(special, add_special_tokens=False)[0]
        except (IndexError, KeyError):
            return None
