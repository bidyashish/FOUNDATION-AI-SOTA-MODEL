"""Grouped-Query Attention with Flash Attention 2/3 fast path and KV-cache integration.

GQA (n_q_heads=128, n_kv_heads=16) is the trick that makes 1M-context shippable —
without it the KV cache for a SuperModel 4.7-class model would be ~7TB instead of 880GB.
See  1.3 for the architectural rationale.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sota_model.modeling.kv_cache import PagedKVCache
from sota_model.modeling.rope import RotaryEmbedding, apply_rope


try:  # Flash Attention is the production path; SDPA is the portable fallback.
    from flash_attn import flash_attn_func  # type: ignore

    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope: RotaryEmbedding,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_groups = n_q_heads // n_kv_heads
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_idx = layer_idx
        self.rope = rope

        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[PagedKVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE on (B, H, T, D) layout. Use `logical_position` (not
        # `n_tokens`) so RoPE numbering survives sliding-window eviction.
        q = q.transpose(1, 2)
        k_for_rope = k.transpose(1, 2)
        if kv_cache is not None:
            past_len = kv_cache.logical_position
        else:
            past_len = 0
        if positions is None:
            positions = torch.arange(past_len, past_len + T, device=x.device)
        # Avoid `positions.max().item()` — that's a CPU↔GPU sync per layer.
        cos, sin = self.rope(past_len + T, x.device, x.dtype)
        q, k_rope = apply_rope(q, k_for_rope, cos, sin, positions=positions)

        # Append new K,V to cache (one batch element at a time — caller handles batching of caches).
        if kv_cache is not None and B == 1:
            kv_cache.append(self.layer_idx, k_rope[0].transpose(0, 1), v[0])
            full_k, full_v = kv_cache.gather(self.layer_idx, x.dtype)
            full_k = full_k.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, N, D)
            full_v = full_v.unsqueeze(0).transpose(1, 2)
        else:
            full_k = k_rope
            full_v = v.transpose(1, 2)

        out = self._attend(q, full_k, full_v, attention_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_q_heads * self.head_dim)
        return self.o_proj(out)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Repeat K,V across the GQA group axis. With Flash Attention 3 this can be
        # done implicitly via the grouped kernel; the explicit version below is the
        # portable path and is bandwidth-equivalent under torch.compile.
        if k.shape[1] != q.shape[1]:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Flash Attention requires CUDA + fp16/bf16. Otherwise → SDPA.
        flash_eligible = (
            _HAS_FLASH
            and q.is_cuda
            and mask is None
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if flash_eligible:
            # flash_attn_func expects (B, T, H, D).
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            window = (-1, -1) if self.sliding_window is None else (self.sliding_window, 0)
            out = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True, window_size=window,
            )
            return out.transpose(1, 2)

        # SDPA fallback. Pass `is_causal=True` OR `attn_mask`, never both.
        scale = 1.0 / math.sqrt(self.head_dim)
        if mask is not None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False, scale=scale,
            )
        if self.sliding_window is None:
            return F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True, scale=scale,
            )
        # Causal + sliding window: build explicit keep-mask (True = attend).
        T_q, T_k = q.shape[2], k.shape[2]
        idx_k = torch.arange(T_k, device=q.device)
        idx_q = torch.arange(T_q, device=q.device) + (T_k - T_q)
        causal = idx_q.unsqueeze(1) >= idx_k.unsqueeze(0)
        in_window = (idx_q.unsqueeze(1) - idx_k.unsqueeze(0)) < self.sliding_window
        keep = causal & in_window
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False, scale=scale,
        )
