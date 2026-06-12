"""Rotary Position Embeddings with YaRN scaling for 1M-token context.

Reference: YaRN (Peng et al., 2023). The scaling lets us extend a model trained at
8K seq len out to 1M without retraining from scratch.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def _yarn_corrected_freqs(
    base: float,
    head_dim: int,
    scale: float,
    original_max_position: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN-corrected inverse frequencies for RoPE.

    For each band, α = original_max_position / wavelength:
        α >= beta_fast  → smooth=1, full extrapolation (high freq, short λ)
        α <= beta_slow  → smooth=0, full interpolation (low freq, long λ)
        in between      → smooth blend

    Required for the modelcard 8.7 1M-context targets (MRCR v2, GraphWalks
    BFS @ 1M). Pure linear interpolation blurs short-range signal; pure
    extrapolation breaks long-range generalization.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if scale <= 1.0:
        return inv_freq

    wavelens = 2 * math.pi / inv_freq
    alpha = original_max_position / wavelens
    smooth = torch.clamp((alpha - beta_slow) / (beta_fast - beta_slow), min=0.0, max=1.0)

    return smooth * inv_freq + (1.0 - smooth) * (inv_freq / scale)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 1_000_000.0,
        scale: float = 1.0,
        original_max_position: int = 8_192,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scale = scale
        self.original_max_position = original_max_position
        inv_freq = _yarn_corrected_freqs(base, head_dim, scale, original_max_position)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @lru_cache(maxsize=4)
    def _cached(self, seq_len: int, device_str: str, dtype: torch.dtype):
        device = torch.device(device_str)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        return self._cached(seq_len, str(device), dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q,k of shape (B, H, T, D)."""
    if positions is not None:
        cos = cos[positions]
        sin = sin[positions]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
