"""Merge utilities for staged training.

Two operations supported:

  - `merge_lora_into_base` — LoRA-style adapter merge for the SFT/RLHF
    stages where operators may train an adapter rather than a full
    fine-tune. Modelcard 6.2 doesn't pin this; the merge is here so that a
    deployed checkpoint never has to load its adapters separately.
  - `interpolate_state_dicts` — linear interpolation between two compatible
    checkpoints, used for model souping during stage transitions.

Both operate on `state_dict`s, so the manager `save_checkpoint` produces a
mergeable artifact directly.
"""

from __future__ import annotations

from typing import Mapping

import torch


def merge_lora_into_base(
    base: Mapping[str, torch.Tensor],
    adapter: Mapping[str, torch.Tensor],
    *,
    scale: float = 1.0,
    a_suffix: str = ".lora_a",
    b_suffix: str = ".lora_b",
) -> dict[str, torch.Tensor]:
    """Merge LoRA `A · B` deltas into the corresponding base weight.

    Naming convention: a parameter `foo.weight` has adapters
    `foo.weight.lora_a` (r × in) and `foo.weight.lora_b` (out × r). The
    merged delta is `(B @ A) * scale`, added to the original.
    """
    merged = {k: v.clone() for k, v in base.items()}

    # Find pairs (a, b) keyed by the parameter they decorate.
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, val in adapter.items():
        if key.endswith(a_suffix):
            base_key = key[: -len(a_suffix)]
            pairs.setdefault(base_key, [None, None])[0] = val
        elif key.endswith(b_suffix):
            base_key = key[: -len(b_suffix)]
            pairs.setdefault(base_key, [None, None])[1] = val

    for key, ab in pairs.items():
        a, b = ab[0], ab[1]
        if a is None or b is None:
            continue
        if key not in merged:
            continue
        delta = (b @ a) * scale
        if delta.shape != merged[key].shape:
            raise ValueError(
                f"adapter delta shape {tuple(delta.shape)} != base {tuple(merged[key].shape)} for {key}"
            )
        merged[key] = merged[key] + delta.to(merged[key].dtype)
    return merged


def interpolate_state_dicts(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Linear interp: out = (1 - alpha) * a + alpha * b.

    Used for model souping at stage boundaries. Only keys present in both
    state dicts are interpolated; others are taken from `a` unchanged.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    out: dict[str, torch.Tensor] = {}
    for k, v in a.items():
        if k in b and b[k].shape == v.shape and b[k].dtype == v.dtype:
            out[k] = ((1.0 - alpha) * v + alpha * b[k]).to(v.dtype)
        else:
            out[k] = v.clone()
    return out
