"""Vision encoder for the SOTA multimodal stack.

Modelcard invariant 6 (CLAUDE.md):
    Multimodal resolution: 2576 px / 3.75 MP. The ScreenSpot-Pro and LAB-Bench
    FigQA gains come from this, not from any new modeling trick. Don't
    compromise on it.

That single number drives all the design choices in this file:

- Long-edge cap is `vision_max_image_long_edge_px = 2576`.
- Total-pixel cap is `vision_max_image_pixels = 3_750_000` (3.75 MP).
- Patch size is `vision_patch_size = 14` (the modelcard 1.4 default).
- Worst case 2576 × 1456 (3.75 MP, 2576 long edge) → ~19,180 patches at 14×14;
  2576 × 2576 would be 33,856 patches but is clipped by the MP cap to
  ~3.75 MP / 14^2 ≈ 19,132 patches. Both fit inside the 1M-token context.

The encoder is a plain ViT (RMSNorm + GQA + SwiGLU, same building blocks as
the text path). The output is projected into the LM's d_model and inserted as
image-region tokens delimited by `<|image_start|>` / `<|image_end|>` from the
chat template.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from sota_model.config import ModelConfig
from sota_model.modeling.attention import GroupedQueryAttention
from sota_model.modeling.layers import RMSNorm, SwiGLU
from sota_model.modeling.rope import RotaryEmbedding


@dataclass
class VisionEncoderConfig:
    patch_size: int = 14
    max_long_edge_px: int = 2_576
    max_pixels: int = 3_750_000
    in_channels: int = 3

    d_model: int = 1_280            # vision tower width; smaller than the LM
    n_layers: int = 32
    n_q_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 80
    ffn_dim: int = 5_120

    norm_eps: float = 1e-5
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @classmethod
    def from_model_config(cls, mc: ModelConfig) -> "VisionEncoderConfig":
        return cls(
            patch_size=mc.vision_patch_size,
            max_long_edge_px=mc.vision_max_image_long_edge_px,
            max_pixels=mc.vision_max_image_pixels,
        )

    def max_patches(self) -> int:
        return self.max_pixels // (self.patch_size * self.patch_size)


@dataclass
class ImageInput:
    """Pre-tokenized image: (C, H, W) float tensor, already normalized.

    `H` and `W` are the post-resize dimensions (multiples of patch_size).
    """
    pixels: torch.Tensor


@dataclass
class VisionFeatures:
    """The encoder output, ready to be projected into the LM.

    `features`: (n_patches, d_model_vision)
    `grid`: the (Hp, Wp) patch grid that produced the features — used to
        rebuild 2D structure for downstream tasks like ScreenSpot-Pro.
    """
    features: torch.Tensor
    grid: tuple[int, int]


def preprocess_image(
    pil_or_tensor,
    cfg: VisionEncoderConfig,
) -> ImageInput:
    """Resize within the modelcard caps, pad to a patch-aligned grid, normalize.

    Strict order:
        1. clip the long edge to `max_long_edge_px`
        2. clip total pixels to `max_pixels` (the 3.75 MP invariant)
        3. round both H,W up to the next multiple of `patch_size`
        4. normalize to ImageNet mean/std

    This module avoids a hard PIL dependency: it accepts either a PIL image
    or an already-decoded `(C,H,W)` torch.Tensor in [0,1].
    """
    if hasattr(pil_or_tensor, "size") and hasattr(pil_or_tensor, "convert"):
        try:
            from torchvision.transforms.functional import to_tensor
            arr = to_tensor(pil_or_tensor.convert("RGB"))
        except ImportError:
            # Manual: PIL → tensor without torchvision.
            import numpy as np
            arr = torch.from_numpy(
                np.asarray(pil_or_tensor.convert("RGB"))
            ).permute(2, 0, 1).float() / 255.0
    else:
        arr = pil_or_tensor
        if arr.dtype != torch.float32:
            arr = arr.float() / (255.0 if arr.dtype == torch.uint8 else 1.0)
    if arr.dim() != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected (3,H,W) image, got shape {tuple(arr.shape)}")

    _, h, w = arr.shape
    long_edge = max(h, w)
    if long_edge > cfg.max_long_edge_px:
        s = cfg.max_long_edge_px / long_edge
        h, w = int(round(h * s)), int(round(w * s))
        arr = _resize(arr, h, w)

    if h * w > cfg.max_pixels:
        s = math.sqrt(cfg.max_pixels / (h * w))
        h, w = int(h * s), int(w * s)
        arr = _resize(arr, h, w)

    # Patch-align: round H,W down to the nearest multiple of patch_size so we
    # don't smear edge pixels with reflection padding when not needed.
    ph = (h // cfg.patch_size) * cfg.patch_size
    pw = (w // cfg.patch_size) * cfg.patch_size
    if ph == 0 or pw == 0:
        ph = pw = cfg.patch_size  # tiny image: 1x1 patches
        arr = _resize(arr, ph, pw)
    elif (ph, pw) != (h, w):
        arr = arr[:, :ph, :pw]

    mean = torch.tensor(cfg.image_mean).view(3, 1, 1)
    std = torch.tensor(cfg.image_std).view(3, 1, 1)
    arr = (arr - mean) / std
    return ImageInput(pixels=arr)


def _resize(arr: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        arr.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)


class VisionPatchEmbedding(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.d_model,
            kernel_size=cfg.patch_size, stride=cfg.patch_size, bias=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        # x: (B, C, H, W); for now B is fixed at 1 — multimodal batching is
        # handled by concatenating image features as separate items.
        if x.dim() == 3:
            x = x.unsqueeze(0)
        feat = self.proj(x)
        B, D, Hp, Wp = feat.shape
        return feat.flatten(2).transpose(1, 2), (Hp, Wp)  # (B, N, D)


class VisionTransformerBlock(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig, rope: RotaryEmbedding):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=cfg.d_model,
            n_q_heads=cfg.n_q_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            rope=rope,
        )
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """A small ViT trained jointly with the LM during stage-3 refinement.

    Image flow at inference:
        bytes → preprocess_image() → VisionEncoder.forward() → VisionFeatures
        VisionFeatures → VisionLanguageProjector → LM token sequence
    """

    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = VisionPatchEmbedding(cfg)
        # 2D rotary: split the head_dim in half and rope-rotate over (H,W).
        # In this minimal version we use 1D RoPE on flattened patches; the
        # block layout is preserved by the projector that follows.
        self.rope = RotaryEmbedding(
            head_dim=cfg.head_dim, base=10_000.0, scale=1.0,
            original_max_position=cfg.max_patches(),
        )
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(cfg, self.rope) for _ in range(cfg.n_layers)
        ])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)

    def forward(self, image: ImageInput) -> VisionFeatures:
        x, grid = self.patch_embed(image.pixels)  # (1, N, D)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return VisionFeatures(features=x.squeeze(0), grid=grid)


def build_vision_encoder(model_cfg: ModelConfig) -> VisionEncoder:
    cfg = VisionEncoderConfig.from_model_config(model_cfg)
    return VisionEncoder(cfg)
