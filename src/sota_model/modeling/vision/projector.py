"""Project vision features into the LM's token-embedding space.

Two paths supported:

1. **Linear projection** — vision features → LM d_model directly. Used in
   small experiments and tests because it doesn't require an extra training
   stage.
2. **Pixel-shuffle + MLP** — typical of frontier multimodal stacks. We tile
   2x2 patches into one merged token to bring 2576px image down to a more
   manageable token count without losing OCR-relevant detail.

Modelcard 8.9 ScreenSpot-Pro is a UI-grounding eval; the patch grid must
survive into the LM, otherwise spatial precision is lost. The projector
always returns a (n_image_tokens, lm_d_model) tensor and the original grid.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from sota_model.modeling.layers import RMSNorm
from sota_model.modeling.vision.encoder import VisionFeatures


class VisionLanguageProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        lm_dim: int,
        method: Literal["linear", "pixel_shuffle_mlp"] = "pixel_shuffle_mlp",
        shuffle_factor: int = 2,
    ):
        super().__init__()
        self.method = method
        self.shuffle_factor = shuffle_factor

        if method == "linear":
            self.proj = nn.Linear(vision_dim, lm_dim, bias=False)
        elif method == "pixel_shuffle_mlp":
            in_dim = vision_dim * shuffle_factor * shuffle_factor
            self.proj = nn.Sequential(
                RMSNorm(in_dim),
                nn.Linear(in_dim, lm_dim, bias=False),
                nn.GELU(),
                nn.Linear(lm_dim, lm_dim, bias=False),
            )
        else:
            raise ValueError(f"unknown projector method: {method}")

    def forward(self, vf: VisionFeatures) -> tuple[torch.Tensor, tuple[int, int]]:
        feats, grid = vf.features, vf.grid
        Hp, Wp = grid

        if self.method == "linear":
            return self.proj(feats), grid

        # pixel_shuffle_mlp: reshape to (Hp, Wp, D), tile into 2x2 patches,
        # then project to LM dim. Only works when Hp,Wp divisible by factor.
        f = self.shuffle_factor
        if Hp % f != 0 or Wp % f != 0:
            # fall back to dropping the last partial row/col
            Hp = (Hp // f) * f
            Wp = (Wp // f) * f
            feats = feats[: Hp * Wp]

        D = feats.shape[-1]
        feats = feats.view(Hp, Wp, D)
        feats = feats.view(Hp // f, f, Wp // f, f, D)
        feats = feats.permute(0, 2, 1, 3, 4).contiguous()
        feats = feats.view(Hp // f, Wp // f, f * f * D)
        feats = feats.view(-1, f * f * D)
        return self.proj(feats), (Hp // f, Wp // f)
