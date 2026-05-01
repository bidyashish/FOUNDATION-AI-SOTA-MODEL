"""Top-level transformer model with adaptive-thinking effort head.

Capability targets are defined in  8 (the source of truth):
SWE-bench Verified ≥ 87%, GPQA Diamond ≥ 94%, ARC-AGI-2 ≥ 75%, OSWorld ≥ 78%, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from sota_model.config import ModelConfig
from sota_model.modeling.attention import GroupedQueryAttention
from sota_model.modeling.kv_cache import KVCacheConfig, PagedKVCache
from sota_model.modeling.layers import RMSNorm, SwiGLU
from sota_model.modeling.rope import RotaryEmbedding
from sota_model.modeling.vision import (
    VisionEncoder,
    VisionLanguageProjector,
    build_vision_encoder,
)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    effort_logit: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


class SOTATransformerBlock(nn.Module):
    """One transformer block.

    Reads its effective shape from `cfg.layer_config(layer_idx)` so per-layer
    `ffn_dim` and `sliding_window` overrides flow through automatically.
    Frontier-dense models are not uniform across depth — see
    `ModelConfig.layer_overrides` and presets like `tapered_ffn_overrides`.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        rope: RotaryEmbedding,
        layer_idx: int,
    ):
        super().__init__()
        lc = cfg.layer_config(layer_idx)
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=cfg.d_model,
            n_q_heads=lc.n_q_heads,
            n_kv_heads=lc.n_kv_heads,
            head_dim=lc.head_dim,
            rope=rope,
            sliding_window=lc.sliding_window,
            layer_idx=layer_idx,
        )
        self.post_attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, lc.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[PagedKVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.input_norm(x),
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            positions=positions,
        )
        x = x + self.ffn(self.post_attn_norm(x))
        return x


class EffortHead(nn.Module):
    """Predicts the adaptive-thinking effort tier from late-layer hidden states.

    Output is a single scalar logit per sequence; mapped to {min, low, medium,
    high, max} via thresholds learned during RL post-training.
    """

    def __init__(self, d_model: int, hidden: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Pool over the last 8 tokens of the prompt — small, latency-cheap.
        pooled = h[:, -8:].mean(dim=1)
        return self.proj(pooled).squeeze(-1)


class SOTAModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(
            head_dim=cfg.head_dim,
            base=cfg.rope_base,
            scale=cfg.rope_yarn_scale,
            original_max_position=cfg.rope_yarn_original_max_position,
        )
        self.layers = nn.ModuleList(
            [SOTATransformerBlock(cfg, self.rope, i) for i in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.effort_head = EffortHead(cfg.d_model)

        # Multimodal — built lazily so text-only checkpoints stay small.
        self.vision_encoder: Optional[VisionEncoder] = None
        self.vision_projector: Optional[VisionLanguageProjector] = None
        if cfg.vision_enabled:
            self.vision_encoder = build_vision_encoder(cfg)
            self.vision_projector = VisionLanguageProjector(
                vision_dim=self.vision_encoder.cfg.d_model,
                lm_dim=cfg.d_model,
                method="pixel_shuffle_mlp",
            )

        self.gradient_checkpointing = False

    def make_kv_cache(
        self, dtype: str = "bf16", sliding_window: Optional[int] = None
    ) -> PagedKVCache:
        return PagedKVCache(
            KVCacheConfig(
                n_layers=self.cfg.n_layers,
                n_kv_heads=self.cfg.n_kv_heads,
                head_dim=self.cfg.head_dim,
                dtype=dtype,
                sliding_window=sliding_window,
            ),
            device=next(self.parameters()).device,
        )

    def encode_image(self, image) -> torch.Tensor:
        """Run an image through the vision encoder + projector.

        Returns: (n_image_tokens, d_model) ready to splice into the LM input.
        Raises: RuntimeError if the model was built without vision_enabled.
        """
        if self.vision_encoder is None or self.vision_projector is None:
            raise RuntimeError("vision encoder not enabled — set ModelConfig.vision_enabled=True")
        from sota_model.modeling.vision import ImageInput, preprocess_image
        if not isinstance(image, ImageInput):
            image = preprocess_image(image, self.vision_encoder.cfg)
        device = next(self.parameters()).device
        image.pixels = image.pixels.to(device)
        vf = self.vision_encoder(image)
        toks, _ = self.vision_projector(vf)
        return toks

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[PagedKVCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        compute_effort: bool = False,
        image_features: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
    ) -> ModelOutput:
        """Standard forward.

        Multimodal: if `image_features` is provided (n_img_tokens, d_model),
        the embedded `image_token_id` placeholders in `input_ids` are
        replaced 1-to-1 with rows from `image_features`. The chat template
        emits exactly `n_img_tokens` placeholders between
        `<|image_start|>` / `<|image_end|>`, so the splice is deterministic.
        """
        h = self.embed(input_ids)

        if image_features is not None and image_token_id is not None:
            mask = (input_ids == image_token_id)
            n_slots = int(mask.sum().item())
            if n_slots != image_features.shape[0]:
                raise ValueError(
                    f"image_features has {image_features.shape[0]} tokens "
                    f"but prompt has {n_slots} <|image|> placeholder slots"
                )
            h = h.clone()
            h[mask] = image_features.to(h.dtype).to(h.device)

        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, kv_cache, attention_mask, positions, use_reentrant=False
                )
            else:
                h = block(h, kv_cache=kv_cache, attention_mask=attention_mask, positions=positions)
        h = self.final_norm(h)

        if self.lm_head is None:
            logits = h @ self.embed.weight.T
        else:
            logits = self.lm_head(h)

        effort = self.effort_head(h) if compute_effort else None
        return ModelOutput(logits=logits, effort_logit=effort, hidden_states=None)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True


def build_model(cfg: ModelConfig) -> SOTAModel:
    return SOTAModel(cfg)
