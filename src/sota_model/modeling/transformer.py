"""Top-level transformer model with adaptive-thinking effort head.

Capability targets are defined in modelcard §8 (the source of truth) — at the
UltraModel 5 spec: SWE-bench Verified ≥ 95%, GPQA Diamond ≥ 94%, Terminal-Bench
2.1 ≥ 84%, OSWorld ≥ 85%, GraphWalks BFS @ 1M ≥ 79%.

Architecture reference: docs/ARCHITECTURE.md. Depth recurrence (the looped
prelude → core×loops → coda forward, off unless `ModelConfig.recurrent_core_layers`
is set): docs/LOOPED_TRANSFORMER.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

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
    effort_logit: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


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
            qk_norm=cfg.qk_norm,
            norm_eps=cfg.norm_eps,
        )
        self.post_attn_norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, lc.ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: PagedKVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        cache_layer: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.input_norm(x),
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            positions=positions,
            cache_layer=cache_layer,
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
    """Decoder-only LM: embed → blocks → final norm → LM head (+ effort head).

    `self.layers` holds the `cfg.n_layers` *stored* blocks. Without recurrence
    they run once each. With recurrence (`cfg.recurrent_core_layers > 0`)
    they split into prelude / core / coda and the core is re-entered `loops`
    times with the prelude output injected additively on every entry
    (Huginn-style; Geiping et al. 2025):

        e   = prelude(embed(x))
        s_0 = 0
        s_i = core(s_{i-1} + e)        i = 1..loops
        y   = coda(s_loops)

    `loops=1` reduces to the plain stack bit-for-bit (s_1 = core(e)), so a
    checkpoint trained without recurrence is a valid loops=1 looped model.
    `unrolled_blocks(loops)` is the single source of truth for execution
    order and KV-cache slots; every consumer that walks the layers by hand
    (RLHF, reward model, probes) must iterate it, not `self.layers`.
    """

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
        self.vision_encoder: VisionEncoder | None = None
        self.vision_projector: VisionLanguageProjector | None = None
        if cfg.vision_enabled:
            self.vision_encoder = build_vision_encoder(cfg)
            self.vision_projector = VisionLanguageProjector(
                vision_dim=self.vision_encoder.cfg.d_model,
                lm_dim=cfg.d_model,
                method="pixel_shuffle_mlp",
            )

        self.gradient_checkpointing = False
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Width-aware init (2026 standard, replaces PyTorch layer defaults).

        All weights ~ normal(0, init_std) where init_std defaults to
        1/sqrt(d_model). Residual-out projections (attention `o_proj`, SwiGLU
        `down`) are further scaled by 1/sqrt(2·n_layers) so residual-stream
        variance stays O(1) at depth (GPT-2 / PaLM residual-scaled init).
        """
        std = self.cfg.init_std if self.cfg.init_std is not None else self.cfg.d_model ** -0.5
        # Depth for the residual scale is the *stored* depth, not the unrolled
        # one: with input injection each loop re-reads e rather than stacking
        # residual writes indefinitely, and a loops=1 run must initialise
        # exactly like the non-recurrent model. Revisit if loops_max grows
        # past ~4 (Huginn trains stably at r≈32 with this choice).
        resid_scale = (2 * self.cfg.n_layers) ** -0.5
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                s = std * resid_scale if name.endswith((".o_proj", ".down")) else std
                nn.init.normal_(module.weight, mean=0.0, std=s)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def make_kv_cache(
        self,
        dtype: str = "bf16",
        sliding_window: int | None = None,
        loops: int | None = None,
    ) -> PagedKVCache:
        """Allocate a cache for one sequence.

        Storage is sized for the unrolled depth at `recurrent_loops_max` (the
        provisioning number `implied_scale.kv_cache_*` quotes). `loops` pins
        the sequence's active virtual depth; it defaults to loops_max and is
        a no-op for non-recurrent configs (n_virtual_layers == n_layers).
        """
        loops = self.cfg.resolve_loops(loops)
        cache = PagedKVCache(
            KVCacheConfig(
                n_layers=int(self.cfg.n_virtual_layers()),
                n_kv_heads=self.cfg.n_kv_heads,
                head_dim=self.cfg.head_dim,
                dtype=dtype,
                sliding_window=sliding_window,
            ),
            device=next(self.parameters()).device,
        )
        cache.set_active_layers(int(self.cfg.n_virtual_layers(loops)))
        return cache

    # --- recurrence plumbing ---------------------------------------------

    @property
    def prelude(self) -> list[SOTATransformerBlock]:
        return list(self.layers[: self.cfg.recurrent_prelude_layers])

    @property
    def core(self) -> list[SOTATransformerBlock]:
        n_pre, n_core = self.cfg.recurrent_prelude_layers, self.cfg.recurrent_core_layers
        return list(self.layers[n_pre : n_pre + n_core])

    @property
    def coda(self) -> list[SOTATransformerBlock]:
        start = self.cfg.recurrent_prelude_layers + self.cfg.recurrent_core_layers
        return list(self.layers[start:])

    def unrolled_blocks(
        self, loops: int | None = None
    ) -> Iterator[tuple[SOTATransformerBlock, int, str]]:
        """Yield (block, virtual_layer_idx, phase) in execution order.

        phase ∈ {"prelude", "core", "coda"} ("unique" when recurrence is off,
        in which case this is just enumerate(self.layers)). Virtual indices
        are contiguous 0..n_virtual_layers(loops)-1 and are the KV-cache slots;
        the KV cache is told the count via `set_active_layers`.
        """
        if not self.cfg.recurrent:
            for i, block in enumerate(self.layers):
                yield block, i, "unique"
            return
        loops = self.cfg.resolve_loops(loops)
        v = 0
        for block in self.prelude:
            yield block, v, "prelude"
            v += 1
        for _ in range(loops):
            for block in self.core:
                yield block, v, "core"
                v += 1
        for block in self.coda:
            yield block, v, "coda"
            v += 1

    def _run_block(
        self,
        block: SOTATransformerBlock,
        h: torch.Tensor,
        kv_cache: PagedKVCache | None,
        attention_mask: torch.Tensor | None,
        positions: torch.Tensor | None,
        cache_layer: int,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                block, h, kv_cache, attention_mask, positions, cache_layer, use_reentrant=False
            )
        return block(
            h, kv_cache=kv_cache, attention_mask=attention_mask, positions=positions,
            cache_layer=cache_layer,
        )

    def run_blocks(
        self,
        h: torch.Tensor,
        kv_cache: PagedKVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        loops: int | None = None,
        stop_after_virtual: int | None = None,
    ) -> torch.Tensor:
        """embed-output → last-block-output (pre final_norm) at `loops`.

        The one place the prelude / core×loops / coda control flow lives.
        Input injection: the prelude output `e` is added to the residual
        stream at every core entry, and the core state starts at zero, so
        loop 1 sees exactly `e` and the plain-stack forward is a special case.

        `stop_after_virtual=v` returns the residual stream right after virtual
        layer v (for probes / feature extraction) instead of the final output.
        """
        if not self.cfg.recurrent:
            for block, v, _ in self.unrolled_blocks():
                h = self._run_block(block, h, kv_cache, attention_mask, positions, v)
                if v == stop_after_virtual:
                    break
            return h

        loops = self.cfg.resolve_loops(loops)
        if kv_cache is not None:
            expected = int(self.cfg.n_virtual_layers(loops))
            if kv_cache.active_layers != expected:
                raise ValueError(
                    f"kv_cache active depth {kv_cache.active_layers} != {expected} virtual "
                    f"layers at loops={loops}; allocate it with make_kv_cache(loops=...)"
                )

        e = h                                   # prelude output (after the prelude runs)
        s: torch.Tensor | None = None        # core state; None ≡ s_0 = 0
        for block, v, phase in self.unrolled_blocks(loops):
            if phase == "prelude":
                e = self._run_block(block, e, kv_cache, attention_mask, positions, v)
            elif phase == "core":
                if self._is_core_entry(v):
                    # Loop entry: s_{i-1} + e. With s_0 = 0 the first entry is just e.
                    s = e if s is None else s + e
                s = self._run_block(block, s, kv_cache, attention_mask, positions, v)
            else:  # coda — recurrence implies R ≥ 1, so the core has run and s is set
                s = self._run_block(block, s, kv_cache, attention_mask, positions, v)
            if v == stop_after_virtual:
                return e if s is None else s
        return s

    def _is_core_entry(self, virtual_idx: int) -> bool:
        n_pre, n_core = self.cfg.recurrent_prelude_layers, self.cfg.recurrent_core_layers
        return (virtual_idx - n_pre) % n_core == 0

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
        kv_cache: PagedKVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        compute_effort: bool = False,
        image_features: torch.Tensor | None = None,
        image_token_id: int | None = None,
        loops: int | None = None,
    ) -> ModelOutput:
        """Standard forward.

        `loops`: core recurrence count for this pass (recurrent configs only;
        None → recurrent_loops_max). Training samples it per step via
        `cfg.sample_train_loops`; inference derives it from the effort tier
        via `cfg.loops_for_effort` and holds it for the whole sequence.

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

        h = self.run_blocks(
            h, kv_cache=kv_cache, attention_mask=attention_mask, positions=positions, loops=loops
        )
        h = self.final_norm(h)

        logits = h @ self.embed.weight.T if self.lm_head is None else self.lm_head(h)

        effort = self.effort_head(h) if compute_effort else None
        return ModelOutput(logits=logits, effort_logit=effort, hidden_states=None)

    def num_parameters(self) -> int:
        """Stored (checkpoint) parameters — the core counted once."""
        return sum(p.numel() for p in self.parameters())

    def num_compute_parameters(self, loops: int | None = None) -> int:
        """Parameters touched per token at `loops` (the N of 6·N·D)."""
        loops = self.cfg.resolve_loops(loops)
        core = sum(p.numel() for blk in self.core for p in blk.parameters())
        return self.num_parameters() + (loops - 1) * core

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True


def build_model(cfg: ModelConfig) -> SOTAModel:
    return SOTAModel(cfg)
