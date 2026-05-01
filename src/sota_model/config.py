"""Model, training, and inference configuration.

The defaults here mirror the SuperModel 4.7-class spec from :
- 200B dense parameters, 110 layers, d_model 16384
- 128 query heads / 16 KV heads (GQA), head_dim 128
- 1M-token context with RoPE base 1e6 + YaRN scaling
- bf16 training, GQA-shaped KV cache
- Adaptive thinking with effort-tier token budgets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


EffortTier = Literal["min", "low", "medium", "high", "max"]


@dataclass
class LayerConfig:
    """Effective per-layer architecture spec.

    Frontier-dense models are NOT uniform. Two heterogeneities the rest of
    this codebase supports without redesigning the paged KV cache:

      - **Per-layer sliding_window** — some layers are full-attention, others
        windowed. Already pattern-based via `sliding_window_layer_stride`;
        layer_overrides lets operators set it explicitly per layer.
      - **Per-layer ffn_dim** — taper FFN width to land precise param targets
        without changing d_model or n_layers. Frontier-dense convention is
        wider FFN at the network's edges (where representation-shaping
        dominates) and narrower in the middle.

    Per-layer `n_kv_heads` or `head_dim` is NOT supported here because the
    paged KV cache assumes a uniform `(n_layers, n_kv_heads, head_dim)` shape;
    changing that requires a cache redesign (left as a future op).
    """
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_dim: int
    sliding_window: Optional[int]

    @property
    def n_kv_groups(self) -> int:
        return self.n_q_heads // self.n_kv_heads


# Fields a `layer_overrides` dict entry is allowed to set. Restricted to the
# subset the KV cache layout can absorb without redesign (see LayerConfig
# docstring).
_LAYER_OVERRIDE_FIELDS: tuple[str, ...] = ("ffn_dim", "sliding_window")


@dataclass
class ModelConfig:
    # Topology
    vocab_size: int = 200_000
    d_model: int = 16_384
    n_layers: int = 110
    n_q_heads: int = 128
    n_kv_heads: int = 16
    head_dim: int = 128
    ffn_dim: int = 65_536
    norm_eps: float = 1e-5
    tie_embeddings: bool = False

    # Position
    max_position_embeddings: int = 1_048_576
    rope_base: float = 1_000_000.0
    rope_yarn_scale: float = 8.0
    rope_yarn_original_max_position: int = 8_192

    # Sliding-window attention applied to a subset of layers (modelcard 1.3 sketch).
    # `sliding_window_layer_stride=k` means every k-th layer (excluding 0) uses
    # sliding-window attention; the rest are full attention. Override per-layer
    # via `layer_overrides[i] = {"sliding_window": null}` (force full attention)
    # or `{"sliding_window": 8192}` (force a custom window).
    sliding_window_size: int = 32_768
    sliding_window_layer_stride: int = 2

    # Sparse per-layer overrides. Keys: layer index 0..n_layers-1. Values: dict
    # whose keys are a subset of `_LAYER_OVERRIDE_FIELDS`. Layers not in this
    # dict inherit the defaults above plus the stride-based sliding window.
    layer_overrides: dict[int, dict] = field(default_factory=dict)

    # Multimodal
    vision_enabled: bool = True
    vision_patch_size: int = 14
    vision_max_image_pixels: int = 3_750_000
    vision_max_image_long_edge_px: int = 2_576

    # Adaptive thinking budgets in tokens
    thinking_budgets: dict[EffortTier, int] = field(
        default_factory=lambda: {
            "min": 0,
            "low": 256,
            "medium": 2_048,
            "high": 8_192,
            "max": 32_768,
        }
    )
    thinking_token_min_floor: int = 32

    # Reserved special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        # YAML loads dict keys as strings; coerce to int.
        if self.layer_overrides:
            self.layer_overrides = {int(k): v for k, v in self.layer_overrides.items()}
        for idx, ov in self.layer_overrides.items():
            if not 0 <= idx < self.n_layers:
                raise ValueError(f"layer_overrides key {idx} out of range [0, {self.n_layers})")
            unknown = set(ov) - set(_LAYER_OVERRIDE_FIELDS)
            if unknown:
                raise ValueError(
                    f"layer_overrides[{idx}]: unsupported fields {sorted(unknown)}; "
                    f"supported = {_LAYER_OVERRIDE_FIELDS}"
                )

    @property
    def n_kv_groups(self) -> int:
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads for GQA")
        return self.n_q_heads // self.n_kv_heads

    def _default_sliding_window(self, layer_idx: int) -> Optional[int]:
        """Stride-based default — layers 2, 4, 6, ... use sliding window."""
        if layer_idx > 0 and layer_idx % self.sliding_window_layer_stride == 0:
            return self.sliding_window_size
        return None

    def layer_config(self, layer_idx: int) -> LayerConfig:
        """Effective per-layer config: defaults + sparse overrides.

        Use `layer_overrides[i] = {"ffn_dim": ..., "sliding_window": ...}` in
        the YAML to make any layer non-uniform. `sliding_window: null` forces
        full attention; a positive int forces a custom window.
        """
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")
        spec = LayerConfig(
            n_q_heads=self.n_q_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            ffn_dim=self.ffn_dim,
            sliding_window=self._default_sliding_window(layer_idx),
        )
        for k, v in self.layer_overrides.get(layer_idx, {}).items():
            setattr(spec, k, v)
        return spec

    def estimate_params_billions(self) -> float:
        """Heterogeneous-aware param count.

        Iterates layers via `layer_config(i)` so per-layer ffn_dim shows up in
        the total. Attention shape (n_q, n_kv, head_dim) is uniform per the
        KV-cache invariant.
        """
        emb = self.vocab_size * self.d_model
        attn_per_layer = (
            self.d_model * self.n_q_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_q_heads * self.head_dim * self.d_model
        )
        attn = self.n_layers * attn_per_layer
        ffn = sum(3 * self.d_model * self.layer_config(i).ffn_dim for i in range(self.n_layers))
        norms = self.n_layers * 2 * self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return (emb + attn + ffn + norms + head) / 1e9

    def per_layer_param_breakdown(self) -> list[dict]:
        """Diagnostic table — one row per layer with its effective shape.

        Useful when calibrating `layer_overrides` against a target param count
        or when auditing whether the heterogeneity matches the modelcard sketch.
        """
        out: list[dict] = []
        attn_per_layer = (
            self.d_model * self.n_q_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_q_heads * self.head_dim * self.d_model
        )
        for i in range(self.n_layers):
            lc = self.layer_config(i)
            out.append({
                "layer": i,
                "ffn_dim": lc.ffn_dim,
                "sliding_window": lc.sliding_window,
                "attn_params": attn_per_layer,
                "ffn_params": 3 * self.d_model * lc.ffn_dim,
            })
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["model"])


def tapered_ffn_overrides(
    n_layers: int,
    edge_layers: int = 4,
    middle_ffn_dim: int = 49_152,
) -> dict[int, dict]:
    """Preset: full ffn_dim at the first/last `edge_layers`, narrower in the middle.

    Frontier-dense convention: representation-shaping load concentrates at the
    network's ends; middle layers refine and have less marginal gain from FFN
    width. Tapering is the cheapest way to land a specific param target without
    changing d_model or n_layers.

    Example: at n_layers=110, edge_layers=4, middle_ffn_dim=49152:
        - Layers 0..3 and 106..109 keep the default ffn_dim.
        - Layers 4..105 (102 layers) drop to 49152.

    Use:
        cfg = ModelConfig(layer_overrides=tapered_ffn_overrides(110))
        print(f"{cfg.estimate_params_billions():.1f} B")
    """
    if edge_layers < 0 or 2 * edge_layers >= n_layers:
        raise ValueError(f"edge_layers={edge_layers} invalid for n_layers={n_layers}")
    return {
        i: {"ffn_dim": middle_ffn_dim}
        for i in range(edge_layers, n_layers - edge_layers)
    }


def hybrid_attention_overrides(
    n_layers: int,
    full_attention_layers: tuple[int, ...] = (0, -1),
) -> dict[int, dict]:
    """Preset: force specific layers to full attention (override the stride pattern).

    Negative indices count from the end (so `-1` means the last layer). Useful
    for guaranteeing the first and last layers are full-attention regardless of
    the stride pattern, which helps long-context retrieval evals (MRCR, GraphWalks).
    """
    overrides: dict[int, dict] = {}
    for raw in full_attention_layers:
        idx = raw if raw >= 0 else n_layers + raw
        if not 0 <= idx < n_layers:
            raise ValueError(f"layer index {raw} out of range for n_layers={n_layers}")
        overrides[idx] = {"sliding_window": None}
    return overrides


@dataclass
class TrainingConfig:
    stage: Literal["foundation", "long_context", "refinement"] = "foundation"
    # Optimizer. AdamW remains the safe frontier-dense default in 2026; Distributed
    # Shampoo (Google scale) and Muon (Keller Jordan, gaining traction in
    # modded-NanoGPT) are documented alternatives but not yet validated at 400B+
    # publicly. Switching is one config line; the rest of the trainer is agnostic.
    optimizer: Literal["adamw", "shampoo", "muon"] = "adamw"
    lr: float = 3e-4
    beta1: float = 0.9
    # 2026 frontier convention: β2=0.95, lower than Adam's 0.999 default. Gives
    # faster adaptation through long-context staging and stage transitions.
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2_000
    total_steps: int = 1_000_000
    seq_len: int = 8_192
    global_batch_tokens: int = 4_194_304
    grad_accum: int = 16
    # Mixed-precision policy. 2026 standard for frontier dense pretraining is FP8
    # mixed (E4M3 forward, E5M2 backward) via NVIDIA Transformer Engine on Blackwell;
    # bf16 remains the fallback for Ampere/Hopper-only stacks. DeepSeek-V3 (2024)
    # and Llama 4 (2025) shipped FP8 native pretraining without quality loss.
    mixed_precision: Literal["fp8", "bf16", "fp16", "fp32"] = "fp8"
    grad_checkpointing: bool = True
    zero_stage: int = 3
    tp_degree: int = 8
    pp_degree: int = 8
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 10

    # Stage-2 long-context overrides applied by training.schedule
    long_doc_mix_ratio: float = 0.4
    sliding_window_layers_enabled: bool = True

    # Stage-3 refinement overrides
    refinement_sources: tuple[str, ...] = (
        "filtered_web_top10pct",
        "code_pr_review",
        "olympiad_math",
        "expert_qa_traces",
        "instruction_following",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["training"])


@dataclass
class InferenceConfig:
    # Sampling defaults. 2026 convention: 0.7 / 0.95 still standard for
    # general assistants; 0.5–0.6 temperature for reasoning-heavy paths
    # (math, code) is increasingly common but tuned per route, not as a
    # global default. top_k=40 is unchanged.
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_new_tokens: int = 8_192
    use_cache: bool = True
    cache_implementation: Literal["paged", "static"] = "paged"
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    # KV cache element type. 2026 frontier convention: FP8 by default on
    # Blackwell (E4M3 cuts memory in half vs bf16 with negligible quality
    # loss); int8 for Hopper/Ampere; FP4 / MXFP4 for memory-bound serving
    # past 1M context. The cache module supports `bf16` and `int8` natively;
    # `fp8`/`fp4` paths require the operator to wire matching kernels.
    kv_cache_dtype: Literal["fp8", "bf16", "fp16", "int8", "fp4"] = "fp8"
    page_block_size: int = 16
    enable_prefix_cache: bool = True

    # Adaptive thinking
    adaptive_thinking: bool = True
    default_effort: EffortTier = "high"
    thinking_visible_to_user: bool = False

    # Long-context. 1M is the standard 2026 frontier window;
    # context compaction at 200k drives the 10M-token agentic-search runs
    # (BrowseComp, DeepSearchQA) referenced in modelcard 8.8.
    context_compaction_trigger: int = 200_000
    max_context_tokens: int = 1_048_576
    max_agentic_context_tokens: int = 10_485_760  # 10M, with compaction

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["inference"])


def load_implied(path: str | Path) -> dict:
    """Load the IMPLIED sections of the YAML config.

    These are the operator-committed values that  does not pin
    numerically but that any compliant build must commit to:

      - implied_scale (param count, KV math)
      - implied_training_corpus (token count + source mix)
      - implied_compute (FLOPs, GPU-hours, weeks, $)
      - implied_schedule_split (3-stage compute share + LR)
      - capability_targets (modelcard 8.1 scoreboard — release gates)
      - safety_thresholds (modelcard 4 / 5 explicit numbers)
      - implied_serving_minimums (deployment requirements)
      - implied_special_tokens_required
      - implied_multilingual_coverage

    Use this for release gating, capacity planning, and validation. The core
    ModelConfig / TrainingConfig / InferenceConfig dataclasses do NOT consume
    these — they are operator-facing metadata.
    """
    data = yaml.safe_load(Path(path).read_text())
    keys = (
        "implied_scale",
        "implied_training_corpus",
        "implied_compute",
        "implied_schedule_split",
        "capability_targets",
        "safety_thresholds",
        "implied_serving_minimums",
        "implied_special_tokens_required",
        "implied_multilingual_coverage",
    )
    return {k: data.get(k) for k in keys if k in data}
