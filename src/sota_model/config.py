"""Model, training, and inference configuration.

The dataclass defaults mirror `configs/sota_ultra_5.yaml` (UltraModel 5), the
single spec this repo ships; the config gate checks they stay in lockstep.
- ~627B dense parameters (verify via ModelConfig().estimate_params_billions()),
  128 layers, d_model 18432
- 144 query heads / 18 KV heads (GQA), head_dim 128
- 1M-token context with RoPE base 1e6 + YaRN scaling
- FP8 mixed-precision training (E4M3 fwd / E5M2 bwd) by default; bf16 fallback
  for Ampere/Hopper-only stacks. GQA-shaped KV cache, fp8 by default.
- Adaptive thinking with effort-tier token budgets
- Optional depth recurrence ("looped transformer"): a weight-shared middle
  core run `loops` times. Off by default (`recurrent_core_layers=0`); the
  effort tier picks the loop count at inference. See docs/LOOPED_TRANSFORMER.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    sliding_window: int | None

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
    d_model: int = 18_432
    n_layers: int = 128
    n_q_heads: int = 144
    n_kv_heads: int = 18
    head_dim: int = 128
    ffn_dim: int = 73_728
    norm_eps: float = 1e-5
    tie_embeddings: bool = False

    # Attention stabilization — RMSNorm on Q and K per head, applied before
    # RoPE (Gemma-2/3 / Qwen-3 / OLMo-2 convention). The 2026-standard
    # replacement for attention-logit softcapping; required for stable fp8
    # training at frontier width. On for UltraModel 5; a variant config can
    # switch it off.
    qk_norm: bool = True
    # Weight init: normal(0, init_std); None → 1/sqrt(d_model). Residual-out
    # projections (attention o_proj, SwiGLU down) are additionally scaled by
    # 1/sqrt(2·n_layers) so residual-stream variance stays O(1) with depth
    # (GPT-2 / PaLM convention).
    init_std: float | None = None

    # Position. The YaRN scale must cover the full extension ratio:
    # max_position_embeddings / rope_yarn_original_max_position
    # (1,048,576 / 8,192 = 128). A smaller scale silently extrapolates past
    # scale × original and breaks the 1M retrieval targets (GraphWalks, MRCR).
    max_position_embeddings: int = 1_048_576
    rope_base: float = 1_000_000.0
    rope_yarn_scale: float = 128.0
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

    # --- Depth recurrence ("looped transformer") ---------------------------
    # The stored stack of `n_layers` unique blocks is split into three bands:
    #
    #     prelude  layers [0, P)              run once
    #     core     layers [P, P+R)            weight-shared, run `loops` times
    #     coda     layers [P+R, n_layers)     run once
    #
    # with P = recurrent_prelude_layers, R = recurrent_core_layers. Each loop
    # re-enters the core with the prelude output added back (input injection,
    # Yang et al. 2023 / Geiping et al. 2025): s_0 = 0, s_i = core(s_{i-1} + e).
    # At loops=1 the forward is identical to the plain stack, so params and
    # the KV layout at loops=1 match a non-recurrent model of the same shape.
    #
    # One execution of a block is a *virtual layer*; the KV cache is indexed
    # by virtual layer, so a sequence run at `loops` uses
    # n_virtual_layers(loops) = P + loops·R + (n_layers − P − R) cache slots.
    # Compute (6·N·D) scales with virtual depth; stored params do not — that
    # decoupling is the whole point (Saunshi et al. 2025, Ouro 2025).
    #
    # recurrent_core_layers = 0 disables recurrence (every layer unique; the
    # flagship behavior). Then prelude must be 0, loops_max 1, and every
    # effort tier maps to 1 loop.
    recurrent_prelude_layers: int = 0
    recurrent_core_layers: int = 0
    # Ceiling on loops. Sizes the KV cache and bounds the training sampler
    # (`sample_train_loops` draws uniformly from 1..loops_max per optimizer
    # step, so every inference depth is trained; expected_train_loops() is
    # the mean the 6·N·D anchor uses).
    recurrent_loops_max: int = 1
    # Inference: effort tier → loop count (the latent-depth axis of the
    # effort ladder, next to `thinking_budgets`, the token axis). Fixed per
    # sequence once chosen — the cache layout depends on it.
    recurrent_loops_per_effort: dict[EffortTier, int] = field(
        default_factory=lambda: {"min": 1, "low": 1, "medium": 1, "high": 1, "max": 1}
    )

    # Multimodal
    vision_enabled: bool = True
    vision_patch_size: int = 14
    vision_max_image_pixels: int = 3_750_000
    vision_max_image_long_edge_px: int = 2_576

    # Adaptive thinking budgets in tokens. The card's "xhigh" effort (USAMO
    # attempts at ~100K tokens) maps onto `max`.
    thinking_budgets: dict[EffortTier, int] = field(
        default_factory=lambda: {
            "min": 0,
            "low": 1_024,
            "medium": 8_192,
            "high": 32_768,
            "max": 131_072,
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
        self._validate_recurrence()

    def _validate_recurrence(self) -> None:
        n_pre, n_core = self.recurrent_prelude_layers, self.recurrent_core_layers
        tiers = ("min", "low", "medium", "high", "max")
        if n_pre < 0 or n_core < 0 or self.recurrent_loops_max < 1:
            raise ValueError(
                "recurrence: prelude/core layers must be >= 0 and recurrent_loops_max >= 1"
            )
        if set(self.recurrent_loops_per_effort) != set(tiers):
            raise ValueError(
                f"recurrent_loops_per_effort must map exactly the tiers {tiers}; "
                f"got {sorted(self.recurrent_loops_per_effort)}"
            )
        if n_core == 0:
            if n_pre != 0 or self.recurrent_loops_max != 1 or any(
                v != 1 for v in self.recurrent_loops_per_effort.values()
            ):
                raise ValueError(
                    "recurrence disabled (recurrent_core_layers=0) but prelude/loops set: "
                    "set recurrent_prelude_layers=0, recurrent_loops_max=1, all tiers → 1"
                )
            return
        if n_pre + n_core > self.n_layers:
            raise ValueError(
                f"recurrence: prelude {n_pre} + core {n_core} exceeds n_layers {self.n_layers}"
            )
        for tier, loops in self.recurrent_loops_per_effort.items():
            if not 1 <= loops <= self.recurrent_loops_max:
                raise ValueError(
                    f"recurrent_loops_per_effort[{tier}]={loops} outside "
                    f"[1, recurrent_loops_max={self.recurrent_loops_max}]"
                )

    @property
    def n_kv_groups(self) -> int:
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads for GQA")
        return self.n_q_heads // self.n_kv_heads

    # --- recurrence geometry ------------------------------------------------

    @property
    def recurrent(self) -> bool:
        return self.recurrent_core_layers > 0

    @property
    def recurrent_coda_layers(self) -> int:
        return self.n_layers - self.recurrent_prelude_layers - self.recurrent_core_layers

    def resolve_loops(self, loops: int | None) -> int:
        """Validate a per-forward loop count; None → full depth (loops_max)."""
        if loops is None:
            return self.recurrent_loops_max
        if not 1 <= loops <= self.recurrent_loops_max:
            raise ValueError(
                f"loops={loops} outside [1, recurrent_loops_max={self.recurrent_loops_max}]"
            )
        return loops

    def n_virtual_layers(self, loops: float | None = None) -> float:
        """Blocks executed per token at `loops` — the KV-cache slot count.

        P + loops·R + C. Accepts a fractional `loops` so the training-mean
        depth (expected_train_loops) can be plugged in for the FLOPs anchor.
        Returns an int whenever `loops` is an int.
        """
        if loops is None:
            loops = self.recurrent_loops_max
        return (
            self.recurrent_prelude_layers
            + loops * self.recurrent_core_layers
            + self.recurrent_coda_layers
        )

    def loops_for_effort(self, tier: EffortTier) -> int:
        return self.recurrent_loops_per_effort[tier]

    def expected_train_loops(self) -> float:
        """Mean loop count under the uniform 1..loops_max training sampler."""
        return (1 + self.recurrent_loops_max) / 2

    def sample_train_loops(self, step: int, seed: int = 0) -> int:
        """Loop count for optimizer step `step` — uniform on 1..loops_max.

        Deterministic in (seed, step) so every rank (TP/PP/DP) and every
        micro-batch of the step unrolls the same virtual depth; pipeline
        stages that hold the core must agree on how many times to re-enter.
        """
        if not self.recurrent:
            return 1
        return random.Random(seed * 1_000_003 + step).randint(1, self.recurrent_loops_max)

    def layer_role(self, layer_idx: int) -> str:
        """'unique' (no recurrence) or 'prelude' / 'core' / 'coda'."""
        if not self.recurrent:
            return "unique"
        n_pre, n_core = self.recurrent_prelude_layers, self.recurrent_core_layers
        if layer_idx < n_pre:
            return "prelude"
        if layer_idx < n_pre + n_core:
            return "core"
        return "coda"

    def _default_sliding_window(self, layer_idx: int) -> int | None:
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

    def _attn_params_per_layer(self) -> int:
        return (
            self.d_model * self.n_q_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_q_heads * self.head_dim * self.d_model
        )

    def _params_per_layer(self, layer_idx: int) -> int:
        """Attention + SwiGLU + the block's norms (incl. QK-norm) at layer_idx."""
        norms = 2 * self.d_model + (2 * self.head_dim if self.qk_norm else 0)
        return (
            self._attn_params_per_layer()
            + 3 * self.d_model * self.layer_config(layer_idx).ffn_dim
            + norms
        )

    def _embedding_params(self) -> int:
        emb = self.vocab_size * self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return emb + head

    def estimate_params_billions(self) -> float:
        """Heterogeneous-aware *stored* param count.

        Iterates layers via `layer_config(i)` so per-layer ffn_dim shows up in
        the total. Attention shape (n_q, n_kv, head_dim) is uniform per the
        KV-cache invariant. Under recurrence the core is counted once — this
        is what a checkpoint holds; `estimate_compute_params_billions` is
        what a token touches.
        """
        layers = sum(self._params_per_layer(i) for i in range(self.n_layers))
        return (self._embedding_params() + layers) / 1e9

    def estimate_compute_params_billions(self, loops: float | None = None) -> float:
        """Params touched per token at `loops` — the N in 6·N·D.

        Prelude and coda once, the core `loops` times (fractional loops allowed
        for the training mean). Equals estimate_params_billions() when
        recurrence is off or loops=1.
        """
        if loops is None:
            loops = self.recurrent_loops_max
        n_pre, n_core = self.recurrent_prelude_layers, self.recurrent_core_layers
        stored = sum(self._params_per_layer(i) for i in range(self.n_layers))
        core = sum(self._params_per_layer(i) for i in range(n_pre, n_pre + n_core))
        return (self._embedding_params() + stored + (loops - 1) * core) / 1e9

    def kv_cache_bytes_per_token(
        self, bytes_per_element: int = 2, loops: int | None = None
    ) -> int:
        """Per-token KV-cache bytes at the uniform cache shape.

        n_virtual_layers(loops) × n_kv_heads × head_dim × 2 (K+V) ×
        bytes_per_element. bytes_per_element: 2 for bf16/fp16 (default), 1 for
        fp8/int8. `loops` defaults to recurrent_loops_max (the provisioning
        worst case; = n_layers when recurrence is off). Uniform across layers
        by the paged-KV-cache invariant (see LayerConfig), so per-layer
        overrides never change this number. Single home of the formula behind
        `implied_scale.kv_cache_kib_per_token_bf16*` (validated by
        scripts/validate_config.py).
        """
        n_virtual = int(self.n_virtual_layers(self.resolve_loops(loops)))
        return n_virtual * self.n_kv_heads * self.head_dim * 2 * bytes_per_element

    def training_flops(self, total_tokens: float, loops: float | None = None) -> float:
        """Pretraining FLOPs by the standard 6·N·D estimate.

        N = estimate_compute_params_billions(loops) × 1e9 — dense, every
        touched param counts, and under recurrence the core counts once per
        loop. `loops` defaults to expected_train_loops() (the mean of the
        uniform training sampler; 1 when recurrence is off, which reduces to
        6 × stored params × D). D = total_tokens. Single home of the formula
        behind `implied_compute.training_flops_point_estimate` (validated by
        scripts/validate_config.py); the GPU-hour / wall-clock / dollar bands
        in that section all derive from this number via `gpu_hours`.
        """
        if loops is None:
            loops = self.expected_train_loops()
        return 6.0 * self.estimate_compute_params_billions(loops) * 1e9 * total_tokens

    def per_layer_param_breakdown(self) -> list[dict]:
        """Diagnostic table — one row per layer with its effective shape.

        Useful when calibrating `layer_overrides` against a target param count
        or when auditing whether the heterogeneity matches the modelcard sketch.
        """
        out: list[dict] = []
        attn_per_layer = self._attn_params_per_layer()
        for i in range(self.n_layers):
            lc = self.layer_config(i)
            out.append({
                "layer": i,
                "role": self.layer_role(i),
                "ffn_dim": lc.ffn_dim,
                "sliding_window": lc.sliding_window,
                "attn_params": attn_per_layer,
                "ffn_params": 3 * self.d_model * lc.ffn_dim,
            })
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["model"])


def gpu_hours(flops: float, peak_flops_per_gpu: float, mfu: float) -> float:
    """GPU-hours to spend `flops` at a sustained fraction `mfu` of peak.

    flops ÷ (peak_flops_per_gpu × mfu) ÷ 3600 — the formula behind the
    `implied_compute.*_hours_band` derivation comments in the YAMLs, e.g.
    UltraModel 5: gpu_hours(1.354e26, 7.5e15, 0.40 … 0.50) ≈ 12.5M … 10.0M
    B300-hours (B300 fp8 dense peak 7.5 PFLOPS at 40–50% MFU).
    """
    return flops / (peak_flops_per_gpu * mfu) / 3600.0


def tapered_ffn_overrides(
    n_layers: int,
    edge_layers: int = 4,
    middle_ffn_dim: int = 55_296,
) -> dict[int, dict]:
    """Preset: full ffn_dim at the first/last `edge_layers`, narrower in the middle.

    Frontier-dense convention: representation-shaping load concentrates at the
    network's ends; middle layers refine and have less marginal gain from FFN
    width. Tapering is the cheapest way to land a specific param target without
    changing d_model or n_layers.

    Example: at n_layers=128, edge_layers=4, middle_ffn_dim=55296 (3 × d_model):
        - Layers 0..3 and 124..127 keep the default ffn_dim (73728).
        - Layers 4..123 (120 layers) drop to 55296 → ~505 B instead of 627 B.

    Use:
        cfg = ModelConfig(layer_overrides=tapered_ffn_overrides(128))
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
    # Peak LR, width-scaled µP-style (LR ∝ 1/d_model) with fp8 headroom.
    lr: float = 2.5e-4
    beta1: float = 0.9
    # 2026 frontier convention: β2=0.95, lower than Adam's 0.999 default. Gives
    # faster adaptation through long-context staging and stage transitions.
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 3_000
    # Full-corpus optimizer-step budget: 36T tokens ÷ 8M tokens/step ≈ 4.29M.
    total_steps: int = 4_300_000
    seq_len: int = 8_192
    # The global batch decomposes exactly — enforced in __post_init__:
    #   global_batch_tokens = seq_len × micro_batch_size × grad_accum × dp_degree
    # Defaults: 8192 × 1 × 16 × 64 = 8,388,608 (the UltraModel 5 shape).
    global_batch_tokens: int = 8_388_608
    micro_batch_size: int = 1   # sequences per replica per micro-step
    grad_accum: int = 16
    # Mixed-precision policy.
    #   2026 default: FP8 mixed (E4M3 fwd / E5M2 bwd) via NVIDIA Transformer
    #     Engine on Blackwell. DeepSeek-V3 (2024) and Llama 4 (2025) shipped
    #     FP8 native pretraining without quality loss.
    #   2026+ Rubin: NVFP4 mixed via Transformer Engine 2 — 35 PFLOPS train,
    #     50 PFLOPS inference per Rubin GPU (3.5x / 5x Blackwell). Set
    #     mixed_precision="nvfp4" once the operator's launcher is on TE2.
    #   Fallback: bf16 for Ampere/Hopper-only stacks.
    mixed_precision: Literal["nvfp4", "fp8", "mxfp8", "bf16", "fp16", "fp32"] = "fp8"
    grad_checkpointing: bool = True
    zero_stage: int = 3
    # GB300 NVL72 layout — one replica = one rack (tp × pp = 9 × 8 = 72 GPUs,
    # the NVLink domain). TP must divide n_kv_heads: 18 shards at 9 or 6, not 8.
    tp_degree: int = 9
    pp_degree: int = 8
    # Data-parallel replica count. world_size = tp × pp × dp; one replica spans
    # tp × pp GPUs. Defaults land 9 × 8 × 64 = 4608 GPUs (64 racks).
    dp_degree: int = 64
    # Cosine-schedule floor as a fraction of peak LR (both trainer paths).
    lr_min_ratio: float = 0.1
    # Z-loss coefficient (PaLM / OLMo-2): penalizes log²Z to keep output
    # logits from drifting — matters under fp8. 0 disables; 1e-4 standard.
    z_loss_coef: float = 1e-4
    # Seed for corpus interleave + shuffle buffer (reproducible data order).
    data_seed: int = 1234
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

    def __post_init__(self) -> None:
        if self.global_batch_tokens % self.seq_len:
            raise ValueError(
                f"global_batch_tokens ({self.global_batch_tokens}) must be a "
                f"multiple of seq_len ({self.seq_len})"
            )
        expected = self.micro_batch_size * self.grad_accum * self.dp_degree
        if self.sequences_per_step != expected:
            raise ValueError(
                "batch arithmetic broken: global_batch_tokens / seq_len = "
                f"{self.sequences_per_step} sequences/step, but micro_batch_size "
                f"× grad_accum × dp_degree = {self.micro_batch_size} × "
                f"{self.grad_accum} × {self.dp_degree} = {expected}"
            )

    @property
    def sequences_per_step(self) -> int:
        return self.global_batch_tokens // self.seq_len

    @property
    def world_size(self) -> int:
        return self.tp_degree * self.pp_degree * self.dp_degree

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["training"])


@dataclass
class InferenceConfig:
    # Sampling defaults. 2026 frontier serving samples with temperature +
    # top_p only (0.7 / 0.95; lower temperature per reasoning route, not as a
    # global default). top_k and repetition penalty are 2023-era local-runtime
    # knobs — the sampler treats top_k <= 0 and penalty 1.0 as no-ops.
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    # Chat default; agentic computer-use raises the per-turn cap to 128K.
    max_new_tokens: int = 32_768
    use_cache: bool = True
    cache_implementation: Literal["paged", "static"] = "paged"
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    # KV cache element type.
    #   Rubin (2026+): NVFP4 — block-of-16 FP4 with FP8 (E4M3) per-block scale
    #     + per-tensor FP32 master scale. ~4.5 effective bits/value.
    #     ~5x throughput of B200 MXFP4 inference, retains ~0.5pp more accuracy.
    #   Blackwell (2025+): fp8 by default; mxfp4/fp4 for memory-bound past 1M.
    #   Hopper / Ampere: bf16 / int8 (the cache module supports both natively).
    # The cache module ships kernels for `bf16` and `int8`; `fp8` / `fp4` /
    # `nvfp4` paths require the operator to wire matching kernels.
    kv_cache_dtype: Literal["nvfp4", "fp8", "mxfp4", "bf16", "fp16", "int8", "fp4"] = "fp8"
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
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["inference"])


def load_implied(path: str | Path) -> dict:
    """Load the IMPLIED sections of the YAML config.

    These are the operator-committed values that the system card does not pin
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
