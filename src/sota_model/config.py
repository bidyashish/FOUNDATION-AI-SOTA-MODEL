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
from typing import Literal

import yaml


EffortTier = Literal["min", "low", "medium", "high", "max"]


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

    # Sliding-window attention applied to a subset of layers (modelcard 1.3 sketch)
    sliding_window_size: int = 32_768
    sliding_window_layer_stride: int = 2

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

    @property
    def n_kv_groups(self) -> int:
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads for GQA")
        return self.n_q_heads // self.n_kv_heads

    def estimate_params_billions(self) -> float:
        emb = self.vocab_size * self.d_model
        attn = self.n_layers * (
            self.d_model * self.n_q_heads * self.head_dim
            + 2 * self.d_model * self.n_kv_heads * self.head_dim
            + self.n_q_heads * self.head_dim * self.d_model
        )
        ffn = self.n_layers * (3 * self.d_model * self.ffn_dim)
        norms = self.n_layers * 2 * self.d_model
        head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return (emb + attn + ffn + norms + head) / 1e9

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data["model"])


@dataclass
class TrainingConfig:
    stage: Literal["foundation", "long_context", "refinement"] = "foundation"
    #  does not prescribe an optimizer; AdamW is the framework-neutral
    # default. Operators are free to swap to whatever the production run actually uses.
    optimizer: Literal["adamw"] = "adamw"
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2_000
    total_steps: int = 1_000_000
    seq_len: int = 8_192
    global_batch_tokens: int = 4_194_304
    grad_accum: int = 16
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
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
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_new_tokens: int = 8_192
    use_cache: bool = True
    cache_implementation: Literal["paged", "static"] = "paged"
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    kv_cache_dtype: Literal["bf16", "fp16", "int8"] = "bf16"
    page_block_size: int = 16
    enable_prefix_cache: bool = True

    # Adaptive thinking
    adaptive_thinking: bool = True
    default_effort: EffortTier = "high"
    thinking_visible_to_user: bool = False

    # Long-context
    context_compaction_trigger: int = 200_000
    max_context_tokens: int = 1_048_576

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
