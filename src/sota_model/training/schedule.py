"""Three-stage pretraining schedule.

Stage 1 — Foundation (~70% of compute): seq 8K, lr 3e-4, broad data mix.
Stage 2 — Long context (~20%): seq up to 1M, RoPE base 1e6 + YaRN x8, sliding window.
Stage 3 — Quality refinement (~10%): top-decile data, lr 5e-5, instruction-style.

See  1.1.1 for the data composition the final model trains on.
"""

from __future__ import annotations

from dataclasses import dataclass

from sota_model.config import TrainingConfig


@dataclass
class StageConfig:
    name: str
    cfg: TrainingConfig
    fraction_of_compute: float


def three_stage_schedule(base: TrainingConfig) -> list[StageConfig]:
    foundation = TrainingConfig(
        stage="foundation",
        lr=3e-4,
        seq_len=8_192,
        warmup_steps=base.warmup_steps,
        total_steps=base.total_steps,
        global_batch_tokens=base.global_batch_tokens,
        grad_accum=base.grad_accum,
        mixed_precision=base.mixed_precision,
        grad_checkpointing=base.grad_checkpointing,
        zero_stage=base.zero_stage,
        tp_degree=base.tp_degree,
        pp_degree=base.pp_degree,
    )

    long_context = TrainingConfig(
        stage="long_context",
        lr=1e-4,
        seq_len=32_768,
        warmup_steps=500,
        total_steps=int(base.total_steps * 0.25),
        global_batch_tokens=base.global_batch_tokens,
        grad_accum=base.grad_accum * 2,  # halve micro-batch to fit longer seq
        long_doc_mix_ratio=0.4,
        sliding_window_layers_enabled=True,
        mixed_precision=base.mixed_precision,
        grad_checkpointing=True,
        zero_stage=base.zero_stage,
        tp_degree=base.tp_degree,
        pp_degree=base.pp_degree,
    )

    refinement = TrainingConfig(
        stage="refinement",
        lr=5e-5,
        seq_len=8_192,
        warmup_steps=200,
        total_steps=int(base.total_steps * 0.1),
        global_batch_tokens=base.global_batch_tokens // 4,
        grad_accum=base.grad_accum,
        mixed_precision=base.mixed_precision,
        grad_checkpointing=base.grad_checkpointing,
        zero_stage=base.zero_stage,
        tp_degree=base.tp_degree,
        pp_degree=base.pp_degree,
    )

    return [
        StageConfig("foundation", foundation, 0.70),
        StageConfig("long_context", long_context, 0.20),
        StageConfig("refinement", refinement, 0.10),
    ]
