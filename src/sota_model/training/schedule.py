"""Three-stage pretraining schedule.

Stage 1 — Foundation (70% of corpus tokens): seq 8K, base LR, broad data mix.
Stage 2 — Long context (20%): seq 32K, RoPE base 1e6 + YaRN x128, sliding window.
Stage 3 — Quality refinement (10%): top-decile data, quarter global batch.

`base.total_steps` is the full-corpus optimizer-step budget at the base global
batch — `implied_training_corpus.total_tokens / training.global_batch_tokens`
(sota_ultra_5.yaml: 36T / 8M ≈ 4.3M; sota_4_7.yaml: 25T / 4M ≈ 6M). Stages
split the *tokens* 70 / 20 / 10, so step counts are 0.70 / 0.20 / 0.40 of the
budget — refinement runs at a quarter batch, hence 4× the steps per token.

The LR ladder derives from the operator-committed `training.lr` rather than
being hardcoded: foundation trains at `base.lr`; long-context and refinement
default to `base.lr / 3` and `base.lr / 6` (exactly the 4.7-class ladder at
lr=3e-4), or are pinned from the YAML's `implied_schedule_split` when built
via `schedule_for_config` (sota_ultra_5.yaml: 2.5e-4 / 8e-5 / 4e-5).

See  1.1.1 for the data composition the final model trains on.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from sota_model.config import TrainingConfig, load_implied


@dataclass
class StageConfig:
    name: str
    cfg: TrainingConfig
    fraction_of_compute: float


def _derived_grad_accum(
    stage: str, global_batch_tokens: int, seq_len: int, micro_batch_size: int, dp_degree: int
) -> int:
    """Gradient accumulation that closes the batch identity:

        global_batch_tokens = seq_len × micro_batch_size × grad_accum × dp_degree
    """
    sequences, rem = divmod(global_batch_tokens, seq_len)
    if rem:
        raise ValueError(
            f"{stage}: global_batch_tokens {global_batch_tokens} not divisible by seq_len {seq_len}"
        )
    accum, rem = divmod(sequences, micro_batch_size * dp_degree)
    if rem or accum < 1:
        raise ValueError(
            f"{stage}: {sequences} sequences/step not divisible by "
            f"micro_batch_size × dp_degree = {micro_batch_size} × {dp_degree}"
        )
    return accum


def three_stage_schedule(
    base: TrainingConfig,
    *,
    long_context_lr: float | None = None,
    refinement_lr: float | None = None,
) -> list[StageConfig]:
    foundation = dataclasses.replace(
        base,
        stage="foundation",
        total_steps=int(base.total_steps * 0.70),
    )

    # Longer sequences need a smaller per-replica micro-batch to fit in HBM;
    # accumulation is re-derived so the token batch stays exactly constant.
    lc_seq_len = 32_768
    lc_micro = max(1, base.micro_batch_size // 2)
    long_context = dataclasses.replace(
        base,
        stage="long_context",
        lr=long_context_lr if long_context_lr is not None else base.lr / 3,
        seq_len=lc_seq_len,
        warmup_steps=500,
        total_steps=int(base.total_steps * 0.20),
        micro_batch_size=lc_micro,
        grad_accum=_derived_grad_accum(
            "long_context", base.global_batch_tokens, lc_seq_len, lc_micro, base.dp_degree
        ),
        long_doc_mix_ratio=0.4,
        sliding_window_layers_enabled=True,
        grad_checkpointing=True,
    )

    # Quarter batch for the same 10% token share → 4× the steps per token.
    rf_batch = base.global_batch_tokens // 4
    refinement = dataclasses.replace(
        base,
        stage="refinement",
        lr=refinement_lr if refinement_lr is not None else base.lr / 6,
        warmup_steps=200,
        total_steps=int(base.total_steps * 0.10 * 4),
        global_batch_tokens=rf_batch,
        grad_accum=_derived_grad_accum(
            "refinement", rf_batch, base.seq_len, base.micro_batch_size, base.dp_degree
        ),
    )

    return [
        StageConfig("foundation", foundation, 0.70),
        StageConfig("long_context", long_context, 0.20),
        StageConfig("refinement", refinement, 0.10),
    ]


def schedule_for_config(config_path: str | Path) -> list[StageConfig]:
    """Build the three-stage schedule from a config YAML.

    Pins the stage-2/3 LRs from `implied_schedule_split` when the YAML carries
    it, so the schedule and the YAML's committed LR ladder cannot drift apart.
    """
    base = TrainingConfig.from_yaml(config_path)
    split = load_implied(config_path).get("implied_schedule_split", {})
    return three_stage_schedule(
        base,
        long_context_lr=split.get("long_context_lr"),
        refinement_lr=split.get("refinement_lr"),
    )
