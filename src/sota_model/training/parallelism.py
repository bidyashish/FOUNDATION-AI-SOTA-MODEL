"""3D parallelism mesh: TP x PP x DP.

A frontier-dense model fits on no single GPU. The partitioning:

    Tensor parallel (TP)           within the NVLink domain. On HGX boxes that
                                   domain is the 8-GPU node; on rack-scale
                                   GB200/GB300 NVL72 it is the whole 72-GPU
                                   rack, so TP > 8 is natural there.
    Pipeline parallel (PP)         across nodes/racks, InfiniBand / NVLink-Switch
    Data parallel (DP, ZeRO-3)     sharding of optimizer state + grads
    Gradient accumulation          to close the global token batch

Reference layouts:
    4.7-class  (1024 H100):       TP=8 × PP=8 × DP=16 — one replica per 8 nodes
    UltraModel 5 (GB300 NVL72):   TP=9 × PP=8 × DP=64 — one replica per rack
                                  (9 × 8 = 72 GPUs = one NVL72 NVLink domain)

TP must divide the sharded model dims (d_model, ffn_dim, n_q_heads, n_kv_heads)
— use `validate_parallel_layout` before launching. Note 18 KV heads shard at
TP=9 or 6, NOT at TP=8.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from sota_model.config import ModelConfig, TrainingConfig


@dataclass
class ParallelismMesh:
    tp_degree: int = 8
    pp_degree: int = 8
    dp_degree: int = 1
    grad_accum: int = 16

    rank: int = 0
    world_size: int = 1
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0

    @property
    def expected_world_size(self) -> int:
        return self.tp_degree * self.pp_degree * self.dp_degree

    def validate(self) -> None:
        if self.world_size != self.expected_world_size:
            raise ValueError(
                f"world_size={self.world_size} but tp*pp*dp = "
                f"{self.tp_degree}*{self.pp_degree}*{self.dp_degree} = {self.expected_world_size}"
            )


def init_mesh(
    tp_degree: int = 8,
    pp_degree: int = 8,
    grad_accum: int = 16,
) -> ParallelismMesh:
    """Read the launcher-provided env vars and slot this rank into the mesh.

    Compatible with `torchrun`, `deepspeed`, and Ray TorchTrainer launchers.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size % (tp_degree * pp_degree) != 0:
        raise ValueError(
            f"world_size={world_size} not divisible by tp*pp={tp_degree * pp_degree}"
        )
    dp_degree = world_size // (tp_degree * pp_degree)

    # Layout: ranks are arranged so that contiguous ranks share TP group (NVLink),
    # then PP group (InfiniBand), then DP group (ethernet / lower-priority).
    tp_rank = rank % tp_degree
    pp_rank = (rank // tp_degree) % pp_degree
    dp_rank = rank // (tp_degree * pp_degree)

    mesh = ParallelismMesh(
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        dp_degree=dp_degree,
        grad_accum=grad_accum,
        rank=rank,
        world_size=world_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
    )
    mesh.validate()
    return mesh


def validate_parallel_layout(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> list[str]:
    """Check the model shape actually shards at the configured TP / PP.

    Hard errors (raise): any tensor-parallel-sharded dimension not divisible
    by tp_degree — d_model, ffn_dim (incl. per-layer overrides), n_q_heads,
    n_kv_heads. These cannot be padded away.

    Soft warnings (returned): uneven pipeline stages (n_layers % pp != 0 —
    supported via custom partitioning, but stages imbalance) and vocab padding
    (Megatron-style embedding sharding pads vocab to a multiple of tp).
    """
    tp, pp = train_cfg.tp_degree, train_cfg.pp_degree
    hard: list[str] = []
    for name, dim in (
        ("d_model", model_cfg.d_model),
        ("n_q_heads", model_cfg.n_q_heads),
        ("n_kv_heads", model_cfg.n_kv_heads),
        ("ffn_dim", model_cfg.ffn_dim),
    ):
        if dim % tp:
            hard.append(f"{name}={dim} not divisible by tp_degree={tp}")
    for idx, ov in model_cfg.layer_overrides.items():
        if "ffn_dim" in ov and ov["ffn_dim"] % tp:
            hard.append(f"layer_overrides[{idx}].ffn_dim={ov['ffn_dim']} not divisible by tp_degree={tp}")
    if hard:
        raise ValueError("model does not shard at this TP layout: " + "; ".join(hard))

    warnings: list[str] = []
    if model_cfg.n_layers % pp:
        warnings.append(
            f"n_layers={model_cfg.n_layers} not divisible by pp_degree={pp}: "
            f"pipeline stages will be uneven (custom partitioning required)"
        )
    if model_cfg.vocab_size % tp:
        padded = ((model_cfg.vocab_size + tp - 1) // tp) * tp
        warnings.append(
            f"vocab_size={model_cfg.vocab_size} not divisible by tp_degree={tp}: "
            f"embedding will be padded to {padded}"
        )
    return warnings


def deepspeed_config_for(
    train_batch_size_global: int,
    grad_accum: int,
    mixed_precision: str = "bf16",
    zero_stage: int = 3,
) -> dict:
    # fp8 / nvfp4 / mxfp8 are Transformer Engine-managed (per-layer autocast on
    # Blackwell/GB300 and Rubin); DeepSpeed runs bf16 master weights and
    # communication underneath TE on those paths.
    te_managed = mixed_precision in ("fp8", "nvfp4", "mxfp8")
    return {
        "train_batch_size": train_batch_size_global,
        "gradient_accumulation_steps": grad_accum,
        "fp16": {"enabled": mixed_precision == "fp16"},
        "bf16": {"enabled": mixed_precision == "bf16" or te_managed},
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        # Activation checkpointing is configured per-layer in the model itself,
        # so we don't enable DeepSpeed's competing implementation here.
    }
