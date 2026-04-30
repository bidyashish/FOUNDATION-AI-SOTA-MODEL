"""3D parallelism mesh: TP x PP x DP.

A SuperModel 4.7-class dense model fits on no single GPU. The actual partitioning
across a 1024-GPU cluster is:

    Tensor parallel (TP=8)         within a node, NVLink-bandwidth
    Pipeline parallel (PP=8)       across nodes, InfiniBand-bandwidth
    Data parallel (DP=remaining)   ZeRO-3 sharding of optimizer state + grads
    Gradient accumulation          to reach 4M tokens per step at the global batch

For 1024 H100s: TP=8, PP=8, DP=16 → 8*8*16 = 1024 GPUs, with each DP replica
holding 1/8 of layers in pipeline and 1/8 of width in tensor-parallel shard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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


def deepspeed_config_for(
    train_batch_size_global: int,
    grad_accum: int,
    mixed_precision: str = "bf16",
    zero_stage: int = 3,
) -> dict:
    return {
        "train_batch_size": train_batch_size_global,
        "gradient_accumulation_steps": grad_accum,
        "fp16": {"enabled": mixed_precision == "fp16"},
        "bf16": {"enabled": mixed_precision == "bf16"},
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
