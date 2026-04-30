#!/usr/bin/env python3
"""Phase 3 — Pretrain. GPUs.

Reads packed `.npy` shards from Phase 2, runs the 3-stage schedule.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sota_model.config import ModelConfig, TrainingConfig
from sota_model.training.pretrain import train_one_stage
from sota_model.training.schedule import three_stage_schedule


def packed_shard_iter(packed_dir: Path, seq_len: int, micro_batch: int = 1):
    shards = sorted(packed_dir.glob("packed_*.npy"))
    if not shards:
        raise SystemExit(f"no packed shards in {packed_dir}")
    arrays = [np.load(s, mmap_mode="r") for s in shards]
    print(f"loaded {len(shards)} shards, {sum(a.shape[0] for a in arrays):,} sequences")
    rng = np.random.default_rng(seed=int(os.environ.get("RANK", 0)))

    while True:
        rows = []
        for _ in range(micro_batch):
            arr = arrays[int(rng.integers(0, len(arrays)))]
            idx = int(rng.integers(0, arr.shape[0]))
            seq = np.array(arr[idx][:seq_len], dtype=np.int64)
            if seq.shape[0] < seq_len:
                continue
            rows.append(torch.from_numpy(seq)[None, :])
        if rows:
            yield {"input_ids": torch.cat(rows, dim=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sota_4_7.yaml"))
    parser.add_argument("--packed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/"))
    parser.add_argument("--stage", choices=["foundation", "long_context", "refinement", "all"], default="all")
    parser.add_argument("--micro-batch", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig.from_yaml(args.config)
    base_train = TrainingConfig.from_yaml(args.config)
    schedule = three_stage_schedule(base_train)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    stages = [s for s in schedule if args.stage in (s.name, "all")]
    print(f"running stages: {[s.name for s in stages]}")

    for stage in stages:
        print(f"\n=== STAGE: {stage.name} ({stage.fraction_of_compute:.0%} compute) ===")
        data_iter = packed_shard_iter(args.packed_dir, stage.cfg.seq_len, args.micro_batch)
        ckpt = args.output_dir / stage.name
        ckpt.mkdir(parents=True, exist_ok=True)
        train_one_stage(model_cfg, stage.cfg, data_iter, ckpt, rank, world_size)


if __name__ == "__main__":
    main()
