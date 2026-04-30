"""Pretraining entrypoint.

Multi-node runs go through DeepSpeed (ZeRO-3 sharding of optimizer state,
gradients, and parameters); single-node smoke runs use plain PyTorch.

The two paths are kept structurally separate to avoid the trap of creating a
PyTorch optimizer alongside a DeepSpeed engine — at ZeRO-3 scale, the optimizer
must be DeepSpeed-managed or the sharded gradient state never gets stepped.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from sota_model.config import ModelConfig, TrainingConfig
from sota_model.modeling.transformer import build_model
from sota_model.training.corpus import (
    CorpusLoader,
    CorpusLoaderConfig,
    CorpusSource,
    iter_jsonl_dir,
    resolve_sources_from_yaml,
)
from sota_model.training.parallelism import deepspeed_config_for


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _build_ds_config(train_cfg: TrainingConfig) -> dict:
    cfg = deepspeed_config_for(
        train_batch_size_global=train_cfg.global_batch_tokens // train_cfg.seq_len,
        grad_accum=train_cfg.grad_accum,
        mixed_precision=train_cfg.mixed_precision,
        zero_stage=train_cfg.zero_stage,
    )
    cfg["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": train_cfg.lr,
            "betas": [train_cfg.beta1, train_cfg.beta2],
            "weight_decay": train_cfg.weight_decay,
        },
    }
    cfg["scheduler"] = {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_max_lr": train_cfg.lr,
            "warmup_min_lr": 0.0,
            "warmup_num_steps": train_cfg.warmup_steps,
            "total_num_steps": train_cfg.total_steps,
            "cos_min_ratio": 0.1,
        },
    }
    cfg["gradient_clipping"] = train_cfg.grad_clip
    return cfg


def _train_with_deepspeed(model, train_cfg, data_iter, ckpt_dir, rank):
    import deepspeed
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=_build_ds_config(train_cfg),
    )
    device = engine.device
    engine.train()
    for step, batch in enumerate(data_iter):
        if step >= train_cfg.total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        out = engine(input_ids)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        engine.backward(loss)
        engine.step()  # handles grad clip + optimizer step + LR + zero_grad
        if step % train_cfg.log_every_steps == 0 and rank == 0:
            print(f"[{train_cfg.stage}] step={step} loss={loss.item():.4f}")
        if step % train_cfg.save_every_steps == 0 and step > 0:
            engine.save_checkpoint(str(ckpt_dir), tag=f"{train_cfg.stage}_step{step}")


def _train_with_pytorch(model, train_cfg, data_iter, ckpt_dir, rank):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )
    model.train()
    accumulator = 0
    optimizer.zero_grad()
    for step, batch in enumerate(data_iter):
        if step >= train_cfg.total_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16,
            enabled=train_cfg.mixed_precision != "fp32" and device.type == "cuda",
        ):
            out = model(input_ids)
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / train_cfg.grad_accum
        loss.backward()
        accumulator += 1
        if accumulator % train_cfg.grad_accum == 0:
            lr_now = cosine_lr(step, train_cfg.warmup_steps, train_cfg.total_steps,
                               train_cfg.lr, train_cfg.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        if step % train_cfg.log_every_steps == 0 and rank == 0:
            print(f"[{train_cfg.stage}] step={step} loss={loss.item() * train_cfg.grad_accum:.4f}")
        if step % train_cfg.save_every_steps == 0 and step > 0 and rank == 0:
            torch.save(
                {"step": step, "state_dict": model.state_dict()},
                ckpt_dir / f"{train_cfg.stage}_step{step}.pt",
            )


def train_one_stage(model_cfg, train_cfg, data_iter, ckpt_dir, rank=0, world_size=1):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    if train_cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()

    use_deepspeed = world_size > 1
    if use_deepspeed:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            print("WARNING: world_size > 1 but DeepSpeed not installed; falling back to PyTorch.")
            use_deepspeed = False

    if use_deepspeed:
        _train_with_deepspeed(model, train_cfg, data_iter, ckpt_dir, rank)
    else:
        _train_with_pytorch(model, train_cfg, data_iter, ckpt_dir, rank)


def _build_data_iter(
    args, model_cfg: ModelConfig, train_cfg: TrainingConfig
):
    """Build the real data iterator from `--data-root` + tokenizer.

    Falls back to a uniform-random emitter only when `--smoke` is passed —
    the smoke path exists for CI-style sanity checks of the trainer wiring.
    """
    if args.smoke:
        def smoke_iter():
            while True:
                yield {"input_ids": torch.randint(
                    0, model_cfg.vocab_size, (1, train_cfg.seq_len), dtype=torch.long
                )}
        return smoke_iter()

    from sota_model.tokenizer import load_tokenizer, make_byte_fallback

    if args.tokenizer is not None and args.tokenizer.exists():
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        print("WARNING: no trained tokenizer at --tokenizer; using byte fallback. "
              "Production runs MUST train and load a 200K BPE first.")
        tokenizer = make_byte_fallback()

    sources = resolve_sources_from_yaml(args.config, args.data_root)
    if not sources:
        raise SystemExit(
            f"no corpus sources found under {args.data_root}; "
            "expected per-source subdirectories matching implied_training_corpus.source_mix_pct"
        )

    loader = CorpusLoader(
        sources,
        tokenizer,
        CorpusLoaderConfig(seq_len=train_cfg.seq_len, pad_token_id=model_cfg.pad_token_id),
    )
    return loader.batches(batch_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sota_4_7.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument(
        "--stage", choices=["foundation", "long_context", "refinement", "all"], default="all"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("./data/corpus"),
        help="Root directory of per-source corpus shards (one subdir per implied_training_corpus source).",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("./tokenizer"),
        help="Trained tokenizer directory.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use a random-token emitter; for CI / wiring smoke tests only.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = ModelConfig.from_yaml(args.config)
    train_cfg = TrainingConfig.from_yaml(args.config)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    data_iter = _build_data_iter(args, model_cfg, train_cfg)
    train_one_stage(model_cfg, train_cfg, data_iter, args.output_dir, rank, world_size)


if __name__ == "__main__":
    main()
