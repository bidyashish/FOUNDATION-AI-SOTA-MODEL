"""Supervised fine-tuning on ≥1M curated chat examples.

Modelcard 1.1.2 / 6.2.1: SFT is the first post-training stage. It takes the
pretrained base model and fits it to the chat surface (the same `ChatTemplate`
used by serving), with an explicit assistant-only loss mask. Without the mask
the model loses calibration on user-message likelihood, which downstream hurts
RLHF stability.

Loss masking rules:
  - tokens emitted by `<|im_start|>system` / `<|im_start|>user`: ignore_index
  - tokens emitted by `<|im_start|>tool`: ignore_index
  - tokens emitted by `<|im_start|>assistant`: train
  - the closing `<|im_end|>` of an assistant turn: train (so the model
    learns to stop)

This module exposes `SFTTrainer` (wraps a `SOTAModel` with AdamW + cosine LR
schedule), `pack_sft_examples` (packs masked sequences into BlockPacker-style
batches), and a CLI entrypoint.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import torch
import torch.nn.functional as F

from sota_model.config import ModelConfig, TrainingConfig
from sota_model.inference.chat_template import ChatTemplate, Message
from sota_model.modeling.transformer import SOTAModel, build_model


# `ignore_index` for cross-entropy when the position should not contribute
# to the loss. Matches PyTorch's default for `cross_entropy`.
IGNORE_INDEX: int = -100


@dataclass
class SFTConfig:
    seq_len: int = 8_192
    lr: float = 5e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    total_steps: int = 100_000
    log_every_steps: int = 10
    save_every_steps: int = 1_000
    pad_token_id: int = 0
    grad_accum: int = 4
    target_examples: int = 1_000_000

    @classmethod
    def from_training_config(cls, t: TrainingConfig) -> "SFTConfig":
        return cls(
            seq_len=t.seq_len,
            lr=t.lr * 0.2,
            beta1=t.beta1, beta2=t.beta2, weight_decay=t.weight_decay,
            grad_clip=t.grad_clip,
            warmup_steps=max(100, t.warmup_steps // 10),
            total_steps=max(10_000, t.total_steps // 10),
        )


@dataclass
class SFTExample:
    """A single curated chat example.

    `messages` is a list of dicts compatible with `ChatTemplate.render`.
    `metadata` is operator-supplied (source, quality bucket, license tag).
    """
    messages: list[dict]
    metadata: dict = field(default_factory=dict)


def _segment_render(template: ChatTemplate, msg: Message) -> str:
    """Render a single message in the same way ChatTemplate would."""
    return template._render_message(msg)


def _ids_for_message(tokenizer, template: ChatTemplate, msg: Message) -> list[int]:
    return tokenizer.encode(_segment_render(template, msg), add_special_tokens=False)


def _ids_for_system(tokenizer, template: ChatTemplate, content: str, tools=None) -> list[int]:
    return tokenizer.encode(template._render_system(content, tools), add_special_tokens=False)


def build_masked_example(
    tokenizer,
    template: ChatTemplate,
    example: SFTExample,
) -> tuple[list[int], list[int]]:
    """Render an example into (input_ids, labels) with the assistant-only mask.

    Returns labels with IGNORE_INDEX on every non-assistant position.
    """
    input_ids: list[int] = []
    labels: list[int] = []

    for raw_msg in example.messages:
        msg = ChatTemplate._coerce(raw_msg)
        if msg.role == "system":
            ids = _ids_for_system(tokenizer, template, msg.content or "", tools=None)
        else:
            ids = _ids_for_message(tokenizer, template, msg)
        input_ids.extend(ids)
        # Train on assistant turns; ignore everything else.
        labels.extend(ids if msg.role == "assistant" else [IGNORE_INDEX] * len(ids))

    return input_ids, labels


def pack_sft_examples(
    tokenizer,
    examples: Iterable[SFTExample],
    seq_len: int = 8_192,
    pad_token_id: int = 0,
    template: Optional[ChatTemplate] = None,
) -> Iterator[dict]:
    """Pack masked examples into fixed-length training blocks.

    Examples that exceed `seq_len` are split at message boundaries. Examples
    that don't fill `seq_len` are left-padded with `pad_token_id` and the
    label is `IGNORE_INDEX` on the pad positions.
    """
    template = template or ChatTemplate()
    buf_ids: list[int] = []
    buf_lbl: list[int] = []
    for ex in examples:
        ids, lbl = build_masked_example(tokenizer, template, ex)
        if len(ids) > seq_len:
            ids = ids[:seq_len]
            lbl = lbl[:seq_len]
        if len(buf_ids) + len(ids) > seq_len:
            yield _pad(buf_ids, buf_lbl, seq_len, pad_token_id)
            buf_ids, buf_lbl = [], []
        buf_ids.extend(ids)
        buf_lbl.extend(lbl)
    if buf_ids:
        yield _pad(buf_ids, buf_lbl, seq_len, pad_token_id)


def _pad(ids: list[int], lbl: list[int], seq_len: int, pad: int) -> dict:
    pad_n = seq_len - len(ids)
    if pad_n > 0:
        ids = ids + [pad] * pad_n
        lbl = lbl + [IGNORE_INDEX] * pad_n
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(lbl, dtype=torch.long),
    }


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


class SFTTrainer:
    """Wraps a SOTAModel with AdamW + cosine LR for chat-style SFT.

    Pure PyTorch; for multi-node ZeRO-3 wire it through the same DeepSpeed
    path used by `training/pretrain.py::_train_with_deepspeed`.
    """

    def __init__(self, model: SOTAModel, cfg: SFTConfig):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay,
        )
        self._step = 0
        self._accum = 0
        self.optimizer.zero_grad()

    def step(self, batch: dict, device: torch.device) -> float:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        out = self.model(input_ids)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        scaled = loss / self.cfg.grad_accum
        scaled.backward()
        self._accum += 1
        if self._accum % self.cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            for pg in self.optimizer.param_groups:
                pg["lr"] = _cosine_lr(self._step, self.cfg.warmup_steps, self.cfg.total_steps, self.cfg.lr)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._step += 1
        return float(loss.item())

    def train(
        self,
        batch_iter: Iterable[dict],
        device: torch.device,
        ckpt_dir: Optional[Path] = None,
    ) -> None:
        self.model.train()
        for step, batch in enumerate(batch_iter):
            if step >= self.cfg.total_steps:
                break
            loss = self.step(batch, device)
            if step % self.cfg.log_every_steps == 0:
                print(f"[sft] step={step} loss={loss:.4f}")
            if ckpt_dir and step > 0 and step % self.cfg.save_every_steps == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"step": step, "state_dict": self.model.state_dict()},
                    ckpt_dir / f"sft_step{step}.pt",
                )


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training loop.")
    parser.add_argument("--config", type=Path, default=Path("configs/sota_4_7.yaml"))
    parser.add_argument("--data", type=Path, required=True, help="JSONL of SFT examples")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/sft"))
    args = parser.parse_args()

    model_cfg = ModelConfig.from_yaml(args.config)
    train_cfg = TrainingConfig.from_yaml(args.config)
    sft_cfg = SFTConfig.from_training_config(train_cfg)

    from sota_model.tokenizer import load_tokenizer, make_byte_fallback
    tokenizer = (
        load_tokenizer(args.config.parent / "tokenizer")
        if (args.config.parent / "tokenizer").exists()
        else make_byte_fallback()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    trainer = SFTTrainer(model, sft_cfg)

    examples = _stream_examples(args.data)
    batches = pack_sft_examples(tokenizer, examples, seq_len=sft_cfg.seq_len, pad_token_id=sft_cfg.pad_token_id)
    trainer.train(batches, device=device, ckpt_dir=args.output_dir)


def _stream_examples(path: Path) -> Iterator[SFTExample]:
    import json
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield SFTExample(messages=row["messages"], metadata=row.get("metadata", {}))


if __name__ == "__main__":
    main()
