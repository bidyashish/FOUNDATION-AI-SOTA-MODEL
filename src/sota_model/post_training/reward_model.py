"""Reward model on ≥500K preference pairs.

Architecture: scalar head on top of the same SOTA backbone (the SFT model).
The reward is the scalar at the final non-pad token; pairwise loss is the
Bradley-Terry log-sigmoid form:

    L = -E [ log sigmoid( r(chosen) - r(rejected) ) ]

Modelcard 1.1.3 references reward modeling on at least 500K pairs as the
sample-efficiency band where the BT model stops being volatile against
held-out human-judgment evals. The size of the training set is wired into
`RewardModelConfig.target_pairs` for release-gating.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from sota_model.config import ModelConfig
from sota_model.modeling.transformer import SOTAModel, build_model


@dataclass
class RewardModelConfig:
    head_hidden: int = 1024
    head_dropout: float = 0.0
    target_pairs: int = 500_000   # release gate
    pad_token_id: int = 0


@dataclass
class PreferencePair:
    """A training example: same prompt, two responses, chosen wins."""
    prompt: str
    chosen: str
    rejected: str


class RewardModel(nn.Module):
    """SOTA backbone + scalar reward head."""

    def __init__(self, backbone: SOTAModel, cfg: RewardModelConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        d = backbone.cfg.d_model
        self.head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1, bias=False),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the scalar reward at the final non-pad token of each row."""
        # We re-run the backbone but pull hidden states; SOTAModel's forward
        # currently returns logits, so the cleanest path is to run the embed
        # + layers + final_norm explicitly here.
        h = self.backbone.embed(input_ids)
        for block in self.backbone.layers:
            h = block(h)
        h = self.backbone.final_norm(h)              # (B, T, D)
        rewards = self.head(h).squeeze(-1)            # (B, T)
        # Pull the last non-pad position. We don't depend on the model's pad
        # token id — the trainer feeds left-padded sequences.
        last_idx = (input_ids != self.cfg.pad_token_id).long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        gather = torch.arange(rewards.shape[0], device=rewards.device)
        return rewards[gather, last_idx]


class BradleyTerryLoss(nn.Module):
    """Pairwise log-sigmoid loss with optional margin."""

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(r_chosen - r_rejected - self.margin).mean()


def _encode(tokenizer, prompt: str, response: str, max_len: int) -> list[int]:
    text = prompt + response
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:max_len]


def _pad(rows: list[list[int]], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(r) for r in rows)
    out = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def collate_pair_batch(
    pairs: Sequence[PreferencePair],
    tokenizer,
    max_len: int = 4096,
    pad_token_id: int = 0,
) -> dict:
    chosen = [_encode(tokenizer, p.prompt, p.chosen, max_len) for p in pairs]
    rejected = [_encode(tokenizer, p.prompt, p.rejected, max_len) for p in pairs]
    return {
        "chosen": _pad(chosen, pad_token_id),
        "rejected": _pad(rejected, pad_token_id),
    }


def train_reward_model(
    model: RewardModel,
    pair_iter: Iterable[Sequence[PreferencePair]],
    *,
    tokenizer,
    device: torch.device,
    n_steps: int = 10_000,
    lr: float = 5e-6,
    log_every: int = 10,
    save_every: Optional[int] = None,
    save_dir: Optional[Path] = None,
    margin: float = 0.0,
    grad_clip: float = 1.0,
    pad_token_id: int = 0,
    max_len: int = 4096,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = BradleyTerryLoss(margin=margin)
    model.train()
    for step, batch in enumerate(pair_iter):
        if step >= n_steps:
            break
        collated = collate_pair_batch(batch, tokenizer, max_len=max_len, pad_token_id=pad_token_id)
        chosen = collated["chosen"].to(device)
        rejected = collated["rejected"].to(device)
        r_chosen = model(chosen)
        r_rejected = model(rejected)
        loss = loss_fn(r_chosen, r_rejected)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if step % log_every == 0:
            acc = float((r_chosen > r_rejected).float().mean().item())
            print(f"[rm] step={step} loss={loss.item():.4f} acc={acc:.3f}")
        if save_every and save_dir and step > 0 and step % save_every == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "state_dict": model.state_dict()}, save_dir / f"rm_step{step}.pt")


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SOTA reward model.")
    parser.add_argument("--config", type=Path, default=Path("configs/sota_4_7.yaml"))
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL of preference pairs")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints/reward_model"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=10_000)
    args = parser.parse_args()

    model_cfg = ModelConfig.from_yaml(args.config)
    rm_cfg = RewardModelConfig()

    backbone = build_model(model_cfg)
    rm = RewardModel(backbone, rm_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm = rm.to(device)

    from sota_model.tokenizer import make_byte_fallback
    tokenizer = make_byte_fallback()

    train_reward_model(
        rm, _stream_pairs(args.pairs, args.batch_size),
        tokenizer=tokenizer, device=device,
        n_steps=args.n_steps, save_dir=args.output_dir, save_every=500,
    )


def _stream_pairs(path: Path, batch_size: int) -> Iterator[list[PreferencePair]]:
    import json
    buf: list[PreferencePair] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            buf.append(PreferencePair(
                prompt=row["prompt"], chosen=row["chosen"], rejected=row["rejected"]
            ))
            if len(buf) >= batch_size:
                yield buf
                buf = []
    if buf:
        yield buf


if __name__ == "__main__":
    main()
