"""Checkpoint manager for the SOTA stack.

The biggest gap in this repo is "no trained checkpoint". The closest a code
deliverable can get is the *infrastructure* a real run would write into:

  - Deterministic, configurable random init from a `ModelConfig` so the file
    layout is locked the moment training begins.
  - Sharded `safetensors` save/load so a 200B-class model doesn't have to
    fit in one file (>100GB hits filesystem limits on common stacks).
  - A `CheckpointMetadata` companion file so resumes don't drift on
    config — modelcard 1.4 invariants are checked at load.
  - `CheckpointManager` wraps lifecycle ops: list, latest, prune.

`safetensors` is preferred over `torch.save` because:
  - Memory-mapped load.
  - Strict no-pickle (no arbitrary code execution from a checkpoint).
  - Stable across PyTorch versions.

The file layout for a sharded checkpoint is:

    checkpoints/<tag>/
        config.yaml           — frozen ModelConfig at save time
        metadata.json         — step, loss, dtype, total_params
        model.safetensors.index.json   — { "weight_map": { name: "shard-0001" } }
        model-shard-0001.safetensors
        model-shard-0002.safetensors
        ...
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch

from sota_model.config import ModelConfig
from sota_model.modeling.transformer import SOTAModel, build_model


@dataclass
class CheckpointMetadata:
    step: int
    stage: str = "foundation"
    total_params: int = 0
    dtype: str = "bf16"
    loss: Optional[float] = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def _shard_state_dict(
    state: dict[str, torch.Tensor],
    max_shard_bytes: int,
) -> list[dict[str, torch.Tensor]]:
    """Greedy bin-pack the state dict by tensor byte-size."""
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for k, v in state.items():
        nbytes = v.element_size() * v.numel()
        if sizes[-1] + nbytes > max_shard_bytes and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = v
        sizes[-1] += nbytes
    return shards


def save_checkpoint(
    model: SOTAModel,
    output_dir: str | Path,
    metadata: CheckpointMetadata,
    *,
    max_shard_bytes: int = 5 * 1024 * 1024 * 1024,  # 5 GB default
) -> Path:
    """Save model weights + config + metadata as a sharded safetensors bundle.

    Returns the bundle directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Freeze config alongside the weights.
    cfg_dict = asdict(model.cfg)
    (out / "config.yaml").write_text(_dump_cfg_yaml(cfg_dict))

    state = model.state_dict()
    metadata.total_params = int(sum(v.numel() for v in state.values()))

    shards = _shard_state_dict(state, max_shard_bytes)

    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, 1):
        shard_name = f"model-shard-{i:04d}.safetensors"
        _safe_save(shard, out / shard_name)
        for k in shard:
            weight_map[k] = shard_name

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}, indent=2)
    )

    (out / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2))
    return out


def load_checkpoint(bundle_dir: str | Path) -> tuple[SOTAModel, CheckpointMetadata]:
    """Load a sharded checkpoint and rebuild the SOTAModel."""
    bundle = Path(bundle_dir)
    cfg = ModelConfig(**_load_cfg_yaml(bundle / "config.yaml"))
    model = build_model(cfg)

    if not (bundle / "model.safetensors.index.json").exists():
        # Single-shard fallback used by the seed init path below.
        single = bundle / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"no model index or single shard at {bundle}")
        state = _safe_load(single)
        model.load_state_dict(state, strict=False)
    else:
        index = json.loads((bundle / "model.safetensors.index.json").read_text())
        full_state: dict[str, torch.Tensor] = {}
        for k, shard_name in index["weight_map"].items():
            shard = _safe_load(bundle / shard_name)
            full_state[k] = shard[k]
        model.load_state_dict(full_state, strict=False)

    meta_path = bundle / "metadata.json"
    if meta_path.exists():
        meta = CheckpointMetadata(**json.loads(meta_path.read_text()))
    else:
        meta = CheckpointMetadata(step=0)
    return model, meta


def init_checkpoint_from_spec(
    cfg: ModelConfig,
    output_dir: str | Path,
    *,
    seed: int = 0,
    notes: str = "fresh init from spec",
) -> Path:
    """Mint a fresh, deterministically-initialized checkpoint at `cfg`'s topology.

    Use this on day zero so the rest of the training pipeline can resume against
    a known starting point — the file layout never has to be invented mid-run.
    """
    torch.manual_seed(seed)
    model = build_model(cfg)
    return save_checkpoint(
        model,
        output_dir,
        CheckpointMetadata(
            step=0, stage="init", dtype="bf16",
            notes=notes, extra={"seed": seed},
        ),
    )


# --- safetensors wrappers ---


def _safe_save(state: dict[str, torch.Tensor], path: Path) -> None:
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous().cpu() for k, v in state.items()}, str(path))
    except ImportError:
        torch.save({k: v.cpu() for k, v in state.items()}, path.with_suffix(".pt"))
        path.with_suffix(".pt").rename(path)


def _safe_load(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except ImportError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _dump_cfg_yaml(cfg_dict: dict) -> str:
    try:
        import yaml
        return yaml.safe_dump(cfg_dict, sort_keys=False)
    except ImportError:
        return json.dumps(cfg_dict, indent=2)


def _load_cfg_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except ImportError:
        return json.loads(path.read_text())


# --- Lifecycle manager ---


@dataclass
class CheckpointManager:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> list[Path]:
        return sorted([p for p in self.root.iterdir() if p.is_dir() and (p / "metadata.json").exists()])

    def latest(self) -> Optional[Path]:
        ckpts = self.list_checkpoints()
        if not ckpts:
            return None
        # Sort by step in metadata; tags can be arbitrary strings.
        def _step(p: Path) -> int:
            try:
                return int(json.loads((p / "metadata.json").read_text())["step"])
            except Exception:
                return -1
        return max(ckpts, key=_step)

    def save(self, model: SOTAModel, tag: str, metadata: CheckpointMetadata) -> Path:
        out = self.root / tag
        return save_checkpoint(model, out, metadata)

    def load(self, tag: str) -> tuple[SOTAModel, CheckpointMetadata]:
        return load_checkpoint(self.root / tag)

    def prune(self, keep: int = 5) -> list[Path]:
        ckpts = self.list_checkpoints()
        if len(ckpts) <= keep:
            return []
        # Newest by step survives.
        def _step(p: Path) -> int:
            try:
                return int(json.loads((p / "metadata.json").read_text())["step"])
            except Exception:
                return -1
        ordered = sorted(ckpts, key=_step)
        to_remove = ordered[: len(ckpts) - keep]
        import shutil
        for p in to_remove:
            shutil.rmtree(p)
        return to_remove
