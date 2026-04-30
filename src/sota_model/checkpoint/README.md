# `sota_model/checkpoint/`

Checkpoint lifecycle for the SOTA stack. Sized for 200B-class models that can't fit in a single file (>100GB hits filesystem limits in many production stacks), so saves are sharded `safetensors` bundles.

```
__init__.py    re-exports
manager.py     save_checkpoint, load_checkpoint, init_checkpoint_from_spec, CheckpointManager
merge.py       merge_lora_into_base, interpolate_state_dicts (model souping)
```

## Code

- **`save_checkpoint(model, output_dir, metadata, max_shard_bytes=5GB)`** — greedy bin-pack the state dict into shards, write `safetensors` files, emit `model.safetensors.index.json` mapping each tensor name to its shard, freeze `config.yaml` next to the weights, write `metadata.json`.
- **`load_checkpoint(bundle_dir)`** — reads `config.yaml` to rebuild the `SOTAModel`, then loads the index and re-assembles. Returns `(model, CheckpointMetadata)`.
- **`init_checkpoint_from_spec(cfg, output_dir, seed=0)`** — mints a fresh, deterministically-initialized checkpoint. Use this on day zero so every later resume has a known starting point and the file layout never has to be invented mid-run.
- **`CheckpointManager(root)`** — directory of bundles; `latest()`, `save(...)`, `load(tag)`, `prune(keep=5)`.
- **`merge_lora_into_base(base, adapter, scale=1)`** — merge `(B @ A) * scale` LoRA deltas into the corresponding base weights.
- **`interpolate_state_dicts(a, b, alpha)`** — `(1-α)·a + α·b`; for stage-boundary model souping.

`safetensors` is preferred over `torch.save` because it's mmap-friendly, stable across PyTorch versions, and forbids arbitrary-code unpickle. The module silently falls back to `torch.save` if `safetensors` isn't installed.

## Data

Bundle layout:

```
checkpoints/<tag>/
    config.yaml                          # frozen ModelConfig at save time
    metadata.json                        # CheckpointMetadata (step, stage, dtype, total_params, ...)
    model.safetensors.index.json         # { "weight_map": { tensor_name: "model-shard-0001.safetensors" } }
    model-shard-0001.safetensors
    model-shard-0002.safetensors
    ...
```

`CheckpointMetadata`:
- `step: int`
- `stage: str` — `init` / `foundation` / `long_context` / `refinement` / `sft` / `rm` / `ppo`
- `total_params: int` — set automatically on save
- `dtype: str` — `bf16` / `fp16` / `fp32`
- `loss: float | None`
- `notes: str`
- `extra: dict` — operator-supplied (e.g. `{"seed": 42, "git_sha": "abc123"}`)

## Working

```python
from sota_model.checkpoint import (
    init_checkpoint_from_spec, load_checkpoint, save_checkpoint,
    CheckpointManager, CheckpointMetadata,
)

# Day zero — mint a fresh checkpoint at the configured topology.
init_checkpoint_from_spec(model_cfg, "./checkpoints/init", seed=0)

# After each training step…
mgr = CheckpointManager("./checkpoints")
mgr.save(
    model,
    tag=f"foundation_step{step}",
    metadata=CheckpointMetadata(step=step, stage="foundation", loss=cur_loss),
)

# Resume…
model, meta = mgr.load(tag="foundation_step5000")
print(f"resuming from step {meta.step}, loss {meta.loss}")
```

### Operator checklist

- [ ] Run `init_checkpoint_from_spec(...)` once at the start of training so the file layout is committed.
- [ ] Save every `save_every_steps` (configurable in `TrainingConfig` / `SFTConfig`).
- [ ] Prune old checkpoints to keep disk pressure bounded — default keep is 5.
- [ ] When merging LoRA adapters back into a base run, call `save_checkpoint` AFTER the merge so the deployment artifact is single-stage.
