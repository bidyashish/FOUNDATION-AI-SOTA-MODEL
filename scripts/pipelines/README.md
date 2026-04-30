# `scripts/pipelines/`

Three standalone phases. Run in order. Each phase has a stable on-disk output that the next phase consumes.

```
data/raw/*.jsonl
    │  Phase 1 (CPU)
    ▼
data/cleaned/*.jsonl
    │  Phase 2 (CPU)
    ▼
data/packed/*.npy
    │  Phase 3 (GPU)
    ▼
checkpoints/{stage}/{stage}_step{N}.pt
```

Each phase has its own report file, restart story, and is reusable across runs.

## Phase 1 — Clean

```bash
python scripts/pipelines/01_clean_data.py \
    --input "data/raw/*.jsonl" \
    --output data/cleaned/ \
    --shard-size 100000 \
    --report data/cleaned/report.json
```

Filters: language → min-length → dedup → quality → toxicity → PII → benchmark contamination (modelcard 9.2).

## Phase 2 — Tokenize + pack

```bash
python scripts/pipelines/02_tokenize_and_pack.py \
    --input "data/cleaned/*.jsonl" \
    --output data/packed/seq_8192/ \
    --tokenizer-name path/to/tokenizer.json \
    --seq-len 8192
```

Pack at the seq_len for the stage you'll train: 8192 (Stage 1), 32768+ (Stage 2).

## Phase 3 — Train

```bash
# Single-node smoke
python scripts/pipelines/03_pretrain.py \
    --config configs/sota_4_7.yaml \
    --packed-dir data/packed/seq_8192/ \
    --output-dir checkpoints/

# Multi-node full
deepspeed --num_gpus 8 --num_nodes 128 scripts/pipelines/03_pretrain.py \
    --config configs/sota_4_7.yaml \
    --packed-dir data/packed/ \
    --output-dir checkpoints/
```

The 3-stage schedule (foundation → long_context → refinement) runs through `train_one_stage`. ZeRO-3 sharding is automatic when world_size > 1.

## Smoke run on bundled samples

```bash
python scripts/pipelines/01_clean_data.py \
    --input "data/samples/pretrain.jsonl" \
    --output data/_smoke_cleaned/ --shard-size 100

python scripts/pipelines/02_tokenize_and_pack.py \
    --input "data/_smoke_cleaned/*.jsonl" \
    --output data/_smoke_packed/ \
    --seq-len 512 --shard-tokens 100000
```

For a real training smoke, override `n_layers` and `d_model` in code — the production config will OOM a laptop.
