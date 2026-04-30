# `sota_model/training/`

Pretraining-side: data filters, packing, schedule, parallelism, entry point.

```
data.py           Filters (lang, dedup, quality, toxicity, PII, contamination), BlockPacker
classifiers/      Trainable replacements for the heuristic stubs in data.py.
                  See classifiers/README.md.
corpus.py         CorpusLoader: filter chain + tokenizer + packer + mix-ratio interleave.
                  Replaces dummy_iter() from pretrain.py for production runs.
schedule.py       3-stage schedule: foundation → long_context → refinement
parallelism.py    TP × PP × DP mesh init + DeepSpeed ZeRO-3 config helper
pretrain.py       CLI entry, AdamW + cosine LR, optional DeepSpeed init
sample_loader.py  JSONL helpers used by tests
```

---

## `data.py` — filter pipeline

### Order matters

```
LanguageDetector       → MinLengthFilter → DuplicateRemover
       ↓
QualityScorer → ToxicityFilter → PIIRedactor → BenchmarkContaminationFilter
       ↓
yield tokenizer.encode(doc["text"])
```

Cheap filters run first (language, length) so expensive filters (quality, dedup) only see promising candidates.

### `BenchmarkContaminationFilter` — load-bearing

The default URL blocklist mirrors `` 9.2. Extending is fine; **narrowing breaks the eval suite**. If a training shard contains HLE / GPQA / SWE-bench problem text, the model has effectively memorized the eval — every reported number becomes meaningless.

The filter checks both URL substring matches (normalized: lowercase, slashes stripped) and text-level patterns (eval names like "GPQA Diamond" appearing in prose).

### Stubs to replace before real training

| Stub | Replacement (now shipping) | Notes |
|---|---|---|
| `LanguageDetector` (data.py tag-and-pass) | `classifiers.CharNgramLanguageDetector` (script-family) → `classifiers.TrainedLanguageDetector` (multiclass logistic) | Pass `backend=` to `data.py::QualityScorer` etc. |
| `QualityScorer` (heuristic) | `classifiers.TrainedQualityScorer` (logistic on hashing trick) | Default backend stays heuristic for day-zero compat |
| `ToxicityFilter` (single regex) | `classifiers.BlocklistToxicityFilter.from_files([...])` (audited JSON blocklists) AND `classifiers.TrainedToxicityFilter` (logistic) | Compose deterministic + fuzzy paths |
| `DuplicateRemover` | MinHash-LSH at Jaccard 0.85 over 5-grams (still TODO) | Current is a sha256 fingerprint stand-in |

### `BlockPacker`

Concatenates tokenized streams into fixed-length blocks. `reset_mask=True` records document boundaries so attention doesn't cross them — without this, the model learns to ignore document structure, which hurts retrieval and instruction-following.

`seq_len` ramps across stages: 8192 (Stage 1) → 32768 (Stage 2 entry) → 131072 → 1048576 (Stage 2 finish).

---

## `schedule.py` — three-stage pretraining

| Stage | Compute | Wall-clock | LR | Seq | Goal |
|---|---|---|---|---|---|
| Foundation | ~70% | ~8 weeks | 3e-4 | 8K | Next-token prediction over broad mix |
| Long context | ~20% | ~4 weeks | 1e-4 | 32K → 1M | Extend RoPE with YaRN; long-doc mix 40% |
| Refinement | ~10% | ~4 weeks | 5e-5 | 8K | Top-decile data only; instruction-style |

Stage transitions are picked by validation-loss elbows + downstream benchmark deltas, not fixed step counts. `three_stage_schedule(base)` returns `[StageConfig, ...]` with each stage's `TrainingConfig` adjusted accordingly.

### Why this split

- **Stage 1** is where the model learns the bulk of language and the world. Spending 70% of compute here is well-supported empirically.
- **Stage 2** is short relative to Stage 1 because long-context capability is mostly about the position-encoding extension, not about needing as many gradient steps.
- **Stage 3** is the lowest-LR phase; it polishes the model on curated data and fixes pathologies introduced by web-scale noise. Skipping Stage 3 measurably hurts MMLU / GPQA.

---

## `parallelism.py` — TP × PP × DP

### Why all three

A 200B dense model does not fit on any single GPU's HBM (~400 GB at bf16). One axis isn't enough:

- **Pure DP** can't fit the model.
- **Pure TP** is bandwidth-bound past ~8-way (NVLink within a node OK; cross-node TP all-reduces become latency-dominant).
- **Pure PP** has bubble overhead unless the micro-batch count is high — but you can't grow micro-batches indefinitely without OOM.

**TP × PP × DP** lets each axis solve a different bottleneck:

| Axis | Solves | Communication | Typical degree |
|---|---|---|---|
| TP | Memory within node | NVLink all-reduce per layer | 8 |
| PP | Memory across nodes | InfiniBand activation passing | 8 |
| DP (ZeRO-3) | Throughput + optimizer state size | Reduce-scatter grads, all-gather params | 16+ |

For a 1024-GPU run: TP=8 × PP=8 × DP=16 = 1024.

### `init_mesh()`

Reads `RANK` and `WORLD_SIZE` from the launcher and assigns each rank to a `(tp_rank, pp_rank, dp_rank)` triple. Layout: contiguous ranks share TP groups (NVLink), then PP groups, then DP groups. Compatible with `torchrun`, `deepspeed`, and Ray TorchTrainer.

### `deepspeed_config_for()`

Generates a ZeRO-3 config tuned for this scale:
- `overlap_comm: true` — overlap gradient comm with backward compute
- `reduce_bucket_size: 5e8` — large enough to amortize NCCL latency
- `stage3_prefetch_bucket_size: 5e8` — prefetch params before they're needed
- Activation checkpointing left to the model (we already wire it via `enable_gradient_checkpointing()`); enabling DeepSpeed's competing implementation double-counts.

---

## `pretrain.py` — entry point

Production:

```bash
sota-pretrain \
    --config configs/sota_4_7.yaml \
    --output-dir ./checkpoints \
    --data-root ./data/corpus \
    --tokenizer ./tokenizer
```

Wiring smoke test (no real corpus needed):

```bash
sota-pretrain --config configs/sota_4_7.yaml --output-dir ./checkpoints --smoke
```

Single-node runs work directly. Multi-node uses a DeepSpeed or torchrun launcher; the script reads `LOCAL_RANK` / `WORLD_SIZE` and initializes accordingly.

### Optimizer

AdamW with the config's `lr`, `betas=(beta1, beta2)`, `weight_decay`. `` doesn't prescribe a specific optimizer.

### LR schedule

Cosine decay to 10% of peak after `warmup_steps` linear warmup. Same shape across all three stages, just with different peak LRs.

### Data path

`_build_data_iter` resolves sources from `implied_training_corpus.source_mix_pct` and instantiates a `CorpusLoader` over them — see [`corpus.py`](./corpus.py) and the layout under `--data-root`:

```
<data-root>/
    web/*.jsonl
    code/*.jsonl
    academic/*.jsonl
    books_reference/*.jsonl
    math_structured/*.jsonl
    dialogue_instructions/*.jsonl
```

`--smoke` retains the random-token dummy iterator for CI / wiring sanity checks; production runs MUST pass `--data-root` and `--tokenizer`.

### Checkpointing

Saves every `save_every_steps` steps. Files: `{stage}_step{N}.pt`. Production runs should call `checkpoint.save_checkpoint(...)` directly for the sharded safetensors layout — see [`checkpoint/README.md`](../checkpoint/README.md). DeepSpeed runs use DeepSpeed's sharded checkpoint format for ZeRO-3 resume.

---

## `sample_loader.py` — JSONL helpers

Used by `tests/test_samples.py` and onboarding scripts. Reads `data/samples/*.jsonl` into Python lists. No external dependencies — keeps the test suite hermetic.
