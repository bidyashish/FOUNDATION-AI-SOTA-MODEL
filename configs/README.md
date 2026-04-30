# `configs/`

Central knob for the model. Everything in here traces back to a target in [``](../). When code and config disagree, config wins; when config and `` disagree, `` wins.

## Files

```
sota_4_7.yaml    SuperModel 4.7-class spec — model + training + inference
```

## How the YAML maps to dataclasses

```
sota_4_7.yaml
├── model:         → src/sota_model/config.py::ModelConfig
├── training:      → src/sota_model/config.py::TrainingConfig
└── inference:     → src/sota_model/config.py::InferenceConfig
```

Loaders: `ModelConfig.from_yaml(path)`, `TrainingConfig.from_yaml(path)`, `InferenceConfig.from_yaml(path)`.

---

## Back-trace: every field vs ``

┌─────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐  
  │             Section             │                                               What's in it                                               │                Why it had to be implied                │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤  
  │ implied_scale                   │ 427B params, 880 KB/tok, 880 GB at 1M                                                                    │ modelcard places SM4.7 in frontier-dense band but pins │
  │                                 │                                                                                                          │  no number                                             │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤  
  │ implied_training_corpus         │ 18T tokens, 6-bucket source mix (web 45 / code 20 / academic 10 / books 10 / math 10 / dialogue 5)       │ 1.1.1 names sources, doesn't pin tokens               │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤  
  │ implied_compute                 │ 10²⁵–10²⁶ FLOPs, 1–4M H100·hr, 16–24 weeks, $10–50M                                                      │ required to actually train at this scale               │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ implied_schedule_split          │ 70/20/10 compute share + LR ladder 3e-4 / 1e-4 / 5e-5                                                    │ modelcard implies 3 stages, doesn't fix the split      │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ capability_targets              │ every 8.1 scoreboard number — SWE-bench 87%, GPQA 94%, ARC-AGI-2 75%, OSWorld 78%, ScreenSpot-Pro 87%,  │ directly modelcard-pinned — these are release gates    │  
  │                                 │ MMMLU 91%, MILU 89%, etc.                                                                                │                                                        │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤  
  │ safety_thresholds               │ 4.1 single-turn 97.9%, 4.5.3 election 100%, 5.1.1 Claude Code 91%, 5.2.1 ART k=100 ≤ 6%, 5.2.2.3    │ directly modelcard-pinned — these are deployment gates │
  │                                 │ browser 0%                                                                                               │                                                        │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ implied_serving_minimums        │ 4× A100 int8 / 8× A100 bf16 / 16× H100 optimal, edge layer, pre/post safety gate, observability list     │ required to deploy at modelcard scale                  │  
  │                                 │ including modelcard 7 distress monitor                                                                  │                                                        │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ implied_special_tokens_required │ the 16 chat-surface tokens                                                                               │ cross-validated against tokenizer.SPECIAL_TOKENS       │  
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤  
  │ implied_multilingual_coverage   │ the 44 modelcard 8.12 languages by tier                                                                 │ cross-validated against tokenizer.MODELCARD_LANGUAGES  │
  └─────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘  
                                                                  


A `grep` over `` confirms what's actually pinned by the system card vs what's an operator choice. **`` is silent on architectural scalars** (verified: zero hits for `d_model`, `n_layers`, `n_kv_heads`, `ffn_dim`, `AdamW`, `bf16`, `X B parameters`, `X T tokens`, `temperature=0.X`, `top_p=0.X`).

| Field | Value | Source | Modelcard evidence |
|---|---|---|---|
| **model:** | | | |
| `vocab_size` | 200,000 | operator | silent — chosen for 8.12 multilingual coverage |
| `d_model` | 16,384 | operator | no `d_model` string in modelcard |
| `n_layers` | 110 | operator | no `n_layers` string in modelcard |
| `n_q_heads` | 128 | operator | silent |
| `n_kv_heads` | 16 | operator | silent on GQA ratio (implied only by 1M-context cluster-mem feasibility) |
| `head_dim` | 128 | operator | silent |
| `ffn_dim` | 65,536 | operator | no `ffn_dim` string in modelcard |
| `max_position_embeddings` | 1,048,576 | **MODELCARD** | 8.7 — 2 hits for `1M tokens/context` |
| `rope_base` | 1e6 | operator | implied by 1M context (avoid wrap-around), specific value not pinned |
| `rope_yarn_scale` | 8.0 | operator | implied by 8K→1M extension, value not pinned |
| `rope_yarn_original_max_position` | 8,192 | operator | Stage-1 seq len choice |
| `sliding_window_size` | 32,768 | operator | modelcard mentions sliding-window attention idea, not size |
| `sliding_window_layer_stride` | 2 | operator | silent |
| `vision_max_image_long_edge_px` | 2,576 | **MODELCARD** | invariant — 4 hits for `2576px` |
| `vision_max_image_pixels` | 3,750,000 | **MODELCARD** | invariant — 4 hits for `3.75 MP` |
| `vision_patch_size` | 14 | operator | silent |
| `thinking_budgets` keys (min/low/medium/high/max) | — | **MODELCARD** | effort tier names appear in 8 evals (`Max effort`, `High effort`) |
| `thinking_budgets` token values (0/256/2048/8192/32768) | — | operator | modelcard names tiers but pins no token budgets per tier |
| `thinking_token_min_floor` | 32 | operator | silent |
| **training:** | | | |
| `optimizer` | adamw | operator | no `AdamW`, `optimizer`, or training-hyperparam disclosure in modelcard |
| `lr` | 3e-4 | operator | silent |
| `beta1, beta2` | 0.9, 0.95 | operator | silent |
| `weight_decay` | 0.1 | operator | silent |
| `grad_clip` | 1.0 | operator | silent |
| `warmup_steps` | 2,000 | operator | silent |
| `total_steps` | 1,000,000 | operator | silent |
| `seq_len` | 8,192 | operator | Stage-1 choice |
| `global_batch_tokens` | 4,194,304 | operator | silent |
| `grad_accum` | 16 | operator | silent |
| `mixed_precision` | bf16 | operator | no `bf16/bfloat16` string in modelcard |
| `grad_checkpointing` | true | operator | silent |
| `zero_stage` | 3 | operator | silent on parallelism / sharding |
| `tp_degree, pp_degree` | 8, 8 | operator | silent |
| **inference:** | | | |
| `temperature` | 0.7 | operator | modelcard mentions `temperature` but pins NO numeric value |
| `top_p` | 0.95 | operator | modelcard mentions `top_p` but pins NO numeric value |
| `top_k` | 40 | operator | silent |
| `repetition_penalty` | 1.1 | operator | silent |
| `max_new_tokens` | 8,192 | operator | silent |
| `kv_cache_dtype` | bf16 | operator | silent |
| `page_block_size` | 16 | operator | silent on KV cache implementation |
| `enable_prefix_cache` | true | operator | silent |
| `adaptive_thinking` | true | **MODELCARD** | 4.1.1 — 34 hits for `adaptive thinking` |
| `default_effort` | high | operator | tier name pinned, default not |
| `thinking_visible_to_user` | false | operator | silent |
| `context_compaction_trigger` | 200,000 | **MODELCARD** | 4.5 — 4 hits for `200k` |
| `max_context_tokens` | 1,048,576 | **MODELCARD** | 8.7 1M context |

**Summary: 7 of 32 listed config fields are directly pinned by `` text.** The other 25 are operator choices that fit within modelcard constraints but are not numerically specified by the system card.

### What this means for "model size" and "training size"

`` does **not** disclose:
- Total parameter count (no `X B parameters` hits)
- Training token count (no `X T tokens` hits)
- Layer count, hidden dim, FFN dim, head counts (zero string matches)
- Optimizer choice or hyperparameters
- Mixed-precision dtype



This is consistent with how frontier-model providers publish system cards — capability and safety reports, not architecture specs. The current YAML lands at **~427B params with ~880 GB KV at 1M context**, a defensible point in the frontier-dense band, but **that number does not come from ``** — it comes from the operator choices made in this YAML.

If you need a different size, change `n_layers`, `d_model`, `ffn_dim` together. Nothing in `` will tell you which set of values is "correct" — there is no published correct answer.

---

## Implied sections — what an operator MUST commit to

`sota_4_7.yaml` carries additional sections beyond `model:` / `training:` / `inference:` that capture the values needed to **deliver** modelcard targets, even though modelcard doesn't pin them numerically. These are loaded via `sota_model.config.load_implied(path)` — they are NOT consumed by the dataclasses.

| Section | What it pins | Modelcard tie |
|---|---|---|
| `implied_scale` | param count (~427B), KV bytes/token, GB at 1M | Frontier-dense band; modelcard places SM4.7 between SM4.6 and Ultramodel |
| `implied_training_corpus` | 18T tokens, 6-bucket source mix | 1.1.1 names sources; doesn't pin tokens |
| `implied_compute` | 10²⁵–10²⁶ FLOPs, 1–4M H100·hr, 16–24 wks, $10–50M | Frontier-dense norms; modelcard silent |
| `implied_schedule_split` | 70/20/10 compute share, LR ladder 3e-4 / 1e-4 / 5e-5 | Modelcard implies 3 stages, doesn't fix the split |
| `capability_targets` | every 8.1 scoreboard number as a release gate | **Pinned** — these are modelcard 8 numbers |
| `safety_thresholds` | 4.1 / 4.4 / 4.5 / 5.1 / 5.2 numerical thresholds | **Pinned** — explicit numbers from 4 / 5 |
| `implied_serving_minimums` | inference GPUs, edge layer, safety gate, observability | Required to actually deploy at modelcard scale |
| `implied_special_tokens_required` | the 16 chat-surface tokens | Cross-validated against `tokenizer.SPECIAL_TOKENS` |
| `implied_multilingual_coverage` | the 44 modelcard 8.12 languages by tier | Cross-validated against `tokenizer.MODELCARD_LANGUAGES` |

### Reading these in code

```python
from sota_model import load_implied

implied = load_implied("configs/sota_4_7.yaml")

# Use capability_targets as release gates
targets = implied["capability_targets"]
assert measured["swe_bench_verified"] >= targets["swe_bench_verified_pct_min"]

# Use safety_thresholds as deployment gates
thresh = implied["safety_thresholds"]
assert measured["single_turn_violative_harmless_rate"] >= thresh["single_turn_violative_harmless_rate_min_pct"]
```

### Why split into "config" vs "implied"

The dataclass-loaded sections (`model:` / `training:` / `inference:`) are what the **runtime** consumes — change one and the model behaves differently. The `implied_*` and `*_targets` / `*_thresholds` sections are what an **operator** commits to — change one and you've moved the goalposts for what counts as a successful build. Mixing them in one section invites the wrong people to edit the wrong knobs.

---

## `model:` — why each value

| Field | Value | Why |
|---|---|---|
| `vocab_size` | 200,000 | Big enough to compress major non-English languages without byte-fallback dominating; modelcard 8.12 evaluates 44 languages with <-3.6% gap to English on GMMLU. Smaller vocabs (50K–128K) hurt low-resource languages disproportionately. |
| `d_model` | 16,384 | Sets the model "width." Combined with `n_layers=110` and `ffn_dim=65,536`, gives ~200B dense params — the SuperModel 4.7-class scale referenced throughout ``. |
| `n_layers` | 110 | "Depth." Frontier dense models live in the 96–110 range; deeper helps reasoning chains, beyond ~110 returns diminish and pipeline-parallel bubbles grow. |
| `n_q_heads` | 128 | One query head per 128-dim slice of d_model; standard ratio. |
| `n_kv_heads` | 16 | **Load-bearing for 1M context.** Ratio 128:16 = 8 means the KV cache is 1/8 the size of a vanilla MHA cache. See [`../src/sota_model/modeling/README.md`](../src/sota_model/modeling/README.md) GQA for the math (~880 GB at 1M tokens vs. ~7 TB without). |
| `head_dim` | 128 | Standard. Combined with 128 Q-heads, full attention dim = 16,384 = `d_model`. |
| `ffn_dim` | 65,536 | 4× `d_model` — long-standing default. SwiGLU's gate halves the effective param count vs. plain MLP, so this lands near "4×-equivalent." |
| `norm_eps` | 1e-5 | RMSNorm stability constant. Smaller risks fp16 overflow; larger leaks signal. |
| `tie_embeddings` | false | At 200B scale the LM head matters; tying constrains it unhelpfully. Smaller models tie embeddings to save params — not relevant here. |
| `max_position_embeddings` | 1,048,576 | 1M context window. Required for `` 8.7 (MRCR v2, GraphWalks 1M) and 8.8 (10M-token agentic search via 4.5 compaction on top of 1M). |
| `rope_base` | 1,000,000 | RoPE base raised from the classic 10,000 so high-frequency dimensions don't wrap around at long context. Standard for 1M-context models. |
| `rope_yarn_scale` | 8.0 | YaRN extends an 8K-trained model to 64K (=8× scale) cleanly; combined with Stage-2 long-context training reaches 1M. Reference: Peng et al. 2023. |
| `rope_yarn_original_max_position` | 8,192 | The seq length used in Stage-1 pretraining. YaRN interpolates relative to this. |
| `sliding_window_size` | 32,768 | A subset of layers use sliding-window attention to bound KV memory at long context. 32K is large enough that local-coherence isn't lost, small enough to matter. |
| `sliding_window_layer_stride` | 2 | Every 2nd layer (excluding layer 0) uses sliding window; the rest stay full-attention so global retrieval (MRCR v2) still works. |
| `vision_max_image_long_edge_px` | 2,576 | **`` invariant.** This resolution bump (vs. 1568px in prior models) is what unlocks ScreenSpot-Pro 79.5%→87.6% and LAB-Bench FigQA 78.6%→86.4%. |
| `vision_max_image_pixels` | 3,750,000 | 3.75 MP total cap, paired with the long-edge cap above. |
| `thinking_budgets` | min/low/medium/high/max → 0/256/2048/8192/32768 | Five effort tiers; `` 4.1.1 makes adaptive thinking a first-class mode chosen per-query by the model itself. |
| `thinking_token_min_floor` | 32 | Prevents the effort head from collapsing all queries to 0 thinking on hard problems where it should at least try. |

---

## `training:` — why each value

| Field | Value | Why |
|---|---|---|
| `optimizer` | adamw | `` doesn't prescribe an optimizer; AdamW is the framework-neutral default. Operators free to swap. |
| `lr` | 3e-4 | Stage-1 default; Stage-2 drops to 1e-4 and Stage-3 to 5e-5 (see `training/schedule.py`). |
| `beta1, beta2` | 0.9, 0.95 | β₂=0.95 (vs. classic 0.999) is standard at LLM scale — faster adaptation to changing gradient statistics in the first ~10K steps. |
| `weight_decay` | 0.1 | Standard for large LMs; smaller WD overfits, larger smooths to no avail. |
| `grad_clip` | 1.0 | Empirical default. Spikes do happen in early training; clip prevents them from corrupting AdamW's running statistics. |
| `warmup_steps` | 2,000 | ~0.2% of total. Long enough to stabilize early dynamics, short enough not to waste compute. |
| `total_steps` | 1,000,000 | Stage-1 length. At global batch 4M tokens × 1M steps = 4T tokens for Stage 1 alone; Stages 2 and 3 add proportionally less compute. |
| `seq_len` | 8,192 | Stage-1 default. Stage 2 ramps to 32K → 131K → 1M. |
| `global_batch_tokens` | 4,194,304 | 4M tokens/step. Critical-batch-size analyses suggest this regime is near-optimal for ~200B models on web-scale data. |
| `grad_accum` | 16 | Lets micro-batch fit on a per-replica memory budget without reducing the global batch. |
| `mixed_precision` | bf16 | Required at this scale. fp16 loss scaling fails at >100B; fp32 wastes half the memory bandwidth. |
| `grad_checkpointing` | true | Trades ~30% extra forward FLOPs for ~6× activation-memory savings. Mandatory at 110 layers + 1M context. |
| `zero_stage` | 3 | Optimizer state + gradients + parameters all sharded across DP replicas. Without ZeRO-3 the optimizer state alone (8 bytes/param × 200B = 1.6 TB) doesn't fit. |
| `tp_degree` | 8 | One node × 8 GPUs over NVLink. Past 8-way, inter-node TP all-reduces become latency-bound. |
| `pp_degree` | 8 | Across 8 nodes. 110 layers / 8 ≈ 14 layers/stage — small enough that pipeline bubbles stay manageable. |

For the layout math (TP × PP × DP = world_size), see [`../src/sota_model/training/README.md`](../src/sota_model/training/README.md).

---

## `inference:` — why each value

| Field | Value | Why |
|---|---|---|
| `temperature, top_p, top_k, repetition_penalty` | 0.7, 0.95, 40, 1.1 | ``-aligned defaults. Tuned empirically; deviations should be measured against the 8 evaluation suite. |
| `max_new_tokens` | 8,192 | Default per-call answer cap. Independent of thinking budget. |
| `cache_implementation` | paged | vLLM-style paged attention. The alternative ("static") doesn't survive arbitrary-length agentic conversations. |
| `kv_cache_dtype` | bf16 | Default. int8 available for memory-tight deployments at small honesty/MASK regression cost (see modelcard 6.3.3). |
| `page_block_size` | 16 | Standard. Trade-off: smaller = less internal fragmentation, more block-table overhead. 16 is the sweet spot for 1M-token contexts. |
| `enable_prefix_cache` | true | Shared system prompts are reused across many requests. Prefix caching removes the repeated prefill cost. |
| `adaptive_thinking` | true | **First-class mode** per `` 4.1.1. Set to false to force a fixed effort tier. |
| `default_effort` | high | Used when adaptive thinking is disabled, or when the effort head is uncertain. |
| `thinking_visible_to_user` | false | Hidden reasoning channel; user-facing API hides `<\|thinking\|>...<\|/thinking\|>` blocks. |
| `context_compaction_trigger` | 200,000 | **`` invariant** (4.5). Required for 10M-token BrowseComp / DeepSearchQA runs. |
| `max_context_tokens` | 1,048,576 | Hard ceiling per request before compaction must fire. |

---

## How to override

Don't edit `sota_4_7.yaml` for ad-hoc experiments — copy it:

```bash
cp configs/sota_4_7.yaml configs/sota_4_7_smoke.yaml
# edit the copy
sota-pretrain --config configs/sota_4_7_smoke.yaml --output-dir ./checkpoints/smoke
```

Programmatic overrides happen in code, not YAML:

```python
cfg = ModelConfig.from_yaml("configs/sota_4_7.yaml")
cfg = dataclasses.replace(cfg, n_layers=8, d_model=512)  # tiny smoke variant
```
