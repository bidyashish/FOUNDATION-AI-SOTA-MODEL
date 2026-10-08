# `configs/`

Central knob for the model. Everything in here traces back to a target in the UltraModel 5 System Card. When code and config disagree, config wins; when config and the system card disagree, the system card wins.

## Files

```
sota_ultra_5.yaml               UltraModel 5 — THE spec (model + training + inference + implied_* + gates)
sota_ultra_5_looped.yaml        UltraModel 5 looped variant — same stored stack, depth-recurrent core
                                (prelude 32 / shared core 64 / coda 32, 1–4 loops); all compute/KV numbers re-derived
sota_ultra_5_looped_2pass.yaml  2-pass serving profile of the looped checkpoint — same weights, every effort
                                tier at 2 loops (192 virtual layers, 937B compute params); nothing is trained from it
```

`sota_ultra_5.yaml` is the single source for every default in the repo: the `config.py`
dataclass defaults are a verbatim copy of its `model:` / `training:` / `inference:` sections,
and `scripts/validate_config.py` fails if they drift. The looped file is derived from it and
changes only the recurrence block (and everything that arithmetic implies). The 2-pass file is
a *serving profile* of the looped checkpoint: it flattens the loop ladder and inherits every
training number from the looped file (`implied_compute.serving_profile_of`, gate-checked).

### The spec: `sota_ultra_5.yaml` (UltraModel 5)

UltraModel 5 is the generally-available configuration of MathsSchool's Mythos-class frontier
model (same weights as SecretUltraModel 5, plus production safeguards and fallback to the
latest prior-generation model). Headline numbers, every one gate-checked:

| Knob | UltraModel 5 |
|---|---|
| `d_model` / `n_layers` / `n_q_heads` / `n_kv_heads` / `ffn_dim` | 18432 / 128 / 144 / 18 / 73728 |
| total params (dense) | ~627B (verify: `ModelConfig().estimate_params_billions()` → 627.06) |
| KV @ 1M, bf16 / fp8 | 1152 / 576 GiB (1152 / 576 KiB per token) |
| training tokens | 36T (≈2.9× Chinchilla at 627B; mix web 35 / code 22 / math_structured 13 / synthetic_reasoning 10 / academic 8 / books_reference 7 / dialogue_instructions 5) |
| `global_batch_tokens` / `lr` | 8M / 0.00025 (ladder 0.00025 / 8e-05 / 4e-05 across the three stages) |
| `total_steps` (= corpus tokens ÷ batch) | 4.3M (split 70/20/10 by `training/schedule.py`) |
| `implied_compute` 6·N·D anchor (gate-checked) | 1.354e+26 (10M to 13M B300-hr fp8, 13 to 17 weeks, $50M to 91M; `ModelConfig.training_flops(36e12)`) |
| TP × PP × DP (world size) | 9 × 8 × 64 = 4608 B300s — one GB300 NVL72 rack per replica; TP 9 divides the 18 KV heads (TP 8 does not) |
| batch identity (seq × micro × accum × dp) | 8192 × 1 × 16 × 64 = 8,388,608 (enforced by `TrainingConfig`) |
| `thinking_budgets` (min/low/medium/high/max) | 0/1024/8192/32768/131072 (the card's "xhigh" maps onto `max`) |
| `capability_targets` / `safety_thresholds` | §8 scoreboard @ UltraModel 5 / SecretUltraModel 5 (47 gates) / §4–§5 @ UltraModel 5 (14 gates) |

### Looped variant: `sota_ultra_5_looped.yaml` (depth-recurrent UltraModel 5)

Same width, heads, FFN, corpus, batch, LR ladder, gates and serving stack as
`sota_ultra_5.yaml`; the only architectural delta is that stored layers 32–95 form a
weight-shared core re-entered up to 4× per token, with the effort tier choosing the depth
(design + research: [`docs/LOOPED_TRANSFORMER.md`](../docs/LOOPED_TRANSFORMER.md)). Deltas,
all gate-checked:

| Knob | UltraModel 5 | UltraModel 5 looped |
|---|---|---|
| `recurrent_prelude_layers` / `recurrent_core_layers` / coda | 0 / 0 / — (every layer unique) | 32 / 64 / 32 (band edges on PP-8 stage boundaries) |
| `recurrent_loops_max`; `recurrent_loops_per_effort` | 1; all tiers → 1 | 4; min 1 / low 1 / medium 2 / high 3 / max 4 |
| stored params (checkpoint) | 627.06B | **627.06B — unchanged** |
| virtual layers per token | 128 | 128 / 192 / 256 / 320 at loops 1–4 (`virtual_layers_at_loops_max: 320`) |
| compute params per token | 627B | 627B → 1557B (`compute_params_billions_at_loops_max`); training mean 1092B at mean loops 2.5 |
| `implied_compute` 6·N·D anchor | 1.354×10²⁶ (10–13M B300-hr) | 2.358×10²⁶ = 6 × 1091.83e9 × 36e12 (17–22M B300-hr, 23–28 wks, $88–153M); `ModelConfig.training_flops(36e12)` |
| KV @ 1M, bf16 / fp8 | 1152 / 576 GiB | 2880 / 1440 GiB at loops 4 (`kv_cache_kib_per_token_bf16_at_loops_1: 1152` keeps the loops-1 anchor) |
| training tokens, batch, `total_steps`, LR ladder, TP×PP×DP | 36T, 8M, 4.3M, 2.5e-4 ladder, 9×8×64 | identical |

Extra gate checks on this file (`scripts/validate_config.py`): geometry `P + R + C = n_layers`,
monotone effort→loops ladder, the three `implied_scale` recurrence fields against
`ModelConfig.estimate_compute_params_billions()` / `n_virtual_layers()` /
`kv_cache_bytes_per_token(loops=1)`, and the FLOPs anchor reports which N it used.

The 1M context, 200K compaction trigger, and 2576px / 3.75 MP image cap are unchanged
(all modelcard-pinned). The `implied_*` section structure is identical, so
`load_implied()`, `resolve_sources_from_yaml()`, and `ReleaseGate` work against either file.

### 2-pass serving profile: `sota_ultra_5_looped_2pass.yaml`

Serves the **same checkpoint** as `sota_ultra_5_looped.yaml` with every effort tier fixed at
2 loops — same base weights, fewer passes. Nothing is trained from this file: the `training:`
section, corpus, schedule split and the 6·N·D anchor are the looped run's, and
`implied_compute.serving_profile_of` makes the gate check that they are identical. Arithmetic,
efficiency and caveats: [`docs/LOOPED_2PASS_PROFILE.md`](../docs/LOOPED_2PASS_PROFILE.md).

| Knob | UltraModel 5 looped (base) | 2-pass profile |
|---|---|---|
| `recurrent_loops_max`; `recurrent_loops_per_effort` | 4; min 1 / low 1 / medium 2 / high 3 / max 4 | 2; every tier → 2 |
| stored params (checkpoint) | 627.06B | **627.06B — the same file** |
| virtual layers / compute params per token | 128–320 / 627–1557B by tier | 192 / 936.90B at every tier (`compute_params_billions_at_loops_max: 937`) |
| KV per token bf16 / fp8 (provisioned at `loops_max`) | 2880 / 1440 KiB | 1728 / 864 KiB (0.6×); `kv_cache_kib_per_token_bf16_at_loops_1: 1152` unchanged |
| KV @ 1M, bf16 / fp8 | 2880 / 1440 GiB | 1728 / 864 GiB |
| `training:`, corpus, LR ladder, 6·N·D anchor, hour/$ bands, `compute_params_billions_train_mean` | 36T, 2.358×10²⁶, 1092B at mean loops 2.5 | identical — inherited from the base run |

Extra gate checks on a profile: the `model:` section equals the base's bar the two ladder
fields, `training:` / `implied_training_corpus` / `implied_schedule_split` equal the base's,
`recurrent_loops_max` is within the base's trained range, and the FLOPs anchor equals the
base's and its 6·N·D — so a profile can never claim a cheaper pretraining run than the one
that produced its weights.

## How the YAML maps to dataclasses

```
sota_ultra_5.yaml   (and sota_ultra_5_looped.yaml, sota_ultra_5_looped_2pass.yaml)
├── model:         → src/sota_model/config.py::ModelConfig
├── training:      → src/sota_model/config.py::TrainingConfig
└── inference:     → src/sota_model/config.py::InferenceConfig
```

Loaders: `ModelConfig.from_yaml(path)`, `TrainingConfig.from_yaml(path)`, `InferenceConfig.from_yaml(path)`.
`ModelConfig()` with no arguments is the flagship model; the gate's "dataclass defaults vs
flagship" check keeps that true.

---

## Back-trace: every field vs the system card

The system card pins **7** numerical/structural choices; everything else below is an
operator commitment that must stay derivable (the gate checks the arithmetic).

| Section | What's in it | Why it had to be implied |
|---|---|---|
| `implied_scale` | 627B params, 1152 KiB/tok, 1152 GiB at 1M (bf16) | modelcard places UltraModel 5 in the frontier-dense band but pins no number |
| `implied_training_corpus` | 36T tokens, 7-bucket source mix (web 35 / code 22 / math_structured 13 / synthetic_reasoning 10 / academic 8 / books_reference 7 / dialogue_instructions 5) | 1.1.1 names sources, doesn't pin tokens |
| `implied_compute` | 10^26 to 3×10^26 FLOPs, 10M to 13M B300·hr fp8, 13 to 17 weeks, $50M to 91M | required to actually train at this scale |
| `implied_schedule_split` | 70/20/10 compute share + LR ladder 0.00025 / 8e-05 / 4e-05 | modelcard implies 3 stages, doesn't fix the split |
| `capability_targets` | every §8 scoreboard number — SWE-bench Verified 95%, Terminal-Bench 2.1 84%, GPQA 94%, OSWorld 85%, … (47 gates) | directly modelcard-pinned — these are release gates |
| `safety_thresholds` | §4 single-turn harmless 96.9%, §5 ART k=100 ≤ 4.8%, browser-use 0%, … (14 gates) | directly modelcard-pinned — these are deployment gates |
| `implied_serving_minimums` | 8 x A100 80GB int8 / 16 x A100 80GB bf16 / 24 x H100 80GB / 8 x B200 192GB / 6 x B300 279GB / 4 x Rubin R100 288GB, edge layer, pre/post safety gate, observability list | required to deploy at modelcard scale |
| `implied_special_tokens_required` | the 16 chat-surface tokens | cross-validated against `tokenizer.SPECIAL_TOKENS` |
| `implied_multilingual_coverage` | the 42 modelcard 8.12 languages by tier (+ the Indic MILU set) | cross-validated against `tokenizer.MODELCARD_LANGUAGES` |

A `grep` over the system card confirms what's actually pinned vs what's an operator choice. **The system card is silent on architectural scalars** (verified: zero hits for `d_model`, `n_layers`, `n_kv_heads`, `ffn_dim`, `AdamW`, `bf16`, `X B parameters`, `X T tokens`, `temperature=0.X`, `top_p=0.X`).

| Field | Value | Source | Modelcard evidence |
|---|---|---|---|
| **model:** | | | |
| `vocab_size` | 200,000 | operator | silent — chosen for 8.12 multilingual coverage |
| `d_model` | 18,432 | operator | no `d_model` string in modelcard |
| `n_layers` | 128 | operator | no `n_layers` string in modelcard |
| `n_q_heads` | 144 | operator | silent |
| `n_kv_heads` | 18 | operator | silent on GQA ratio (implied only by 1M-context cluster-mem feasibility) |
| `head_dim` | 128 | operator | silent |
| `ffn_dim` | 73,728 | operator | no `ffn_dim` string in modelcard |
| `qk_norm` | true | operator | silent — fp8 attention-logit stabilizer |
| `max_position_embeddings` | 1,048,576 | **MODELCARD** | 8.7 — 2 hits for `1M tokens/context` |
| `rope_base` | 1e6 | operator | implied by 1M context (avoid wrap-around), specific value not pinned |
| `rope_yarn_scale` | 128.0 | operator | must cover the 8K→1M extension ratio (1,048,576 / 8,192 = 128), value not pinned |
| `rope_yarn_original_max_position` | 8,192 | operator | Stage-1 seq len choice |
| `sliding_window_size` | 32,768 | operator | modelcard mentions sliding-window attention idea, not size |
| `sliding_window_layer_stride` | 2 | operator | silent |
| `recurrent_*` | off (core 0, loops 1) | operator | silent — see the looped variant |
| `vision_max_image_long_edge_px` | 2,576 | **MODELCARD** | invariant — 4 hits for `2576px` |
| `vision_max_image_pixels` | 3,750,000 | **MODELCARD** | invariant — 4 hits for `3.75 MP` |
| `vision_patch_size` | 14 | operator | silent |
| `thinking_budgets` keys (min/low/medium/high/max) | — | **MODELCARD** | effort tier names appear in 8 evals (`Max effort`, `High effort`) |
| `thinking_budgets` token values (0/1024/8192/32768/131072) | — | operator | modelcard names tiers but pins no token budgets per tier |
| `thinking_token_min_floor` | 32 | operator | silent |
| **training:** | | | |
| `optimizer` | adamw | operator | no `AdamW`, `optimizer`, or training-hyperparam disclosure in modelcard |
| `lr` | 0.00025 | operator | silent |
| `beta1, beta2` | 0.9, 0.95 | operator | silent |
| `weight_decay` | 0.1 | operator | silent |
| `grad_clip` | 1.0 | operator | silent |
| `warmup_steps` | 3,000 | operator | silent |
| `total_steps` | 4,300,000 | operator | silent — full-corpus budget: 36T ÷ 8M tokens/step |
| `seq_len` | 8,192 | operator | Stage-1 choice |
| `global_batch_tokens` | 8,388,608 | operator | silent |
| `micro_batch_size`, `grad_accum` | 1, 16 | operator | silent |
| `mixed_precision` | fp8 | operator | no `fp8`/`bf16` string in modelcard |
| `grad_checkpointing` | true | operator | silent |
| `zero_stage` | 3 | operator | silent on parallelism / sharding |
| `tp_degree, pp_degree, dp_degree` | 9, 8, 64 | operator | silent |
| `z_loss_coef` | 0.0001 | operator | silent — fp8 logit-drift guard |
| **inference:** | | | |
| `temperature` | 0.7 | operator | modelcard mentions `temperature` but pins NO numeric value |
| `top_p` | 0.95 | operator | modelcard mentions `top_p` but pins NO numeric value |
| `top_k` | 0 (off) | operator | silent |
| `repetition_penalty` | 1.0 (off) | operator | silent |
| `max_new_tokens` | 32,768 | operator | silent |
| `kv_cache_dtype` | fp8 | operator | silent |
| `page_block_size` | 16 | operator | silent on KV cache implementation |
| `enable_prefix_cache` | true | operator | silent |
| `adaptive_thinking` | true | **MODELCARD** | 4.1.1 — 34 hits for `adaptive thinking` |
| `default_effort` | high | operator | tier name pinned, default not |
| `thinking_visible_to_user` | false | operator | silent |
| `context_compaction_trigger` | 200,000 | **MODELCARD** | 4.5 — 4 hits for `200k` |
| `max_context_tokens` | 1,048,576 | **MODELCARD** | 8.7 1M context |

**Summary: 7 of the listed config fields are directly pinned by the system card text.** The rest are operator choices that fit within modelcard constraints but are not numerically specified by the system card.

### What this means for "model size" and "training size"

The system card does **not** disclose:
- Total parameter count (no `X B parameters` hits)
- Training token count (no `X T tokens` hits)
- Layer count, hidden dim, FFN dim, head counts (zero string matches)
- Optimizer choice or hyperparameters
- Mixed-precision dtype

This is consistent with how frontier-model providers publish system cards — capability and safety reports, not architecture specs. The YAML lands at **~627B params with 1152 GiB of bf16 KV at 1M context**, a defensible point at the top of the frontier-dense band, but **that number does not come from the system card** — it comes from the operator choices made in this YAML.

If you need a different size, change `n_layers`, `d_model`, `ffn_dim` together (and keep `n_q_heads × head_dim = d_model`). Nothing in the system card will tell you which set of values is "correct" — there is no published correct answer. [`docs/PERMUTATIONS.md`](../docs/PERMUTATIONS.md) maps the design space.

---

## Implied sections — what an operator MUST commit to

Both configs carry additional sections beyond `model:` / `training:` / `inference:` that capture the values needed to **deliver** modelcard targets, even though modelcard doesn't pin them numerically. These are loaded via `sota_model.config.load_implied(path)` — they are NOT consumed by the dataclasses.

| Section | What it pins | Modelcard tie |
|---|---|---|
| `implied_scale` | param count (~627B), KV KiB/token, GiB at 1M | Frontier-dense band; the card places UltraModel 5 a tier above its predecessors |
| `implied_scale` (looped variant only) | `compute_params_billions_at_loops_max` / `_train_mean`, `virtual_layers_at_loops_max`, `kv_cache_kib_per_token_bf16_at_loops_1` | Stored vs compute params decouple under depth recurrence; see `docs/LOOPED_TRANSFORMER.md` §3 |
| `implied_training_corpus` | 36T tokens, 7-bucket source mix, pipeline knobs | 1.1.1 names sources; doesn't pin tokens |
| `implied_compute` | 10^26 to 3×10^26 FLOPs, 10M to 13M B300·hr fp8, 13 to 17 wks, $50M to 91M | Frontier-dense norms; modelcard silent |
| `implied_schedule_split` | 70/20/10 compute share, LR ladder 0.00025 / 8e-05 / 4e-05 | Modelcard implies 3 stages, doesn't fix the split |
| `capability_targets` | every §8 scoreboard number as a release gate | **Pinned** — these are modelcard §8 numbers |
| `safety_thresholds` | §4 / §5 numerical thresholds | **Pinned** — explicit numbers from §4 / §5 |
| `implied_serving_minimums` | inference GPUs, edge layer, safety gate, observability | Required to actually deploy at modelcard scale |
| `implied_special_tokens_required` | the 16 chat-surface tokens | Cross-validated against `tokenizer.SPECIAL_TOKENS` |
| `implied_multilingual_coverage` | the 42 modelcard 8.12 languages by tier | Cross-validated against `tokenizer.MODELCARD_LANGUAGES` |

### Reading these in code

```python
from sota_model import load_implied

implied = load_implied("configs/sota_ultra_5.yaml")

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
| `vocab_size` | 200,000 | Big enough to compress major non-English languages without byte-fallback dominating; modelcard 8.12 evaluates 42 languages on GMMLU. Smaller vocabs (50K–128K) hurt low-resource languages disproportionately. |
| `d_model` | 18,432 | Sets the model "width." Combined with `n_layers=128` and `ffn_dim=73,728`, gives ~627B dense params (verify via `ModelConfig().estimate_params_billions()`) — the top of the 400–700B frontier-dense band. |
| `n_layers` | 128 | "Depth." Frontier dense models live in the 100–180 band; deeper helps reasoning chains, but wall-clock and pipeline-parallel bubbles grow with depth. 128 divides evenly into 8 PP stages of 16. |
| `n_q_heads` | 144 | One query head per 128-dim slice of d_model (144 × 128 = 18,432); standard ratio. |
| `n_kv_heads` | 18 | **Load-bearing for 1M context.** Ratio 144:18 = 8 means the KV cache is 1/8 the size of a vanilla MHA cache. See [`../src/sota_model/modeling/README.md`](../src/sota_model/modeling/README.md) GQA for the math (1152 GiB at 1M tokens vs ~9 TiB without). 18 also fixes the TP degree: 9 (one NVL72 rack = TP 9 × PP 8). |
| `head_dim` | 128 | Standard. Combined with 144 Q-heads, full attention dim = 18,432 = `d_model`. |
| `ffn_dim` | 73,728 | 4× `d_model` — deliberately FFN-heavy. SwiGLU has 3 projections, so 4×d costs 12·d² per layer vs 8·d² for a standard 4×d MLP (8/3×d ≈ 2.67×d would param-match). |
| `norm_eps` | 1e-5 | RMSNorm stability constant. Smaller risks fp16 overflow; larger leaks signal. |
| `tie_embeddings` | false | At 600B scale the LM head matters; tying constrains it unhelpfully. Smaller models tie embeddings to save params — not relevant here. |
| `qk_norm` | true | Per-head RMSNorm on Q and K before RoPE — the 2026-standard attention-logit stabilizer for fp8 training (replaces softcapping, which FA3 doesn't support). A variant config can switch it off. |
| `max_position_embeddings` | 1,048,576 | 1M context window. Required for system card §8.7 (MRCR v2, GraphWalks 1M) and 8.8 (10M-token agentic search via 4.5 compaction on top of 1M). |
| `rope_base` | 1,000,000 | RoPE base raised from the classic 10,000 so high-frequency dimensions don't wrap around at long context. Standard for 1M-context models. |
| `rope_yarn_scale` | 128.0 | Scale × original_max_position must cover the window: 128 × 8,192 = 1,048,576 (the 1M window). A smaller scale (e.g. 8 → only 64K) silently extrapolates beyond its reach and breaks the 1M retrieval targets. Reference: Peng et al. 2023. |
| `rope_yarn_original_max_position` | 8,192 | The seq length used in Stage-1 pretraining. YaRN interpolates relative to this. |
| `sliding_window_size` | 32,768 | A subset of layers use sliding-window attention to bound KV memory at long context. 32K is large enough that local-coherence isn't lost, small enough to matter. |
| `sliding_window_layer_stride` | 2 | Every 2nd layer (excluding layer 0) uses sliding window; the rest stay full-attention so global retrieval (MRCR v2) still works. |
| `vision_max_image_long_edge_px` | 2,576 | **The system card invariant.** This resolution (vs 1568px in prior generations) is what unlocks the ScreenSpot-Pro and LAB-Bench FigQA targets. |
| `vision_max_image_pixels` | 3,750,000 | 3.75 MP total cap, paired with the long-edge cap above. |
| `thinking_budgets` | min/low/medium/high/max → 0/1024/8192/32768/131072 | Five effort tiers; system card §4.1.1 makes adaptive thinking a first-class mode chosen per-query by the model itself. The card's "xhigh" (USAMO ~100K tokens/attempt) maps onto `max`. |
| `thinking_token_min_floor` | 32 | Prevents the effort head from collapsing all queries to 0 thinking on hard problems where it should at least try. |

---

## `training:` — why each value

| Field | Value | Why |
|---|---|---|
| `optimizer` | adamw | The system card doesn't prescribe an optimizer; AdamW is the framework-neutral default. Operators free to swap (shampoo / muon once validated at scale). |
| `lr` | 0.00025 | Stage-1 peak, width-scaled (LR ∝ 1/d_model, µP-style) with fp8 headroom; Stage-2 drops to 8e-05 (0.32×) and Stage-3 to 4e-05 (0.16×) — pinned in `implied_schedule_split`, consumed by `training/schedule.py`. |
| `beta1, beta2` | 0.9, 0.95 | β₂=0.95 (vs. classic 0.999) is standard at LLM scale — faster adaptation to changing gradient statistics in the first ~10K steps. |
| `weight_decay` | 0.1 | Standard for large LMs; smaller WD overfits, larger smooths to no avail. |
| `grad_clip` | 1.0 | Empirical default. Spikes do happen in early training; clip prevents them from corrupting AdamW's running statistics. |
| `warmup_steps` | 3,000 | ≈0.07% of the 4.3M-step budget (~25B tokens). Long enough to stabilize early dynamics, short enough not to waste compute. |
| `total_steps` | 4,300,000 | Full-corpus budget: 36T tokens ÷ 8M tokens/step ≈ 4.3M optimizer steps (gate-checked). `training/schedule.py` splits the tokens 70/20/10 across the three stages. |
| `seq_len` | 8,192 | Stage-1 default. Stage 2 ramps to 32K → 131K → 1M. |
| `global_batch_tokens` | 8,388,608 | 8M tokens/step. Critical-batch-size analyses put 600B-class models on web-scale data in the 4–8M regime; larger model → larger stable batch. |
| `micro_batch_size`, `grad_accum` | 1, 16 | One 8K sequence per micro-step fits the per-GPU activation budget at 1M-context Stage 2; accumulation keeps the global batch. |
| `mixed_precision` | fp8 | TE fp8 (E4M3 fwd / E5M2 bwd) on Blackwell; bf16 is the Ampere/Hopper fallback. fp16 loss scaling fails at >100B; fp32 wastes memory bandwidth. |
| `grad_checkpointing` | true | Trades ~30% extra forward FLOPs for ~6× activation-memory savings. Mandatory at 128 layers + 1M context. |
| `zero_stage` | 3 | Optimizer state + gradients + parameters all sharded across DP replicas. Without ZeRO-3 the optimizer state alone (8 bytes/param × 627B ≈ 5 TB) doesn't fit. |
| `tp_degree` | 9 | Must divide d_model, ffn_dim and both head counts; 18 KV heads → 9. 9 × 8 = 72 = one GB300 NVL72 NVLink domain, so TP all-reduces never leave the rack. |
| `pp_degree` | 8 | 128 layers / 8 = 16 layers per stage, even — no ragged stage and manageable pipeline bubbles. |
| `dp_degree` | 64 | One replica per rack; 64 racks = 4608 GPUs = the `implied_compute` reference platform. |
| `z_loss_coef` | 0.0001 | Softmax z-loss keeps the logit scale from drifting under fp8 — part of the stability kit with QK-norm. |

For the layout math (TP × PP × DP = world_size), see [`../src/sota_model/training/README.md`](../src/sota_model/training/README.md).

---

## `inference:` — why each value

| Field | Value | Why |
|---|---|---|
| `temperature, top_p, top_k, repetition_penalty` | 0.7, 0.95, 0, 1.0 | Nucleus sampling only; top-k and repetition penalty are off (0 / 1.0) because they measurably hurt long reasoning traces. Deviations should be measured against the §8 evaluation suite. |
| `max_new_tokens` | 32,768 | Default per-call answer cap. Independent of thinking budget. |
| `cache_implementation` | paged | vLLM-style paged attention. The alternative ("static") doesn't survive arbitrary-length agentic conversations. |
| `kv_cache_dtype` | fp8 | Blackwell default — halves KV memory vs bf16 (1152 → 576 KiB/token). bf16 on Hopper/Ampere; int8 for memory-tight deployments at small honesty/MASK regression cost (see modelcard 6.3.3). |
| `page_block_size` | 16 | Standard. Trade-off: smaller = less internal fragmentation, more block-table overhead. 16 is the sweet spot for 1M-token contexts. |
| `enable_prefix_cache` | true | Shared system prompts are reused across many requests. Prefix caching removes the repeated prefill cost. |
| `adaptive_thinking` | true | **First-class mode** per system card §4.1.1. Set to false to force a fixed effort tier. |
| `default_effort` | high | Used when adaptive thinking is disabled, or when the effort head is uncertain. |
| `thinking_visible_to_user` | false | Hidden reasoning channel; user-facing API hides `<\|thinking\|>...<\|/thinking\|>` blocks. |
| `context_compaction_trigger` | 200,000 | **The system card invariant** (4.5). Required for 10M-token BrowseComp / DeepSearchQA runs. |
| `max_context_tokens` | 1,048,576 | Hard ceiling per request before compaction must fire. |

---

## Validating changes

Every numeric relationship in these YAMLs is enforced by the config gate — run it after
any edit, before committing:

```bash
make check          # py_compile everything + validate all three configs
make validate       # just the config gate
python scripts/validate_config.py configs/sota_ultra_5.yaml   # one config
```

It checks ~20 invariants per config: head/hidden arithmetic, YaRN context coverage,
params vs `implied_scale`, KV-cache math, the batch identity
(seq × micro × accum × dp = global batch), `total_steps` vs the corpus commitment,
schedule closure at 70/20/10 with the `implied_schedule_split` LR ladder, TP/PP
shardability, recurrence geometry, serving-profile inheritance (`serving_profile_of`), mix
percentages, gate numericity, the special-token/language counts, and — for the flagship —
that the `config.py` dataclass defaults equal the YAML. CI runs the same gate on every push (`.github/workflows/check.yml`).

## How to override

Don't edit `sota_ultra_5.yaml` (or either looped file) for ad-hoc experiments — copy it:

```bash
cp configs/sota_ultra_5.yaml configs/sota_ultra_5_smoke.yaml
# edit the copy
sota-pretrain --config configs/sota_ultra_5_smoke.yaml --output-dir ./checkpoints/smoke
```

Programmatic overrides happen in code, not YAML:

```python
cfg = ModelConfig.from_yaml("configs/sota_ultra_5.yaml")
cfg = dataclasses.replace(cfg, n_layers=8, d_model=512)  # tiny smoke variant
```
