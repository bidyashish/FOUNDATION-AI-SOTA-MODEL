# Looped (depth-recurrent) transformer — design, numbers, and the research it rests on

> Companion to [`ARCHITECTURE.md`](./ARCHITECTURE.md) (the stack) and
> [`TRANSFORMER_FLOW.md`](./TRANSFORMER_FLOW.md) (shapes). Config:
> [`configs/sota_ultra_5_looped.yaml`](../configs/sota_ultra_5_looped.yaml). Code:
> `ModelConfig.recurrent_*` → `SOTAModel.run_blocks` / `unrolled_blocks`.
> Recurrence is **off by default** (`recurrent_core_layers: 0`); the flagship
> `sota_ultra_5.yaml` is bit-for-bit unchanged.

## 0. One-paragraph summary

A looped transformer re-runs a weight-shared block of layers several times per token.
Stored parameters (what the checkpoint holds, what weight memory costs) stay fixed; the
*compute* a token receives — and with it the model's reasoning depth — scales with the loop
count. The 2025 generation of this idea (Huginn, Ouro, Mixture-of-Recursions, Relaxed
Recursive Transformers, and the Saunshi et al. theory) established that (a) looping buys
reasoning depth roughly as well as unique layers do, (b) it does **not** buy knowledge
capacity, which tracks stored params, and (c) a model trained at random loop counts can be
served at any of them, which makes loop count a natural *second axis of adaptive effort*
next to visible thinking tokens. This repo implements the Huginn-style **prelude → core ×
loops → coda** topology with additive input injection, trains it with a per-step uniform
loop sampler, serves it with the effort tier picking the depth, and re-derives every
downstream number (compute params, 6·N·D, GPU-hours, KV footprint) so `make check` holds.

## 1. What changed in the knowledge, and what that changed in the design

The stack-only UltraModel 5 design (`ARCHITECTURE.md`) encodes a 2024 assumption: depth is
a compile-time constant, and "more compute per token" means a bigger model. Since then:

| Work | Year | What it showed | What we take from it |
|---|---|---|---|
| Dehghani et al., *Universal Transformer* | 2018–19 | Weight-tied depth with ACT halting generalises algorithmically | The idea; ACT-style halting is *not* adopted (see §7) |
| Giannou et al., *Looped Transformers as Programmable Computers* | 2023 | A looped block can emulate a general-purpose instruction set | Depth recurrence is expressive enough to be the reasoning substrate |
| Yang, Lee, Nowak, Papailiopoulos, *Looped Transformers are Better at Learning Learning Algorithms* | 2023–24 | Re-injecting the input at every loop (input injection) is what makes deep loops trainable and stops drift | **Additive input injection** at every core entry (§2) |
| Bae et al., *Relaxed Recursive Transformers* | ICLR 2025 | Layer tying converts a pretrained stack into a looped one; per-loop LoRA "relaxes" the tie; depth-wise batching recovers throughput | Keep the stored stack the same shape so a non-recurrent checkpoint *is* a valid loops=1 model; per-loop LoRA deferred (§7) |
| Saunshi et al., *Reasoning with Latent Thoughts: On the Power of Looped Transformers* | ICLR 2025 | A k-layer block looped L times matches a kL-layer model on reasoning benchmarks; perplexity (knowledge) still tracks params; scaling laws in *effective depth* | Compute/FLOP accounting must use **unrolled** depth; data budget (36T) stays anchored to **stored** params |
| Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* (Huginn-0125) | Feb 2025 | Prelude / recurrent core / coda at 3.5B params, 800B tokens; loop count sampled per step (log-normal Poisson, mean 32); truncated BPTT over the last 8 loops; KL-based adaptive exit; test-time depth up to 64+ | The **P / R / C topology**, random per-step loop sampling, and "depth is chosen at inference" |
| Bae et al., *Mixture-of-Recursions* | 2025 | Middle-Cycle sharing (unique first/last, shared middle) beats other sharing patterns; token-wise routers pick recursion depth; recursion-wise KV caching vs recursive KV sharing | Middle-Cycle = our prelude/core/coda; **per-virtual-layer KV** (recursion-wise caching); token-wise routing deferred (§7) |
| ByteDance Seed, *Ouro: Scaling Latent Reasoning via Looped Language Models* | Oct 2025 | Whole-stack looping ×4 at 1.4B/2.6B on 7.7T tokens matches 4B–12B dense models on reasoning; entropy-regularised learned exit | **loops_max = 4** is where the 2025 evidence is dense; whole-stack looping (P = C = 0) supported by the same code |

The distorted picture the old docs encoded was "params = compute = depth". The corrected
picture has three separate numbers per model: **stored params** (627B, fixed), **virtual
depth** (128 → 320, chosen per sequence), **compute params per token** (627B → 1557B,
follows depth). Every derivation below keeps them apart.

## 2. The topology

```
                           ┌───────────────────────────────────────┐
input_ids ─► embed ─► PRELUDE (P=32 unique layers) ─► e            │
                                       │                           │
                      s_0 = 0          ▼                           │
                     ┌──────► (+ e) ─► CORE (R=64 shared layers) ─┐│  × loops  (1..4)
                     │                                             ││
                     └─────────────── s_i ◄────────────────────────┘│
                                       │                           │
                                       ▼                           │
                                CODA (C=32 unique layers) ─► final RMSNorm ─► lm_head
                                                            └─► effort head
```

* `s_0 = 0`, `s_i = core(s_{i-1} + e)` for `i = 1..loops`, `y = coda(s_loops)`.
* **Input injection is additive, no adapter.** Huginn concatenates `(e, s)` and projects;
  Yang et al. add. Adding keeps the core's input width `d_model`, adds no parameters, and —
  because `s_0 = 0` — makes `loops = 1` reduce to `coda(core(e))`, i.e. the plain stack.
  A checkpoint trained without recurrence is therefore a valid looped model at depth 1
  (verified numerically: identical logits, see §8).
* **No loop-index embedding.** The core is told nothing about which loop it is in; the
  state carries that implicitly (Huginn finds this sufficient). Deferred, see §7.
* **The stored stack is unchanged**: 128 layers, same width, heads and FFN as UltraModel 5.
  `P + R + C = n_layers` is enforced in `ModelConfig.__post_init__`.

### Why 32 / 64 / 32 and loops_max 4

* Middle-Cycle sharing (MoR) and Huginn both keep unique layers at both ends: the prelude
  maps tokens into a latent space the core can iterate on; the coda maps the iterated
  latent back to logits. 25 / 50 / 25 is Huginn's ratio (2 / 4 / 2) scaled.
* The band edges (32, 96) are multiples of 16 = layers per pipeline stage at PP 8, so
  prelude = stages 0–1, core = stages 2–5, coda = stages 6–7; no stage mixes once-run and
  looped layers (`validate_parallel_layout` warns otherwise).
* `loops_max = 4` (not Huginn's 32): at `d_model` 18432 the unrolled depth at 4 is already
  320 — beyond any 2026 dense stack — and KV grows linearly with depth (§3.3). Ouro's
  evidence is at ×4. Raising it later changes no stored weights.

## 3. The arithmetic (all gate-checked by `scripts/validate_config.py`)

Notation: `N_stored` = params in the checkpoint; `N_core` = params of the 64 core layers;
`N_compute(r) = N_stored + (r − 1)·N_core` = params touched per token at `r` loops;
`V(r) = P + r·R + C` = virtual layers.

### 3.1 Parameters

| | Formula | Value |
|---|---|---|
| per layer (attn + SwiGLU + norms) | `18432·(18432 + 2·2304) + 18432² + 3·18432·73728 + 2·18432 + 2·128` | 4.841 B |
| `N_core` | 64 × per layer | 309.84 B |
| `N_stored` | embed + 128 layers + head | **627.06 B** (= UltraModel 5) |
| `N_compute(1)` | | 627.06 B |
| `N_compute(2)` | | 936.90 B |
| `N_compute(3)` | | 1 246.75 B |
| `N_compute(4)` | | **1 556.59 B** (2.48× stored) |
| `N_compute(2.5)` (training mean) | | **1 091.83 B** |

Code: `ModelConfig.estimate_params_billions()` (stored) and
`estimate_compute_params_billions(loops)`; on the module, `num_parameters()` and
`num_compute_parameters(loops)`.

### 3.2 Training compute

Training samples `r ~ Uniform{1,…,4}` per optimizer step (§4), so the expected per-token
compute is `N_compute(E[r]) = N_compute(2.5)`:

```
FLOPs = 6 · N_compute(2.5) · D = 6 × 1091.83e9 × 36e12 = 2.358e26     (1.74× UltraModel 5)
B300-hours @ 7.5 PFLOPS fp8, 40–50 % MFU  = 2.358e26 / (7.5e15 × 0.40…0.50) / 3600 = 17.5M – 21.8M
wall clock @ 4608 GPUs                     = 22.6 – 28.2 weeks
cost @ $5–7 / B300-hour                    = $88M – $153M
```

Per fixed loop count at 36T: 1 → 1.35e26, 2 → 2.02e26, 3 → 2.69e26, 4 → 3.36e26. The
YAML's `implied_compute` carries the 2.5 mean. If the sampler changes (e.g. Huginn's
heavy-tailed Poisson), `expected_train_loops()` changes and the gate forces the YAML to
follow.

### 3.3 KV cache

The paged cache is indexed by **virtual** layer (MoR's "recursion-wise caching"): each
core pass writes its own K/V. Cache storage is allocated at `V(loops_max)`; a sequence run
at fewer loops declares its active depth (`PagedKVCache.set_active_layers`) and uses the
first `V(r)` slots.

```
bytes/token(r) = V(r) × 18 KV heads × 128 head_dim × 2 (K,V) × 2 B (bf16)
   r=1: 128 → 1152 KiB     r=2: 192 → 1728 KiB     r=3: 256 → 2304 KiB     r=4: 320 → 2880 KiB
@ 1M tokens, r=4: 2880 GiB bf16 / 1440 GiB fp8 (the default kv_cache_dtype)
```

This is the real cost of looping and the reason `loops_max` stays at 4. Cross-loop KV
sharing (MoR's "recursive KV sharing", Huginn's choice) would hold KV at 1152 KiB
regardless of depth at some quality cost; it is the first optimisation to evaluate (§7).

### 3.4 Memory at training

Activations scale with virtual depth too: with gradient checkpointing, one checkpointed
boundary per *executed* block, i.e. up to 320 per micro-batch rather than 128. Weight,
gradient and optimizer memory are unchanged (stored params). The reference layout
(TP 9 × PP 8 × DP 64, micro-batch 1) therefore still fits weights exactly as before and
needs up to 2.5× the activation memory on the core stages at `r = 4`.

## 4. Training

* **Loop sampling.** `ModelConfig.sample_train_loops(step, seed)` → `Uniform{1..loops_max}`,
  deterministic in `(seed, step)`. One draw per **optimizer step** (`step // grad_accum`),
  so all micro-batches of a step and all TP/PP/DP ranks unroll the same depth — pipeline
  stages that hold the core have to agree on how many times to re-enter. Both trainers
  (`training/pretrain.py`, DeepSpeed and plain PyTorch) pass `loops=` into the forward.
  Uniform rather than Huginn's log-normal Poisson because every served depth (1–4) is a
  first-class product tier and should see equal training signal; Ouro trains at a fixed 4
  with learned exits, which we don't have.
* **Backprop.** Full BPTT through all loops (at most 4, so Huginn's truncation to the last
  8 is moot).
* **Init.** Residual-out scale stays `1/√(2·n_layers)` with the *stored* depth: at `r = 1`
  the model must initialise identically to the non-recurrent one, and input injection
  re-reads `e` each loop rather than stacking writes indefinitely. Revisit if `loops_max`
  grows well past 4.
* **Schedule / corpus.** Unchanged (36T, 70/20/10, same LR ladder). Saunshi et al.: loops
  do not add knowledge capacity, so the data budget is anchored to stored params.
* **Post-training.** PPO policy, reward model and the safety probes walk the depth through
  `SOTAModel.run_blocks` (never `model.layers` directly) at `loops_max`; RL rollouts are
  produced at max effort so the trained depth matches. Invariants 3 and 7 are untouched —
  the loop count is an architectural choice, not a reward signal, and nothing in the
  recurrence reads the thinking channel.

## 5. Inference: effort → depth

`ModelConfig.recurrent_loops_per_effort` is the second axis of the effort ladder:

| tier | thinking tokens (`thinking_budgets`) | loops | virtual depth | compute params |
|---|---|---|---|---|
| min | 0 | 1 | 128 | 627 B |
| low | 1 024 | 1 | 128 | 627 B |
| medium | 8 192 | 2 | 192 | 937 B |
| high | 32 768 | 3 | 256 | 1 247 B |
| max | 131 072 | 4 | 320 | 1 557 B |

The two axes trade differently: thinking tokens are *serial* (latency ∝ tokens, KV grows
with tokens), loops are *per-token* (every token gets deeper, latency ∝ loops, KV grows
with depth). A query that needs long exploration wants tokens; a query that needs a few
hard inferences wants depth. The effort head picks the tier; the tier sets both.

**Prefill protocol** (`inference/engine.py::_run`). The loop count is a per-sequence
constant because the KV layout depends on it, but the effort head only exists after a
prefill. So: prefill at the *probe* depth (the forced tier's loops, else
`InferenceConfig.default_effort`'s), read the effort logit, and if the decided tier wants a
different depth, allocate a new cache and prefill once more at that depth. Decode, thinking
and compaction (`_compact` re-prefills into a `reset()` cache, which keeps its active depth)
all carry the same `loops`. Non-recurrent configs map every tier to 1 loop, so the
re-prefill never fires there.

## 6. Parallelism

TP is unaffected (same per-layer shapes). PP needs the band edges on stage boundaries
(§2); `validate_parallel_layout` warns when they aren't. The simple 1F1B schedule runs the
core stages `loops` times per micro-batch; with per-step loop sampling, the pipeline
bubble fraction is unchanged within a step and the step time scales with `V(r)/128`.
Relaxed Recursive Transformers' *depth-wise batching* (serving different sequences at
different loop indices in one batch) is the throughput story for serving mixed effort
tiers; it is a serving-engine concern and out of scope here.

## 7. Deliberately not included (follow-ups, in priority order)

1. **Cross-loop KV sharing** (reuse loop-1 K/V for all loops): cuts the 2.5× KV cost back
   to 1×. Needs an ablation at this scale; MoR reports a small quality cost.
2. **Learned exit / adaptive per-token depth** (Huginn's KL exit, Ouro's entropy-gated
   exits, MoR's routers): turns `loops` from a tier constant into a learned per-token
   choice. Requires the loss machinery and interacts with the paged-cache layout.
3. **Per-loop LoRA relaxation** (Relaxed Recursive Transformers): lets loop *i* differ
   slightly from loop *j* at ~0.1 % extra params.
4. **Loop-index conditioning** (a learned per-loop embedding or scale): cheap, untested here.
5. **Heavy-tailed loop sampling** and **truncated BPTT** become relevant only if `loops_max`
   grows well past 4.

## 8. Verification performed

On a CPU toy shape (8 stored layers, `d_model` 64, P/R/C = 2/4/2, `loops_max` 4) and the
whole-stack case (P = C = 0, R = 8):

* `loops = 1` logits are **bit-identical** to the same weights in a non-recurrent
  `SOTAModel` (max |Δ| = 0.0).
* prefill + token-by-token decode through the paged cache matches the single full forward
  to fp16 cache precision (max |Δ| ≈ 3e-3) at loops 1, 2, 4, with a sliding-window
  override, and for the whole-stack shape.
* Gradient checkpointing runs and produces non-zero core gradients at `loops = 2`.
* `run_blocks(stop_after_virtual=v)` matches a manual prelude + partial-core run (probe path).
* The cache rejects a forward whose `loops` disagrees with its active depth.
* `make check` passes for `sota_ultra_5.yaml` and `sota_ultra_5_looped.yaml`;
  `make smoke` passes.

Two pre-existing bugs in the cached decode path surfaced during this verification and were
fixed because the looped flow depends on that path: `PagedKVCache.gather` returned only the
*previous* passes' tokens for every layer but the last (the current tokens' K/V were
written but not read), and the SDPA fallback used `is_causal=True` for single-token decode,
which PyTorch aligns top-left (a 1-token query saw only key 0). Both are in
`modeling/kv_cache.py` / `modeling/attention.py` and are now covered by the equivalence
check above.
