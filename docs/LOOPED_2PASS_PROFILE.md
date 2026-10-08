# UltraModel 5 looped — the 2-pass serving profile

`configs/sota_ultra_5_looped_2pass.yaml` serves the **same checkpoint** as
`configs/sota_ultra_5_looped.yaml` with every effort tier fixed at **2 passes** through the
shared core. Nothing is trained from it. It exists to answer one question with arithmetic
that the config gate checks: *what does the looped checkpoint cost to serve if we stop
offering the 3- and 4-pass depths?*

## 0. Summary

| | UltraModel 5 (flagship, 1 pass) | looped base at its `max` tier (4 passes) | **2-pass profile (every tier)** |
|---|---|---|---|
| stored params (checkpoint) | 627.06 B | 627.06 B | **627.06 B — same file** |
| virtual layers per token | 128 | 320 | **192** |
| compute params per token | 627.06 B | 1 556.59 B | **936.90 B** |
| forward FLOPs per token (2·N) | 1.25 TFLOP | 3.11 TFLOP | **1.87 TFLOP** |
| KV per token, bf16 / fp8 | 1152 / 576 KiB | 2880 / 1440 KiB | **1728 / 864 KiB** |
| KV at 1M context, fp8 | 576 GiB | 1440 GiB | **864 GiB** |
| KV at 8K context, bf16 | 9.7 GB | 24.2 GB | **14.5 GB** |
| effort → loops ladder | 1 / 1 / 1 / 1 / 1 | 1 / 1 / 2 / 3 / 4 | **2 / 2 / 2 / 2 / 2** |
| pretraining cost (6·N·D) | 1.354×10²⁶ | 2.358×10²⁶ | **inherited: 2.358×10²⁶, already spent** |

Relative to the depth the base's `high`/`max` tiers use, the profile is **0.60×** on all three
per-token axes at once — compute, sequential depth, KV bytes. Relative to the flagship it is
1.49× compute and 1.5× depth and KV, in exchange for 64 extra virtual layers per token.

## 1. Why a serving profile

The looped design (`LOOPED_TRANSFORMER.md`) separates three numbers that the old
stack-only picture conflated: stored params (fixed by the checkpoint), virtual depth
(chosen per sequence), and compute per token (follows depth). Once those are separate, "how
deep do we serve?" is a deployment decision, not a training one. A *serving profile* is a
config that changes only that decision.

The pattern has a public precedent. A Microsoft page (since edited, so treat the claim as
unverified) described GPT-6.1 Sol as using "the same base model weights as GPT-6 Sol with
two inference passes instead of three ... optimized for a more efficient serving profile".
This repo takes the *pattern* from that — same weights, fewer passes, a file that says so —
not any fact about any vendor's model.

Why no retraining is needed: the base run samples the loop count uniformly from 1..4 on
every optimizer step (`ModelConfig.sample_train_loops`), so 2 passes received 25 % of the
base's steps and is a trained depth. The profile pins it; it does not introduce it.

Why 2 and not 1 or 3: 1 pass *is* the flagship (no latent depth). 2 is the smallest depth
that adds latent reasoning over the flagship, and it is the depth the base's `medium` tier
already serves, so the base's release gates already exercise it. A 3-pass profile is the
same file with `3` written in six places; the gate accepts any `recurrent_loops_max` ≤ the
base's 4.

## 2. What is in the file

Everything in `configs/sota_ultra_5_looped_2pass.yaml` is a copy of the looped base except:

| Key | base | profile | why |
|---|---|---|---|
| `recurrent_loops_max` | 4 | **2** | KV storage is provisioned at `loops_max` (`SOTAModel.make_kv_cache`), so this is where the memory saving comes from |
| `recurrent_loops_per_effort` | 1 / 1 / 2 / 3 / 4 | **2 / 2 / 2 / 2 / 2** | flat latent axis; the visible axis (`thinking_budgets`) is unchanged |
| `implied_scale.compute_params_billions_at_loops_max` | 1557 | **937** | 627.06 + 1 × 309.84 (core) = 936.90 B |
| `implied_scale.virtual_layers_at_loops_max` | 320 | **192** | 32 + 2×64 + 32 |
| `implied_scale.kv_cache_kib_per_token_bf16` / `..._gib_at_1m_context_bf16` | 2880 / 2880 | **1728 / 1728** | 192 × 18 × 128 × 2 × 2 B = 1 769 472 B |
| `implied_compute.serving_profile_of` | — | **`configs/sota_ultra_5_looped.yaml`** | tells the gate whose checkpoint this is |

Deliberately **unchanged**: `recurrent_prelude_layers` / `recurrent_core_layers` (shared
weights), every other `model:` field, the whole `training:` section, the 36T corpus and its
mix, the 70/20/10 schedule split and LR ladder, `compute_params_billions_train_mean: 1092`
(the base run's mean of 2.5 loops, not this file's 1.5), the 6·N·D anchor 2.358×10²⁶ and
every hour / week / dollar band, all capability and safety gates, the serving-stack lists.

### What the gate does with `serving_profile_of`

`scripts/validate_config.py` runs its ordinary checks on the profile (geometry, ladder
monotone, KV and compute-param fields at loops 2, batch identity, TP/PP shardability, …)
and, because the key is present, adds five that bind the profile to its base:

1. the `model:` section equals the base's in every field except the two ladder fields;
2. `recurrent_loops_max` is within `1..base.recurrent_loops_max`;
3. `implied_training_corpus` and `implied_schedule_split` equal the base's;
4. the `training:` section equals the base's;
5. `training_flops_point_estimate` equals the base's anchor *and* equals 6·N·D computed from
   the **base** model at the base's mean loops (not from this file's loops).

Check 5 is the one that matters for honesty: a 2-pass model trained from scratch would cost
6 × 781.98e9 × 36e12 = 1.689×10²⁶, and without the profile semantics the gate would have
demanded that number. The profile did not run that job; its weights are the base's.
`make check` validates all three configs by default.

## 3. Efficiency arithmetic

All numbers come from `ModelConfig` helpers on the two YAMLs
(`estimate_compute_params_billions`, `n_virtual_layers`, `kv_cache_bytes_per_token`);
nothing below is measured.

### 3.1 Compute per token (prefill and decode)

Compute params at `loops` = stored + (loops − 1) × core, core = 309.84 B.

| passes | compute params | forward FLOPs / token | vs 4-pass | vs flagship |
|---|---|---|---|---|
| 1 | 627.06 B | 1.254 TFLOP | 0.40× | 1.00× |
| **2** | **936.90 B** | **1.874 TFLOP** | **0.60×** | **1.49×** |
| 3 | 1 246.75 B | 2.493 TFLOP | 0.80× | 1.99× |
| 4 | 1 556.59 B | 3.113 TFLOP | 1.00× | 2.48× |

In the compute-bound regime (large batch, short context) tokens per GPU-second scale with
1/FLOPs, so the profile delivers ≈ 1.66× the 4-pass throughput per GPU and ≈ 0.67× the
flagship's. Prefill FLOPs scale the same way, so time-to-first-token on long prompts is
≈ 0.6× the 4-pass figure.

### 3.2 Sequential depth (latency)

Per-token decode latency in the latency-bound regime (batch 1, small context) is
proportional to the number of layer evaluations in series: 192 vs 320 vs 128. The profile's
time-per-output-token is ≈ 0.6× the 4-pass tier's and ≈ 1.5× the flagship's.

### 3.3 KV cache (memory and bandwidth)

KV is indexed by *virtual* layer. Per token: 192 × 18 heads × 128 dims × 2 (K+V) × 2 B =
1 769 472 B = 1728 KiB bf16, 864 KiB fp8 (the default `kv_cache_dtype`).

| | 1 pass | **2 passes** | 4 passes |
|---|---|---|---|
| KV / token, fp8 | 576 KiB | **864 KiB** | 1440 KiB |
| KV at 200K (compaction trigger), fp8 | 118 GB | **177 GB** | 295 GB |
| KV at 1M, fp8 | 619 GB | **928 GB** | 1546 GB |

Two consequences. In the bandwidth-bound regime (long-context decode) every decode step reads
the whole KV, so the profile moves 0.6× the bytes of the 4-pass tier. And because the paged
cache is allocated at `loops_max`, the profile's cache is 0.6× the size for the same
sequence count, or 1.67× the sequences in the same memory.

### 3.4 Fleet size per replica

Memory floor = weights + KV at the stated context, divided by HBM and rounded up. No
activation or runtime headroom is included, so read this table for the *relative* movement,
not as a sizing recommendation; the headline strings in `implied_serving_minimums` are the
same in both files because they are short-context and weight-bound.

| platform, precision (weights / KV) | context | 1 pass | **2 passes** | 4 passes |
|---|---|---|---|---|
| B300 279 GB, fp8 / fp8 | 1M | 1246 GB → 5 | **1555 GB → 6** | 2173 GB → 8 |
| B200 192 GB, fp8 / fp8 | 1M | 1246 GB → 7 | **1555 GB → 9** | 2173 GB → 12 |
| Rubin R100 288 GB, NVFP4 / NVFP4 | 1M | 623 GB → 3 | **777 GB → 3** | 1087 GB → 4 |
| B300 279 GB, int4 / fp8 | 200K | 431 GB → 2 | **490 GB → 2** | 608 GB → 3 |
| B300 279 GB, fp8 / fp8 | 200K | 745 GB → 3 | **804 GB → 3** | 922 GB → 4 |
| Rubin R100 288 GB, NVFP4 / NVFP4 | 200K | 373 GB → 2 | **402 GB → 2** | 461 GB → 2 |

The profile recovers the flagship's GPU count at every row except 1M on B200. At 1M on B300
it needs 6 GPUs where the 4-pass tier needs 8; at the 200K compaction trigger with int4
weights it fits a two-GPU B300 replica, which the 4-pass tier does not.

### 3.5 Operational effects that the arithmetic hides

* **One prefill, always.** The engine prefills at a probe depth, reads the effort head, and
  re-prefills if the decided tier wants a different loop count (`inference/engine.py`).
  With a flat ladder every tier shares one depth, so the second prefill never happens.
* **Uniform KV shape across requests.** Every sequence in a batch has the same per-token KV
  size, so the paged cache packs without per-depth fragmentation and capacity planning is a
  single number instead of a tier-weighted average.
* **Predictable cost per token.** The base's cost depends on the tier mix the effort head
  produces; the profile's does not.

### 3.6 What it does not save

Weight memory (627 GB fp8, 314 GB NVFP4) is unchanged. The visible thinking-token budgets
are unchanged, so a `max`-tier request still emits up to 131 072 thinking tokens. The
profile reduces the per-token cost of those tokens by 0.6× versus the 4-pass tier; it does
not reduce their count.

## 4. What it costs in quality, and how to gate it

The profile removes one pass from the base's `high` tier and two from `max`. The 2025
evidence in `LOOPED_TRANSFORMER.md` §1 says reasoning quality tracks effective depth while
knowledge tracks stored params, so the expected direction is: knowledge-heavy gates
unchanged, reasoning-heavy gates at the hard tiers at risk. **None of that is measured
here.** The profile copies `capability_targets` and `safety_thresholds` unchanged, which
means it must clear the same Terminal-Bench 2.1 ≥ 84 % and every other floor with loops
fixed at 2, or it does not ship. Run `evaluation/release_gate.py` against the profile YAML;
the base's gate results do not transfer.

If the profile misses a gate, the operational answer is the one the public precedent
implies by "optimized for a more efficient serving profile": post-train at the serving
depth. The SFT, reward-model and PPO stages pin the loop count to `recurrent_loops_max`, so
pointing them at the profile YAML post-trains at exactly 2 passes with no code change. That
is a separate post-training run inside the existing 6–10 week band; it is not pretraining.
Invariant 3 (no chain-of-thought supervision) and Invariant 7 (welfare directive) apply to
it exactly as to the base.

## 5. How to use it

**Serve.** The server builds the model from `--config` and loads a raw state dict, and the
stored shapes are identical, so the looped checkpoint loads into the profile unchanged:

```bash
sota-serve --config configs/sota_ultra_5_looped_2pass.yaml --checkpoint <looped-state-dict>
```

**Post-train at 2 passes.** Pass the profile YAML as `--config` to the SFT, reward-model and
PPO entry points; each pins `recurrent_loops_max`, which is now 2.

**Validate.** `make check` covers the profile by default, or run it alone:

```bash
python scripts/validate_config.py configs/sota_ultra_5_looped_2pass.yaml
```

**Known gap (follow-up, not changed here).** `checkpoint.load_checkpoint(bundle)` rebuilds
the model from the bundle's frozen `config.yaml`, which for a looped bundle says
`recurrent_loops_max: 4`. A bundle loaded that way serves the base ladder regardless of the
profile. The server path above does not use the bundle loader, so serving works today; a
`config` override on `load_checkpoint` would let pipelines resume into a profile too.

## 6. What it is not

* **Not a new model.** No weights are created or changed; the gate rejects any `model:` drift
  beyond the two ladder fields.
* **Not a pretraining config.** The gate pins the compute anchor to the base run's. Do not
  pass this file to `sota-pretrain`; if you want a model *trained* for 2 passes, copy the
  looped base, set its `recurrent_loops_max` to 2 without `serving_profile_of`, and let the
  gate demand the 1.689×10²⁶ anchor that job actually costs.
* **Not a Rubin port.** The serving-minimum rows for Rubin are memory arithmetic at NVFP4;
  the code has no Transformer Engine path yet (see the Rubin notes in `PRECISION.md`).
* **Not adaptive.** Depth is fixed per profile. Learned per-token exit remains a follow-up in
  `LOOPED_TRANSFORMER.md` §7.

## 7. Verification performed

* `make check` passes for all three configs. On the profile the gate reports, among the
  ordinary checks: stored stack shared with the base, `loops_max` 2 within 1..4, corpus /
  schedule split / `training:` identical, `compute_params_billions_at_loops_max` 936.90 B at
  loops 2, `compute_params_billions_train_mean` 1091.83 B at the base's mean 2.5, 192 virtual
  layers, KV 1 769 472 B/token and 1728 GiB at 1M, FLOPs 2.358×10²⁶ = 6·N·D of the base run
  and equal to the base's anchor.
* The value-only diff between the base and the profile is exactly the six keys in §2.
* `ruff check src scripts` is clean.
