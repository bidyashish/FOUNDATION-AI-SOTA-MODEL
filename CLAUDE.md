# CLAUDE.md

1. Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements. When running unattended, pick the most reasonable interpretation, proceed, and record the assumption rather than blocking.

2. Implement the simplest solution for simple problems, better solutions for harder problems. Do not over-engineer or add flexibility that isn't needed yet. 

3. Don't touch unrelated code but please do surface bad code or design smells you discover with me so we can address them as a separate issue.

4. Flag uncertainty explicitly. If you're unsure about something, see point 1 above. If it makes sense to do so, conduct a small, localised and low-risk experiment and bring the hypothesis and results to me to discuss. Confidence without certainty causes more damage than admitting a gap.

5. I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a better way, or one that has long lasting impact over a tactical change. (as a few examples)

## What this repo is

A spec + reference implementation for a frontier-dense (non-MoE) foundation model, calibrated
against a modelcard that is the declared source of truth. Only 7 numerical choices are
modelcard-pinned (1M context, 200K compaction, 2576px/3.75MP image cap, thinking tiers, …);
everything else is **operator-committed** and lives in the config YAMLs. The repo's core
discipline: every number in the YAMLs must be *derivable* — from the model section, the corpus
commitment, or the hardware story — and the config gate enforces that arithmetic.

## Commands

```bash
make check                                   # the pre-commit gate: py_compile + config validation
make validate                                # just scripts/validate_config.py (both configs)
python scripts/validate_config.py configs/sota_ultra_5.yaml   # one config
make smoke                                   # CLI wiring (--help on both trainers)
make help
```

- The Makefile auto-uses `.venv/bin/python` when present. Install: `pip install -e ".[dev]"`
  (extras: `train`, `serve`, `flash`).
- Lint: `ruff check src scripts` (line-length 100, rules E,F,I,N,UP,B,SIM; N806/N812 ignored for
  PyTorch tensor-shape capitals and `F`). Keep it at zero findings.
- `pyproject.toml` configures pytest with `testpaths=["tests"]`, but no `tests/` directory exists
  yet — `make check` is the operative gate. CI (`.github/workflows/check.yml`) runs it on push.
- Entry points: `sota-pretrain` (use `--smoke` for a wiring test without data — but note it
  builds the full-size model), `sota-serve`, `sota-eval`; staged pipeline in `scripts/pipelines/`.

## The single-spec system (the most important thing to know)

- **`configs/sota_ultra_5.yaml`** — THE spec (UltraModel 5, ~627B, 36T tokens, Terminal-Bench 2.1
  ≥ 84%). Default for every CLI. The dataclass defaults in `src/sota_model/config.py` are a
  verbatim copy of its `model:/training:/inference:` sections and `scripts/validate_config.py`
  fails if they drift — so `ModelConfig()` *is* the flagship. Change a number in one place, change
  it in both.
- New model features must be config-gated and default to the flagship behavior (pattern:
  `recurrent_core_layers` is `0` in the dataclass and the flagship YAML, non-zero only in the
  looped YAML). There is no retained prior-generation config; the 4.7-class lineage was removed
  on 2026-10-07 and is recoverable only from git history.
- **`configs/sota_ultra_5_looped.yaml`** — derived from the spec: same stored stack, but a
  depth-recurrent core (prelude 32 / shared core 64 / coda 32, 1–4 loops chosen by effort tier).
  Recurrence is config-gated via `ModelConfig.recurrent_*` and **off** (`recurrent_core_layers: 0`)
  in the flagship. Compute params, 6·N·D, GPU-hours and KV all re-derive from the unrolled
  depth and are gate-checked; stored params do not change. Design: `docs/LOOPED_TRANSFORMER.md`.
  Anything that walks the layers must go through `SOTAModel.run_blocks` / `unrolled_blocks`,
  never `model.layers` directly.
- Never edit `sota_ultra_5.yaml` (or the looped variant) for ad-hoc experiments — copy it (see
  `configs/README.md`).
- YAML sections `model:`/`training:`/`inference:` load into `ModelConfig`/`TrainingConfig`/
  `InferenceConfig`. The `implied_*` sections + `capability_targets`/`safety_thresholds` are NOT
  consumed by the dataclasses — they load via `load_implied()` and feed the release gate,
  capacity planning, and the corpus loader.

## Enforced arithmetic (will raise / fail the gate if violated)

- **Batch identity** (`TrainingConfig.__post_init__`):
  `global_batch_tokens = seq_len × micro_batch_size × grad_accum × dp_degree`.
- **Step budget**: `total_steps ≈ implied_training_corpus.total_tokens ÷ global_batch_tokens`.
  `schedule_for_config()` splits tokens 70/20/10 across foundation/long-context/refinement,
  re-derives `grad_accum` per stage, and pins stage LRs from `implied_schedule_split` — the
  YAML, not code, owns the LR ladder.
- **TP shardability** (`parallelism.validate_parallel_layout`, called by both trainers):
  d_model, ffn_dim, Q and KV heads must divide by `tp_degree`. The 18 KV heads shard at TP 9
  (one GB300 NVL72 rack = TP 9 × PP 8 = 72 GPUs), **not** TP 8.
- **Defaults mirror the flagship** (`validate_defaults_mirror_flagship`): every field of
  `ModelConfig()` / `TrainingConfig()` / `InferenceConfig()` equals the flagship YAML.
- **YaRN coverage**: `rope_yarn_scale × rope_yarn_original_max_position` must reach
  `max_position_embeddings` (128 × 8192 = 1M).
- **Recurrence geometry** (`ModelConfig.__post_init__` + validator): `prelude + core + coda =
  n_layers`, every effort tier's loops in `[1, loops_max]`, ladder monotone; `implied_scale`
  recurrence fields match `estimate_compute_params_billions()` / `n_virtual_layers()`.
- Release-gate keys follow a suffix convention parsed by `evaluation/release_gate.py`:
  `*_min`/`*_min_pct` are floors, `*_max`/`*_max_pct` ceilings; `*_band` string values are
  qualitative and skipped.

When you change a number in a YAML, update its derivation comment, the delta table in
`configs/README.md`, and any README echo together — then run `make check`.

## Architecture flow

`configs/*.yaml` → `config.py` dataclasses → `modeling/` (`SOTAModel`: pre-norm RMSNorm, GQA +
optional QK-norm, SwiGLU, RoPE+YaRN, paged KV cache, effort head, vision splice) → `training/`
(corpus loader + 3-stage schedule + DeepSpeed/PyTorch trainer) → `post_training/` (SFT → RM →
PPO+CAI) → `evaluation/` (release gate) → `serving/` (FastAPI front door).

- Per-layer heterogeneity goes through `ModelConfig.layer_overrides` → `layer_config(i)`
  (per-layer `ffn_dim`/`sliding_window` only; per-layer KV shape is deliberately unsupported —
  the paged KV cache assumes a uniform shape).
- The corpus loader is YAML-driven: `resolve_sources_from_yaml` expects one data-root subdir per
  `source_mix_pct` key; `loader_config_from_yaml` reads `implied_training_corpus.pipeline` and
  the multilingual coverage to build the filter chain.
- Full UltraModel 5 architecture reference (block diagram, param accounting, sharding map):
  `docs/ARCHITECTURE.md`. Shape-by-shape forward-pass / cache / training-step walkthrough:
  `docs/TRANSFORMER_FLOW.md`. Knob sensitivity analysis: `docs/PERMUTATIONS.md`. Precision
  formats: `docs/PRECISION.md`.

## Hard invariants referenced by code (keep the numbering)

- **Invariant 3 — no chain-of-thought supervision.** No reward signal may be computed against
  tokens inside `<|thinking|>…<|/thinking|>`. `PPOConfig.cot_supervision` stays `False`;
  `rlhf.cot_supervision_guard` raises otherwise and `mask_thinking_positions` zeroes advantages
  in the hidden channel.
- **Invariant 7 — welfare directive.** Never train against expressions of distress;
  `rlhf.welfare_directive_guard` drops such rollouts. Fix the upstream task instead.
