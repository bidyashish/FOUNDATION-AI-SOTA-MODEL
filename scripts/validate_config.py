#!/usr/bin/env python3
"""Config consistency gate — `make check` runs this on every config.

Codifies every "does it add up?" invariant in the YAML specs so drift is
caught at commit time instead of by hand:

  architecture   q_heads × head_dim = d_model, GQA ratio, YaRN covers the
                 full context window, params estimate matches implied_scale
  kv cache       per-token and @-1M bytes match the implied_scale fields
  training       batch identity (seq × micro × accum × dp = global batch),
                 total_steps ≈ corpus ÷ batch, schedule closes the corpus
                 at 70/20/10 with the implied_schedule_split LR ladder
  compute        training FLOPs point estimate = 6·N·D from the model
                 section and the corpus commitment
  parallelism    model shape shards at the configured TP/PP
  data           source mix sums to 100, pipeline keys are recognized
  gates          capability/safety thresholds are numeric (or *_band)
  surfaces       16 special tokens, multilingual tier counts

Exit 0 = all PASS; exit 1 = any FAIL. WARNs don't fail the gate.
Runs config-math checks with PyYAML alone; schedule/parallelism checks
additionally need the package importable (torch installed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sota_model.config import (  # noqa: E402
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
    load_implied,
)

KNOWN_PIPELINE_KEYS = {
    "min_doc_chars", "drop_duplicates", "quality_filter", "toxicity_filter",
    "pii_redactor", "contamination_filter", "restrict_to_coverage_languages",
    "shuffle_buffer_blocks",
}


class Gate:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, name: str, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"  {tag}  {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name: str, detail: str = "") -> None:
        print(f"  WARN  {name}" + (f" — {detail}" if detail else ""))


def validate(path: Path, gate: Gate) -> None:
    print(f"\n{path}")
    model = ModelConfig.from_yaml(path)
    implied = load_implied(path)

    # --- architecture ------------------------------------------------------
    gate.check(
        model.n_q_heads * model.head_dim == model.d_model,
        "q_heads × head_dim = d_model",
        f"{model.n_q_heads} × {model.head_dim} vs {model.d_model}",
    )
    gate.check(
        model.n_q_heads % model.n_kv_heads == 0,
        "GQA group ratio integral",
        f"{model.n_q_heads}/{model.n_kv_heads} = {model.n_q_heads / model.n_kv_heads:g}:1",
    )
    yarn_reach = model.rope_yarn_scale * model.rope_yarn_original_max_position
    gate.check(
        yarn_reach >= model.max_position_embeddings,
        "YaRN covers the context window",
        f"{model.rope_yarn_scale:g} × {model.rope_yarn_original_max_position} = "
        f"{yarn_reach:g} vs window {model.max_position_embeddings}",
    )

    scale = implied.get("implied_scale", {})
    params_b = model.estimate_params_billions()
    committed = scale.get("total_params_billions")
    if committed is not None:
        gate.check(
            abs(params_b - committed) / committed < 0.005,
            "params estimate matches implied_scale",
            f"{params_b:.2f}B vs committed {committed}B",
        )

    kv_bytes = model.kv_cache_bytes_per_token()  # bf16: 2 bytes/element
    kib_field = scale.get("kv_cache_kib_per_token_bf16")
    if kib_field is not None:
        gate.check(
            kv_bytes == kib_field * 1024,
            "KV bytes/token matches implied_scale",
            f"{kv_bytes:,} B vs {kib_field} KiB",
        )
    gib_field = scale.get("kv_cache_gib_at_1m_context_bf16")
    if gib_field is not None:
        gate.check(
            kv_bytes * 1_048_576 == gib_field * 1024**3,
            "KV @1M matches implied_scale",
            f"{kv_bytes * 1_048_576 / 1024**3:.0f} GiB vs {gib_field} GiB",
        )

    # --- training ----------------------------------------------------------
    try:
        train = TrainingConfig.from_yaml(path)
        gate.check(True, "batch identity",
                   f"{train.global_batch_tokens:,} tok = {train.seq_len} × "
                   f"{train.micro_batch_size} × {train.grad_accum} × {train.dp_degree} "
                   f"(world {train.world_size})")
    except ValueError as e:
        gate.check(False, "batch identity", str(e))
        train = None
    InferenceConfig.from_yaml(path)  # loads or raises

    corpus_t = implied.get("implied_training_corpus", {}).get("total_tokens_trillions")
    if train is not None and corpus_t:
        exact = corpus_t * 1e12 / train.global_batch_tokens
        gate.check(
            abs(train.total_steps - exact) / exact < 0.01,
            "total_steps ≈ corpus ÷ batch",
            f"{train.total_steps:,} vs {exact:,.0f} ({(train.total_steps - exact) / exact:+.2%})",
        )

    flops_committed = implied.get("implied_compute", {}).get("training_flops_point_estimate")
    if flops_committed is not None and corpus_t:
        # float() guards the PyYAML quirk: an unsigned exponent (1.354e26
        # instead of 1.354e+26) silently loads as a string.
        flops_committed = float(flops_committed)
        flops = model.training_flops(corpus_t * 1e12)
        gate.check(
            abs(flops - flops_committed) / flops_committed < 0.01,
            "training FLOPs = 6·N·D",
            f"{flops:.4g} vs committed {flops_committed:.4g}",
        )

    split = implied.get("implied_schedule_split", {})
    if split:
        pct_sum = split.get("foundation_pct", 0) + split.get("long_context_pct", 0) + split.get("refinement_pct", 0)
        gate.check(pct_sum == 100, "schedule split pcts sum to 100", str(pct_sum))
        if train is not None and split.get("foundation_lr") is not None:
            gate.check(split["foundation_lr"] == train.lr,
                       "foundation_lr == training.lr",
                       f"{split['foundation_lr']} vs {train.lr}")

    # Schedule + parallel-layout checks need the full package (torch).
    try:
        from sota_model.training.parallelism import validate_parallel_layout
        from sota_model.training.schedule import schedule_for_config
    except ImportError as e:
        gate.warn("schedule/parallelism checks skipped", f"import failed: {e}")
    else:
        if train is not None and corpus_t:
            sched = schedule_for_config(path)
            total = sum(s.cfg.total_steps * s.cfg.global_batch_tokens for s in sched)
            gate.check(
                abs(total / 1e12 - corpus_t) / corpus_t < 0.01,
                "schedule closes the corpus",
                f"{total / 1e12:.2f}T vs {corpus_t}T across "
                + " / ".join(f"{s.name} {s.cfg.total_steps:,}" for s in sched),
            )
            ladder_ok = (
                not split
                or (sched[1].cfg.lr == split.get("long_context_lr", sched[1].cfg.lr)
                    and sched[2].cfg.lr == split.get("refinement_lr", sched[2].cfg.lr))
            )
            gate.check(ladder_ok, "stage LRs match implied_schedule_split",
                       " / ".join(f"{s.cfg.lr:.1e}" for s in sched))
        if train is not None:
            try:
                for w in validate_parallel_layout(model, train):
                    gate.warn("layout", w)
                gate.check(True, "model shards at TP/PP",
                           f"tp{train.tp_degree} × pp{train.pp_degree} × dp{train.dp_degree}")
            except ValueError as e:
                gate.check(False, "model shards at TP/PP", str(e))

    # --- data --------------------------------------------------------------
    corpus = implied.get("implied_training_corpus", {})
    mix = corpus.get("source_mix_pct", {})
    if mix:
        gate.check(sum(mix.values()) == 100, "source mix sums to 100",
                   f"{sum(mix.values())} across {len(mix)} sources")
    pipe = corpus.get("pipeline") or {}
    unknown = set(pipe) - KNOWN_PIPELINE_KEYS
    if unknown:
        gate.check(False, "pipeline keys recognized", f"unknown: {sorted(unknown)}")
    elif pipe:
        gate.check(True, "pipeline keys recognized", f"{len(pipe)} keys")

    # --- gates -------------------------------------------------------------
    for section in ("capability_targets", "safety_thresholds"):
        entries = implied.get(section, {})
        bad = [
            k for k, v in entries.items()
            if not isinstance(v, (int, float)) and "_band" not in k
        ]
        gate.check(not bad, f"{section} numeric ({len(entries)} gates)",
                   f"non-numeric: {bad}" if bad else "")

    # --- surfaces ----------------------------------------------------------
    tokens = implied.get("implied_special_tokens_required", [])
    if tokens:
        gate.check(len(tokens) == 16, "16 special tokens", str(len(tokens)))
    coverage = implied.get("implied_multilingual_coverage", {})
    if coverage:
        n = (1 if coverage.get("english_baseline") else 0) + sum(
            len(coverage.get(t) or []) for t in ("high_resource", "mid_resource", "low_resource")
        )
        gate.check(n == 42, "42-language coverage set", f"counted {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "configs", nargs="*", type=Path,
        default=[Path("configs/sota_ultra_5.yaml"), Path("configs/sota_4_7.yaml")],
    )
    args = parser.parse_args()

    gate = Gate()
    for path in args.configs:
        validate(path, gate)

    print(f"\n{'OK — all checks passed' if gate.failures == 0 else f'{gate.failures} FAILURE(S)'}")
    sys.exit(1 if gate.failures else 0)


if __name__ == "__main__":
    main()
