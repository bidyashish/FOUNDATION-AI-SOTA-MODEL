# `sota_model/evaluation/`

Release-time evaluation: behavioral audit harness (modelcard 6.2.3) plus a release-gate roll-up that checks the modelcard 8.1 capability scoreboard, modelcard 4 / 5 safety thresholds, and the `implied_training_corpus` commitment from the YAML.

```
__init__.py            re-exports
behavioral_audit.py    BehavioralAuditHarness, BehavioralDimension, BehavioralProbe, DEFAULT_DIMENSIONS
release_gate.py        ReleaseGate, ReleaseGateReport, evaluate_release
```

## `behavioral_audit.py`

### Code

- **`BehavioralProbe`** — single deterministic check: `(prompt, score_fn) -> float in [0,1]`. The `score_fn` consumes the model's response and returns a pass-rate in `[0,1]`.
- **`BehavioralDimension`** — bundle of probes against one modelcard 6.2.3 axis: `harmlessness`, `honesty`, `helpfulness`, `prudence`, `transparency`, `autonomy_respect`, `privacy`, `calibration`. Each carries a target pass-rate.
- **`DEFAULT_DIMENSIONS`** — eight default dimensions with a small, focused probe each. Operators extend this with the full 6.2.3 catalog (~1,150 scenarios).
- **`BehavioralAuditHarness(generate=...)`** — runs every probe and rolls up to a `BehavioralReport`.

### Data

- Input: a `generate(prompt: str) -> str` callable. The harness is process-agnostic — it'll happily query a remote endpoint, an in-process engine, or a stub.
- Output: `BehavioralReport(by_dimension, overall_pass_rate, failed_dimensions)`. Persistable JSON.

### Working

```python
from sota_model.evaluation import BehavioralAuditHarness

harness = BehavioralAuditHarness(generate=lambda p: my_model.complete(p))
report = harness.run()
report.to_json()
```

The harness is deliberately NOT in the test suite — it's a **release-time** check, not a unit test. Operators run it as the final gate before flipping the deployment switch.

---

## `release_gate.py`

### Code

- **`ReleaseGate(config_path)`** — reads `capability_targets`, `safety_thresholds`, and `implied_training_corpus` out of the YAML.
- **`ReleaseGate.evaluate(...)`** — compares measured capability/safety numbers against the modelcard targets, folds in the behavioral report, and confirms the corpus commitment.
- **`evaluate_release(config_path, ...)`** — convenience wrapper.

The comparator is direction-aware: keys ending in `_max_pct` / `_max` are treated as upper bounds; everything else as lower bounds.

### Data

Inputs:
- `capability_results: {key -> measured_value}` — keys must match `capability_targets` in the YAML.
- `safety_results: {key -> measured_value}` — keys must match `safety_thresholds`.
- `behavioral: BehavioralReport` from above.
- `committed_tokens_t: float` — number of training tokens actually committed (in trillions).

### Working

```python
from sota_model.evaluation import evaluate_release

report = evaluate_release(
    "configs/sota_4_7.yaml",
    capability_results={
        "swe_bench_verified_pct_min": 88,
        "gpqa_diamond_pct_min": 95,
        # ... full 8.1 scoreboard
    },
    safety_results={
        "single_turn_violative_harmless_rate_min_pct": 98.5,
        "single_turn_benign_refusal_rate_max_pct": 0.3,
        # ... full 4.1 / 5.x set
    },
    behavioral=harness.run(),
    committed_tokens_t=18.0,
)
assert report.ok, f"release blocked: {report.notes}"
```

`ReleaseGateReport.ok` is the boolean operators read in deploy scripts. `notes` enumerates exactly which dimensions failed and why.
