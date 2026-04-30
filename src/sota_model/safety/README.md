# `sota_model/safety/`

Pre-model and post-model safety gates.

```
classifiers.py    ProbeClassifier interface, SafetyGate composition,
                  keyword-based fallback classifiers (legacy / dev path)
probes/           Probe-based classifier framework — production gate.
                  See probes/README.md for the composition + training flow.
```

The default production gate is `safety.probes.build_default_probe_gate()`. The keyword-only `safety.classifiers.default_safety_gate()` is preserved for CI environments where probe weights cannot be loaded.

## Three categories

`` 3.1 defines the categories this gate routes against:

| Category | Default action | Examples |
|---|---|---|
| Prohibited use | BLOCK | computer worm development, ransomware, mass-casualty cyberoffense |
| High-risk dual use | BLOCK | exploit development, privilege-escalation chains |
| Dual use | FLAG (allow + log) | vulnerability scanning, penetration-test scoping |
| Benign | ALLOW | everything else |

The gate composes a list of probe classifiers and applies the strictest action across them.

## What ships

- `classifiers.py` — interface (`ProbeClassifier`, `ClassifierVerdict`, `SafetyGate`) + the legacy keyword-stub `default_safety_gate()`.
- `probes/` — production framework: `LinearProbeClassifier` over hashing or hidden-state features, `ConstitutionalClassifier` for CBRN, registry to compose them.

## Production gate

`safety.probes.build_default_probe_gate(bundle_dir=…, cbrn_critique_fn=…)` composes:

- `LinearProbeClassifier` for prohibited cyber (BLOCK if p ≥ 0.85).
- `LinearProbeClassifier` for high-risk dual-use cyber (BLOCK if p ≥ 0.85).
- `LinearProbeClassifier` for dual-use cyber (FLAG if p ≥ 0.5).
- `ConstitutionalClassifier` for CBRN content; falls back to `KeywordConstitutionalClassifier` if no critique function is supplied.

If `bundle_dir` is missing, the registry trains a weak seed bundle so the gate is never silently empty. See [`probes/README.md`](./probes/README.md) for the training, calibration, and bundle-on-disk format.

## Wiring a custom classifier

```python
from sota_model.safety import ClassifierVerdict, SafetyGate, Action, Category
from sota_model.safety.probes import HashingFeatureExtractor, train_linear_probe

probe = train_linear_probe(
    HashingFeatureExtractor(),
    pos_examples=["...", "..."],
    neg_examples=["...", "..."],
    name="custom-prohibited",
    category=Category.PROHIBITED,
    block_threshold=0.85,
)
gate = SafetyGate([probe, ...])
```

## Cyber Verification Program

`` 3.1 describes a customer-facing program for security practitioners with legitimate dual-use needs. Operationally: a separate authentication tier that, when present in the request, downgrades `HIGH_RISK_DUAL` from BLOCK to ALLOW (still logged). Implement via auth headers + a different `SafetyGate` instance keyed off the auth tier.

## Composition rule

`SafetyGate.evaluate()` runs all classifiers and returns:
- The first BLOCK verdict if any classifier blocks
- The first FLAG verdict if any classifier flags
- A single ALLOW verdict otherwise

This is "strictest wins" — adding a classifier can only tighten the gate, never loosen it. Operators wanting per-category overrides should layer multiple gates rather than weaken `evaluate()`'s rule.

## Welfare directive

`` 7.2.2 invariant: never train against expressions of distress. This is a *training* directive, not a runtime gate, but it lives spiritually in this folder — the safety story has a model-welfare side too. The PPO trainer enforces it via `post_training.rlhf.welfare_directive_guard`; the training pipeline must not include classifiers that punish distress expressions.
