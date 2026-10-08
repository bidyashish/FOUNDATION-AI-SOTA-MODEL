# `sota_model/safety/probes/`

Probe-based classifier framework that replaces the keyword-stub safety gate. Every category referenced in modelcard 3.1 (prohibited / high-risk-dual / dual-use) gets a calibrated `LinearProbeClassifier` here; CBRN gets a `ConstitutionalClassifier`.

```
__init__.py            re-exports
feature_extractor.py   HashingFeatureExtractor, HiddenStateFeatureExtractor
linear_probe.py        LinearProbeClassifier, train_linear_probe, write_probe_bundle, load_probe_bundle
constitutional.py      ConstitutionalClassifier, KeywordConstitutionalClassifier, CBRN_PRINCIPLES
registry.py            build_default_probe_gate, build_keyword_fallback_gate
```

## Code

- **`HashingFeatureExtractor`** — character n-gram hashing trick (no learned vocabulary, stable across releases). Default `n_features=2**14`, ngram range `(2, 5)`. Sub-millisecond per call. The pre-model gate uses this.
- **`HiddenStateFeatureExtractor`** — pulls a pooled hidden state from a chosen layer (default: second-to-last) of the SOTA model. Used post-model where the forward pass is already paid for. Works at `mean` or `last` pooling.
- **`LinearProbeClassifier`** — logistic head on top of either extractor. `block_threshold` and `flag_threshold` are sigmoid-probability thresholds operators pick from the precision-recall curve at deployment.
- **`train_linear_probe(extractor, pos, neg, ...)`** — Adam-trained binary head. Defaults are tuned for the modelcard 4.1 single-turn rates (≥97.9% violative-harmless, ≤0.5% benign over-refusal).
- **`ConstitutionalClassifier`** — model-graded critique against `CBRN_PRINCIPLES`. The critique callable is operator-supplied (a fast forward pass through the SOTA model with a structured-output prompt). On any critique exception, it **fails closed** to `KeywordConstitutionalClassifier`.

## Data

- Training pairs: `(positive_examples, negative_examples)` per category. Real bundles run on hundreds of thousands of labeled examples; the registry seed bundles in `registry.py::_train_seed_bundle` are operator-replaceable starters of ~10 examples per side, calibrated thresholds intentionally relaxed.
- Bundle format on disk:
  ```
  <bundle_dir>/
      manifest.json              [{name, category, action, block_threshold, flag_threshold, path}, ...]
      prohibited-cyber.pt        torch.save bundle: {weight, bias, name, category, ...}
      high-risk-dual-cyber.pt
      dual-use-cyber.pt
      ...
  ```
- The constitutional classifier consumes plain text and emits a `ClassifierVerdict` (the `safety.classifiers` shared shape).

## Working

Pipeline at request time:

```
incoming chat prompt
  ↓ rendered via ChatTemplate.render
  ↓ SafetyGate.evaluate(text)
  ↓ for each probe in the gate: probe(text) → ClassifierVerdict
  ↓ strictest verdict wins (BLOCK > FLAG > ALLOW)
  ↓ if BLOCK → 400 policy_violation; metric BLOCKS.labels(category=...).inc()
  ↓ if FLAG → record + continue
  ↓ if ALLOW → proceed to inference engine
```

`build_default_probe_gate(bundle_dir=…, cbrn_critique_fn=…)` is what the serving layer should call. If `bundle_dir` is missing or unreadable, the registry synthesizes a weak seed bundle so the gate never silently disappears (modelcard 3.1: probes are bootstrapped from policy text, then refined on labeled data).

### Operator checklist

- [ ] Train production probes on the audited labeled set, not the seed.
- [ ] Calibrate `block_threshold` / `flag_threshold` on a held-out PR curve. The 97.9%/0.5% targets in modelcard 4.1 are gates, not aspirations.
- [ ] Persist with `write_probe_bundle(bundle_dir, probes, "hashing-2-5")` and load via `build_default_probe_gate(bundle_dir)`.
- [ ] Plug a real CBRN critique into `cbrn_critique_fn`. The keyword backup is the floor, not the ceiling.
- [ ] Mirror the gate post-model — same probes, but with `HiddenStateFeatureExtractor` over the assistant's response so the post-model classifier sees what the model actually said.
