# `sota_model/training/classifiers/`

Trainable replacements for the heuristic stubs in `training/data.py`. Every classifier here implements the `(doc) -> doc | None` filter contract so it drops directly into `PretrainingPipeline.filters`.

```
__init__.py    re-exports
base.py        HashingTextVectorizer + LogisticTextClassifier + train_logistic
quality.py     HeuristicQualityScorer (legacy) + TrainedQualityScorer
toxicity.py    BlocklistToxicityFilter + TrainedToxicityFilter
language.py    CharNgramLanguageDetector + TrainedLanguageDetector
```

## Code

- **`HashingTextVectorizer`** — character n-gram + word n-gram hashing trick into a fixed-size feature space. No learned vocabulary, no drift across retrains. Default `2^16` features, char `(2,5)` + word `(1,2)`.
- **`LogisticTextClassifier`** — multiclass softmax head. `predict(text)` returns `(label, prob_max, all_probs)`; `predict_proba(text, label)` returns the calibrated probability for one class.
- **`train_logistic(texts, labels, ...)`** — Adam-trained reference loop. Pure PyTorch; no scikit-learn dependency.
- **`HeuristicQualityScorer`** — punctuation-density + line-layout heuristic. Same scoring as the original stub, factored out so `data.py::QualityScorer` can fall back to it when no trained backend is provided.
- **`TrainedQualityScorer`** — logistic over `(high_quality, low_quality)`. Threshold on `predict_proba(text, "high_quality")`. Default 0.7.
- **`BlocklistToxicityFilter`** — exact-match regex blocklist. `from_default_seed()` mirrors the legacy stub's single CSAM-adjacent pattern; `from_files([...])` loads operator-rotated audited blocklists out-of-band.
- **`TrainedToxicityFilter`** — logistic over `(toxic, safe)`. Block if `predict_proba(text, "toxic") >= block_threshold`.
- **`CharNgramLanguageDetector`** — script-family detector via Unicode names (LATIN / CYRILLIC / ARABIC / DEVANAGARI / …). Coarse first cut; passes the candidate set forward to a trained detector.
- **`TrainedLanguageDetector`** — multiclass logistic over the modelcard 8.12 44-language set.

## Data

- Quality: pair lists of `(high_quality_texts, low_quality_texts)`. Production runs label ~5–50K examples per side; the in-line tests use ~10 per side as a smoke baseline.
- Toxicity: pair lists of `(toxic_texts, safe_texts)`, plus operator-supplied JSON blocklist files for the deterministic pattern path.
- Language: list of `(text, lang_code)` pairs; the classifier auto-discovers the label set, but `accepted=MODELCARD_LANGUAGES` locks the output set to the 44 evaluated languages.
- All three persist as `LogisticTextClassifier.save(path)` → single torch file containing weights + label set + vectorizer config; reload with the matching `.load(path)` classmethod.

## Working

Pipeline at training time:

```
raw doc {"text", "url", "lang", ...}
  ↓ LanguageDetector(accepted=...) — drops out-of-set, optionally tags candidates
  ↓ MinLengthFilter(min_chars=100)
  ↓ DuplicateRemover (sha256 short-fingerprint)
  ↓ QualityScorer(backend=TrainedQualityScorer.load(path)) — drops below threshold
  ↓ ToxicityFilter(backend=BlocklistToxicityFilter.from_files([...]))
  ↓ PIIRedactor — emails / SSNs / phones → tokens
  ↓ BenchmarkContaminationFilter — modelcard 9.2 URL + eval text blocklist
  ↓ tokenize + BlockPacker → training shards
```

The legacy `QualityScorer` and `ToxicityFilter` in `training/data.py` are now shells over a swappable `backend`. Production wiring:

```python
from sota_model.training.classifiers import TrainedQualityScorer, TrainedToxicityFilter
from sota_model.training.data import PretrainingPipeline, QualityScorer, ToxicityFilter

pipeline = PretrainingPipeline(
    tokenizer=load_tokenizer("./tokenizer"),
    filters=[
        # ...
        QualityScorer(backend=TrainedQualityScorer.load("classifiers/quality.pt")),
        ToxicityFilter(backend=TrainedToxicityFilter.load("classifiers/toxicity.pt")),
        # ...
    ],
)
```

### Operator checklist

- [ ] Label ≥ 5K examples per class for each scorer before going past CI smoke.
- [ ] Re-train monthly as the corpus grows; persist version-tagged artifacts.
- [ ] Audit toxicity blocklists out-of-band and rotate via `from_files([...])` rather than editing source.
- [ ] Confirm `TrainedLanguageDetector` covers all 44 languages from `MODELCARD_LANGUAGES` so 8.12 evals don't silently drop low-resource tiers.
