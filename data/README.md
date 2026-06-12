# Sample data

These files are tiny illustrative samples — not training data. They exist so that:
- Tests and the config gate can read concrete fixtures without any external download.
- New contributors can see exactly what schema each pipeline stage expects.
- The real `CorpusLoader` path is exercisable end-to-end against the shipped tree (see below).

The real pretraining corpus is **36 trillion tokens** at the current spec
(`configs/sota_ultra_5.yaml::implied_training_corpus`; the 2026 frontier-dense band is 20–40T —
see [`README.md` 2.2](../README.md)).

## Layout

```
data/samples/
    corpus/                 ◄── mirrors implied_training_corpus.source_mix_pct —
        web/                    one subdir per source key, loadable by the REAL
        code/                   corpus path (resolve_sources_from_yaml)
        math_structured/
        synthetic_reasoning/
        academic/
        books_reference/
        dialogue_instructions/
    pretrain.jsonl          web + book + technical text snippets (legacy flat)
    code.jsonl              code-domain samples (Python, Rust, SQL)
    multilingual.jsonl      GMMLU/MILU-style examples in 13 languages
    chat.jsonl              multi-turn assistant conversations
    tool_use.jsonl          tool-call traces (web_search, code_exec, file_read, computer use)
    contamination.jsonl     adversarial samples that MUST be filtered out
```

The `corpus/` tree is keyed to the **UltraModel 5 source mix** (web 35 / code 22 /
math_structured 13 / synthetic_reasoning 10 / academic 8 / books_reference 7 /
dialogue_instructions 5 — per-source token budgets in the YAML). Every doc is ≥200 chars so it
survives the configured `pipeline.min_doc_chars` filter.

### Smoke-running the real corpus path

```python
from sota_model.tokenizer import make_byte_fallback
from sota_model.training.corpus import (
    CorpusLoader, loader_config_from_yaml, resolve_sources_from_yaml,
)

sources = resolve_sources_from_yaml("configs/sota_ultra_5.yaml", "data/samples/corpus")
loader = CorpusLoader(
    sources, make_byte_fallback(),
    loader_config_from_yaml("configs/sota_ultra_5.yaml", seq_len=512),
)
batch = next(loader.batches(batch_size=1))   # {"input_ids": LongTensor(1, 512)}
```

`training/sample_loader.py` provides the no-dependency loaders: `load_corpus_samples()` returns
the tree as `{source: [docs]}`; `load_pretrain_samples()` returns flat files + tree combined.

## Schema

### Pretraining docs (`corpus/**.jsonl`, `pretrain.jsonl`, `code.jsonl`, `multilingual.jsonl`)

```json
{
  "id": "string, stable across reruns",
  "url": "source URL or null for synthetic",
  "lang": "ISO 639-1 code",
  "text": "the actual training text",
  "tokens": "approximate token count (optional, for shard balancing)"
}
```

The pipeline filters by `url`, `lang`, and `text` content. Anything missing those fields is
dropped. `synthetic_reasoning` docs are plain-text reasoning traces (problem → steps → answer)
with `url: null` — they are pretraining text, **not** chat-formatted.

### `chat.jsonl`

```json
{
  "id": "string",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "thinking": "...", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

`thinking` is optional; when present, training masks the loss on user/system turns and computes
loss on `thinking` + `content`.

### `tool_use.jsonl`

```json
{
  "id": "string",
  "tools": [{"name": "...", "schema": {}}],
  "messages": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "thinking": "...",
      "tool_calls": [{"name": "web_search", "arguments": {"q": "..."}}]
    },
    {"role": "tool", "name": "web_search", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### `contamination.jsonl`

Each entry is something `BenchmarkContaminationFilter` must reject (blocklisted URLs, verbatim
eval text). Used as negative-test fixtures.

## How to extend

1. Match the schemas above exactly; keep every `text` ≥200 chars (the ultra pipeline's
   `min_doc_chars`) or the doc will silently vanish from corpus smoke runs.
2. New corpus sources go in `corpus/<source>/` where `<source>` is a `source_mix_pct` key —
   the loader ignores directories that don't match the active config's mix.
3. Run `make check` (and the smoke snippet above if you touched `corpus/`).
4. Do not commit anything that resembles a real eval question; the contamination filter exists
   for a reason.
