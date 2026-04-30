# Sample data

These files are tiny illustrative samples — not training data. They exist so that:
- `tests/` can read concrete fixtures without any external download.
- New contributors can see exactly what schema each pipeline stage expects.
- Sanity-runs of `BlockPacker`, `PretrainingPipeline`, and the chat-template renderer have something to chew on.

The real pretraining corpus is **15–20 trillion tokens** across web, code, academic, books, math, and dialogue (see [`README.md` 2.2](../README.md) and [`` 1.1.1](../)).

## Layout

```
data/samples/
    pretrain.jsonl        web + book + technical text snippets
    code.jsonl            code-domain pretraining samples (Python, Rust, SQL)
    multilingual.jsonl    GMMLU/MILU-style examples in 5 languages
    chat.jsonl            multi-turn assistant conversations
    tool_use.jsonl        tool-call traces (web_search, code_exec, file_read)
    contamination.jsonl   adversarial samples that MUST be filtered out
```

## Schema

### `pretrain.jsonl`, `code.jsonl`, `multilingual.jsonl`

```json
{
  "id": "string, stable across reruns",
  "url": "source URL or NULL for synthetic",
  "lang": "ISO 639-1 code",
  "text": "the actual training text",
  "tokens": "approximate token count (optional, for shard balancing)"
}
```

The pipeline filters by `url`, `lang`, and `text` content. Anything missing those fields is dropped.

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

`thinking` is optional; when present, training masks the loss on user/system turns and computes loss on `thinking` + `content`.

### `tool_use.jsonl`

```json
{
  "id": "string",
  "tools": [{"name": "...", "schema": {...}}, ...],
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

Each entry is something `BenchmarkContaminationFilter` must reject. Used as negative-test fixtures.

## How to extend

1. Match the schemas above exactly.
2. Re-run `pytest tests/test_data.py` — the contamination, packing, and filter tests read these files directly.
3. Do not commit anything that resembles a real eval question; the contamination filter exists for a reason.
