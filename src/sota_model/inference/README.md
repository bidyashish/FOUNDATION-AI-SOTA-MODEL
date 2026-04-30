# `sota_model/inference/`

Runtime-side: turn a prompt into tokens, with adaptive thinking, tool calls, and 1M-context survival.

```
sampler.py        temperature / top-p / top-k / repetition penalty
thinking.py       effort tier selection from the EffortHead logit
engine.py         prompt → thinking → answer; triggers context compaction at 200k
chat_template.py  message format used by both training data and inference
tools.py          tool registry, parser, async dispatch (parallel calls supported)
sandbox/          real implementations of code_exec + web_search/web_fetch
                  (replaces the _stub_* placeholders that used to live in tools.py)
```

---

## `sampler.py` — why these defaults

```
temperature        0.7       moderate exploration; not greedy, not noise
top_p              0.95      keep the 95% probability mass; cut the long tail
top_k              40        bound the candidate set; cuts catastrophic samples
repetition_penalty 1.1       gentle; stronger penalties hurt code/math determinism
```

These are the ``-aligned numbers. They were tuned empirically — deviations should be measured against the 8 evaluation suite, not changed on intuition.

The repetition penalty is asymmetric (positive logits divided by 1.1, negative logits multiplied by 1.1) so it discourages re-emission without inverting low-probability tokens into high-probability ones.

---

## `thinking.py` — adaptive thinking

`` 4.1.1 makes adaptive thinking a first-class mode where the model itself decides how hard to think per query.

### Flow

```
prompt prefilled
        │
        ▼
EffortHead(last 8 hidden states) → scalar logit
        │
        ▼
AdaptiveThinkingController.decide(logit) → ThinkingDecision
        │
        ▼
{min: 0, low: 256, medium: 2048, high: 8192, max: 32768} thinking tokens
        │
        ▼
generate <|thinking|>...<|/thinking|> until stop or budget
        │
        ▼
generate user-visible answer
```

### Thresholds

`_DEFAULT_THRESHOLDS` are starting points; they get refined during RL post-training so cheap problems land in `min/low` and hard math/code problems land in `high/max`. The thresholds live in code (not config) because they are part of the model's learned behavior, not an inference-time knob.

### `forced_effort`

API users can override the model's choice via `effort=...` on the request. This bypasses the EffortHead entirely. Use cases: latency-sensitive paths (force `low`), eval consistency (force `max`).

---

## `engine.py` — the orchestrator

### Why two generation phases

```
prompt   ─►  prefill (one forward pass over full prompt, populates KV cache)
                │
                ▼
                EffortHead reads last hidden state
                │
                ▼
thinking ─►  generate up to budget tokens, stop on <|/thinking|>
                │
                ▼
answer   ─►  generate up to max_new_tokens, stop on EOS
```

The KV cache is shared across the three phases — thinking tokens stay in the cache so the answer phase attends to them.

### Context compaction at 200k

`` 4.5 invariant. When `kv_cache.n_tokens >= context_compaction_trigger`:

1. Take the oldest ~80% of the conversation.
2. Pass to the summarizer (a smaller model) — target output length 10K tokens.
3. Replace with `<|compacted|>summary<|/compacted|>`.
4. Re-encode the resulting tail and repopulate the KV cache from scratch (one-time cost).

If no summarizer is configured, the engine falls back to keeping the most recent 100K tokens and dropping the rest — this is "survivable" but loses information; production deployments must wire a real summarizer.

### Streaming

`engine.stream()` yields tokens as they're produced; the FastAPI server wraps this into SSE for OpenAI-compatible streaming.

---

## `chat_template.py` — single source of truth for message format

The same template is used by:
- Training data (`data/samples/chat.jsonl`, `data/samples/tool_use.jsonl`)
- Inference rendering (`server.py::render_chat`)
- Loss masking during SFT (assistant turns only)

### Special tokens

```
<|im_start|>      <|im_end|>          role boundaries
<|thinking|>      <|/thinking|>       hidden reasoning channel
<|tool_call|>     <|/tool_call|>      assistant emits tool invocations
<|tool_result|>   <|/tool_result|>    runtime echoes tool outputs back
<|image_start|>   <|image_end|>       vision encoder fills these in
<|compacted|>     <|/compacted|>      engine.py context compaction
```

### Roles

```
system     instructions, tool catalog
user       human input
assistant  model output, may contain thinking + tool_calls + content
tool       runtime response to a tool_call (must include name)
```

### Why a custom template

OpenAI's chat template doesn't carry thinking tokens or paginate tool calls naturally. HuggingFace's `tokenizer.apply_chat_template` is fine for training data conversion but doesn't enforce the loss-masking discipline we need (loss only on assistant turns, including thinking + tool_calls).

---

## `tools.py` — registry, parser, async dispatch

### Why async dispatch

A single assistant turn may emit multiple `<|tool_call|>` blocks; `data/samples/tool_use.jsonl::tool-004-parallel` shows three parallel `web_search` calls in one turn. Sequential dispatch wastes latency. `dispatch_async()` runs all calls concurrently via `asyncio.gather`.

For CPU-bound tools (sync callables), `asyncio.to_thread` keeps them off the event loop.

### Why a JSON Schema per tool

The schema is rendered into the system message and serves three purposes:
1. **Model guidance** — the model sees what arguments are valid before emitting calls.
2. **Validation** (in production: validate before dispatch; not enforced in the stubs here).
3. **Documentation** — agents can introspect via `registry.catalog()`.

### Built-in implementations

`builtin_registry()` returns `{code_exec, web_search, web_fetch}` wired to the real implementations in [`sandbox/`](./sandbox/README.md):

- `code_exec` → `CodeExecSandbox` (subprocess + rlimits + timeout + output truncation).
- `web_search` / `web_fetch` → `AllowlistedWebTool` (domain allowlist, modelcard 9.2 contamination denylist, JSON cache, offline mode by default).

Operators harden `code_exec` further with kernel-level isolation (gVisor / firecracker / Docker+seccomp) and plug a real search/fetch backend into `AllowlistedWebTool` via `searcher` / `fetcher` callables. The `ToolRegistry` interface stays stable across both.

### Errors are model-visible

Tool exceptions become `ToolResult(error=...)` and reach the model as a structured tool message. This lets the model recover (retry, ask the user, switch approach) instead of silently hanging.

### Safety integration

Every prompt passes through `safety/probes/build_default_probe_gate()` before dispatch in the FastAPI server. The default gate composes:

- `LinearProbeClassifier` for prohibited / high-risk-dual / dual-use cyber.
- `ConstitutionalClassifier` for CBRN content (with a keyword fail-closed backup).

See [`safety/probes/README.md`](../safety/probes/README.md). The legacy keyword stub remains accessible as `safety.classifiers.default_safety_gate()` for environments where probe weights cannot be loaded.
