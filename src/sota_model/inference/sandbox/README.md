# `sota_model/inference/sandbox/`

Real, sandboxed implementations of the two tools the model leans on most: `code_exec` and `web_search` / `web_fetch`. Replaces the `_stub_*` placeholders that used to live in `inference/tools.py`.

```
__init__.py    re-exports
code_exec.py   CodeExecSandbox + SandboxConfig + CodeExecResult
web.py         AllowlistedWebTool + WebFetchResult + WebSearchResult
```

## Code

### `CodeExecSandbox`

Subprocess-isolated Python/bash runner. Per call:

- Fresh `tempfile.TemporaryDirectory` as cwd / `HOME` / `TMPDIR`.
- Environment scrubbed to a tiny allowlist (`PATH`, `LANG`, `LC_ALL`).
- `setrlimit` floors on memory (`RLIMIT_AS`), CPU time (`RLIMIT_CPU`), file size (`RLIMIT_FSIZE`), open files (`RLIMIT_NOFILE`), child processes (`RLIMIT_NPROC` where supported).
- Hard timeout via `subprocess.run(..., timeout=…)`.
- Output truncated to `SandboxConfig.max_output_bytes`.
- Python invoked with `-I -S -B` (isolated, no site, no bytecode write).

What it does NOT do: kernel-level isolation, namespace separation, network blocking. Documented as **best-effort soft isolation** — production hardens this with gVisor / firecracker / Docker w/ seccomp, layered on top of the same `CodeExecSandbox` interface.

### `AllowlistedWebTool`

`web_search(q, n)` and `web_fetch(url)` with:

- **Domain allowlist**: hard-coded conservative default (`example.com`, `wikipedia.org`); operators extend at construction time.
- **Domain denylist**: mirrors the modelcard 9.2 contamination URL set so the model never fetches eval-discussing pages at inference time.
- **Offline mode** (default `True`): no network call ever; returns cache hits or a clearly-marked "no result" sentinel. This is what evals and CI run with.
- **Disk cache**: keyed by SHA-256 of (kind, params); deterministic replay across runs.
- **Request budget**: `request_budget` calls per session; raises on overflow.
- **Pluggable backends**: pass `searcher: (q, n) -> list[dict]` and `fetcher: (url) -> (status, content_type, text)` to wire a real provider (Brave, Bing, Google CSE) without touching this file.

## Data

- `code_exec`: takes `{"language": "python"|"bash", "code": "..."}` and returns `{"stdout", "stderr", "exit_code", "timed_out", "truncated", "ok"}`.
- `web_search`: takes `{"q", "n"}` and returns `[{"title", "url", "snippet"}, ...]`.
- `web_fetch`: takes `{"url"}` and returns the readable text body.
- All three serialize through the existing JSON-schema-validated tool-call surface in `inference/tools.py`.

## Working

The high-level wiring:

```
ChatRequest
  ↓ render via ChatTemplate
  ↓ engine generates assistant turn
  ↓ assistant emits <|tool_call|>{"name": "code_exec", ...}<|/tool_call|>
  ↓ inference/tools.py::parse_tool_calls
  ↓ inference/tools.py::dispatch → registry["code_exec"](language, code)
  ↓ CodeExecSandbox.run(...) → CodeExecResult.to_dict()
  ↓ wrapped as <|tool_result|>{...}<|/tool_result|>
  ↓ next assistant turn picks it up
```

The `builtin_registry()` factory in `inference/tools.py` wires `CodeExecSandbox` and `AllowlistedWebTool` into the same `ToolRegistry` shape the rest of the engine speaks.

### Operator checklist

- [ ] Replace the conservative default allowlist with the production search-provider list.
- [ ] Plug a real `searcher` and `fetcher` (or proxy through an internal search service).
- [ ] Wrap `CodeExecSandbox` in a kernel-isolated runner before exposing to untrusted prompts.
- [ ] Persist the `web_cache_dir` to a shared volume so eval reproducibility survives across worker restarts.
- [ ] Audit logs: every `dispatch()` call should be observable in the agentic-safety log stream (modelcard 5.1).
