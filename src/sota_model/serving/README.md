# `sota_model/serving/`

Production-style API surface.

```
server.py    FastAPI app: /v1/chat/completions (sync + SSE streaming),
             /v1/models, /metrics, /healthz, /readyz
```

## Endpoints

```
POST /v1/chat/completions     OpenAI-compatible chat (streaming via SSE if stream=true)
POST /v1/completions          legacy text completions (not implemented yet)
GET  /v1/models               list deployed models
GET  /metrics                 Prometheus scrape
GET  /healthz                 liveness probe (always returns ok)
GET  /readyz                  readiness probe (model loaded, GPU available)
```

## Why OpenAI-compatible

Anyone with a `openai` Python client can swap base URL and use this server. The API is the de facto industry contract; deviating from it loses ecosystem (LangChain, LlamaIndex, custom SDKs all assume `chat/completions`).

The two extensions over the standard schema:
- `effort` — adaptive thinking tier override (`min`/`low`/`medium`/`high`/`max`)
- `Usage.thinking_tokens` — exposes the thinking-token count separately from completion tokens (billing transparency)

## Why FastAPI

Async out of the box (matches the asyncio-based `tools.py` dispatcher), Pydantic validation for free, OpenAPI docs auto-generated at `/docs`. The alternatives (Flask, Starlette raw) need extra glue for what FastAPI does natively.

## Why SSE for streaming, not WebSockets

OpenAI-compatible clients expect SSE. WebSockets are nicer for bidirectional traffic, but chat-completion is one-way (server → client) once the request is in. SSE works through every HTTP-aware proxy without surprises.

## Prometheus metrics

| Metric | Type | Why |
|---|---|---|
| `sota_requests_total{endpoint,status}` | counter | Throughput, error rate |
| `sota_safety_blocks_total{category}` | counter | Watch the safety gate; sudden spikes mean something changed (model, prompt, traffic) |
| `sota_ttft_seconds` | histogram | P50/P99 time to first token — user-experience metric |
| `sota_tpot_seconds` | histogram | P50/P99 time per output token — capacity metric |

Operators add their own metrics for per-model GPU utilization, KV-cache page utilization, prefix-cache hit rate (modelcard 5.5 monitoring list).

## Safety gate placement

```
request → safety_gate.evaluate(prompt) ─► BLOCK?  return 400 with reason
                                       └─► allow ─► engine.generate ─► response
```

The gate runs **before** model invocation. Every request that gets to the model has been pre-classified. Output-side classifiers (post-model) are TODO; today the model's own refusal training carries the load.

## Authentication & rate limiting

Not implemented in this scaffold. Production deploys put this behind:
- API gateway (Kong / AWS API Gateway / CloudFlare) for auth + rate limit
- Edge load balancer for TLS termination + DDoS

The server is designed to be deployed behind such a layer, not exposed to the internet directly.

## Tokenizer slot

`main()` currently loads `gpt2` as a placeholder. Operators must replace with the real 200K-vocab BPE before serving — see modelcard 1.1.1 for the training-data pipeline that produces it.

## Checkpoint loading

`create_app(checkpoint_path=...)` loads weights via `torch.load`. For ZeRO-3 sharded checkpoints, swap in `deepspeed.utils.zero_to_fp32_consolidated_state_dict()` first. For int8 / int4 deployments, weights load through AWQ / GPTQ paths instead.
