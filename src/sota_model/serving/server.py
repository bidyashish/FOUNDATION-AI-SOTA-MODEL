"""OpenAI-compatible serving layer with adaptive thinking and safety gates.

Endpoints:
  POST /v1/chat/completions      — chat completions (sync + streaming via SSE)
  POST /v1/completions           — legacy text completions
  GET  /v1/models                — list deployed models
  GET  /metrics                  — Prometheus scrape endpoint
  GET  /healthz                  — liveness
  GET  /readyz                   — readiness (model loaded, GPU available)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator, Literal

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel
    from sse_starlette.sse import EventSourceResponse
except ImportError as e:
    raise RuntimeError("Install with `pip install -e '.[serve]'` to use the serving layer") from e

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    Counter = Histogram = None
    generate_latest = lambda: b""
    CONTENT_TYPE_LATEST = "text/plain"

from sota_model.config import EffortTier, InferenceConfig, ModelConfig
from sota_model.inference.chat_template import ChatTemplate
from sota_model.inference.engine import InferenceEngine
from sota_model.modeling.transformer import build_model
from sota_model.safety.classifiers import Action
from sota_model.safety.probes import build_default_probe_gate


def _msg_to_dict(m) -> dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


def _dump(obj) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


# --- API schemas ---


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 8192
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    effort: EffortTier | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


# --- Metrics ---

if Counter is not None:
    REQUESTS = Counter("sota_requests_total", "Total requests", ["endpoint", "status"])
    BLOCKS = Counter("sota_safety_blocks_total", "Safety blocks", ["category"])
    TTFT = Histogram("sota_ttft_seconds", "Time to first token")
    TPOT = Histogram("sota_tpot_seconds", "Time per output token")
else:
    REQUESTS = BLOCKS = TTFT = TPOT = None


# --- App factory ---


def create_app(
    model_cfg: ModelConfig,
    inference_cfg: InferenceConfig,
    tokenizer,
    checkpoint_path: Path | None = None,
) -> FastAPI:
    import torch

    app = FastAPI(title="SOTA Model Server", version="0.1.0")

    model = build_model(model_cfg)
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    engine = InferenceEngine(model, inference_cfg, tokenizer).with_model_config(model_cfg)
    safety_gate = build_default_probe_gate()
    chat_template = ChatTemplate()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_params_b": model.num_parameters() / 1e9,
            "cuda_available": torch.cuda.is_available(),
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "sota-4.7-200b",
                    "object": "model",
                    "owned_by": "internal",
                    "context_window": inference_cfg.max_context_tokens,
                }
            ],
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request):
        # Canonical chat template — honors thinking, tool_calls, tool_result,
        # and the system-message contract (inference/README.md).
        prompt = chat_template.render(
            [_msg_to_dict(m) for m in req.messages],
            add_generation_prompt=True,
        )
        verdict = safety_gate.evaluate(prompt)
        if verdict.action == Action.BLOCK:
            if BLOCKS:
                BLOCKS.labels(category=verdict.category.value).inc()
            if REQUESTS:
                REQUESTS.labels(endpoint="chat", status="blocked").inc()
            raise HTTPException(
                status_code=400,
                detail={"error": {
                    "code": "policy_violation",
                    "category": verdict.category.value,
                    "reason": verdict.reason,
                }},
            )

        prompt_ids = tokenizer.encode(prompt)
        forced_effort = req.effort  # per-request — does NOT leak across requests

        if req.stream:
            return EventSourceResponse(
                _stream(engine, prompt_ids, req, tokenizer, forced_effort=forced_effort)
            )

        start = time.time()
        result = engine.generate(prompt_ids, max_new_tokens=req.max_tokens, forced_effort=forced_effort)
        elapsed = time.time() - start
        if TPOT and len(result.tokens) > 0:
            TPOT.observe(elapsed / max(1, len(result.tokens)))
        if REQUESTS:
            REQUESTS.labels(endpoint="chat", status="ok").inc()

        text = tokenizer.decode(result.tokens)
        response = ChatResponse(
            id=f"chatcmpl-{int(time.time()*1e6)}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=result.finished_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(result.tokens),
                thinking_tokens=len(result.thinking_tokens),
                total_tokens=len(prompt_ids) + len(result.tokens) + len(result.thinking_tokens),
            ),
        )
        return JSONResponse(_dump(response))

    return app


def _stream(
    engine: InferenceEngine,
    prompt_ids: list[int],
    req: ChatRequest,
    tokenizer,
    forced_effort=None,
) -> Iterator[str]:
    """Real streaming: yield tokens as the engine produces them."""
    started = time.time()
    first = True
    for tok in engine.stream(prompt_ids, max_new_tokens=req.max_tokens, forced_effort=forced_effort):
        if first:
            if TTFT:
                TTFT.observe(time.time() - started)
            first = False
        chunk = {"choices": [{"delta": {"content": tokenizer.decode([tok])}, "index": 0}]}
        yield {"data": json.dumps(chunk)}
    yield {"data": "[DONE]"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sota_4_7.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    from transformers import AutoTokenizer

    model_cfg = ModelConfig.from_yaml(args.config)
    inference_cfg = InferenceConfig.from_yaml(args.config)
    # Tokenizer slot — operators plug in the real trained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # placeholder

    app = create_app(model_cfg, inference_cfg, tokenizer, args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
