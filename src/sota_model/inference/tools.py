"""Tool registry, parser, and dispatch.

Tools are plain Python callables registered with a JSON Schema. The model emits
`<|tool_call|>{"name": ..., "arguments": {...}}<|/tool_call|>` blocks; the
runtime parses them, runs the callable, and returns
`<|tool_result|>{"name": ..., "content": ...}<|/tool_result|>` for the next turn.

Parallel tool calls are supported: the model may emit multiple tool_call blocks
in a single assistant turn; they're dispatched concurrently.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


_TOOL_CALL_PATTERN = re.compile(r"<\|tool_call\|>(\{.*?\})<\|/tool_call\|>", re.DOTALL)


@dataclass
class Tool:
    name: str
    schema: dict
    func: Callable[..., Any]
    is_async: bool = False
    description: str = ""


@dataclass
class ToolInvocation:
    name: str
    arguments: dict


@dataclass
class ToolResult:
    name: str
    content: str
    error: str | None = None

    def to_message(self) -> dict:
        return {
            "role": "tool",
            "name": self.name,
            "content": self.content if self.error is None else json.dumps({"error": self.error}),
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        schema: dict,
        func: Callable[..., Any],
        description: str = "",
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            schema=schema,
            func=func,
            is_async=inspect.iscoroutinefunction(func),
            description=description,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def catalog(self) -> list[dict]:
        return [
            {"name": t.name, "schema": t.schema, "description": t.description}
            for t in self._tools.values()
        ]


def parse_tool_calls(assistant_text: str) -> list[ToolInvocation]:
    invocations: list[ToolInvocation] = []
    for match in _TOOL_CALL_PATTERN.finditer(assistant_text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if "name" not in payload:
            continue
        invocations.append(
            ToolInvocation(name=payload["name"], arguments=payload.get("arguments", {}))
        )
    return invocations


async def _run_one(registry: ToolRegistry, inv: ToolInvocation) -> ToolResult:
    if inv.name not in registry:
        return ToolResult(name=inv.name, content="", error=f"unknown tool: {inv.name}")
    tool = registry[inv.name]
    try:
        if tool.is_async:
            result = await tool.func(**inv.arguments)
        else:
            result = await asyncio.to_thread(tool.func, **inv.arguments)
        if not isinstance(result, str):
            result = json.dumps(result)
        return ToolResult(name=inv.name, content=result)
    except Exception as e:  # tool-author bugs should bubble up as model-visible errors
        return ToolResult(name=inv.name, content="", error=f"{type(e).__name__}: {e}")


async def dispatch_async(
    registry: ToolRegistry,
    invocations: list[ToolInvocation],
) -> list[ToolResult]:
    """Run all invocations concurrently."""
    if not invocations:
        return []
    return await asyncio.gather(*[_run_one(registry, inv) for inv in invocations])


def dispatch(registry: ToolRegistry, invocations: list[ToolInvocation]) -> list[ToolResult]:
    return asyncio.run(dispatch_async(registry, invocations))


# --- Built-in tools ---


def builtin_registry(
    *,
    sandbox_cfg=None,
    web_cache_dir: str | None = None,
    web_searcher=None,
    web_fetcher=None,
    offline_web: bool = True,
) -> ToolRegistry:
    """Registry with real (sandboxed) implementations.

    Defaults are conservative: `code_exec` runs in a subprocess with rlimits;
    `web_*` is offline-only unless an operator-supplied `web_searcher` /
    `web_fetcher` is passed.
    """
    from sota_model.inference.sandbox import (
        AllowlistedWebTool,
        CodeExecSandbox,
    )

    reg = ToolRegistry()

    sandbox = CodeExecSandbox(sandbox_cfg)
    reg.register(
        "code_exec",
        schema={
            "type": "object",
            "properties": {
                "language": {"type": "string", "enum": ["python", "bash"]},
                "code": {"type": "string"},
            },
            "required": ["language", "code"],
        },
        func=lambda language, code: sandbox(language, code),
        description="Execute code in a sandboxed environment.",
    )

    web = AllowlistedWebTool(
        cache_dir=web_cache_dir,
        offline=offline_web,
        searcher=web_searcher,
        fetcher=web_fetcher,
    )
    reg.register(
        "web_search",
        schema={
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "n": {"type": "integer", "default": 5},
            },
            "required": ["q"],
        },
        func=web.make_search_callable(),
        description="Search the public web (allowlisted).",
    )
    reg.register(
        "web_fetch",
        schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        func=web.make_fetch_callable(),
        description="Fetch a URL and return its readable text (allowlisted).",
    )
    return reg
