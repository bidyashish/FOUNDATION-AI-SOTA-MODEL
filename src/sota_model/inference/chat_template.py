"""Chat template: messages → token sequence.

The format intentionally matches what `data/samples/chat.jsonl` and
`data/samples/tool_use.jsonl` use, so training data and inference data share a
single source of truth.

Special tokens (must exist in the tokenizer):
    <|im_start|> <|im_end|>          role boundaries
    <|thinking|> <|/thinking|>       hidden reasoning channel (not user-visible)
    <|tool_call|> <|/tool_call|>     emitted by the assistant
    <|tool_result|> <|/tool_result|> echoed back from the tool runtime
    <|image_start|> <|image_end|>    multimodal placeholder (vision encoder fills in)
    <|compacted|>  <|/compacted|>    inserted by the context-compaction path
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    name: str
    arguments: dict

    def render(self) -> str:
        return f"<|tool_call|>{json.dumps({'name': self.name, 'arguments': self.arguments})}<|/tool_call|>"


@dataclass
class Message:
    role: Role
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None
    name: str | None = None  # required for role=tool; tool name being responded to


class ChatTemplate:
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    THINKING_OPEN = "<|thinking|>"
    THINKING_CLOSE = "<|/thinking|>"
    TOOL_RESULT_OPEN = "<|tool_result|>"
    TOOL_RESULT_CLOSE = "<|/tool_result|>"

    def __init__(self, default_system: str | None = None):
        self.default_system = default_system

    def render(
        self,
        messages: Iterable[Message | dict],
        tools: list[dict] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        msgs = [self._coerce(m) for m in messages]
        parts: list[str] = []

        # If a system message isn't first, prepend the default.
        if self.default_system and (not msgs or msgs[0].role != "system"):
            parts.append(self._render_system(self.default_system, tools))
        elif msgs and msgs[0].role == "system":
            sys_msg = msgs[0]
            msgs = msgs[1:]
            parts.append(self._render_system(sys_msg.content or "", tools))
        elif tools:
            # No system message, but tools were provided: emit a minimal system
            # block so the model sees the tool catalog.
            parts.append(self._render_system("", tools))

        for m in msgs:
            parts.append(self._render_message(m))

        if add_generation_prompt:
            parts.append(f"{self.IM_START}assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _coerce(m) -> Message:
        if isinstance(m, Message):
            return m
        return Message(
            role=m["role"],
            content=m.get("content"),
            thinking=m.get("thinking"),
            tool_calls=[ToolCall(**tc) for tc in m["tool_calls"]] if m.get("tool_calls") else None,
            name=m.get("name"),
        )

    def _render_system(self, content: str, tools: list[dict] | None) -> str:
        body = content
        if tools:
            tool_lines = ["", "Available tools:"]
            for t in tools:
                tool_lines.append(f"- {t['name']}: {json.dumps(t.get('schema', {}))}")
            body = (body + "\n".join(tool_lines)).strip()
        return f"{self.IM_START}system\n{body}{self.IM_END}"

    def _render_message(self, m: Message) -> str:
        if m.role == "tool":
            if not m.name:
                raise ValueError("tool messages require `name`")
            inner = f"{self.TOOL_RESULT_OPEN}{json.dumps({'name': m.name, 'content': m.content})}{self.TOOL_RESULT_CLOSE}"
            return f"{self.IM_START}tool\n{inner}{self.IM_END}"

        if m.role == "assistant":
            inner_parts: list[str] = []
            if m.thinking:
                inner_parts.append(f"{self.THINKING_OPEN}{m.thinking}{self.THINKING_CLOSE}")
            if m.tool_calls:
                inner_parts.extend(tc.render() for tc in m.tool_calls)
            if m.content:
                inner_parts.append(m.content)
            return f"{self.IM_START}assistant\n{''.join(inner_parts)}{self.IM_END}"

        return f"{self.IM_START}{m.role}\n{m.content or ''}{self.IM_END}"
