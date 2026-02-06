"""
Base LLM provider interface for ClawAI Agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Tool call
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ToolCall:
    """
    A single tool invocation requested by the LLM.
    """
    id: str
    name: str
    arguments: dict[str, Any]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LLMResponse:
    """
    Normalized response from an LLM provider.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Literal[
        "stop",
        "tool_calls",
        "length",
        "error",
        "timeout",
    ] = "stop"
    usage: TokenUsage | None = None
    raw: Any | None = None   # provider-native payload (debug / tracing)

    # -----------------------------

    @property
    def has_content(self) -> bool:
        return bool(self.content)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_final(self) -> bool:
        return self.finish_reason in ("stop", "length")

    @property
    def requires_tool_execution(self) -> bool:
        return self.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Streaming (optional but future-proof)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LLMDelta:
    """
    Partial streaming chunk from LLM.
    """
    content: str | None = None
    tool_call: ToolCall | None = None
    done: bool = False


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provider responsibilities:
    - Translate ClawAI messages → provider API
    - Normalize provider response → LLMResponse / LLMDelta
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

    # ---------------------------------------------------------------------

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Execute a single non-streaming chat completion.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Optional streaming interface.

        Default: fallback to non-streaming.
        Providers may override.
        """
        response = await self.chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield LLMDelta(content=response.content, done=True)

    # ---------------------------------------------------------------------

    @abstractmethod
    def get_default_model(self) -> str:
        """Return provider default model name."""
        raise NotImplementedError
  
