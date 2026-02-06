"""
LiteLLM provider implementation for ClawAI Agent.
"""

from __future__ import annotations

import os
import json
from typing import Any

import litellm
from litellm import acompletion

from clawai.llm.base import (
    LLMProvider,
    LLMResponse,
    ToolCall,
    TokenUsage,
)


class LiteLLMProvider(LLMProvider):
    """
    LLM provider backed by LiteLLM.

    Supports OpenAI, Anthropic, Gemini, OpenRouter, vLLM, Zhipu, Groq, etc.
    """

    # ---------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        timeout: float = 60.0,
    ):
        super().__init__(api_key=api_key, api_base=api_base, timeout=timeout)
        self.default_model = default_model

        self._detect_provider_mode()
        self._configure_env()
        self._configure_litellm()

    # ---------------------------------------------------------------------
    # Provider detection
    # ---------------------------------------------------------------------

    def _detect_provider_mode(self) -> None:
        self.is_openrouter = bool(
            (self.api_key and self.api_key.startswith("sk-or-")) or
            (self.api_base and "openrouter" in self.api_base)
        )
        self.is_vllm = bool(self.api_base) and not self.is_openrouter

    # ---------------------------------------------------------------------
    # Environment configuration
    # ---------------------------------------------------------------------

    def _configure_env(self) -> None:
        if not self.api_key:
            return

        if self.is_openrouter:
            os.environ.setdefault("OPENROUTER_API_KEY", self.api_key)
            return

        model = self.default_model.lower()

        if self.is_vllm:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        elif "anthropic" in model:
            os.environ.setdefault("ANTHROPIC_API_KEY", self.api_key)
        elif "openai" in model or "gpt" in model:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
        elif "gemini" in model:
            os.environ.setdefault("GEMINI_API_KEY", self.api_key)
        elif "deepseek" in model:
            os.environ.setdefault("DEEPSEEK_API_KEY", self.api_key)
        elif any(k in model for k in ("glm", "zhipu", "zai")):
            os.environ.setdefault("ZHIPUAI_API_KEY", self.api_key)
        elif "groq" in model:
            os.environ.setdefault("GROQ_API_KEY", self.api_key)

    def _configure_litellm(self) -> None:
        if self.api_base:
            litellm.api_base = self.api_base
        litellm.suppress_debug_info = True

    # ---------------------------------------------------------------------
    # Chat
    # ---------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        model = self._normalize_model_name(model or self.default_model)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self.timeout,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            raw = await acompletion(**kwargs)
            return self._parse_response(raw)

        except Exception as e:
            return LLMResponse(
                content=None,
                finish_reason="error",
                raw=str(e),
            )

    # ---------------------------------------------------------------------
    # Parsing
    # ---------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []

        for tc in getattr(message, "tool_calls", []) or []:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}

            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                )
            )

        usage = None
        if getattr(response, "usage", None):
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            raw=response,
        )

    # ---------------------------------------------------------------------
    # Model normalization
    # ---------------------------------------------------------------------

    def _normalize_model_name(self, model: str) -> str:
        m = model

        if self.is_openrouter and not m.startswith("openrouter/"):
            return f"openrouter/{m}"

        if self.is_vllm:
            return f"hosted_vllm/{m}"

        if "gemini" in m.lower() and not m.startswith("gemini/"):
            return f"gemini/{m}"

        if any(k in m.lower() for k in ("glm", "zhipu")) and not any(
            m.startswith(p) for p in ("zai/", "zhipu/", "openrouter/")
        ):
            return f"zai/{m}"

        return m

    # ---------------------------------------------------------------------

    def get_default_model(self) -> str:
        return self.default_model
  
