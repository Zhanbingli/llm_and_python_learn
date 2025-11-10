"""
Unified LLM provider interface with OpenAI-compatible + offline backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


Message = Dict[str, str]


@dataclass
class LLMConfig:
    """Configuration shared across providers."""

    provider: str = "echo"
    model: str = "gpt-3.5-turbo"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: Optional[int] = None
    max_retries: int = 2


@dataclass
class LLMResponse:
    """Standard response envelope."""

    text: str
    model: str
    total_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None


class BaseLLMProvider:
    """Interface used by the rest of the app."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Message]] = None,
        **overrides: Any,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIChatProvider(BaseLLMProvider):
    """Wraps OpenAI-compatible chat completion endpoints."""

    def __init__(self, config: LLMConfig):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIChatProvider")
        api_key = config.api_key or os.getenv(config.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing API key for provider; set {config.api_key_env} or pass api_key"
            )
        super().__init__(config)
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Message]] = None,
        **overrides: Any,
    ) -> LLMResponse:
        if not prompt and not messages:
            raise ValueError("prompt or messages must be provided")
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        payload = {
            "model": overrides.get("model", self.config.model),
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
            "messages": list(messages),
        }
        response = self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", None) if usage else None
        raw = response.model_dump() if hasattr(response, "model_dump") else None
        return LLMResponse(
            text=text,
            model=response.model,
            total_tokens=tokens,
            raw=raw,
        )


class EchoProvider(BaseLLMProvider):
    """
    Offline fallback useful for unit tests.
    Simply mirrors the prompt and marks metadata in the response.
    """

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Message]] = None,
        **overrides: Any,
    ) -> LLMResponse:
        if not prompt and not messages:
            raise ValueError("prompt or messages must be provided")
        if prompt:
            content = prompt
        else:
            content = " | ".join(msg["content"] for msg in messages or [])
        text = f"[echo:{self.config.model}] {content}"
        return LLMResponse(text=text, model=self.config.model, total_tokens=len(content.split()))


def build_provider(config: Optional[LLMConfig] = None) -> BaseLLMProvider:
    config = config or LLMConfig()
    provider = config.provider.lower()
    if provider in {"openai", "deepseek", "azure"}:
        return OpenAIChatProvider(config)
    if provider == "echo":
        return EchoProvider(config)
    raise ValueError(f"Unknown provider: {config.provider}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    echo_provider = build_provider()
    result = echo_provider.generate(prompt="如何管理感冒患者？")
    logging.info("Echo response: %s", result.text)

