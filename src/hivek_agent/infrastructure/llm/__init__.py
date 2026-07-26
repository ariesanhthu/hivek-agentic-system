"""LLM gateway assembly: provider selection + caching."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

from hivek_agent.config import Settings, get_settings
from hivek_agent.infrastructure.llm.base import (
    LLMCompletion,
    LLMError,
    LLMGateway,
    LLMSchemaError,
    ModelT,
    context_hash,
    estimate_tokens,
    extract_json_object,
    parse_into,
)
from hivek_agent.infrastructure.llm.gemini import GeminiLLM
from hivek_agent.infrastructure.llm.mock import MockLLM

logger = logging.getLogger(__name__)

__all__ = [
    "CachingLLM",
    "GeminiLLM",
    "LLMCompletion",
    "LLMError",
    "LLMGateway",
    "LLMSchemaError",
    "MockLLM",
    "ModelT",
    "build_llm",
    "context_hash",
    "estimate_tokens",
    "extract_json_object",
    "parse_into",
]


class CachingLLM(LLMGateway):
    """Wraps a gateway with an in-process LRU keyed by
    `input_context_hash + prompt_version + model`, exactly as the blueprint requires.

    Deliberately per-process: it is a cost guard, not a source of truth. Swap in Redis
    behind the same interface when there are multiple workers.
    """

    def __init__(
        self, inner: LLMGateway, *, max_entries: int = 512, ttl_seconds: int = 900
    ) -> None:
        self._inner = inner
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    def _get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > self._ttl_seconds:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def _put(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1800,
        fallback_models: list[str] | None = None,
    ) -> LLMCompletion:
        # Creative sampling must not be cached, or every variant is identical.
        if temperature > 0.3:
            return await self._inner.complete(
                system=system,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                fallback_models=fallback_models,
            )

        key = context_hash("complete", system, prompt, model, temperature)
        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            return cached.model_copy(update={"cache_hit": True})

        self.misses += 1
        result = await self._inner.complete(
            system=system,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            fallback_models=fallback_models,
        )
        self._put(key, result)
        return result

    async def complete_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[ModelT],
        model: str,
        temperature: float = 0.4,
        max_output_tokens: int = 1800,
        fallback_models: list[str] | None = None,
    ) -> tuple[ModelT, LLMCompletion]:
        key = context_hash("structured", system, prompt, model, schema.__name__, temperature)
        cached = self._get(key)
        if cached is not None:
            self.hits += 1
            value, completion = cached
            return value, completion.model_copy(update={"cache_hit": True})

        self.misses += 1
        result = await self._inner.complete_structured(
            system=system,
            prompt=prompt,
            schema=schema,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            fallback_models=fallback_models,
        )
        self._put(key, result)
        return result


def build_llm(settings: Settings | None = None) -> LLMGateway:
    settings = settings or get_settings()

    if settings.resolved_llm_provider == "gemini":
        try:
            gateway: LLMGateway = GeminiLLM(
                settings.gemini_api_key,
                timeout_seconds=settings.gemini_timeout_seconds,
                max_retries=settings.gemini_max_retries,
            )
            logger.info("llm provider=gemini fast=%s", settings.gemini_model_fast)
        except LLMError as exc:
            logger.error("gemini init failed (%s) - falling back to mock provider", exc)
            gateway = MockLLM()
    else:
        gateway = MockLLM()
        logger.info("llm provider=mock (set GEMINI_API_KEY to use a real model)")

    return CachingLLM(gateway)
