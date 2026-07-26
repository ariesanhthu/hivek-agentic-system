"""Gemini gateway via google-genai.

Uses Gemini's native structured output (`response_schema`) so the model is constrained
to the Pydantic schema at decode time rather than being asked nicely in the prompt.
Falls back to prompt-and-parse only if the SDK rejects the schema.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from google import genai
from google.genai import types as genai_types

from hivek_agent.infrastructure.llm.base import (
    LLMCompletion,
    LLMError,
    LLMGateway,
    LLMSchemaError,
    ModelT,
    estimate_tokens,
    parse_into,
)

logger = logging.getLogger(__name__)

# Transient conditions worth retrying on the same model.
_RETRYABLE = ("500", "502", "503", "504", "deadline", "timeout", "unavailable")

# Conditions that mean "this key cannot use this model at all" - retrying is pointless,
# but a different model may work.
_MODEL_UNAVAILABLE = ("not found", "404", "permission_denied", "403")

# A hard zero quota: the model is simply not enabled for this key/tier. Free-tier keys
# report Pro models this way. Distinct from a per-minute rate limit, which is transient
# and *must* be retried on the same model rather than silently downgrading it.
_ZERO_QUOTA_MARKERS = ("limit: 0", "limit:0", "perday", "per day")

# Rejections of the response_schema itself. Only these justify falling back to an
# unconstrained prompt-and-parse call.
_SCHEMA_REJECTED = ("invalid_argument", "400", "response_schema", "invalid json schema")

_RETRY_DELAY_PATTERN = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
_MAX_BACKOFF_SECONDS = 30.0


class GeminiLLM(LLMGateway):
    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        if not api_key.strip():
            raise LLMError("GEMINI_API_KEY is empty")
        self._client = genai.Client(api_key=api_key)
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

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
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        response, latency_ms, served_by = await self._call(
            model=model, prompt=prompt, config=config, fallbacks=fallback_models
        )
        return self._to_completion(response, served_by, system + prompt, latency_ms)

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
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

        try:
            response, latency_ms, served_by = await self._call(
                model=model, prompt=prompt, config=config, fallbacks=fallback_models
            )
        except _SchemaRejected as exc:
            # ONLY a genuine schema rejection (some unions/tuples are unsupported)
            # justifies dropping response_schema. Catching every LLMError here was a bug:
            # an unreachable model would fall through to an unconstrained call, which
            # returns prose, and the caller then failed on "no JSON object found" -
            # hiding the real cause (the model was never reachable).
            logger.warning("response_schema rejected, retrying prompt-and-parse: %s", exc)
            fallback_prompt = (
                f"{prompt}\n\nReturn ONLY a JSON object matching this schema:\n"
                f"{schema.model_json_schema()}"
            )
            completion = await self.complete(
                system=system,
                prompt=fallback_prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                fallback_models=fallback_models,
            )
            return parse_into(schema, completion.text), completion

        completion = self._to_completion(response, served_by, system + prompt, latency_ms)

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed, completion
        if not completion.text:
            raise LLMSchemaError(f"empty response for {schema.__name__}")
        return parse_into(schema, completion.text), completion

    async def _call(
        self, *, model: str, prompt: str, config, fallbacks: list[str] | None = None
    ) -> tuple[object, int, str]:
        """Call `model`, degrading through `fallbacks` if it is unavailable to this key.

        Returns the response, latency, and the model that actually served it - the
        caller must report the real model, not the one it asked for.
        """
        candidates = [model, *(fallbacks or [])]
        last_error: Exception | None = None

        for index, candidate in enumerate(candidates):
            try:
                return await self._call_one(model=candidate, prompt=prompt, config=config)
            except _ModelUnavailable as exc:
                last_error = exc.__cause__ or exc
                remaining = candidates[index + 1 :]
                if not remaining:
                    break
                logger.warning(
                    "model %s unavailable to this key (%s) - falling back to %s",
                    candidate,
                    exc.reason,
                    remaining[0],
                )

        raise LLMError(f"gemini failed on {candidates}: {last_error}")

    async def _call_one(self, *, model: str, prompt: str, config) -> tuple[object, int, str]:
        last_error: Exception | None = None
        delay = 0.0

        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model, contents=prompt, config=config
                    ),
                    timeout=self._timeout_seconds,
                )
                return response, int((time.perf_counter() - started) * 1000), model
            except TimeoutError as exc:
                last_error = exc
                delay = min(float(2**attempt), _MAX_BACKOFF_SECONDS)
                logger.warning("gemini timeout model=%s attempt=%s", model, attempt + 1)
            except Exception as exc:  # google-genai raises provider-specific types
                last_error = exc
                # Order matters: a zero quota means this model will never work, while a
                # plain 429 just means we are going too fast and must back off here.
                if _is_model_unavailable(exc):
                    raise _ModelUnavailable(model, _reason(exc)) from exc
                if _is_schema_rejected(exc):
                    raise _SchemaRejected(str(exc)) from exc
                if _is_rate_limited(exc):
                    delay = _retry_delay(exc, attempt)
                    logger.warning(
                        "gemini rate limited model=%s attempt=%s - waiting %.1fs",
                        model,
                        attempt + 1,
                        delay,
                    )
                elif _is_retryable(exc):
                    delay = min(float(2**attempt), _MAX_BACKOFF_SECONDS)
                    logger.warning(
                        "gemini retryable error model=%s attempt=%s: %s",
                        model,
                        attempt + 1,
                        exc.__class__.__name__,
                    )
                else:
                    raise LLMError(f"gemini call failed: {exc.__class__.__name__}: {exc}") from exc

            if attempt < self._max_retries:
                await asyncio.sleep(delay)

        raise LLMError(f"gemini failed after {self._max_retries + 1} attempts: {last_error}")

    def _to_completion(
        self, response: object, model: str, prompt_text: str, latency_ms: int
    ) -> LLMCompletion:
        text = getattr(response, "text", None) or ""
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        finish = str(getattr(candidates[0], "finish_reason", "")) if candidates else "unknown"
        return LLMCompletion(
            text=text,
            model=model,
            input_tokens=getattr(usage, "prompt_token_count", None) or estimate_tokens(prompt_text),
            output_tokens=getattr(usage, "candidates_token_count", None) or estimate_tokens(text),
            latency_ms=latency_ms,
            finish_reason=finish,
        )


class _ModelUnavailable(Exception):
    """This key cannot use this model at all. Try another rather than retrying it."""

    def __init__(self, model: str, reason: str) -> None:
        super().__init__(f"{model}: {reason}")
        self.model = model
        self.reason = reason


class _SchemaRejected(Exception):
    """The API rejected response_schema. Retrying unconstrained may still work."""


def _blob(exc: Exception) -> str:
    return f"{exc.__class__.__name__} {exc}".lower()


def _is_rate_limited(exc: Exception) -> bool:
    """A 429 that is NOT a hard zero quota - i.e. we are simply going too fast.

    Worth retrying on the same model. Conflating this with a zero quota silently
    downgrades every request to a weaker model the moment traffic spikes.
    """
    blob = _blob(exc)
    if "429" not in blob and "resource_exhausted" not in blob:
        return False
    return not _is_zero_quota(exc)


def _is_zero_quota(exc: Exception) -> bool:
    blob = _blob(exc)
    if "429" not in blob and "resource_exhausted" not in blob:
        return False
    return any(marker in blob for marker in _ZERO_QUOTA_MARKERS)


def _is_retryable(exc: Exception) -> bool:
    return any(token in _blob(exc) for token in _RETRYABLE)


def _is_model_unavailable(exc: Exception) -> bool:
    return any(token in _blob(exc) for token in _MODEL_UNAVAILABLE) or _is_zero_quota(exc)


def _is_schema_rejected(exc: Exception) -> bool:
    blob = _blob(exc)
    return any(token in blob for token in _SCHEMA_REJECTED)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honour the server's requested delay when it gives one, else exponential."""
    match = _RETRY_DELAY_PATTERN.search(str(exc))
    if match:
        return min(float(match.group(1)) + 1.0, _MAX_BACKOFF_SECONDS)
    return min(float(2**attempt), _MAX_BACKOFF_SECONDS)


def _reason(exc: Exception) -> str:
    text = str(exc).lower()
    if _is_zero_quota(exc):
        return "quota limit is 0 (this model is not enabled for this key/tier)"
    if "not found" in text or "404" in text:
        return "model not found for this API version"
    if "permission" in text or "403" in text:
        return "no permission for this model"
    return exc.__class__.__name__
