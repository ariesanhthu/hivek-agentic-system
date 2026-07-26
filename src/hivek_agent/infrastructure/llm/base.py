"""LLM gateway contract.

Every call is structured: the caller passes a Pydantic model and gets a validated
instance back, or an error. No node parses free text.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(RuntimeError):
    """Provider failure. Callers decide whether to degrade or surface it."""


class LLMSchemaError(LLMError):
    """The model returned output that would not validate against the schema."""


class LLMCompletion(BaseModel):
    """Envelope carrying the parsed value plus the metering the harness records."""

    text: str = ""
    model: str = "mock"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cache_hit: bool = False
    finish_reason: str = "stop"


class LLMGateway(Protocol):
    provider_name: str

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1800,
        fallback_models: list[str] | None = None,
    ) -> LLMCompletion: ...

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
        """Return a schema-valid instance. Raises LLMSchemaError if it cannot."""
        ...


def context_hash(*parts: Any) -> str:
    """Stable hash of prompt inputs. Used as the cache key and stored on each asset
    so a draft can be traced back to the exact context that produced it."""
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def estimate_tokens(text: str) -> int:
    """Cheap heuristic (~4 chars/token). Only used for budgeting, never for billing."""
    return max(1, len(text) // 4)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences even when told not to, so this is defensive
    by design rather than trusting the response to be clean.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    cleaned = cleaned.strip().strip("`").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            # Include what actually came back: "no JSON found" without the payload is
            # unactionable, and the text is model output, not user data or secrets.
            raise LLMSchemaError(
                f"no JSON object found in model output "
                f"(len={len(text)}, open_brace={start != -1}, close_brace={end != -1}): "
                f"{cleaned[:200]!r}"
            ) from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMSchemaError(
                f"invalid JSON in model output: {exc.msg} | {cleaned[:200]!r}"
            ) from exc

    if not isinstance(parsed, dict):
        raise LLMSchemaError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


def parse_into[T: BaseModel](schema: type[T], text: str) -> T:
    try:
        return schema.model_validate(extract_json_object(text))
    except ValidationError as exc:
        raise LLMSchemaError(
            f"output failed {schema.__name__} validation: {exc.error_count()} errors"
        ) from exc
