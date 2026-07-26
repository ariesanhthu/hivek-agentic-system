"""Deterministic mock LLM.

Required by the blueprint so tests and demos never depend on a live API. It is
deterministic (seeded by the prompt hash), so the same context yields the same draft
and assertions about generated content are stable.

It fills schemas from their own field definitions rather than hardcoding one shape,
so it keeps working when a schema changes.
"""

from __future__ import annotations

import random
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from hivek_agent.infrastructure.llm.base import (
    LLMCompletion,
    LLMGateway,
    ModelT,
    context_hash,
    estimate_tokens,
)

# Vietnamese-first, since the product writes Vietnamese social copy.
_HOOKS = (
    "Bạn đang bỏ lỡ điều gì trong tuần này?",
    "Ba điều ít người nói về chủ đề này.",
    "Câu hỏi nhận được nhiều nhất tháng qua.",
)
_BODIES = (
    "Đây là bản nháp mô phỏng do mock provider tạo ra. Nội dung chỉ dùng cho kiểm thử "
    "và demo, không phải văn bản thật từ mô hình ngôn ngữ.",
    "Bản nháp này minh họa luồng tạo nội dung end-to-end mà không cần gọi API thật.",
)
_CTAS = ("Nhắn tin để được tư vấn.", "Xem chi tiết trong phần bình luận.")


class MockLLM(LLMGateway):
    provider_name = "mock"

    def __init__(self, *, model_name: str = "mock-agent") -> None:
        self._model_name = model_name

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
        rng = random.Random(context_hash(system, prompt))
        text = f"{rng.choice(_HOOKS)}\n\n{rng.choice(_BODIES)}\n\n{rng.choice(_CTAS)}"
        return LLMCompletion(
            text=text,
            model=self._model_name,
            input_tokens=estimate_tokens(system + prompt),
            output_tokens=estimate_tokens(text),
            latency_ms=1,
        )

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
        rng = random.Random(context_hash(system, prompt, schema.__name__))
        payload = {
            name: _mock_value(name, field, rng)
            for name, field in schema.model_fields.items()
            if field.is_required() or rng.random() < 0.75
        }
        instance = schema.model_validate(payload)
        rendered = instance.model_dump_json()
        return instance, LLMCompletion(
            text=rendered,
            model=self._model_name,
            input_tokens=estimate_tokens(system + prompt),
            output_tokens=estimate_tokens(rendered),
            latency_ms=1,
        )


def _mock_value(name: str, field: FieldInfo, rng: random.Random) -> Any:
    annotation = field.annotation
    origin = get_origin(annotation)

    # Optional[X] -> unwrap to X.
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if origin is not None and type(None) in get_args(annotation) and len(args) == 1:
        annotation = args[0]
        origin = get_origin(annotation)

    # Literal unions: always pick a legal member so validation cannot fail.
    literal_args = get_args(annotation)
    if literal_args and all(isinstance(arg, str) for arg in literal_args) and origin is not list:
        return rng.choice(literal_args)

    if annotation is str:
        return _mock_string(name, rng)
    if annotation is bool:
        return rng.random() < 0.5
    if annotation is int:
        return rng.randint(1, 5)
    if annotation is float:
        return round(rng.uniform(0.55, 0.95), 2)
    if origin is list:
        inner = get_args(annotation)[0] if get_args(annotation) else str
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return []
        return [_mock_string(name, rng)]
    if origin is dict:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {}
    return None


def _mock_string(name: str, rng: random.Random) -> str:
    lowered = name.lower()
    if "hook" in lowered:
        return rng.choice(_HOOKS)
    if "cta" in lowered:
        return rng.choice(_CTAS)
    if "body" in lowered or "content" in lowered or "text" in lowered:
        return rng.choice(_BODIES)
    if "hashtag" in lowered:
        return "#hivek"
    if "summary" in lowered or "rationale" in lowered or "reason" in lowered:
        return "Giải thích mô phỏng từ mock provider."
    return f"mock-{lowered}"
