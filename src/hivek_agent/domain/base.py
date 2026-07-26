"""Shared model base.

The Next.js client speaks camelCase; Python speaks snake_case. Rather than hand-write
a translation layer, every domain model carries a camelCase alias generator:

  - FastAPI serialises responses with `by_alias=True` (its default), so the wire is
    camelCase and the existing TypeScript types need no change;
  - `model_dump()` defaults to `by_alias=False`, so documents are stored snake_case;
  - `populate_by_name=True` means models still load from those snake_case documents.

So the alias only affects the HTTP boundary, which is the only place it matters.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class HivekModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
