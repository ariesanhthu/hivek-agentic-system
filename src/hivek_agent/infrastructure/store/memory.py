"""In-memory document store.

The zero-infra default: the harness boots and the full chat flow runs with no
MongoDB. Also what the unit tests run against, so tests need no network.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from hivek_agent.infrastructure.store.base import (
    INDEX_SPECS,
    DocumentStore,
    DuplicateKeyError,
    matches,
)


class MemoryStore(DocumentStore):
    backend_name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._unique: dict[str, list[tuple[str, ...]]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        await self.ensure_indexes()

    async def close(self) -> None:
        self._data.clear()

    async def ping(self) -> bool:
        return True

    async def ensure_indexes(self) -> None:
        for collection, keys, unique in INDEX_SPECS:
            if unique:
                self._unique.setdefault(collection, []).append(keys)

    def _violates_unique(self, collection: str, document: dict[str, Any]) -> bool:
        for keys in self._unique.get(collection, []):
            if any(key not in document for key in keys):
                continue
            signature = tuple(document[key] for key in keys)
            for existing in self._data.get(collection, []):
                if all(key in existing for key in keys) and (
                    tuple(existing[key] for key in keys) == signature
                ):
                    return True
        return False

    async def insert(self, collection: str, document: dict[str, Any]) -> None:
        async with self._lock:
            if self._violates_unique(collection, document):
                raise DuplicateKeyError(f"duplicate key in {collection}")
            self._data.setdefault(collection, []).append(copy.deepcopy(document))

    async def find_one(self, collection: str, query: dict[str, Any]) -> dict[str, Any] | None:
        for document in self._data.get(collection, []):
            if matches(document, query):
                return copy.deepcopy(document)
        return None

    async def find(
        self,
        collection: str,
        query: dict[str, Any],
        *,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        rows = [
            copy.deepcopy(document)
            for document in self._data.get(collection, [])
            if matches(document, query)
        ]
        for field, direction in reversed(sort or []):
            rows.sort(key=lambda row: _sort_key(row.get(field)), reverse=direction < 0)
        if skip:
            rows = rows[skip:]
        if limit:
            rows = rows[:limit]
        return rows

    async def update_one(
        self,
        collection: str,
        query: dict[str, Any],
        document: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> bool:
        async with self._lock:
            rows = self._data.setdefault(collection, [])
            for index, existing in enumerate(rows):
                if matches(existing, query):
                    rows[index] = copy.deepcopy(document)
                    return True
            if upsert:
                # An upsert that inserts is still an insert, and Mongo would apply the
                # unique index to it. Skipping the check here let this store accept rows
                # Mongo rejects, which is the one thing it must never do - a test suite
                # that passes on data production refuses is worse than no test at all.
                if self._violates_unique(collection, document):
                    raise DuplicateKeyError(f"duplicate key in {collection}")
                rows.append(copy.deepcopy(document))
                return True
            return False

    async def count(self, collection: str, query: dict[str, Any]) -> int:
        return sum(1 for document in self._data.get(collection, []) if matches(document, query))

    async def delete(self, collection: str, query: dict[str, Any]) -> int:
        async with self._lock:
            rows = self._data.get(collection, [])
            keep = [document for document in rows if not matches(document, query)]
            removed = len(rows) - len(keep)
            self._data[collection] = keep
            return removed


def _sort_key(value: Any) -> tuple[int, Any]:
    """Order a value the way MongoDB's BSON comparison does.

    BSON ranks Null below numbers and strings, so a missing field sorts first ascending
    and last descending. The rank prefix also keeps Python from comparing across types,
    which would raise instead of ordering.
    """
    if value is None:
        return (0, "")
    return (1, value)
