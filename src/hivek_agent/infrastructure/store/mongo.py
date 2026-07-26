"""MongoDB document store (Atlas or self-hosted).

Shares the `hivek` database with the main backend, so every collection this service
touches carries the `agentic_` prefix and nothing here reads or writes the backend's
own collections. Uses PyMongo 4.13+'s native AsyncMongoClient - motor is not needed.
"""

from __future__ import annotations

import logging
from typing import Any

from pymongo import AsyncMongoClient
from pymongo.errors import DuplicateKeyError as PyMongoDuplicateKeyError
from pymongo.errors import PyMongoError

from hivek_agent.infrastructure.store.base import (
    INDEX_SPECS,
    DocumentStore,
    DuplicateKeyError,
    StoreError,
)

logger = logging.getLogger(__name__)


class MongoStore(DocumentStore):
    backend_name = "mongo"

    def __init__(self, uri: str, database: str, *, timeout_ms: int = 5000) -> None:
        self._uri = uri
        self._database_name = database
        self._timeout_ms = timeout_ms
        self._client: AsyncMongoClient | None = None

    @property
    def _db(self):
        if self._client is None:
            raise StoreError("MongoStore.connect() was not awaited")
        return self._client[self._database_name]

    async def connect(self) -> None:
        self._client = AsyncMongoClient(
            self._uri,
            serverSelectionTimeoutMS=self._timeout_ms,
            appname="hivek-agentic-system",
            tz_aware=True,
        )
        await self.ping()
        await self.ensure_indexes()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.admin.command("ping")
            return True
        except PyMongoError as exc:
            raise StoreError(f"mongo ping failed: {exc.__class__.__name__}") from exc

    async def ensure_indexes(self) -> None:
        for collection, keys, unique in INDEX_SPECS:
            try:
                await self._db[collection].create_index(
                    [(key, 1) for key in keys],
                    unique=unique,
                    name="_".join(keys) + ("_uniq" if unique else "_idx"),
                )
            except PyMongoError as exc:
                # An index that already exists with different options must not stop boot.
                logger.warning(
                    "index create skipped collection=%s keys=%s reason=%s",
                    collection,
                    keys,
                    exc.__class__.__name__,
                )

    async def insert(self, collection: str, document: dict[str, Any]) -> None:
        try:
            await self._db[collection].insert_one(dict(document))
        except PyMongoDuplicateKeyError as exc:
            raise DuplicateKeyError(f"duplicate key in {collection}") from exc
        except PyMongoError as exc:
            raise StoreError(f"insert failed on {collection}: {exc.__class__.__name__}") from exc

    async def find_one(self, collection: str, query: dict[str, Any]) -> dict[str, Any] | None:
        try:
            document = await self._db[collection].find_one(query, projection={"_id": False})
        except PyMongoError as exc:
            raise StoreError(f"find_one failed on {collection}: {exc.__class__.__name__}") from exc
        return document

    async def find(
        self,
        collection: str,
        query: dict[str, Any],
        *,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
        skip: int = 0,
    ) -> list[dict[str, Any]]:
        try:
            cursor = self._db[collection].find(query, projection={"_id": False})
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            return [document async for document in cursor]
        except PyMongoError as exc:
            raise StoreError(f"find failed on {collection}: {exc.__class__.__name__}") from exc

    async def update_one(
        self,
        collection: str,
        query: dict[str, Any],
        document: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> bool:
        try:
            result = await self._db[collection].replace_one(query, dict(document), upsert=upsert)
        except PyMongoDuplicateKeyError as exc:
            raise DuplicateKeyError(f"duplicate key in {collection}") from exc
        except PyMongoError as exc:
            raise StoreError(f"update failed on {collection}: {exc.__class__.__name__}") from exc
        return bool(result.modified_count or result.upserted_id)

    async def count(self, collection: str, query: dict[str, Any]) -> int:
        try:
            return await self._db[collection].count_documents(query)
        except PyMongoError as exc:
            raise StoreError(f"count failed on {collection}: {exc.__class__.__name__}") from exc

    async def delete(self, collection: str, query: dict[str, Any]) -> int:
        try:
            result = await self._db[collection].delete_many(query)
        except PyMongoError as exc:
            raise StoreError(f"delete failed on {collection}: {exc.__class__.__name__}") from exc
        return result.deleted_count

    async def graph_neighbors(
        self,
        collection: str,
        *,
        workspace_id: str,
        start_id: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """Traverse the edge collection with $graphLookup.

        This is the Neo4j replacement: real multi-hop traversal, executed in the same
        database as the rest of the data. Always scoped to one workspace.
        """
        pipeline = [
            {"$match": {"workspace_id": workspace_id, "from_id": start_id}},
            {
                "$graphLookup": {
                    "from": collection,
                    "startWith": "$to_id",
                    "connectFromField": "to_id",
                    "connectToField": "from_id",
                    "as": "downstream",
                    "maxDepth": max(0, max_depth - 1),
                    "restrictSearchWithMatch": {"workspace_id": workspace_id},
                }
            },
            {"$project": {"_id": False}},
        ]
        try:
            return [doc async for doc in await self._db[collection].aggregate(pipeline)]
        except PyMongoError as exc:
            raise StoreError(f"graph traversal failed: {exc.__class__.__name__}") from exc
