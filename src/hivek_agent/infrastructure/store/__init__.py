"""Store selection.

Chooses Mongo when a usable URI is configured, otherwise the in-memory store. A
Mongo that fails to connect at boot degrades to memory with a loud warning rather
than crashing the service - a demo should still start when Atlas is unreachable.
"""

from __future__ import annotations

import logging

from hivek_agent.config import Settings, get_settings
from hivek_agent.infrastructure.store.base import (
    ALL_COLLECTIONS,
    ASSERTIONS,
    ASSETS,
    AUDIT,
    BRAND_PROFILES,
    CONFLICTS,
    CONVERSATIONS,
    EDGES,
    EDIT_EVENTS,
    ENTITIES,
    EVENTS,
    FEEDBACK,
    MESSAGES,
    NODE_RUNS,
    OUTBOUND_ACTIONS,
    PERFORMANCE,
    PLANS,
    PREFERENCES,
    PUBLICATIONS,
    PUBLISH_REQUESTS,
    REPLY_DECISIONS,
    RUNS,
    SOCIAL_ACCOUNTS,
    SOCIAL_CREDENTIALS,
    SYNC_CURSORS,
    THREADS,
    VOICE_PROFILES,
    WEBHOOK_EVENTS,
    DocumentStore,
    DuplicateKeyError,
    StoreError,
)
from hivek_agent.infrastructure.store.memory import MemoryStore
from hivek_agent.infrastructure.store.mongo import MongoStore

logger = logging.getLogger(__name__)

__all__ = [
    "ALL_COLLECTIONS",
    "ASSERTIONS",
    "ASSETS",
    "AUDIT",
    "BRAND_PROFILES",
    "CONVERSATIONS",
    "CONFLICTS",
    "EDGES",
    "EDIT_EVENTS",
    "ENTITIES",
    "EVENTS",
    "FEEDBACK",
    "MESSAGES",
    "NODE_RUNS",
    "PERFORMANCE",
    "PLANS",
    "PREFERENCES",
    "PUBLICATIONS",
    "PUBLISH_REQUESTS",
    "REPLY_DECISIONS",
    "RUNS",
    "SOCIAL_ACCOUNTS",
    "SOCIAL_CREDENTIALS",
    "SYNC_CURSORS",
    "THREADS",
    "VOICE_PROFILES",
    "WEBHOOK_EVENTS",
    "OUTBOUND_ACTIONS",
    "DocumentStore",
    "DuplicateKeyError",
    "MemoryStore",
    "MongoStore",
    "StoreError",
    "build_store",
]


async def build_store(settings: Settings | None = None) -> DocumentStore:
    settings = settings or get_settings()

    if settings.resolved_store_backend == "memory":
        if settings.mongo_uri_is_placeholder:
            logger.warning(
                "MONGODB_URI still contains the '<db_password>' template - "
                "using in-memory store. Data will not persist across restarts."
            )
        store: DocumentStore = MemoryStore()
        await store.connect()
        logger.info("store backend=memory (no persistence)")
        return store

    mongo = MongoStore(
        settings.mongodb_uri,
        settings.mongodb_db_name,
        timeout_ms=settings.mongodb_timeout_ms,
    )
    try:
        await mongo.connect()
    except StoreError as exc:
        logger.error(
            "mongo unreachable (%s) - falling back to in-memory store. "
            "Check MONGODB_URI and Atlas IP allowlist.",
            exc,
        )
        await mongo.close()
        fallback: DocumentStore = MemoryStore()
        await fallback.connect()
        return fallback

    logger.info("store backend=mongo db=%s", settings.mongodb_db_name)
    return mongo
