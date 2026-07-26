"""Document store abstraction.

Two implementations (memory, mongo) satisfy this protocol, so every repository and
node above it is storage-agnostic and testable without infrastructure.

The query language is deliberately tiny - equality and a handful of comparison/set
operators - because
the in-memory implementation must reproduce it exactly. Anything richer belongs in a
Mongo-specific method on a repository, not here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Collections owned by this service. The `agentic_` prefix keeps them clear of the
# main backend's collections (users, campaigns, influencers, ...) in the shared DB.
COLLECTION_PREFIX = "agentic_"

RUNS = f"{COLLECTION_PREFIX}runs"
NODE_RUNS = f"{COLLECTION_PREFIX}node_runs"
EVENTS = f"{COLLECTION_PREFIX}events"
THREADS = f"{COLLECTION_PREFIX}threads"
AUDIT = f"{COLLECTION_PREFIX}audit"

ASSERTIONS = f"{COLLECTION_PREFIX}assertions"
ENTITIES = f"{COLLECTION_PREFIX}entities"
EDGES = f"{COLLECTION_PREFIX}edges"
CONFLICTS = f"{COLLECTION_PREFIX}conflicts"
BRAND_PROFILES = f"{COLLECTION_PREFIX}brand_profiles"
VOICE_PROFILES = f"{COLLECTION_PREFIX}voice_profiles"

PLANS = f"{COLLECTION_PREFIX}plans"
ASSETS = f"{COLLECTION_PREFIX}assets"

FEEDBACK = f"{COLLECTION_PREFIX}feedback"
PREFERENCES = f"{COLLECTION_PREFIX}preferences"
EDIT_EVENTS = f"{COLLECTION_PREFIX}edit_events"
PERFORMANCE = f"{COLLECTION_PREFIX}performance"

# Social connector + unified inbox collections.  These are intentionally kept in the
# same store abstraction as the existing harness; provider services never reach into
# PyMongo directly, which keeps deduplication tests faithful on MemoryStore.
SOCIAL_ACCOUNTS = f"{COLLECTION_PREFIX}social_accounts"
SOCIAL_CREDENTIALS = f"{COLLECTION_PREFIX}social_credentials"
PUBLICATIONS = f"{COLLECTION_PREFIX}publications"
PUBLISH_REQUESTS = f"{COLLECTION_PREFIX}publish_requests"
WEBHOOK_EVENTS = f"{COLLECTION_PREFIX}webhook_events"
CONVERSATIONS = f"{COLLECTION_PREFIX}conversations"
MESSAGES = f"{COLLECTION_PREFIX}messages"
REPLY_DECISIONS = f"{COLLECTION_PREFIX}reply_decisions"
OUTBOUND_ACTIONS = f"{COLLECTION_PREFIX}outbound_actions"
SYNC_CURSORS = f"{COLLECTION_PREFIX}sync_cursors"

ALL_COLLECTIONS = (
    RUNS,
    NODE_RUNS,
    EVENTS,
    THREADS,
    AUDIT,
    ASSERTIONS,
    ENTITIES,
    EDGES,
    CONFLICTS,
    BRAND_PROFILES,
    VOICE_PROFILES,
    PLANS,
    ASSETS,
    FEEDBACK,
    PREFERENCES,
    EDIT_EVENTS,
    PERFORMANCE,
    SOCIAL_ACCOUNTS,
    SOCIAL_CREDENTIALS,
    PUBLICATIONS,
    PUBLISH_REQUESTS,
    WEBHOOK_EVENTS,
    CONVERSATIONS,
    MESSAGES,
    REPLY_DECISIONS,
    OUTBOUND_ACTIONS,
    SYNC_CURSORS,
)

# (collection, keys, unique) - applied by both backends at startup.
# The unique ones are what make retried writes idempotent rather than duplicating.
INDEX_SPECS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    (RUNS, ("run_id",), True),
    (RUNS, ("workspace_id", "created_at"), False),
    (RUNS, ("workspace_id", "idempotency_key"), False),
    (NODE_RUNS, ("run_id",), False),
    (EVENTS, ("run_id", "seq"), True),
    (THREADS, ("thread_id",), True),
    (AUDIT, ("workspace_id", "at"), False),
    (ASSERTIONS, ("assertion_id",), True),
    (ASSERTIONS, ("workspace_id", "subject_id", "predicate"), False),
    (ENTITIES, ("entity_id",), True),
    (EDGES, ("edge_id",), True),
    (EDGES, ("workspace_id", "from_id"), False),
    (CONFLICTS, ("conflict_id",), True),
    (BRAND_PROFILES, ("workspace_id",), True),
    (VOICE_PROFILES, ("workspace_id",), True),
    (PLANS, ("plan_id",), True),
    (ASSETS, ("asset_id",), True),
    (ASSETS, ("workspace_id", "status"), False),
    (FEEDBACK, ("feedback_id",), True),
    (PREFERENCES, ("workspace_id", "key"), True),
    (EDIT_EVENTS, ("event_id",), True),
    (PERFORMANCE, ("event_id",), True),
    (SOCIAL_ACCOUNTS, ("workspace_id", "account_id"), True),
    (SOCIAL_ACCOUNTS, ("platform", "provider_account_id"), False),
    (SOCIAL_CREDENTIALS, ("workspace_id", "credential_id"), True),
    (PUBLICATIONS, ("workspace_id", "publication_id"), True),
    (PUBLICATIONS, ("platform", "platform_post_id"), True),
    (PUBLISH_REQUESTS, ("idempotency_key",), True),
    (WEBHOOK_EVENTS, ("provider", "provider_event_id"), True),
    (CONVERSATIONS, ("workspace_id", "conversation_id"), True),
    (CONVERSATIONS, ("workspace_id", "last_message_at"), False),
    (CONVERSATIONS, ("platform", "provider_thread_key"), True),
    (MESSAGES, ("workspace_id", "message_id"), True),
    (MESSAGES, ("platform", "provider_message_id"), True),
    (MESSAGES, ("conversation_id", "created_at"), False),
    (REPLY_DECISIONS, ("workspace_id", "decision_id"), True),
    (REPLY_DECISIONS, ("message_id",), True),
    (OUTBOUND_ACTIONS, ("idempotency_key",), True),
    (SYNC_CURSORS, ("workspace_id", "publication_id"), True),
)


class StoreError(RuntimeError):
    """Raised for store-level failures the caller may retry."""


class DuplicateKeyError(StoreError):
    """A unique index rejected the write. Callers treat this as 'already done'."""


@runtime_checkable
class DocumentStore(Protocol):
    """Minimal async document store."""

    backend_name: str

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> bool: ...

    async def ensure_indexes(self) -> None: ...

    async def insert(self, collection: str, document: dict[str, Any]) -> None:
        """Insert one document. Raises DuplicateKeyError if a unique index rejects it."""
        ...

    async def find_one(self, collection: str, query: dict[str, Any]) -> dict[str, Any] | None: ...

    async def find(
        self,
        collection: str,
        query: dict[str, Any],
        *,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
        skip: int = 0,
    ) -> list[dict[str, Any]]: ...

    async def update_one(
        self,
        collection: str,
        query: dict[str, Any],
        document: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> bool:
        """Full-document replace of the first match. Returns True if something changed."""
        ...

    async def count(self, collection: str, query: dict[str, Any]) -> int: ...

    async def delete(self, collection: str, query: dict[str, Any]) -> int: ...


def matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
    """Evaluate the supported query subset against one document.

    Shared by the in-memory store and the tests so both agree with Mongo's semantics
    for the operators we actually use.
    """
    for field, condition in query.items():
        value = _resolve_path(document, field)
        if isinstance(condition, dict):
            for operator, operand in condition.items():
                if operator == "$in":
                    if value not in operand:
                        return False
                elif operator == "$nin":
                    if value in operand:
                        return False
                elif operator == "$ne":
                    if value == operand:
                        return False
                elif operator == "$exists":
                    if (value is not None) != bool(operand):
                        return False
                elif operator == "$gte":
                    if value is None or value < operand:
                        return False
                elif operator == "$lte":
                    if value is None or value > operand:
                        return False
                else:
                    raise StoreError(f"Unsupported query operator: {operator}")
        elif value != condition:
            return False
    return True


def _resolve_path(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
