"""Typed repositories over the document store.

Workspace isolation is enforced here rather than left to each caller: every method
takes `workspace_id` as a required argument and injects it into the query. A node
cannot construct a query that reads another workspace's data, because it never builds
the query itself.
"""

from __future__ import annotations

import uuid
from typing import Any, TypeVar

from pydantic import BaseModel

from hivek_agent.domain import (
    AgentRun,
    BrandOperatingProfile,
    BrandVoiceProfile,
    ContentAsset,
    ContentPlan,
    EditLearningEvent,
    FeedbackEvent,
    GraphEdge,
    KnowledgeAssertion,
    KnowledgeConflict,
    KnowledgeEntity,
    NodeRun,
    OutboundAction,
    PreferenceCandidate,
    ReplyDecision,
    RunEvent,
    SocialAccount,
    SocialConversation,
    SocialCredential,
    SocialMessage,
    SocialPublication,
    SocialSyncCursor,
    SocialWebhookEvent,
    utc_now_iso,
)
from hivek_agent.domain.social import utc_now
from hivek_agent.infrastructure.store import (
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
    PLANS,
    PREFERENCES,
    PUBLICATIONS,
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
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _load[T: BaseModel](schema: type[T], document: dict[str, Any] | None) -> T | None:
    return schema.model_validate(document) if document else None


class RunRepository:
    """Agent runs, per-node telemetry, and the SSE event log."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def create_run(self, run: AgentRun) -> AgentRun:
        await self._store.insert(RUNS, run.model_dump())
        return run

    async def get_run(self, workspace_id: str, run_id: str) -> AgentRun | None:
        return _load(
            AgentRun,
            await self._store.find_one(RUNS, {"run_id": run_id, "workspace_id": workspace_id}),
        )

    async def find_by_idempotency_key(self, workspace_id: str, key: str) -> AgentRun | None:
        """Lets a retried request return the original run instead of starting a new one."""
        return _load(
            AgentRun,
            await self._store.find_one(
                RUNS, {"workspace_id": workspace_id, "idempotency_key": key}
            ),
        )

    async def update_run(self, run: AgentRun) -> None:
        run.updated_at = utc_now_iso()
        await self._store.update_one(
            RUNS, {"run_id": run.run_id, "workspace_id": run.workspace_id}, run.model_dump()
        )

    async def record_node_run(self, node_run: NodeRun) -> None:
        await self._store.insert(NODE_RUNS, node_run.model_dump())

    async def list_node_runs(self, workspace_id: str, run_id: str) -> list[NodeRun]:
        rows = await self._store.find(
            NODE_RUNS, {"run_id": run_id, "workspace_id": workspace_id}, sort=[("at", 1)]
        )
        return [NodeRun.model_validate(row) for row in rows]

    async def append_event(self, workspace_id: str, event: RunEvent) -> None:
        # (run_id, seq) is unique; a duplicate means a retry already logged this frame.
        try:
            await self._store.insert(EVENTS, {**event.model_dump(), "workspace_id": workspace_id})
        except DuplicateKeyError:
            return

    async def list_events(
        self, workspace_id: str, run_id: str, after_seq: int = 0
    ) -> list[RunEvent]:
        rows = await self._store.find(
            EVENTS,
            {"run_id": run_id, "workspace_id": workspace_id, "seq": {"$gte": after_seq}},
            sort=[("seq", 1)],
        )
        return [
            RunEvent.model_validate({k: v for k, v in row.items() if k != "workspace_id"})
            for row in rows
        ]

    async def save_thread(
        self, workspace_id: str, thread_id: str, messages: list[dict[str, Any]]
    ) -> None:
        await self._store.update_one(
            THREADS,
            {"thread_id": thread_id, "workspace_id": workspace_id},
            {
                "thread_id": thread_id,
                "workspace_id": workspace_id,
                "messages": messages,
                "updated_at": utc_now_iso(),
            },
            upsert=True,
        )

    async def get_thread(self, workspace_id: str, thread_id: str) -> list[dict[str, Any]]:
        document = await self._store.find_one(
            THREADS, {"thread_id": thread_id, "workspace_id": workspace_id}
        )
        return document.get("messages", []) if document else []

    async def audit(self, workspace_id: str, action: str, detail: dict[str, Any]) -> str:
        event_id = new_id("audit")
        await self._store.insert(
            AUDIT,
            {
                "audit_id": event_id,
                "workspace_id": workspace_id,
                "action": action,
                "detail": detail,
                "at": utc_now_iso(),
            },
        )
        return event_id


class KnowledgeRepository:
    """Facts, entities, edges and conflicts - all provenance-carrying."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def add_assertion(self, assertion: KnowledgeAssertion) -> KnowledgeAssertion:
        await self._store.insert(ASSERTIONS, assertion.model_dump())
        return assertion

    async def get_assertion(
        self, workspace_id: str, assertion_id: str
    ) -> KnowledgeAssertion | None:
        return _load(
            KnowledgeAssertion,
            await self._store.find_one(
                ASSERTIONS, {"assertion_id": assertion_id, "workspace_id": workspace_id}
            ),
        )

    async def update_assertion(self, assertion: KnowledgeAssertion) -> None:
        await self._store.update_one(
            ASSERTIONS,
            {"assertion_id": assertion.assertion_id, "workspace_id": assertion.workspace_id},
            assertion.model_dump(),
        )

    async def find_live_by_key(
        self, workspace_id: str, subject_id: str, predicate: str
    ) -> list[KnowledgeAssertion]:
        """Assertions still in play for a claim - excludes superseded and rejected."""
        rows = await self._store.find(
            ASSERTIONS,
            {
                "workspace_id": workspace_id,
                "subject_id": subject_id,
                "predicate": predicate,
                "approval_status": {"$nin": ["superseded", "rejected"]},
            },
        )
        return [KnowledgeAssertion.model_validate(row) for row in rows]

    async def list_assertions(
        self, workspace_id: str, *, subject_id: str | None = None, limit: int = 0
    ) -> list[KnowledgeAssertion]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if subject_id:
            query["subject_id"] = subject_id
        rows = await self._store.find(ASSERTIONS, query, sort=[("created_at", 1)], limit=limit)
        return [KnowledgeAssertion.model_validate(row) for row in rows]

    async def add_conflict(self, conflict: KnowledgeConflict) -> None:
        await self._store.insert(CONFLICTS, conflict.model_dump())

    async def list_conflicts(
        self, workspace_id: str, *, unresolved_only: bool = True
    ) -> list[KnowledgeConflict]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if unresolved_only:
            query["resolved"] = False
        rows = await self._store.find(CONFLICTS, query)
        return [KnowledgeConflict.model_validate(row) for row in rows]

    async def resolve_conflict(self, conflict: KnowledgeConflict) -> None:
        await self._store.update_one(
            CONFLICTS,
            {"conflict_id": conflict.conflict_id, "workspace_id": conflict.workspace_id},
            conflict.model_dump(),
        )

    async def upsert_entity(self, entity: KnowledgeEntity) -> None:
        await self._store.update_one(
            ENTITIES,
            {"entity_id": entity.entity_id, "workspace_id": entity.workspace_id},
            entity.model_dump(),
            upsert=True,
        )

    async def add_edge(self, edge: GraphEdge) -> None:
        try:
            await self._store.insert(EDGES, edge.model_dump())
        except DuplicateKeyError:
            return

    async def neighbors(self, workspace_id: str, from_id: str) -> list[GraphEdge]:
        rows = await self._store.find(EDGES, {"workspace_id": workspace_id, "from_id": from_id})
        return [GraphEdge.model_validate(row) for row in rows]

    async def save_brand_profile(self, profile: BrandOperatingProfile) -> None:
        profile.updated_at = utc_now_iso()
        await self._store.update_one(
            BRAND_PROFILES,
            {"workspace_id": profile.workspace_id},
            profile.model_dump(),
            upsert=True,
        )

    async def get_brand_profile(self, workspace_id: str) -> BrandOperatingProfile | None:
        return _load(
            BrandOperatingProfile,
            await self._store.find_one(BRAND_PROFILES, {"workspace_id": workspace_id}),
        )

    async def save_voice_profile(self, profile: BrandVoiceProfile) -> None:
        profile.updated_at = utc_now_iso()
        await self._store.update_one(
            VOICE_PROFILES,
            {"workspace_id": profile.workspace_id},
            profile.model_dump(),
            upsert=True,
        )

    async def get_voice_profile(self, workspace_id: str) -> BrandVoiceProfile | None:
        return _load(
            BrandVoiceProfile,
            await self._store.find_one(VOICE_PROFILES, {"workspace_id": workspace_id}),
        )


class ContentRepository:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def save_plan(self, plan: ContentPlan) -> None:
        await self._store.update_one(
            PLANS,
            {"plan_id": plan.plan_id, "workspace_id": plan.workspace_id},
            plan.model_dump(),
            upsert=True,
        )

    async def get_plan(self, workspace_id: str, plan_id: str) -> ContentPlan | None:
        return _load(
            ContentPlan,
            await self._store.find_one(PLANS, {"plan_id": plan_id, "workspace_id": workspace_id}),
        )

    async def latest_plan(self, workspace_id: str) -> ContentPlan | None:
        rows = await self._store.find(
            PLANS, {"workspace_id": workspace_id}, sort=[("created_at", -1)], limit=1
        )
        return ContentPlan.model_validate(rows[0]) if rows else None

    async def save_asset(self, asset: ContentAsset) -> None:
        asset.updated_at = utc_now_iso()
        await self._store.update_one(
            ASSETS,
            {"asset_id": asset.asset_id, "workspace_id": asset.workspace_id},
            asset.model_dump(),
            upsert=True,
        )

    async def get_asset(self, workspace_id: str, asset_id: str) -> ContentAsset | None:
        return _load(
            ContentAsset,
            await self._store.find_one(
                ASSETS, {"asset_id": asset_id, "workspace_id": workspace_id}
            ),
        )

    async def list_assets(
        self, workspace_id: str, *, status: str | None = None, limit: int = 20
    ) -> list[ContentAsset]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            query["status"] = status
        rows = await self._store.find(ASSETS, query, sort=[("created_at", -1)], limit=limit)
        return [ContentAsset.model_validate(row) for row in rows]

    async def find_by_run(self, workspace_id: str, run_id: str) -> ContentAsset | None:
        return _load(
            ContentAsset,
            await self._store.find_one(ASSETS, {"workspace_id": workspace_id, "run_id": run_id}),
        )

    async def find_by_context_hash(
        self, workspace_id: str, context_hash: str
    ) -> ContentAsset | None:
        """Idempotency: same context + prompt version must not create a second asset."""
        return _load(
            ContentAsset,
            await self._store.find_one(
                ASSETS, {"workspace_id": workspace_id, "context_hash": context_hash}
            ),
        )


class LearningRepository:
    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def add_feedback(self, event: FeedbackEvent) -> None:
        await self._store.insert(FEEDBACK, event.model_dump())

    async def list_feedback(self, workspace_id: str, *, limit: int = 50) -> list[FeedbackEvent]:
        rows = await self._store.find(
            FEEDBACK, {"workspace_id": workspace_id}, sort=[("created_at", -1)], limit=limit
        )
        return [FeedbackEvent.model_validate(row) for row in rows]

    async def add_edit_event(self, event: EditLearningEvent) -> None:
        await self._store.insert(EDIT_EVENTS, event.model_dump())

    async def get_preference(self, workspace_id: str, key: str) -> PreferenceCandidate | None:
        document = await self._store.find_one(
            PREFERENCES, {"workspace_id": workspace_id, "key": key}
        )
        if not document:
            return None
        return PreferenceCandidate.model_validate({k: v for k, v in document.items() if k != "key"})

    async def upsert_preference(self, preference: PreferenceCandidate) -> None:
        preference.updated_at = utc_now_iso()
        # `key` is stored denormalised because it backs the unique index.
        await self._store.update_one(
            PREFERENCES,
            {"workspace_id": preference.workspace_id, "key": preference.key},
            {**preference.model_dump(), "key": preference.key},
            upsert=True,
        )

    async def list_preferences(
        self, workspace_id: str, *, active_only: bool = False
    ) -> list[PreferenceCandidate]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if active_only:
            query["status"] = {"$in": ["repeated", "stable"]}
        rows = await self._store.find(PREFERENCES, query, sort=[("observation_count", -1)])
        return [
            PreferenceCandidate.model_validate({k: v for k, v in row.items() if k != "key"})
            for row in rows
        ]


class SocialRepository:
    """Accounts, encrypted credentials, publications and polling cursors."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def get_account(self, workspace_id: str, account_id: str) -> SocialAccount | None:
        return _load(
            SocialAccount,
            await self._store.find_one(
                SOCIAL_ACCOUNTS, {"workspace_id": workspace_id, "account_id": account_id}
            ),
        )

    async def find_account_by_provider(
        self, workspace_id: str, platform: str, provider_account_id: str
    ) -> SocialAccount | None:
        return _load(
            SocialAccount,
            await self._store.find_one(
                SOCIAL_ACCOUNTS,
                {
                    "workspace_id": workspace_id,
                    "platform": platform,
                    "provider_account_id": provider_account_id,
                },
            ),
        )

    async def find_any_account_by_provider(
        self, platform: str, provider_account_id: str
    ) -> SocialAccount | None:
        """Webhook lookup before the workspace is known.

        This is the only deliberately non-workspace-scoped read.  It returns an account
        identity, never a credential, and the provider account ID is globally owned by
        one platform account.  Every subsequent operation uses the account's workspace.
        """
        return _load(
            SocialAccount,
            await self._store.find_one(
                SOCIAL_ACCOUNTS,
                {"platform": platform, "provider_account_id": provider_account_id},
            ),
        )

    async def list_accounts(
        self, workspace_id: str, *, connected_only: bool = False
    ) -> list[SocialAccount]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if connected_only:
            query["status"] = "connected"
        rows = await self._store.find(SOCIAL_ACCOUNTS, query, sort=[("created_at", 1)])
        return [SocialAccount.model_validate(row) for row in rows]

    async def save_account(self, account: SocialAccount) -> None:
        account.updated_at = utc_now()
        await self._store.update_one(
            SOCIAL_ACCOUNTS,
            {"workspace_id": account.workspace_id, "account_id": account.account_id},
            account.model_dump(),
            upsert=True,
        )

    async def get_credential(
        self, workspace_id: str, credential_id: str
    ) -> SocialCredential | None:
        return _load(
            SocialCredential,
            await self._store.find_one(
                SOCIAL_CREDENTIALS,
                {"workspace_id": workspace_id, "credential_id": credential_id},
            ),
        )

    async def save_credential(self, credential: SocialCredential) -> None:
        credential.updated_at = utc_now()
        await self._store.update_one(
            SOCIAL_CREDENTIALS,
            {
                "workspace_id": credential.workspace_id,
                "credential_id": credential.credential_id,
            },
            credential.model_dump(),
            upsert=True,
        )

    async def save_publication(self, publication: SocialPublication) -> bool:
        existing = await self.find_publication_by_provider_post(
            publication.workspace_id, publication.platform, publication.platform_post_id
        )
        if existing is not None:
            # Registration retries may enrich the local ID/text but must not create a
            # second polling target for the same provider post.
            publication.created_at = existing.created_at
            await self._store.update_one(
                PUBLICATIONS,
                {
                    "workspace_id": existing.workspace_id,
                    "publication_id": existing.publication_id,
                },
                publication.model_copy(
                    update={"publication_id": existing.publication_id}
                ).model_dump(),
            )
            return False
        try:
            await self._store.insert(PUBLICATIONS, publication.model_dump())
            return True
        except DuplicateKeyError:
            return False

    async def get_publication(
        self, workspace_id: str, publication_id: str
    ) -> SocialPublication | None:
        return _load(
            SocialPublication,
            await self._store.find_one(
                PUBLICATIONS,
                {"workspace_id": workspace_id, "publication_id": publication_id},
            ),
        )

    async def list_publications(
        self,
        workspace_id: str,
        *,
        publication_ids: list[str] | None = None,
        sync_enabled_only: bool = False,
        limit: int = 100,
    ) -> list[SocialPublication]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if publication_ids:
            query["publication_id"] = {"$in": publication_ids}
        if sync_enabled_only:
            query["sync_enabled"] = True
        rows = await self._store.find(PUBLICATIONS, query, sort=[("published_at", -1)], limit=limit)
        return [SocialPublication.model_validate(row) for row in rows]

    async def find_publication_by_provider_post(
        self, workspace_id: str, platform: str, platform_post_id: str
    ) -> SocialPublication | None:
        return _load(
            SocialPublication,
            await self._store.find_one(
                PUBLICATIONS,
                {
                    "workspace_id": workspace_id,
                    "platform": platform,
                    "platform_post_id": platform_post_id,
                },
            ),
        )

    async def get_sync_cursor(
        self, workspace_id: str, publication_id: str
    ) -> SocialSyncCursor | None:
        return _load(
            SocialSyncCursor,
            await self._store.find_one(
                SYNC_CURSORS,
                {"workspace_id": workspace_id, "publication_id": publication_id},
            ),
        )

    async def save_sync_cursor(self, cursor: SocialSyncCursor) -> None:
        cursor.last_synced_at = utc_now()
        await self._store.update_one(
            SYNC_CURSORS,
            {"workspace_id": cursor.workspace_id, "publication_id": cursor.publication_id},
            cursor.model_dump(),
            upsert=True,
        )

    async def latest_sync_cursor(self, workspace_id: str) -> SocialSyncCursor | None:
        rows = await self._store.find(
            SYNC_CURSORS,
            {"workspace_id": workspace_id},
            sort=[("last_synced_at", -1)],
            limit=1,
        )
        return SocialSyncCursor.model_validate(rows[0]) if rows else None


class InboxRepository:
    """Conversation/message persistence with provider-ID deduplication."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    async def insert_webhook_event_once(self, event: SocialWebhookEvent) -> bool:
        try:
            await self._store.insert(WEBHOOK_EVENTS, event.model_dump())
            return True
        except DuplicateKeyError:
            return False

    async def update_webhook_event(self, event: SocialWebhookEvent) -> None:
        await self._store.update_one(
            WEBHOOK_EVENTS,
            {
                "workspace_id": event.workspace_id,
                "webhook_event_id": event.webhook_event_id,
            },
            event.model_dump(),
        )

    async def get_conversation(
        self, workspace_id: str, conversation_id: str
    ) -> SocialConversation | None:
        return _load(
            SocialConversation,
            await self._store.find_one(
                CONVERSATIONS,
                {"workspace_id": workspace_id, "conversation_id": conversation_id},
            ),
        )

    async def find_conversation_by_thread(
        self, workspace_id: str, platform: str, provider_thread_key: str
    ) -> SocialConversation | None:
        return _load(
            SocialConversation,
            await self._store.find_one(
                CONVERSATIONS,
                {
                    "workspace_id": workspace_id,
                    "platform": platform,
                    "provider_thread_key": provider_thread_key,
                },
            ),
        )

    async def upsert_conversation(self, conversation: SocialConversation) -> SocialConversation:
        existing = await self.find_conversation_by_thread(
            conversation.workspace_id,
            conversation.platform,
            conversation.provider_thread_key,
        )
        if existing is not None:
            # Preserve user-controlled state while refreshing source/customer context.
            existing.customer_name = conversation.customer_name or existing.customer_name
            existing.customer_username = (
                conversation.customer_username or existing.customer_username
            )
            existing.provider_user_id = conversation.provider_user_id or existing.provider_user_id
            existing.publication_id = conversation.publication_id or existing.publication_id
            existing.source_post_id = conversation.source_post_id or existing.source_post_id
            existing.source_context = conversation.source_context or existing.source_context
            existing.last_message_at = max(existing.last_message_at, conversation.last_message_at)
            existing.updated_at = utc_now()
            await self.update_conversation(existing)
            return existing
        try:
            await self._store.insert(CONVERSATIONS, conversation.model_dump())
            return conversation
        except DuplicateKeyError:
            concurrent = await self.find_conversation_by_thread(
                conversation.workspace_id,
                conversation.platform,
                conversation.provider_thread_key,
            )
            if concurrent is None:
                raise
            return concurrent

    async def update_conversation(self, conversation: SocialConversation) -> None:
        conversation.updated_at = utc_now()
        await self._store.update_one(
            CONVERSATIONS,
            {
                "workspace_id": conversation.workspace_id,
                "conversation_id": conversation.conversation_id,
            },
            conversation.model_dump(),
        )

    async def mark_inbound(self, conversation: SocialConversation, at: Any) -> None:
        conversation.unread_count += 1
        conversation.last_message_at = max(conversation.last_message_at, at)
        if conversation.status == "waiting_for_customer":
            conversation.status = "open"
        await self.update_conversation(conversation)

    async def list_conversations(
        self,
        workspace_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
        skip: int = 0,
    ) -> list[SocialConversation]:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            query["status"] = status
        rows = await self._store.find(
            CONVERSATIONS,
            query,
            sort=[("last_message_at", -1)],
            limit=limit,
            skip=skip,
        )
        return [SocialConversation.model_validate(row) for row in rows]

    async def count_conversations(self, workspace_id: str, *, status: str | None = None) -> int:
        query: dict[str, Any] = {"workspace_id": workspace_id}
        if status:
            query["status"] = status
        return await self._store.count(CONVERSATIONS, query)

    async def insert_message_once(self, message: SocialMessage) -> tuple[SocialMessage, bool]:
        try:
            await self._store.insert(MESSAGES, message.model_dump())
            return message, True
        except DuplicateKeyError:
            existing = await self._store.find_one(
                MESSAGES,
                {
                    "workspace_id": message.workspace_id,
                    "platform": message.platform,
                    "provider_message_id": message.provider_message_id,
                },
            )
            if existing is None:
                raise
            return SocialMessage.model_validate(existing), False

    async def get_message(self, workspace_id: str, message_id: str) -> SocialMessage | None:
        return _load(
            SocialMessage,
            await self._store.find_one(
                MESSAGES, {"workspace_id": workspace_id, "message_id": message_id}
            ),
        )

    async def list_messages(
        self, workspace_id: str, conversation_id: str, *, limit: int = 200
    ) -> list[SocialMessage]:
        rows = await self._store.find(
            MESSAGES,
            {"workspace_id": workspace_id, "conversation_id": conversation_id},
            sort=[("created_at", 1)],
            limit=limit,
        )
        return [SocialMessage.model_validate(row) for row in rows]

    async def latest_message(self, workspace_id: str, conversation_id: str) -> SocialMessage | None:
        rows = await self._store.find(
            MESSAGES,
            {"workspace_id": workspace_id, "conversation_id": conversation_id},
            sort=[("created_at", -1)],
            limit=1,
        )
        return SocialMessage.model_validate(rows[0]) if rows else None

    async def list_approved_replies(
        self, workspace_id: str, *, platform: str | None = None, limit: int = 30
    ) -> list[SocialMessage]:
        query: dict[str, Any] = {
            "workspace_id": workspace_id,
            "direction": "outbound",
            "delivery_status": {"$in": ["sent", "delivered"]},
        }
        if platform:
            query["platform"] = platform
        rows = await self._store.find(MESSAGES, query, sort=[("created_at", -1)], limit=limit)
        return [SocialMessage.model_validate(row) for row in rows]

    async def save_reply_decision(self, decision: ReplyDecision) -> tuple[ReplyDecision, bool]:
        existing = await self.get_reply_decision_for_message(
            decision.workspace_id, decision.message_id
        )
        if existing is not None:
            return existing, False
        try:
            await self._store.insert(REPLY_DECISIONS, decision.model_dump())
            return decision, True
        except DuplicateKeyError:
            existing = await self.get_reply_decision_for_message(
                decision.workspace_id, decision.message_id
            )
            if existing is None:
                raise
            return existing, False

    async def update_reply_decision(self, decision: ReplyDecision) -> None:
        decision.updated_at = utc_now()
        await self._store.update_one(
            REPLY_DECISIONS,
            {"workspace_id": decision.workspace_id, "decision_id": decision.decision_id},
            decision.model_dump(),
        )

    async def get_reply_decision(self, workspace_id: str, decision_id: str) -> ReplyDecision | None:
        return _load(
            ReplyDecision,
            await self._store.find_one(
                REPLY_DECISIONS,
                {"workspace_id": workspace_id, "decision_id": decision_id},
            ),
        )

    async def get_reply_decision_for_message(
        self, workspace_id: str, message_id: str
    ) -> ReplyDecision | None:
        return _load(
            ReplyDecision,
            await self._store.find_one(
                REPLY_DECISIONS,
                {"workspace_id": workspace_id, "message_id": message_id},
            ),
        )

    async def latest_pending_decision(
        self, workspace_id: str, conversation_id: str
    ) -> ReplyDecision | None:
        rows = await self._store.find(
            REPLY_DECISIONS,
            {
                "workspace_id": workspace_id,
                "conversation_id": conversation_id,
                "status": "pending",
            },
            sort=[("created_at", -1)],
            limit=1,
        )
        return ReplyDecision.model_validate(rows[0]) if rows else None

    async def save_outbound_action_once(
        self, action: OutboundAction
    ) -> tuple[OutboundAction, bool]:
        existing = await self.get_outbound_action_by_key(
            action.workspace_id, action.idempotency_key
        )
        if existing is not None:
            return existing, False
        try:
            await self._store.insert(OUTBOUND_ACTIONS, action.model_dump())
            return action, True
        except DuplicateKeyError:
            existing = await self.get_outbound_action_by_key(
                action.workspace_id, action.idempotency_key
            )
            if existing is None:
                raise
            return existing, False

    async def get_outbound_action_by_key(
        self, workspace_id: str, idempotency_key: str
    ) -> OutboundAction | None:
        return _load(
            OutboundAction,
            await self._store.find_one(
                OUTBOUND_ACTIONS,
                {"workspace_id": workspace_id, "idempotency_key": idempotency_key},
            ),
        )

    async def get_outbound_action_by_decision(
        self, workspace_id: str, decision_id: str
    ) -> OutboundAction | None:
        return _load(
            OutboundAction,
            await self._store.find_one(
                OUTBOUND_ACTIONS,
                {"workspace_id": workspace_id, "decision_id": decision_id},
            ),
        )

    async def update_outbound_action(self, action: OutboundAction) -> None:
        action.updated_at = utc_now()
        await self._store.update_one(
            OUTBOUND_ACTIONS,
            {"workspace_id": action.workspace_id, "action_id": action.action_id},
            action.model_dump(),
        )


class Repositories:
    """Bundle handed to nodes so they receive one dependency instead of four."""

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.runs = RunRepository(store)
        self.knowledge = KnowledgeRepository(store)
        self.content = ContentRepository(store)
        self.learning = LearningRepository(store)
        self.social = SocialRepository(store)
        self.inbox = InboxRepository(store)
