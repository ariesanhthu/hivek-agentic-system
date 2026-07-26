"""HTTP request/response schemas.

Field names are camelCase on the wire to match the existing Next.js client, while the
Python side stays snake_case. Pydantic aliases do the translation so neither side
compromises its conventions.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from hivek_agent.domain import (
    FeedbackEventType,
    OutboundAction,
    PlatformId,
    ReplyDecision,
    SocialAccount,
    SocialConversation,
    SocialMessage,
    SocialPublication,
)


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class ChatRequest(CamelModel):
    workspace_id: str = Field(min_length=1)
    user_id: str = "anonymous"
    thread_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)
    platform: PlatformId | None = None
    platforms: list[PlatformId] | None = None
    days: int | None = Field(default=None, ge=1, le=30)
    angle: str | None = None
    goal: str | None = None
    node_id: str | None = None
    plan_id: str | None = None
    user_instruction: str | None = Field(default=None, max_length=1000)
    force_refresh: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "platform": self.platform,
            "platforms": self.platforms,
            "days": self.days,
            "angle": self.angle,
            "goal": self.goal,
            "node_id": self.node_id,
            "plan_id": self.plan_id,
            "user_instruction": self.user_instruction,
            "force_refresh": self.force_refresh,
        }
        return {key: value for key, value in payload.items() if value is not None}


class SocialSetupRequest(CamelModel):
    workspace_id: str
    user_id: str = "anonymous"
    platforms: list[str] = Field(min_length=1)


class BrandSetupRequest(CamelModel):
    workspace_id: str
    user_id: str = "anonymous"
    name: str = Field(min_length=1, max_length=200)
    tone: str = Field(min_length=1, max_length=50)


class DriveSetupRequest(CamelModel):
    workspace_id: str
    user_id: str = "anonymous"
    url: str = Field(min_length=1, max_length=2000)


class DecisionRequest(CamelModel):
    workspace_id: str
    user_id: str = "anonymous"
    decision: FeedbackEventType
    edited_text: str | None = Field(default=None, max_length=8000)
    reason: str | None = Field(default=None, max_length=1000)


class FeedbackRequest(CamelModel):
    workspace_id: str
    asset_id: str | None = None
    event_type: FeedbackEventType
    before_text: str | None = None
    after_text: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(CamelModel):
    status: Literal["ok", "degraded"]
    store_backend: str
    llm_provider: str
    store_reachable: bool
    skills_loaded: int
    warnings: list[str] = Field(default_factory=list)


# --- Social API -----------------------------------------------------------


class SocialSyncRequest(CamelModel):
    publication_ids: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=100)


class SyncErrorResponse(CamelModel):
    publication_id: str
    code: str


class SocialSyncResponse(CamelModel):
    publications_checked: int = 0
    events_found: int = 0
    messages_inserted: int = 0
    duplicates_ignored: int = 0
    decisions_created: int = 0
    auto_replies_sent: int = 0
    errors: list[SyncErrorResponse] = Field(default_factory=list)


class PublicSocialAccount(CamelModel):
    account_id: str
    platform: str
    provider_account_id: str
    display_name: str
    profile_url: str | None = None
    status: str
    capabilities: dict[str, bool]
    connection_mode: str
    auto_reply_enabled: bool
    webhook_status: str
    last_verified_at: Any | None = None

    @classmethod
    def from_domain(cls, account: SocialAccount) -> PublicSocialAccount:
        return cls(
            account_id=account.account_id,
            platform=account.platform,
            provider_account_id=account.provider_account_id,
            display_name=account.display_name,
            profile_url=account.profile_url,
            status=account.status,
            capabilities=account.capabilities.model_dump(),
            connection_mode=account.connection_mode,
            auto_reply_enabled=account.auto_reply_enabled,
            webhook_status=account.webhook_status,
            last_verified_at=account.last_verified_at,
        )


class SocialStatusResponse(CamelModel):
    social_mode: str
    inbound_mode: str
    auto_reply_mode: str
    store_backend: str
    accounts: list[PublicSocialAccount] = Field(default_factory=list)
    publications_tracked: int = 0
    last_synced_at: Any | None = None
    webhook_active: bool = False


class ConversationCapabilities(CamelModel):
    can_send_text: bool
    can_send_attachments: bool = False
    can_open_native_conversation: bool = False
    can_use_automation: bool
    permission_status: Literal["active", "read_only", "expired"]
    policy_notice: str | None = None


class ConversationEnvelope(CamelModel):
    conversation: SocialConversation
    account: PublicSocialAccount
    capabilities: ConversationCapabilities
    latest_message: SocialMessage | None = None
    reply_decision: ReplyDecision | None = None
    messages: list[SocialMessage] = Field(default_factory=list)


class ConversationListResponse(CamelModel):
    items: list[ConversationEnvelope]
    total: int


class ConversationDetailResponse(ConversationEnvelope):
    pass


class MessageListResponse(CamelModel):
    items: list[SocialMessage]


class SendConversationMessageRequest(CamelModel):
    content: str = Field(min_length=1, max_length=4000)
    mode: Literal["reply", "note"] = "reply"
    client_message_id: str = Field(min_length=1, max_length=200)


class SendConversationMessageResponse(CamelModel):
    message: SocialMessage
    outbound_action: OutboundAction


class TakeoverRequest(CamelModel):
    assignee: str = Field(default="Workspace member", max_length=200)


class ConversationMutationResponse(CamelModel):
    conversation: SocialConversation


class EditAndSendRequest(CamelModel):
    content: str = Field(min_length=1, max_length=4000)


class RejectDecisionRequest(CamelModel):
    reason: str | None = Field(default=None, max_length=1000)


class DecisionSendResponse(CamelModel):
    message: SocialMessage
    decision: ReplyDecision
    outbound_action: OutboundAction


class DecisionMutationResponse(CamelModel):
    decision: ReplyDecision


class ActivateSocialAccountRequest(CamelModel):
    workspace_id: str
    account_id: str


class ActivateSocialAccountResponse(CamelModel):
    account: PublicSocialAccount
    activated: bool
    webhook_status: str


class RegisterPublicationRequest(CamelModel):
    workspace_id: str
    publication_id: str
    social_account_id: str
    content_asset_id: str | None = None
    platform: Literal["facebook", "threads", "instagram"]
    platform_post_id: str
    text: str = Field(default="", max_length=20000)
    reply_suggestions: list[str] = Field(default_factory=list)
    permalink: str | None = None
    published_at: Any

    def to_domain(self) -> SocialPublication:
        return SocialPublication.model_validate(self.model_dump())


class RegisterPublicationResponse(CamelModel):
    publication: SocialPublication
    created: bool


class InjectMockEventRequest(CamelModel):
    publication_id: str
    sender_id: str = "mock-customer"
    sender_name: str = "Demo customer"
    text: str = Field(min_length=1, max_length=4000)


class InjectMockEventResponse(CamelModel):
    queued: bool
    provider_message_id: str


class WebhookIngestResponse(CamelModel):
    received: int
    inserted: int
    duplicates: int
    ignored: int
