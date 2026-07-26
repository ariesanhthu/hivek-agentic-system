"""Social account, publication and provider-event contracts.

MongoDB documents use snake_case.  :class:`HivekModel` exposes camelCase aliases at
the HTTP boundary, which keeps the existing Next.js convention without storing two
different shapes.  Timestamps are real datetimes so documents written by the Next.js
Mongo driver (BSON Date) and documents written by Python round-trip identically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from hivek_agent.domain.base import HivekModel

SocialPlatform = Literal["facebook", "threads", "instagram"]
SocialMode = Literal["mock", "sandbox", "live"]
InboundMode = Literal["polling", "webhook", "hybrid"]
AutoReplyMode = Literal["off", "suggestion", "low_risk"]
ChannelType = Literal["comment", "public_reply", "dm", "mention"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class SocialCapabilities(HivekModel):
    publish: bool = False
    read_comments: bool = False
    reply_comments: bool = False
    read_messages: bool = False
    reply_messages: bool = False


class SocialCredential(HivekModel):
    """Encrypted provider credential shared with the Next.js token vault.

    No API response model embeds this class.  It only crosses the repository/vault
    boundary inside the process.
    """

    credential_id: str
    workspace_id: str
    provider: SocialPlatform
    token_ciphertext: str
    token_iv: str
    token_tag: str
    key_version: int = 1
    scopes: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    status: Literal["active", "expired", "revoked", "invalid"] = "active"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SocialAccount(HivekModel):
    account_id: str
    workspace_id: str
    platform: SocialPlatform
    provider_account_id: str
    display_name: str = ""
    profile_url: str | None = None
    scopes: list[str] = Field(default_factory=list)
    credential_id: str
    status: Literal["connected", "reauthorize_required", "disconnected", "error"] = "connected"
    capabilities: SocialCapabilities = Field(default_factory=SocialCapabilities)
    connection_mode: SocialMode = "sandbox"
    auto_reply_enabled: bool = False
    webhook_status: Literal["inactive", "pending", "active", "error", "disabled"] = "inactive"
    webhook_error: str | None = None
    last_verified_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SocialPublication(HivekModel):
    publication_id: str
    workspace_id: str
    social_account_id: str
    content_asset_id: str | None = None
    platform: SocialPlatform
    platform_post_id: str
    text: str = ""
    reply_suggestions: list[str] = Field(default_factory=list)
    permalink: str | None = None
    published_at: datetime
    sync_enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class NormalizedInboundEvent(HivekModel):
    provider_event_id: str
    provider_message_id: str
    platform: SocialPlatform
    channel_type: ChannelType
    provider_account_id: str
    provider_post_id: str | None = None
    provider_parent_id: str | None = None
    provider_thread_key: str
    sender_id: str
    sender_name: str = ""
    text: str
    created_at: datetime
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class SocialWebhookEvent(HivekModel):
    webhook_event_id: str
    workspace_id: str
    provider: SocialPlatform
    provider_event_id: str
    account_id: str
    payload_hash: str
    status: Literal["received", "processed", "ignored", "failed"] = "received"
    error: str | None = None
    received_at: datetime = Field(default_factory=utc_now)
    processed_at: datetime | None = None


class SocialSyncCursor(HivekModel):
    workspace_id: str
    publication_id: str
    cursor: str | None = None
    last_provider_message_id: str | None = None
    last_event_at: datetime | None = None
    last_synced_at: datetime = Field(default_factory=utc_now)


class OutboundAction(HivekModel):
    action_id: str
    workspace_id: str
    conversation_id: str
    message_id: str | None = None
    decision_id: str | None = None
    platform: SocialPlatform
    action_type: Literal["public_reply", "message", "internal_note"]
    target_id: str
    text: str
    idempotency_key: str
    status: Literal["queued", "sending", "sent", "failed", "needs_review"] = "queued"
    provider_request_id: str | None = None
    provider_message_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
