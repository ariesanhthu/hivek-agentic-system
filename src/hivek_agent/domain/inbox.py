"""Unified inbox and reply-decision contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from hivek_agent.domain.base import HivekModel
from hivek_agent.domain.social import ChannelType, SocialPlatform, utc_now


class SocialConversation(HivekModel):
    conversation_id: str
    workspace_id: str
    social_account_id: str
    platform: SocialPlatform
    channel_type: ChannelType
    provider_thread_key: str
    provider_user_id: str = ""
    customer_name: str = ""
    customer_username: str = ""
    publication_id: str | None = None
    source_post_id: str | None = None
    source_context: str = ""
    status: Literal["open", "needs_human", "waiting_for_customer", "snoozed", "resolved"] = "open"
    priority: Literal["normal", "high", "urgent"] = "normal"
    handling_mode: Literal["limited_auto", "suggestion_only", "human"] = "suggestion_only"
    ai_state: Literal[
        "active",
        "suggestion_only",
        "paused_by_user",
        "blocked_missing_data",
        "blocked_conflict",
        "handoff_requested",
    ] = "suggestion_only"
    assignee: str = ""
    tags: list[str] = Field(default_factory=list)
    unread_count: int = 0
    last_message_at: datetime
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SocialMessage(HivekModel):
    message_id: str
    workspace_id: str
    conversation_id: str
    platform: SocialPlatform
    provider_message_id: str
    direction: Literal["inbound", "outbound", "internal"]
    author_type: Literal[
        "customer", "ai_agent", "current_user", "workspace_member", "internal_note", "system"
    ]
    author_name: str = ""
    text: str
    normalized_text: str = ""
    created_at: datetime
    delivery_status: Literal["sent", "delivered", "sending", "failed"] = "sent"
    client_message_id: str | None = None
    is_automated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplyEvidence(HivekModel):
    source_type: Literal["post_reply_suggestion", "confirmed_fact", "approved_reply", "rule"]
    source_id: str
    excerpt: str
    score: float = Field(ge=0, le=1, default=0)


class ReplyDecision(HivekModel):
    decision_id: str
    workspace_id: str
    conversation_id: str
    message_id: str
    intent: str
    risk_labels: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    action: Literal["suggestion", "auto_reply", "human_handoff", "ignore"]
    suggested_text: str | None = None
    evidence: list[ReplyEvidence] = Field(default_factory=list)
    model_used: str = "deterministic"
    status: Literal["pending", "approved", "edited", "sent", "rejected"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
