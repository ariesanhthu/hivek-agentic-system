"""Knowledge contracts: facts carry provenance, confidence and version.

Blueprint rule: every fact used in a post must be traceable to a source, and a
confirmed fact is never silently overwritten.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from hivek_agent.domain.base import HivekModel

SourceType = Literal[
    "user_input",
    "drive_file",
    "website",
    "social_api",
    "approved_memory",
    "performance",
    "system_inference",
]

# Ordered worst -> best. Index position is the precedence score, so a source can
# never be promoted above a user confirmation by an extractor bug.
SOURCE_PRECEDENCE: tuple[SourceType, ...] = (
    "system_inference",
    "performance",
    "website",
    "drive_file",
    "social_api",
    "approved_memory",
    "user_input",
)

ApprovalStatus = Literal["candidate", "confirmed", "superseded", "conflict", "rejected"]
Severity = Literal["blocking", "quality", "optional"]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def source_rank(source_type: SourceType) -> int:
    try:
        return SOURCE_PRECEDENCE.index(source_type)
    except ValueError:
        return 0


class SourceRef(HivekModel):
    """Where a fact came from. Never holds file contents or tokens - IDs only."""

    source_id: str
    source_type: SourceType
    version_id: str | None = None
    observed_at: str = Field(default_factory=utc_now_iso)
    confidence: float = Field(ge=0, le=1, default=0.5)
    approved: bool = False
    excerpt: str | None = Field(
        default=None,
        max_length=280,
        description="Short quote for UI attribution. Truncated; never the full document.",
    )


class KnowledgeAssertion(HivekModel):
    """A single subject-predicate-object fact with provenance and lifecycle."""

    assertion_id: str
    workspace_id: str
    subject_id: str
    predicate: str
    # `list` is included because some facts are genuinely multi-valued (the channels a
    # brand posts on, for example) rather than one scalar.
    object_value: str | float | bool | list[Any] | dict[str, Any]

    source: SourceRef
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.5)
    approval_status: ApprovalStatus = "candidate"
    extractor_version: str = "v1"
    supersedes_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def key(self) -> str:
        """Identity of the *claim*, independent of which source asserted it."""
        return f"{self.subject_id}::{self.predicate}"

    @property
    def is_usable(self) -> bool:
        """Only confirmed or reasonably-confident candidates may reach a draft."""
        if self.approval_status in ("superseded", "rejected", "conflict"):
            return False
        if self.approval_status == "confirmed":
            return True
        return self.confidence >= 0.6


class KnowledgeConflict(HivekModel):
    """Two live assertions on the same key. Never auto-resolved when both are strong."""

    conflict_id: str
    workspace_id: str
    key: str
    assertion_ids: list[str]
    reason: str
    detected_at: str = Field(default_factory=utc_now_iso)
    resolved: bool = False
    resolved_assertion_id: str | None = None


class MissingItem(HivekModel):
    """A gap the system found. Must say where it looked before asking the user."""

    field: str
    severity: Severity
    reason: str
    searched_sources: list[str] = Field(default_factory=list)
    suggested_action: str
    can_infer: bool = False
    ui_target: dict[str, Any] | None = None


class GraphEdge(HivekModel):
    """Typed relation between entities. Traversed via Mongo $graphLookup."""

    edge_id: str
    workspace_id: str
    from_id: str
    to_id: str
    edge_type: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source: SourceRef | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class KnowledgeEntity(HivekModel):
    entity_id: str
    workspace_id: str
    entity_type: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class BrandVoiceProfile(HivekModel):
    """Structured voice - not one long prompt string.

    Rules are learned from edits and only become `stable` after repetition.
    """

    workspace_id: str
    version: int = 1
    tone: str | None = None
    sentence_length_range: tuple[int, int] = (8, 24)
    preferred_openings: list[str] = Field(default_factory=list)
    avoided_openings: list[str] = Field(default_factory=list)
    preferred_cta_types: list[str] = Field(default_factory=list)
    banned_phrases: list[str] = Field(default_factory=list)
    emoji_policy: dict[str, Any] = Field(default_factory=lambda: {"max_per_post": 3})
    pronoun_rules: dict[str, str] = Field(default_factory=dict)
    platform_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    confidence_by_rule: dict[str, float] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)


class BrandOperatingProfile(HivekModel):
    """The 'brand snapshot' shown to the user after setup."""

    workspace_id: str
    version: int = 1
    identity: dict[str, Any] = Field(default_factory=dict)
    products: list[dict[str, Any]] = Field(default_factory=list)
    audiences: list[dict[str, Any]] = Field(default_factory=list)
    offers: list[dict[str, Any]] = Field(default_factory=list)
    approved_claims: list[dict[str, Any]] = Field(default_factory=list)
    blocked_claims: list[dict[str, Any]] = Field(default_factory=list)
    faq: list[dict[str, Any]] = Field(default_factory=list)
    channel_roles: list[dict[str, Any]] = Field(default_factory=list)
    missing_items: list[MissingItem] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    source_coverage: dict[str, Any] = Field(default_factory=dict)
    readiness_score: float = 0.0
    updated_at: str = Field(default_factory=utc_now_iso)
