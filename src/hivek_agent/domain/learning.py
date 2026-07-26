"""Learning contracts: feedback, edit analysis, and the memory lifecycle.

Blueprint rule: a single edit never becomes a permanent rule. A preference must be
observed repeatedly (or explicitly pinned) before it reaches `stable`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from hivek_agent.domain.base import HivekModel
from hivek_agent.domain.knowledge import utc_now_iso

# Kept identical to the client's `FeedbackEventType` union.
FeedbackEventType = Literal[
    "approve",
    "reject",
    "edit",
    "regenerate",
    "publish",
    "pin_as_good",
    "mark_too_ai",
    "mark_wrong_fact",
]

# candidate -> repeated -> stable -> deprecated/rejected
MemoryStatus = Literal["candidate", "repeated", "stable", "deprecated", "rejected"]

PREFERENCE_PROMOTION_THRESHOLD = 2
"""Observations required before a candidate becomes `repeated`, then `stable`."""


class FeedbackEvent(HivekModel):
    feedback_id: str
    workspace_id: str
    asset_id: str | None = None
    node_id: str | None = None
    run_id: str | None = None
    event_type: FeedbackEventType
    before_text: str | None = None
    after_text: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class FeatureDelta(HivekModel):
    """Deterministic diff between before/after text. No LLM involved."""

    length_delta_ratio: float = 0.0
    sentence_count_delta: int = 0
    emoji_delta: int = 0
    question_ratio_delta: float = 0.0
    exclamation_delta: int = 0
    removed_phrases: list[str] = Field(default_factory=list)
    added_phrases: list[str] = Field(default_factory=list)
    structural_changes: list[str] = Field(default_factory=list)


class PreferenceCandidate(HivekModel):
    """One inferred voice rule, with the evidence that produced it."""

    preference_id: str
    workspace_id: str
    rule_type: str
    rule_value: str
    scope: Literal["global", "platform"] = "global"
    platform: str | None = None
    status: MemoryStatus = "candidate"
    observation_count: int = 1
    confidence: float = Field(ge=0, le=1, default=0.3)
    evidence_asset_ids: list[str] = Field(default_factory=list)
    explicit: bool = False
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @property
    def key(self) -> str:
        return f"{self.scope}:{self.platform or '*'}:{self.rule_type}:{self.rule_value}"

    @property
    def is_active(self) -> bool:
        """Only `repeated`/`stable` rules are allowed to steer generation."""
        return self.status in ("repeated", "stable")


class EditLearningEvent(HivekModel):
    event_id: str
    workspace_id: str
    asset_id: str
    before_text: str
    after_text: str
    feature_delta: FeatureDelta
    inferred_preferences: list[PreferenceCandidate] = Field(default_factory=list)
    explicit_reason: str | None = None
    confidence: float = Field(ge=0, le=1, default=0.3)
    created_at: str = Field(default_factory=utc_now_iso)


class PerformanceEvent(HivekModel):
    event_id: str
    workspace_id: str
    asset_id: str
    platform: str
    collected_at: str = Field(default_factory=utc_now_iso)
    horizon: Literal["1h", "24h", "7d"] = "24h"
    raw_metrics: dict[str, float] = Field(default_factory=dict)
    normalized_metrics: dict[str, float] = Field(default_factory=dict)
