"""Content contracts: plan nodes, drafts, validation results.

Field names mirror the TypeScript types already used by the Next.js client
(`src/server/ai/types/agent.types.ts`) so responses need no translation layer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from hivek_agent.domain.base import HivekModel
from hivek_agent.domain.knowledge import SourceRef, utc_now_iso

# Kept identical to the client's `PlatformId` union.
PlatformId = Literal["facebook", "threads", "tiktok"]
FunnelStage = Literal["awareness", "consideration", "conversion", "retention"]
RiskLevel = Literal["green", "amber", "red"]
ValidationDecision = Literal["approve", "revise", "human_review"]
AssetStatus = Literal["draft", "needs_review", "approved", "rejected", "scheduled", "published"]


class ContentPlanNode(HivekModel):
    """A planned post slot. The planner picks the slot; the composer writes it."""

    node_id: str
    workspace_id: str
    plan_id: str
    day_index: int
    platform: PlatformId
    funnel_stage: FunnelStage
    goal: str
    angle: str
    pillar: str | None = None
    required_fact_keys: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ContentPlan(HivekModel):
    plan_id: str
    workspace_id: str
    strategy_summary: str = ""
    days: int = 7
    platforms: list[PlatformId] = Field(default_factory=list)
    nodes: list[ContentPlanNode] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class PostDraft(HivekModel):
    """LLM output, schema-validated. Declares which facts it used and which it lacked."""

    hook: str
    body: str
    cta: str = ""
    first_comment: str = ""
    reply_suggestions: list[str] = Field(default_factory=list)
    hashtags: list[str] = Field(default_factory=list)
    # Traceability: the composer must declare its sources rather than free-associate.
    delivered_fact_ids: list[str] = Field(default_factory=list)
    missing_fact_ids: list[str] = Field(default_factory=list)
    skill_ids_used: list[str] = Field(default_factory=list)
    pattern_notes: str = ""

    @property
    def full_text(self) -> str:
        parts = [self.hook, self.body, self.cta]
        return "\n\n".join(part.strip() for part in parts if part and part.strip())


class ValidationIssue(HivekModel):
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    evidence: str | None = None


class ContentValidationResult(HivekModel):
    """Deterministic checks run first; the LLM judge only scores semantics."""

    risk_level: RiskLevel = "green"
    brand_fit_score: float = Field(ge=0, le=1, default=0.5)
    human_likeness_score: float = Field(ge=0, le=1, default=0.5)
    factual_consistency_score: float = Field(ge=0, le=1, default=0.5)
    platform_fit_score: float = Field(ge=0, le=1, default=0.5)
    sales_pressure_score: float = Field(ge=0, le=1, default=0.5)
    duplication_score: float = Field(ge=0, le=1, default=0.0)
    issues: list[ValidationIssue] = Field(default_factory=list)
    suggested_revision: str = ""
    final_decision: ValidationDecision = "approve"
    deterministic_only: bool = True

    @property
    def blocking_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


class ContentAsset(HivekModel):
    """A stored draft plus the context that produced it - required for reproducibility."""

    asset_id: str
    workspace_id: str
    node_id: str | None = None
    plan_id: str | None = None
    # The run that produced this draft. Required to trace a post back to its run, and
    # to replay an idempotent request with the asset it actually created.
    run_id: str | None = None
    platform: PlatformId
    draft: PostDraft
    validation: ContentValidationResult | None = None
    status: AssetStatus = "draft"
    citations: list[SourceRef] = Field(default_factory=list)

    # Reproducibility trio required by the blueprint.
    prompt_version: str = "v1"
    context_hash: str = ""
    model: str = "mock"

    edited_text: str | None = None
    review_note: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class CompiledContext(HivekModel):
    """The ONLY thing a composer prompt may read. Bounded by a token budget."""

    task: str
    workspace_id: str
    platform: PlatformId | None = None
    immutable_facts: list[dict[str, Any]] = Field(default_factory=list)
    brand_rules: list[dict[str, Any]] = Field(default_factory=list)
    audience_summary: dict[str, Any] = Field(default_factory=dict)
    platform_rules: dict[str, Any] = Field(default_factory=dict)
    relevant_examples: list[dict[str, Any]] = Field(default_factory=list)
    negative_memories: list[dict[str, Any]] = Field(default_factory=list)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[SourceRef] = Field(default_factory=list)
    omitted_sections: list[str] = Field(default_factory=list)
    token_budget: int = 12000
    estimated_tokens: int = 0
    context_hash: str = ""
