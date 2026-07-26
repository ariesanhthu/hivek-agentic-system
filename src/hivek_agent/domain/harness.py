"""Harness state, run lifecycle, and the response contract the frontend consumes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from hivek_agent.domain.base import HivekModel
from hivek_agent.domain.content import (
    CompiledContext,
    ContentAsset,
    ContentPlan,
    ContentValidationResult,
    PostDraft,
)
from hivek_agent.domain.knowledge import MissingItem, SourceRef, utc_now_iso

Intent = Literal[
    "setup",
    "update_knowledge",
    "create_content_plan",
    "create_post",
    "analyze_performance",
    "smalltalk",
]

RunStatus = Literal[
    "running",
    "completed",
    "partial",
    "needs_user_input",
    "needs_approval",
    "failed",
    "cancelled",
]

ToolRisk = Literal["read", "write", "external_side_effect"]


class ToolPolicy(HivekModel):
    """Registry entry. A tool the caller lacks scopes for is never shown to the model."""

    tool_name: str
    risk: ToolRisk
    required_scopes: list[str] = Field(default_factory=list)
    requires_human_approval: bool = False
    timeout_seconds: int = 30
    max_retries: int = 2
    idempotent: bool = True
    description: str = ""


class RunPolicy(HivekModel):
    """Hard ceilings for one run. Prevents unbounded loops and cost blowouts."""

    max_steps: int = 24
    max_tool_calls: int = 32
    token_budget: int = 12000
    allow_llm: bool = True


class ModelRoute(HivekModel):
    task: str
    model_tier: Literal["local", "fast", "reasoning", "creative"]
    model_name: str
    # Tried in order when the primary model is unavailable to this API key (quota 0,
    # model not enabled, region restriction). Degrading to a cheaper model beats
    # failing the user's request.
    fallback_chain: list[str] = Field(default_factory=list)
    max_output_tokens: int = 1800
    temperature: float = 0.7
    cacheable: bool = True


class HarnessState(BaseModel):
    """LangGraph state. Holds IDs and structured data - never tokens or file bodies."""

    run_id: str
    workspace_id: str
    user_id: str
    thread_id: str

    intent: Intent | None = None
    user_message: str = ""
    request_payload: dict[str, Any] = Field(default_factory=dict)
    authorized_tools: list[str] = Field(default_factory=list)

    facts: dict[str, Any] = Field(default_factory=dict)
    source_refs: dict[str, list[SourceRef]] = Field(default_factory=dict)
    missing_items: list[MissingItem] = Field(default_factory=list)

    retrieved_memories: list[dict[str, Any]] = Field(default_factory=list)
    graph_context: list[dict[str, Any]] = Field(default_factory=list)
    compiled_context: CompiledContext | None = None

    plan: ContentPlan | None = None
    draft: PostDraft | None = None
    asset: ContentAsset | None = None
    validation: ContentValidationResult | None = None

    approval_required: bool = False
    approval_payload: dict[str, Any] | None = None

    reply_text: str = ""
    ui_actions: list[dict[str, Any]] = Field(default_factory=list)
    widget: dict[str, Any] | None = None

    status: RunStatus = "running"
    step_count: int = 0
    tool_call_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    audit_event_ids: list[str] = Field(default_factory=list)
    trace_id: str = ""


class UIAction(HivekModel):
    """Rendered as a button by the client. `kind` mirrors the existing AiChatAction union."""

    id: str
    label: str
    kind: Literal["intent", "href", "decision"] = "intent"
    intent: str | None = None
    href: str | None = None
    decision: str | None = None
    variant: Literal["primary", "secondary"] = "secondary"
    payload: dict[str, Any] = Field(default_factory=dict)


class RunEvent(HivekModel):
    """One SSE frame. Event names match the blueprint's frontend contract."""

    event: Literal[
        "run.started",
        "step.started",
        "step.progress",
        "message.delta",
        "fact.extracted",
        "conflict.detected",
        "input.required",
        "approval.required",
        "draft.created",
        "validation.completed",
        "run.completed",
        "run.failed",
    ]
    run_id: str
    seq: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=utc_now_iso)


class AgentRunSummary(HivekModel):
    """Mirrors the client's `AgentRunSummary` TypeScript type field-for-field."""

    run_id: str
    workflow_name: str
    agent_name: str
    input_summary: str = ""
    output_summary: str = ""
    model: str = "mock"
    prompt_version: str = "v1"
    latency_ms: int = 0
    status: Literal["success", "failed"] = "success"
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False


class AgentRun(HivekModel):
    run_id: str
    workspace_id: str
    user_id: str
    thread_id: str
    graph_name: str = "chat"
    intent: Intent | None = None
    status: RunStatus = "running"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    idempotency_key: str | None = None
    error: str | None = None


class NodeRun(HivekModel):
    """Per-node observability record. Redacted - never holds raw prompts or secrets."""

    node_run_id: str
    run_id: str
    workspace_id: str
    graph_name: str
    node_name: str
    prompt_version: str = "v1"
    model: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False
    tool_calls: list[str] = Field(default_factory=list)
    retrieval_ids: list[str] = Field(default_factory=list)
    error_class: str | None = None
    retry_count: int = 0
    at: str = Field(default_factory=utc_now_iso)


class HarnessResponse(HivekModel):
    """What POST /v1/chat/messages returns. The frontend needs nothing else."""

    run_id: str
    thread_id: str
    status: RunStatus
    reply: str = ""
    widget: dict[str, Any] | None = None
    next_actions: list[UIAction] = Field(default_factory=list)
    missing_items: list[MissingItem] = Field(default_factory=list)
    asset: ContentAsset | None = None
    plan: ContentPlan | None = None
    citations: list[SourceRef] = Field(default_factory=list)
    progress: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    agent_run: AgentRunSummary | None = None
    trace_id: str = ""
