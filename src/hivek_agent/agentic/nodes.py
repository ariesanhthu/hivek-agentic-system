"""Graph nodes.

Each node is a plain async method taking and returning `HarnessState`. They know
nothing about LangGraph, so they are unit-testable directly and portable if the
orchestration layer ever changes.

Deterministic-first: routing tries keywords before the model, required-fact checks are
code, and the LLM is only reached for composing, planning prose and semantic judging.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from hivek_agent.agentic.context_compiler import PLATFORM_RULES, ContextCompiler
from hivek_agent.agentic.model_router import ModelRouter
from hivek_agent.agentic.skills import get_skill_registry
from hivek_agent.agentic.tools import DEFAULT_SCOPES, authorized_tools
from hivek_agent.content.composer import PROMPT_VERSION, ContentComposer
from hivek_agent.content.planner import ContentPlanner
from hivek_agent.content.validator import ContentValidator
from hivek_agent.domain import (
    ContentAsset,
    HarnessState,
    Intent,
    KnowledgeAssertion,
    MissingItem,
    NodeRun,
    PlatformId,
    SourceRef,
)
from hivek_agent.infrastructure.llm import LLMError, LLMGateway
from hivek_agent.knowledge.brand_profile import BrandProfileService
from hivek_agent.knowledge.facts import FactService
from hivek_agent.repositories import Repositories, new_id

logger = logging.getLogger(__name__)

SKILL_GUIDANCE_CHARS = 900

# Keyword routing. Accent-insensitive, mirroring the client's `resolveIntent` so the
# UI and backend agree on what a phrase means.
_INTENT_KEYWORDS: tuple[tuple[Intent, tuple[str, ...]], ...] = (
    (
        "setup",
        ("thiet lap", "setup", "bat dau nhanh", "quick start", "ket noi tai khoan", "onboard"),
    ),
    (
        "create_content_plan",
        (
            "len ke hoach",
            "ke hoach dang bai",
            "lich dang",
            "lich noi dung",
            "content plan",
            "ke hoach noi dung",
        ),
    ),
    (
        "create_post",
        (
            "viet bai",
            "tao bai",
            "soan bai",
            "viet post",
            "caption",
            "noi dung bai",
            "y tuong",
            "chien dich",
            "campaign",
        ),
    ),
    (
        "analyze_performance",
        ("phan tich", "hieu qua", "performance", "danh gia noi dung", "bao cao"),
    ),
    ("update_knowledge", ("cap nhat", "sua thong tin", "doi gia", "cap nhat du lieu")),
)

_PLATFORM_KEYWORDS: dict[str, PlatformId] = {
    "facebook": "facebook",
    "fb": "facebook",
    "threads": "threads",
    "tiktok": "tiktok",
    "tik tok": "tiktok",
}


class IntentVerdict(BaseModel):
    """Schema for the fallback LLM router."""

    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


class NodeDeps:
    """Everything the nodes need. Constructed once per process."""

    def __init__(
        self,
        repos: Repositories,
        llm: LLMGateway,
        *,
        token_budget: int = 12000,
        fast_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
    ) -> None:
        self.repos = repos
        self.llm = llm
        self.facts = FactService(repos.knowledge)
        self.brand = BrandProfileService(repos.knowledge, self.facts)
        self.compiler = ContextCompiler(default_budget=token_budget)
        self.router = ModelRouter()
        self.planner = ContentPlanner()
        self.composer = ContentComposer(llm)
        self.validator = ContentValidator(
            llm,
            judge_model=judge_model,
            judge_fallbacks=self.router.route("content_validate").fallback_chain,
        )
        self.skills = get_skill_registry()
        self.fast_model = fast_model


class HarnessNodes:
    def __init__(self, deps: NodeDeps) -> None:
        self.deps = deps

    # --- 1. authorise ---------------------------------------------------
    async def authenticate_and_authorize(self, state: HarnessState) -> HarnessState:
        # TODO(auth): scopes are currently granted per workspace membership. Wire to the
        # main backend's JWT (`hivek_access_token`) when the gateway is in place.
        state.request_payload.setdefault("scopes", list(DEFAULT_SCOPES))
        state.step_count += 1
        return state

    # --- 2. route -------------------------------------------------------
    async def route_request(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        intent = classify_intent_keywords(state.user_message)
        model_used = ""

        if intent is None and state.user_message.strip():
            try:
                route = self.deps.router.route("intent_classification")
                verdict, completion = await self.deps.llm.complete_structured(
                    system=_ROUTER_SYSTEM,
                    prompt=f"Tin nhắn người dùng: {state.user_message}",
                    schema=IntentVerdict,
                    model=route.model_name,
                    temperature=0.0,
                    max_output_tokens=route.max_output_tokens,
                    fallback_models=route.fallback_chain,
                )
                intent = verdict.intent
                model_used = completion.model
            except LLMError as exc:
                logger.warning("intent router failed, defaulting to smalltalk: %s", exc)
                state.warnings.append("Không phân loại được yêu cầu; đã chuyển sang trả lời chung.")

        state.intent = intent or "smalltalk"
        state.authorized_tools = authorized_tools(
            state.intent, state.request_payload.get("scopes", [])
        )
        state.step_count += 1
        await self._record(state, "route_request", started, model=model_used)
        return state

    # --- 3. load facts --------------------------------------------------
    async def load_knowledge(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        usable = await self.deps.facts.usable_facts(state.workspace_id)
        state.facts = {key: assertion.model_dump() for key, assertion in usable.items()}
        state.source_refs = {key: [assertion.source] for key, assertion in usable.items()}
        state.missing_items = await self.deps.brand.detect_gaps(state.workspace_id)
        state.step_count += 1
        await self._record(state, "load_knowledge", started, retrieval_ids=list(usable.keys())[:20])
        return state

    # --- 4. gate on blocking gaps ---------------------------------------
    async def validate_required_facts(self, state: HarnessState) -> HarnessState:
        """Deterministic gate. No model may decide it has enough information."""
        started = time.perf_counter()
        blocking = [item for item in state.missing_items if item.severity == "blocking"]

        if blocking:
            state.status = "needs_user_input"
            state.reply_text = _ask_for_missing(blocking)
            state.widget = _widget_for_missing(blocking[0])
            state.ui_actions = [
                {
                    "id": "open-quick-setup",
                    "label": "Hoàn tất thiết lập",
                    "kind": "intent",
                    "intent": "quick-start",
                    "variant": "primary",
                }
            ]
        state.step_count += 1
        await self._record(state, "validate_required_facts", started)
        return state

    # --- 5. compile context ---------------------------------------------
    async def compile_context(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        platform = _resolve_platform(state)
        task = "content_compose" if state.intent == "create_post" else "content_plan"

        usable = {
            key: KnowledgeAssertion.model_validate(value) for key, value in state.facts.items()
        }
        brand_profile = await self.deps.repos.knowledge.get_brand_profile(state.workspace_id)
        voice = await self.deps.repos.knowledge.get_voice_profile(state.workspace_id)
        preferences = await self.deps.repos.learning.list_preferences(state.workspace_id)
        approved = await self.deps.repos.content.list_assets(
            state.workspace_id, status="approved", limit=6
        )
        rejected = await self.deps.repos.content.list_assets(
            state.workspace_id, status="rejected", limit=3
        )

        skills = [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "guidance": skill.guidance(max_chars=SKILL_GUIDANCE_CHARS),
            }
            for skill in self.deps.skills.select_for_task(task, platform=platform)
        ]

        state.compiled_context = self.deps.compiler.compile(
            task=task,
            workspace_id=state.workspace_id,
            platform=platform,
            facts=usable,
            brand_profile=brand_profile,
            voice_profile=voice,
            preferences=preferences,
            approved_examples=approved,
            rejected_examples=rejected,
            skills=skills,
            required_fact_keys=state.request_payload.get("required_fact_keys", []),
        )
        state.step_count += 1
        await self._record(state, "compile_context", started)
        return state

    # --- 6a. plan -------------------------------------------------------
    async def create_content_plan(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        payload = state.request_payload
        platforms = _resolve_platforms(state)
        usable = {
            key: KnowledgeAssertion.model_validate(value) for key, value in state.facts.items()
        }
        brand_profile = await self.deps.repos.knowledge.get_brand_profile(state.workspace_id)
        recent_assets = await self.deps.repos.content.list_assets(state.workspace_id, limit=10)
        recent = [asset.draft.pattern_notes for asset in recent_assets]

        plan = self.deps.planner.plan(
            workspace_id=state.workspace_id,
            days=int(payload.get("days", 7)),
            platforms=platforms,
            facts=usable,
            brand_profile=brand_profile,
            recent_angles=recent,
            skill_ids=[
                skill.skill_id for skill in self.deps.skills.select_for_task("content_plan")
            ],
        )
        for node in plan.nodes:
            node.plan_id = plan.plan_id
        await self.deps.repos.content.save_plan(plan)

        state.plan = plan
        state.status = "completed"
        state.reply_text = plan.strategy_summary
        state.ui_actions = [
            {
                "id": "open-campaign-planning",
                "label": "Mở lịch nội dung",
                "kind": "href",
                "href": "/campaign-planning",
                "variant": "primary",
            },
            {
                "id": "generate-first-post",
                "label": "Viết bài đầu tiên",
                "kind": "intent",
                "intent": "create_post",
                "variant": "secondary",
            },
        ]
        state.step_count += 1
        await self._record(state, "create_content_plan", started)
        return state

    # --- 6b. compose ----------------------------------------------------
    async def generate_draft(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        context = state.compiled_context
        if context is None:
            state.status = "failed"
            state.errors.append("compile_context did not run before generate_draft")
            return state

        platform = _resolve_platform(state) or "facebook"
        angle = (
            state.request_payload.get("angle")
            or _angle_from_plan(state)
            or "Chia sẻ giá trị hữu ích"
        )
        goal = state.request_payload.get("goal") or "Tăng tương tác với khách hàng mục tiêu"

        # Idempotency: same context + prompt version returns the existing asset instead
        # of paying for a second generation.
        existing = await self.deps.repos.content.find_by_context_hash(
            state.workspace_id, context.context_hash
        )
        if existing is not None and not state.request_payload.get("force_refresh"):
            state.asset = existing
            state.draft = existing.draft
            state.validation = existing.validation
            state.status = "needs_approval" if existing.status == "needs_review" else "completed"
            await self._record(state, "generate_draft", started, cache_hit=True)
            return state

        route = self.deps.router.route("content_compose")
        try:
            draft, completion = await self.deps.composer.compose(
                context=context,
                platform=platform,
                angle=angle,
                goal=goal,
                model=route.model_name,
                temperature=route.temperature,
                max_output_tokens=route.max_output_tokens,
                user_instruction=state.request_payload.get("user_instruction"),
                fallback_models=route.fallback_chain,
            )
        except LLMError as exc:
            # The user gets a plain explanation and a next step; the diagnostic detail
            # stays in the trace. An empty reply with status=failed tells them nothing.
            logger.error("draft generation failed run=%s: %s", state.run_id, exc)
            state.status = "failed"
            state.errors.append(str(exc)[:500])
            state.reply_text = _llm_failure_message(exc)
            state.ui_actions = [
                {
                    "id": "retry-generate",
                    "label": "Thử lại",
                    "kind": "intent",
                    "intent": "create_post",
                    "variant": "primary",
                }
            ]
            await self._record(state, "generate_draft", started, error_class=type(exc).__name__)
            return state

        state.draft = draft
        state.asset = ContentAsset(
            asset_id=new_id("asset"),
            workspace_id=state.workspace_id,
            node_id=state.request_payload.get("node_id"),
            plan_id=state.request_payload.get("plan_id"),
            run_id=state.run_id,
            platform=platform,
            draft=draft,
            citations=context.citations,
            prompt_version=PROMPT_VERSION,
            context_hash=context.context_hash,
            model=completion.model,
        )
        state.step_count += 1
        await self._record(
            state,
            "generate_draft",
            started,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cache_hit=completion.cache_hit,
        )
        return state

    # --- 7. validate ----------------------------------------------------
    async def validate_draft(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        if state.draft is None or state.asset is None:
            return state

        voice = await self.deps.repos.knowledge.get_voice_profile(state.workspace_id)
        result = await self.deps.validator.validate(
            state.draft,
            platform=state.asset.platform,
            context=state.compiled_context,
            voice=voice,
        )
        state.validation = result
        state.asset.validation = result

        # Everything stops at a human. Nothing here can publish.
        state.asset.status = "needs_review"
        state.approval_required = True
        state.status = "needs_approval"
        # Emitted verbatim as the `approval.required` SSE payload, so camelCase to
        # match every other field the client reads.
        state.approval_payload = {
            "assetId": state.asset.asset_id,
            "riskLevel": result.risk_level,
            "decision": result.final_decision,
        }
        await self.deps.repos.content.save_asset(state.asset)
        await self.deps.repos.runs.audit(
            state.workspace_id,
            "draft.created",
            {
                "asset_id": state.asset.asset_id,
                "risk_level": result.risk_level,
                "run_id": state.run_id,
            },
        )

        state.reply_text = _review_message(result, state.asset)
        state.ui_actions = _review_actions(state.asset.asset_id)
        state.step_count += 1
        await self._record(state, "validate_draft", started)
        return state

    # --- 8. setup / smalltalk -------------------------------------------
    async def handle_setup(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        profile = await self.deps.brand.build_profile(state.workspace_id)
        blocking = [item for item in profile.missing_items if item.severity == "blocking"]

        if blocking:
            state.status = "needs_user_input"
            state.reply_text = _ask_for_missing(blocking)
            state.widget = _widget_for_missing(blocking[0])
        else:
            state.status = "completed"
            state.reply_text = (
                f"Workspace đã sẵn sàng (mức độ hoàn thiện {profile.readiness_score:.0%}). "
                "Mình có thể lên kế hoạch nội dung hoặc viết bài đầu tiên."
            )
            state.widget = {"type": "setup-complete"}
            state.ui_actions = [
                {
                    "id": "complete-plan-posts",
                    "label": "Lên kế hoạch đăng bài",
                    "kind": "intent",
                    "intent": "plan-posts",
                    "variant": "primary",
                },
                {
                    "id": "complete-campaign-ideas",
                    "label": "Viết bài đầu tiên",
                    "kind": "intent",
                    "intent": "campaign-ideas",
                    "variant": "secondary",
                },
            ]

        state.missing_items = profile.missing_items
        state.step_count += 1
        await self._record(state, "handle_setup", started)
        return state

    async def handle_smalltalk(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        state.status = "completed"
        state.reply_text = (
            "Mình có thể giúp bạn thiết lập workspace, lên kế hoạch nội dung, "
            "viết bài theo từng kênh và học từ phản hồi của bạn. Bạn muốn bắt đầu từ đâu?"
        )
        state.ui_actions = [
            {
                "id": "suggest-plan-posts",
                "label": "Lên kế hoạch đăng bài",
                "kind": "intent",
                "intent": "plan-posts",
                "variant": "primary",
            },
            {
                "id": "suggest-campaign-ideas",
                "label": "Gợi ý chiến dịch",
                "kind": "intent",
                "intent": "campaign-ideas",
                "variant": "secondary",
            },
        ]
        state.step_count += 1
        await self._record(state, "handle_smalltalk", started)
        return state

    async def analyze_performance(self, state: HarnessState) -> HarnessState:
        started = time.perf_counter()
        feedback = await self.deps.repos.learning.list_feedback(state.workspace_id, limit=50)
        assets = await self.deps.repos.content.list_assets(state.workspace_id, limit=50)
        counts: dict[str, int] = {}
        for event in feedback:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1

        approved = counts.get("approve", 0)
        total = sum(counts.values())
        preferences = await self.deps.repos.learning.list_preferences(
            state.workspace_id, active_only=True
        )

        if total == 0:
            state.reply_text = (
                "Chưa có đủ phản hồi để phân tích. Hãy duyệt hoặc chỉnh sửa vài bài trước, "
                "mình sẽ học từ đó."
            )
        else:
            rules = ", ".join(f"{p.rule_type}={p.rule_value}" for p in preferences[:4]) or "chưa có"
            edits = counts.get("edit", 0)
            rejects = counts.get("reject", 0)
            state.reply_text = (
                f"Đã ghi nhận {total} phản hồi trên {len(assets)} bài "
                f"(duyệt {approved}, sửa {edits}, từ chối {rejects}). "
                f"Quy tắc giọng văn đã ổn định: {rules}."
            )
        state.status = "completed"
        state.step_count += 1
        await self._record(state, "analyze_performance", started)
        return state

    # --- telemetry ------------------------------------------------------
    async def _record(
        self,
        state: HarnessState,
        node_name: str,
        started: float,
        *,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_hit: bool = False,
        error_class: str | None = None,
        retrieval_ids: list[str] | None = None,
    ) -> None:
        """One row per node. Never carries prompts, secrets or personal data."""
        await self.deps.repos.runs.record_node_run(
            NodeRun(
                node_run_id=new_id("noderun"),
                run_id=state.run_id,
                workspace_id=state.workspace_id,
                graph_name="chat",
                node_name=node_name,
                model=model,
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit=cache_hit,
                tool_calls=state.authorized_tools,
                retrieval_ids=retrieval_ids or [],
                error_class=error_class,
            )
        )


# --- pure helpers -------------------------------------------------------


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return stripped.replace("đ", "d")


def classify_intent_keywords(message: str) -> Intent | None:
    """Deterministic router. Returns None when unsure so the caller can escalate."""
    folded = fold(message)
    if not folded.strip():
        return None
    for intent, keywords in _INTENT_KEYWORDS:
        if any(keyword in folded for keyword in keywords):
            return intent
    return None


def _resolve_platform(state: HarnessState) -> PlatformId | None:
    explicit = state.request_payload.get("platform")
    if explicit in PLATFORM_RULES:
        return explicit  # type: ignore[return-value]
    folded = fold(state.user_message)
    for keyword, platform in _PLATFORM_KEYWORDS.items():
        if keyword in folded:
            return platform
    return None


def _resolve_platforms(state: HarnessState) -> list[PlatformId]:
    requested = state.request_payload.get("platforms")
    if isinstance(requested, list) and requested:
        return [item for item in requested if item in PLATFORM_RULES]
    single = _resolve_platform(state)
    return [single] if single else ["facebook", "threads"]


def _angle_from_plan(state: HarnessState) -> str | None:
    if state.plan and state.plan.nodes:
        return state.plan.nodes[0].angle
    return None


def _ask_for_missing(blocking: list[MissingItem]) -> str:
    """Say where we looked before asking - the blueprint requires it."""
    first = blocking[0]
    searched = ", ".join(first.searched_sources[:3]) or "chưa có nguồn nào"
    lines = [
        f"Mình cần thêm {len(blocking)} thông tin bắt buộc trước khi tiếp tục.",
        f"Đã tìm trong: {searched}.",
        "",
    ]
    lines.extend(f"• {item.reason} → {item.suggested_action}" for item in blocking[:3])
    return "\n".join(lines)


def _llm_failure_message(exc: Exception) -> str:
    """Turn a provider error into something the user can act on."""
    blob = str(exc).lower()
    if "limit: 0" in blob or "resource_exhausted" in blob or "quota" in blob:
        return (
            "Hạn mức API của mô hình ngôn ngữ đã hết (quota free tier). "
            "Kế hoạch nội dung và dữ kiện vẫn được lưu đầy đủ; bạn có thể thử lại sau "
            "khi hạn mức đặt lại, hoặc đặt AI_AGENT_PROVIDER=mock để chạy thử luồng."
        )
    if "timeout" in blob or "deadline" in blob:
        return "Mô hình phản hồi quá lâu. Bạn thử lại giúp mình nhé."
    return (
        "Chưa tạo được bản nháp do lỗi từ nhà cung cấp mô hình. "
        "Dữ liệu của bạn không bị ảnh hưởng."
    )


def _widget_for_missing(item: MissingItem) -> dict[str, Any] | None:
    """Map a gap onto the widget the client already renders."""
    mapping = {
        "brand.channels": "social-connect",
        "brand.name": "brand-form",
        "brand.tone": "brand-form",
        "brand.resource_url": "drive-form",
    }
    widget = mapping.get(item.field)
    return {"type": widget} if widget else {"type": "setup-overview"}


def _review_message(result: Any, asset: ContentAsset) -> str:
    counts = len(asset.draft.delivered_fact_ids)
    header = {
        "green": "Bản nháp đã qua kiểm tra tự động và đang chờ bạn duyệt.",
        "amber": "Bản nháp có vài điểm cần bạn xem lại trước khi duyệt.",
        "red": "Bản nháp có vấn đề bắt buộc phải sửa; mình không tự đăng.",
    }[result.risk_level]

    lines = [header, "", asset.draft.full_text, ""]
    if counts:
        lines.append(f"Đã dùng {counts} dữ kiện có nguồn.")
    if asset.draft.missing_fact_ids:
        lines.append("Còn thiếu: " + ", ".join(asset.draft.missing_fact_ids[:3]))
    problems = [issue for issue in result.issues if issue.severity in ("error", "warning")]
    if problems:
        lines.append("")
        lines.extend(f"⚠ {issue.message}" for issue in problems[:4])
    return "\n".join(lines)


def _review_actions(asset_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"approve:{asset_id}",
            "label": "Duyệt bài",
            "kind": "decision",
            "decision": "approve",
            "variant": "primary",
            "payload": {"assetId": asset_id},
        },
        {
            "id": f"regenerate:{asset_id}",
            "label": "Viết lại",
            "kind": "decision",
            "decision": "regenerate",
            "variant": "secondary",
            "payload": {"assetId": asset_id},
        },
        {
            "id": f"reject:{asset_id}",
            "label": "Từ chối",
            "kind": "decision",
            "decision": "reject",
            "variant": "secondary",
            "payload": {"assetId": asset_id},
        },
    ]


_ROUTER_SYSTEM = (
    "Bạn là bộ định tuyến yêu cầu của HIVE-K. Phân loại tin nhắn vào đúng một intent: "
    "setup, update_knowledge, create_content_plan, create_post, analyze_performance, smalltalk. "
    "Chỉ trả JSON. Không gọi công cụ, không trả lời người dùng."
)


def source_from_user(source_id: str = "user/chat") -> SourceRef:
    """User-supplied data is the highest-precedence source and starts approved."""
    return SourceRef(source_id=source_id, source_type="user_input", confidence=0.95, approved=True)


_WORD = re.compile(r"\w+", re.UNICODE)
