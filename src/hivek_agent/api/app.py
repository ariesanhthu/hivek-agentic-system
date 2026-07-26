"""FastAPI application.

Routes are intentionally thin: validate input, call the service, return a typed model.
No prompts, no model calls, no business rules live here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse

from hivek_agent.agentic.graph import build_chat_graph, build_checkpointer
from hivek_agent.agentic.nodes import NodeDeps
from hivek_agent.api.schemas import (
    ActivateSocialAccountRequest,
    ActivateSocialAccountResponse,
    BrandSetupRequest,
    ChatRequest,
    ConversationCapabilities,
    ConversationDetailResponse,
    ConversationEnvelope,
    ConversationListResponse,
    ConversationMutationResponse,
    DecisionMutationResponse,
    DecisionRequest,
    DecisionSendResponse,
    DriveSetupRequest,
    EditAndSendRequest,
    FeedbackRequest,
    HealthResponse,
    InjectMockEventRequest,
    InjectMockEventResponse,
    MessageListResponse,
    PublicSocialAccount,
    RegisterPublicationRequest,
    RegisterPublicationResponse,
    RejectDecisionRequest,
    SendConversationMessageRequest,
    SendConversationMessageResponse,
    SocialSetupRequest,
    SocialStatusResponse,
    SocialSyncRequest,
    SocialSyncResponse,
    TakeoverRequest,
    WebhookIngestResponse,
)
from hivek_agent.api.security import (
    enforce_internal_workspace,
    require_internal_access,
    require_social_access,
)
from hivek_agent.config import get_settings
from hivek_agent.domain import HarnessResponse, NormalizedInboundEvent
from hivek_agent.infrastructure.llm import build_llm
from hivek_agent.infrastructure.social import ProviderAPIError
from hivek_agent.infrastructure.store import build_store
from hivek_agent.repositories import Repositories, new_id
from hivek_agent.service import AgenticService
from hivek_agent.social.runtime import SocialRuntime
from hivek_agent.social.webhook_service import WebhookVerificationError

logger = logging.getLogger(__name__)

SSE_KEEPALIVE_SECONDS = 15
SSE_TIMEOUT_SECONDS = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    store = await build_store(settings)
    llm = build_llm(settings)
    repos = Repositories(store)
    deps = NodeDeps(
        repos,
        llm,
        token_budget=settings.default_token_budget,
        fast_model=settings.gemini_model_fast,
        judge_model=settings.gemini_model_validator,
    )
    checkpointer = await build_checkpointer(settings)
    graph = build_chat_graph(deps, checkpointer=checkpointer)

    app.state.settings = settings
    app.state.store = store
    app.state.service = AgenticService(deps, graph, repos)
    app.state.social = SocialRuntime(repos, llm, settings)
    app.state.internal_request_ids = {}
    app.state.internal_request_lock = asyncio.Lock()

    logger.info(
        "HIVE-K agentic system ready | store=%s llm=%s skills=%d",
        store.backend_name,
        llm.provider_name,
        len(deps.skills.list_skills()),
    )
    try:
        yield
    finally:
        await app.state.social.close()
        await store.close()


def get_service(request: Request) -> AgenticService:
    return request.app.state.service


def get_social_runtime(request: Request) -> SocialRuntime:
    return request.app.state.social


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="HIVE-K Agentic System",
        version="0.1.0",
        description="Stateful content operations harness for multi-account social workspaces.",
        lifespan=lifespan,
    )

    # The Next.js client calls this service from the browser, so CORS is required.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(_: Request, exc: ValidationError) -> JSONResponse:
        # A domain model rejecting a write is a contract bug, not a transient fault.
        # Surface it loudly: an opaque 500 here once looked like "the field just did
        # not save", which is far harder to diagnose than a 422 naming the field.
        logger.error("domain validation failed: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": "payload rejected by domain contract",
                "errors": [
                    {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
                    for err in exc.errors()
                ][:10],
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(LookupError)
    async def lookup_error_handler(_: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderAPIError)
    async def provider_error_handler(_: Request, exc: ProviderAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": "social provider request failed", "code": exc.code},
        )

    # --- health ---------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(request: Request) -> HealthResponse:
        store = request.app.state.store
        service: AgenticService = request.app.state.service
        warnings: list[str] = []

        try:
            reachable = await store.ping()
        except Exception as exc:
            reachable = False
            warnings.append(f"store unreachable: {type(exc).__name__}")

        if store.backend_name == "memory":
            warnings.append("Using in-memory store - data will not persist across restarts.")
        if service.deps.llm.provider_name == "mock":
            warnings.append("Using mock LLM - set GEMINI_API_KEY for real generation.")

        return HealthResponse(
            status="ok" if reachable else "degraded",
            store_backend=store.backend_name,
            llm_provider=service.deps.llm.provider_name,
            store_reachable=reachable,
            skills_loaded=len(service.deps.skills.list_skills()),
            warnings=warnings,
        )

    # --- chat -----------------------------------------------------------
    @app.post("/v1/chat/messages", response_model=HarnessResponse, tags=["chat"])
    async def send_message(
        body: ChatRequest,
        request: Request,
        service: AgenticService = Depends(get_service),
    ) -> HarnessResponse:
        return await service.send_message(
            workspace_id=body.workspace_id,
            user_id=body.user_id,
            thread_id=body.thread_id,
            message=body.message,
            payload=body.to_payload(),
            idempotency_key=request.headers.get("Idempotency-Key"),
        )

    @app.get("/v1/runs/{run_id}/events", tags=["chat"])
    async def stream_run_events(
        run_id: str,
        workspace_id: str,
        request: Request,
        service: AgenticService = Depends(get_service),
    ) -> EventSourceResponse:
        """SSE stream for one run.

        Replays persisted events first so a client that connects after the run started
        still sees the whole story, then follows live ones.
        """
        queue = service.events.subscribe(run_id)

        async def publisher():
            try:
                for event in await service.repos.runs.list_events(workspace_id, run_id):
                    yield {"event": event.event, "data": event.model_dump_json()}

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_SECONDS)
                    except TimeoutError:
                        yield {"event": "ping", "data": "{}"}
                        continue
                    if event is None:
                        break
                    yield {"event": event.event, "data": event.model_dump_json()}
            finally:
                service.events.unsubscribe(run_id, queue)

        return EventSourceResponse(publisher())

    @app.get("/v1/runs/{run_id}", tags=["chat"])
    async def get_run(
        run_id: str, workspace_id: str, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        run = await service.repos.runs.get_run(workspace_id, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        nodes = await service.repos.runs.list_node_runs(workspace_id, run_id)
        return {
            "run": run.model_dump(by_alias=True),
            "nodes": [item.model_dump(by_alias=True) for item in nodes],
            "total_latency_ms": sum(item.latency_ms for item in nodes),
            "total_tokens": sum(item.input_tokens + item.output_tokens for item in nodes),
        }

    # --- setup (mirrors the client's three chat widgets) ------------------
    @app.post("/v1/setup/social", tags=["setup"])
    async def setup_social(
        body: SocialSetupRequest, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        return await service.record_setup_step(
            workspace_id=body.workspace_id,
            step="social",
            value=body.platforms,
            user_id=body.user_id,
        )

    @app.post("/v1/setup/brand", tags=["setup"])
    async def setup_brand(
        body: BrandSetupRequest, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        await service.record_setup_step(
            workspace_id=body.workspace_id,
            step="brand",
            value=body.name,
            user_id=body.user_id,
        )
        return await service.record_setup_step(
            workspace_id=body.workspace_id,
            step="tone",
            value=body.tone,
            user_id=body.user_id,
        )

    @app.post("/v1/setup/drive", tags=["setup"])
    async def setup_drive(
        body: DriveSetupRequest, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        return await service.record_setup_step(
            workspace_id=body.workspace_id,
            step="drive",
            value=body.url,
            user_id=body.user_id,
        )

    # --- knowledge -------------------------------------------------------
    @app.get("/v1/workspaces/{workspace_id}/brand-profile", tags=["knowledge"])
    async def brand_profile(
        workspace_id: str, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        profile = await service.deps.brand.build_profile(workspace_id)
        return profile.model_dump(by_alias=True)

    @app.get("/v1/workspaces/{workspace_id}/missing-items", tags=["knowledge"])
    async def missing_items(
        workspace_id: str, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        gaps = await service.deps.brand.detect_gaps(workspace_id)
        return {"items": [gap.model_dump(by_alias=True) for gap in gaps]}

    @app.post("/v1/workspaces/{workspace_id}/facts/{assertion_id}/confirm", tags=["knowledge"])
    async def confirm_fact(
        workspace_id: str, assertion_id: str, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        assertion = await service.deps.facts.confirm_fact(workspace_id, assertion_id)
        if assertion is None:
            raise HTTPException(status_code=404, detail="assertion not found")
        return assertion.model_dump(by_alias=True)

    @app.get("/v1/workspaces/{workspace_id}/conflicts", tags=["knowledge"])
    async def conflicts(
        workspace_id: str, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        items = await service.repos.knowledge.list_conflicts(workspace_id)
        return {"items": [item.model_dump(by_alias=True) for item in items]}

    # --- the feedback loop ------------------------------------------------
    @app.post("/v1/content-assets/{asset_id}/decision", tags=["feedback"])
    async def decide(
        asset_id: str, body: DecisionRequest, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        return await service.decide(
            workspace_id=body.workspace_id,
            asset_id=asset_id,
            decision=body.decision,
            edited_text=body.edited_text,
            reason=body.reason,
            user_id=body.user_id,
        )

    @app.post("/v1/feedback", tags=["feedback"])
    async def feedback(
        body: FeedbackRequest, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        event = await service.learning.record_feedback(
            workspace_id=body.workspace_id,
            asset_id=body.asset_id,
            event_type=body.event_type,
            before_text=body.before_text,
            after_text=body.after_text,
            reason=body.reason,
        )
        return {"success": True, "feedbackId": event.feedback_id}

    @app.get("/v1/workspaces/{workspace_id}/voice-profile", tags=["feedback"])
    async def voice_profile(
        workspace_id: str, service: AgenticService = Depends(get_service)
    ) -> dict[str, Any]:
        profile = await service.repos.knowledge.get_voice_profile(workspace_id)
        preferences = await service.repos.learning.list_preferences(workspace_id)
        return {
            "profile": profile.model_dump(by_alias=True) if profile else None,
            "preferences": [item.model_dump(by_alias=True) for item in preferences],
        }

    @app.get("/v1/workspaces/{workspace_id}/assets", tags=["content"])
    async def assets(
        workspace_id: str,
        status: str | None = None,
        limit: int = 20,
        service: AgenticService = Depends(get_service),
    ) -> dict[str, Any]:
        items = await service.repos.content.list_assets(
            workspace_id, status=status, limit=min(limit, 100)
        )
        return {"items": [item.model_dump(by_alias=True) for item in items]}

    # --- social accounts, sync and unified inbox -------------------------
    async def conversation_envelope(
        workspace_id: str,
        conversation_id: str,
        runtime: SocialRuntime,
        *,
        include_messages: bool,
    ) -> ConversationEnvelope | ConversationDetailResponse:
        conversation = await runtime.repos.inbox.get_conversation(workspace_id, conversation_id)
        if conversation is None:
            raise LookupError("conversation not found")
        account = await runtime.repos.social.get_account(
            workspace_id, conversation.social_account_id
        )
        if account is None:
            raise LookupError("social account not found")
        messages = await runtime.repos.inbox.list_messages(workspace_id, conversation_id)
        latest = messages[-1] if messages else None
        last_inbound = next(
            (item for item in reversed(messages) if item.direction == "inbound"),
            None,
        )
        if conversation.channel_type == "dm":
            policy_open = bool(
                last_inbound and last_inbound.created_at >= datetime.now(UTC) - timedelta(hours=24)
            )
            permitted = account.capabilities.reply_messages and policy_open
            policy_notice = None if policy_open else "Messenger 24-hour response window expired."
        else:
            permitted = account.capabilities.reply_comments
            policy_notice = None
        can_send = account.status == "connected" and permitted
        if account.status == "reauthorize_required":
            permission_status = "expired"
        else:
            permission_status = "active" if can_send else "read_only"
        capabilities = ConversationCapabilities(
            can_send_text=can_send,
            can_use_automation=(
                can_send
                and account.auto_reply_enabled
                and runtime.settings.auto_reply_mode == "low_risk"
                and conversation.handling_mode != "human"
            ),
            permission_status=permission_status,
            policy_notice=policy_notice,
        )
        decision = await runtime.repos.inbox.latest_pending_decision(workspace_id, conversation_id)
        common = {
            "conversation": conversation,
            "account": PublicSocialAccount.from_domain(account),
            "capabilities": capabilities,
            "latest_message": latest,
            "reply_decision": decision,
            "messages": messages,
        }
        if include_messages:
            return ConversationDetailResponse(**common)
        return ConversationEnvelope(**common)

    @app.get(
        "/v1/workspaces/{workspace_id}/social/status",
        response_model=SocialStatusResponse,
        tags=["social"],
        dependencies=[Depends(require_social_access)],
    )
    async def social_status(
        workspace_id: str,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> SocialStatusResponse:
        accounts = await runtime.repos.social.list_accounts(workspace_id)
        publications = await runtime.repos.social.list_publications(workspace_id, limit=100)
        cursor = await runtime.repos.social.latest_sync_cursor(workspace_id)
        return SocialStatusResponse(
            social_mode=runtime.settings.social_mode,
            inbound_mode=runtime.settings.inbound_mode,
            auto_reply_mode=runtime.settings.auto_reply_mode,
            store_backend=runtime.repos.store.backend_name,
            accounts=[PublicSocialAccount.from_domain(item) for item in accounts],
            publications_tracked=len(publications),
            last_synced_at=cursor.last_synced_at if cursor else None,
            webhook_active=any(item.webhook_status == "active" for item in accounts),
        )

    @app.post(
        "/v1/workspaces/{workspace_id}/social/sync",
        response_model=SocialSyncResponse,
        tags=["social"],
        dependencies=[Depends(require_social_access)],
    )
    async def sync_social(
        workspace_id: str,
        body: SocialSyncRequest,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> SocialSyncResponse:
        result = await runtime.sync.sync(
            workspace_id,
            publication_ids=body.publication_ids,
            limit=body.limit,
        )
        return SocialSyncResponse.model_validate(result.model_dump())

    @app.get(
        "/v1/workspaces/{workspace_id}/conversations",
        response_model=ConversationListResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def list_conversations(
        workspace_id: str,
        status: str | None = None,
        limit: int = 50,
        skip: int = 0,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> ConversationListResponse:
        bounded_limit = max(1, min(limit, 100))
        bounded_skip = max(0, skip)
        conversations = await runtime.repos.inbox.list_conversations(
            workspace_id,
            status=status,
            limit=bounded_limit,
            skip=bounded_skip,
        )
        items = [
            await conversation_envelope(
                workspace_id,
                conversation.conversation_id,
                runtime,
                include_messages=False,
            )
            for conversation in conversations
        ]
        total = await runtime.repos.inbox.count_conversations(workspace_id, status=status)
        return ConversationListResponse(items=items, total=total)

    @app.get(
        "/v1/workspaces/{workspace_id}/conversations/{conversation_id}",
        response_model=ConversationDetailResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def get_conversation(
        workspace_id: str,
        conversation_id: str,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> ConversationDetailResponse:
        result = await conversation_envelope(
            workspace_id, conversation_id, runtime, include_messages=True
        )
        return ConversationDetailResponse.model_validate(result)

    @app.get(
        "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
        response_model=MessageListResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def list_conversation_messages(
        workspace_id: str,
        conversation_id: str,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> MessageListResponse:
        if not await runtime.repos.inbox.get_conversation(workspace_id, conversation_id):
            raise LookupError("conversation not found")
        items = await runtime.repos.inbox.list_messages(workspace_id, conversation_id)
        return MessageListResponse(items=items)

    @app.post(
        "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
        response_model=SendConversationMessageResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def send_conversation_message(
        workspace_id: str,
        conversation_id: str,
        body: SendConversationMessageRequest,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> SendConversationMessageResponse:
        message, action = await runtime.replies.send(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            text=body.content,
            client_message_id=body.client_message_id,
            mode=body.mode,
        )
        return SendConversationMessageResponse(message=message, outbound_action=action)

    @app.post(
        "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/takeover",
        response_model=ConversationMutationResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def takeover_conversation(
        workspace_id: str,
        conversation_id: str,
        body: TakeoverRequest,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> ConversationMutationResponse:
        conversation = await runtime.repos.inbox.get_conversation(workspace_id, conversation_id)
        if conversation is None:
            raise LookupError("conversation not found")
        conversation.handling_mode = "human"
        conversation.ai_state = "paused_by_user"
        conversation.assignee = body.assignee
        await runtime.repos.inbox.update_conversation(conversation)
        await runtime.repos.runs.audit(
            workspace_id,
            "social.conversation_takeover",
            {"conversation_id": conversation_id, "assignee": body.assignee},
        )
        return ConversationMutationResponse(conversation=conversation)

    @app.post(
        "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/resume-ai",
        response_model=ConversationMutationResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def resume_conversation_ai(
        workspace_id: str,
        conversation_id: str,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> ConversationMutationResponse:
        conversation = await runtime.repos.inbox.get_conversation(workspace_id, conversation_id)
        if conversation is None:
            raise LookupError("conversation not found")
        account = await runtime.repos.social.get_account(
            workspace_id, conversation.social_account_id
        )
        if account is None:
            raise LookupError("social account not found")
        low_risk = runtime.settings.auto_reply_mode == "low_risk" and account.auto_reply_enabled
        conversation.handling_mode = "limited_auto" if low_risk else "suggestion_only"
        conversation.ai_state = "active" if low_risk else "suggestion_only"
        conversation.assignee = ""
        await runtime.repos.inbox.update_conversation(conversation)
        await runtime.repos.runs.audit(
            workspace_id,
            "social.conversation_ai_resumed",
            {"conversation_id": conversation_id, "limited_auto": low_risk},
        )
        return ConversationMutationResponse(conversation=conversation)

    @app.post(
        "/v1/workspaces/{workspace_id}/reply-decisions/{decision_id}/approve",
        response_model=DecisionSendResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def approve_reply_decision(
        workspace_id: str,
        decision_id: str,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> DecisionSendResponse:
        message, decision, action = await runtime.replies.send_decision(
            workspace_id=workspace_id, decision_id=decision_id
        )
        return DecisionSendResponse(message=message, decision=decision, outbound_action=action)

    @app.post(
        "/v1/workspaces/{workspace_id}/reply-decisions/{decision_id}/edit-and-send",
        response_model=DecisionSendResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def edit_and_send_reply_decision(
        workspace_id: str,
        decision_id: str,
        body: EditAndSendRequest,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> DecisionSendResponse:
        message, decision, action = await runtime.replies.send_decision(
            workspace_id=workspace_id,
            decision_id=decision_id,
            edited_text=body.content,
        )
        return DecisionSendResponse(message=message, decision=decision, outbound_action=action)

    @app.post(
        "/v1/workspaces/{workspace_id}/reply-decisions/{decision_id}/reject",
        response_model=DecisionMutationResponse,
        tags=["inbox"],
        dependencies=[Depends(require_social_access)],
    )
    async def reject_reply_decision(
        workspace_id: str,
        decision_id: str,
        body: RejectDecisionRequest,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> DecisionMutationResponse:
        decision = await runtime.replies.reject_decision(workspace_id, decision_id)
        await runtime.repos.runs.audit(
            workspace_id,
            "social.reply_rejected",
            {"decision_id": decision_id, "reason": body.reason},
        )
        return DecisionMutationResponse(decision=decision)

    @app.post(
        "/v1/workspaces/{workspace_id}/social/mock-events",
        response_model=InjectMockEventResponse,
        tags=["social"],
        dependencies=[Depends(require_social_access)],
    )
    async def inject_mock_event(
        workspace_id: str,
        body: InjectMockEventRequest,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> InjectMockEventResponse:
        if runtime.settings.social_mode != "mock":
            raise HTTPException(status_code=404, detail="mock event injection is disabled")
        publication = await runtime.repos.social.get_publication(workspace_id, body.publication_id)
        if publication is None:
            raise LookupError("publication not found")
        account = await runtime.repos.social.get_account(
            workspace_id, publication.social_account_id
        )
        if account is None:
            raise LookupError("social account not found")
        provider_message_id = new_id("mockinbound")
        event = NormalizedInboundEvent(
            provider_event_id=provider_message_id,
            provider_message_id=provider_message_id,
            platform=publication.platform,
            channel_type=("public_reply" if publication.platform == "threads" else "comment"),
            provider_account_id=account.provider_account_id,
            provider_post_id=publication.platform_post_id,
            provider_parent_id=publication.platform_post_id,
            provider_thread_key=publication.platform_post_id,
            sender_id=body.sender_id,
            sender_name=body.sender_name,
            text=body.text,
            created_at=datetime.now(UTC),
        )
        runtime.connectors.mock.queue_event(publication.publication_id, event)
        return InjectMockEventResponse(queued=True, provider_message_id=provider_message_id)

    # --- authenticated Next.js-to-agentic bridge -------------------------
    @app.post(
        "/internal/social/accounts/activate",
        response_model=ActivateSocialAccountResponse,
        tags=["internal"],
        dependencies=[Depends(require_internal_access)],
    )
    async def activate_social_account(
        body: ActivateSocialAccountRequest,
        request: Request,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> ActivateSocialAccountResponse:
        enforce_internal_workspace(request.app.state.settings, body.workspace_id)
        account = await runtime.publications.activate(body.workspace_id, body.account_id)
        return ActivateSocialAccountResponse(
            account=PublicSocialAccount.from_domain(account),
            activated=True,
            webhook_status=account.webhook_status,
        )

    @app.post(
        "/internal/publications/register",
        response_model=RegisterPublicationResponse,
        tags=["internal"],
        dependencies=[Depends(require_internal_access)],
    )
    async def register_publication(
        body: RegisterPublicationRequest,
        request: Request,
        runtime: SocialRuntime = Depends(get_social_runtime),
    ) -> RegisterPublicationResponse:
        enforce_internal_workspace(request.app.state.settings, body.workspace_id)
        publication, created = await runtime.publications.register(body.to_domain())
        return RegisterPublicationResponse(publication=publication, created=created)

    # --- provider-authenticated public webhooks ---------------------------
    @app.get("/webhooks/{provider}", tags=["webhooks"])
    async def verify_webhook(provider: str, request: Request) -> Response:
        if provider not in {"meta", "threads"}:
            raise HTTPException(status_code=404, detail="webhook provider not found")
        runtime: SocialRuntime = request.app.state.social
        try:
            challenge = runtime.webhooks.verify_challenge(
                request.query_params.get("hub.mode", ""),
                request.query_params.get("hub.verify_token", ""),
                request.query_params.get("hub.challenge", ""),
            )
        except WebhookVerificationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(content=challenge, media_type="text/plain")

    @app.post(
        "/webhooks/{provider}",
        response_model=WebhookIngestResponse,
        tags=["webhooks"],
    )
    async def ingest_webhook(
        provider: str,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> WebhookIngestResponse:
        if provider not in {"meta", "threads"}:
            raise HTTPException(status_code=404, detail="webhook provider not found")
        raw_body = await request.body()
        if len(raw_body) > 1_000_000:
            raise HTTPException(status_code=413, detail="webhook payload is too large")
        runtime: SocialRuntime = request.app.state.social
        try:
            runtime.webhooks.verify_signature(
                provider,
                raw_body,
                request.headers.get("X-Hub-Signature-256"),
            )
        except WebhookVerificationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="webhook payload is invalid") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="webhook payload must be an object")
        if runtime.settings.inbound_mode == "polling":
            return WebhookIngestResponse(received=0, inserted=0, duplicates=0, ignored=1)
        background_tasks.add_task(runtime.webhooks.process, provider, payload)
        return WebhookIngestResponse(received=1, inserted=0, duplicates=0, ignored=0)

    return app


app = create_app()
