"""Application service.

Sits between the HTTP layer and the graph. Owns run lifecycle, event emission,
idempotency and the decision/feedback loop, so routes stay thin and contain no
prompts or business rules.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from hivek_agent.agentic.nodes import NodeDeps, source_from_user
from hivek_agent.domain import (
    AgentRun,
    AgentRunSummary,
    ContentAsset,
    FeedbackEventType,
    HarnessResponse,
    HarnessState,
    Intent,
    PlatformId,
    RunEvent,
    UIAction,
)
from hivek_agent.learning.edit_analysis import LearningService
from hivek_agent.repositories import Repositories, new_id

logger = logging.getLogger(__name__)

# Maps the client's setup step -> the fact predicate it satisfies.
SETUP_PREDICATES = {
    "social": "brand.channels",
    "brand": "brand.name",
    "tone": "brand.tone",
    "drive": "brand.resource_url",
}


class EventBus:
    """In-process pub/sub for SSE.

    Events are also persisted, so a late subscriber replays from the store rather
    than losing history. Single-process only; a Redis fan-out is the multi-worker
    upgrade and slots in behind this same interface.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[RunEvent | None]]] = defaultdict(list)

    def subscribe(self, run_id: str) -> asyncio.Queue[RunEvent | None]:
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue(maxsize=256)
        self._subscribers[run_id].append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent | None]) -> None:
        listeners = self._subscribers.get(run_id, [])
        if queue in listeners:
            listeners.remove(queue)
        if not listeners:
            self._subscribers.pop(run_id, None)

    def publish(self, event: RunEvent) -> None:
        for queue in list(self._subscribers.get(event.run_id, [])):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client must never block the run that is producing events.
                logger.warning("dropping event for slow subscriber run=%s", event.run_id)

    def close(self, run_id: str) -> None:
        for queue in list(self._subscribers.get(run_id, [])):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass


class AgenticService:
    def __init__(self, deps: NodeDeps, graph: Any, repos: Repositories) -> None:
        self.deps = deps
        self.graph = graph
        self.repos = repos
        self.events = EventBus()
        self.learning = LearningService(repos.learning, repos.knowledge)

    # --- chat -----------------------------------------------------------
    async def send_message(
        self,
        *,
        workspace_id: str,
        user_id: str,
        thread_id: str,
        message: str,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> HarnessResponse:
        if idempotency_key:
            existing = await self.repos.runs.find_by_idempotency_key(workspace_id, idempotency_key)
            if existing is not None:
                # Replay rather than re-run: the caller retried, the work is already done.
                logger.info("idempotent replay run=%s", existing.run_id)
                return await self._replay(existing)

        run = AgentRun(
            run_id=new_id("run"),
            workspace_id=workspace_id,
            user_id=user_id,
            thread_id=thread_id,
            idempotency_key=idempotency_key,
        )
        await self.repos.runs.create_run(run)

        seq = _Counter()
        await self._emit(run, seq, "run.started", {"message": message[:200]})

        state = HarnessState(
            run_id=run.run_id,
            workspace_id=workspace_id,
            user_id=user_id,
            thread_id=thread_id,
            user_message=message,
            request_payload=payload or {},
            trace_id=run.run_id,
        )

        try:
            raw = await self.graph.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
            final = HarnessState.model_validate(raw)
        except Exception as exc:
            logger.exception("graph failed run=%s", run.run_id)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            await self.repos.runs.update_run(run)
            await self._emit(run, seq, "run.failed", {"error": type(exc).__name__})
            self.events.close(run.run_id)
            return HarnessResponse(
                run_id=run.run_id,
                thread_id=thread_id,
                status="failed",
                reply="Xin lỗi, đã có lỗi khi xử lý yêu cầu. Vui lòng thử lại.",
                warnings=[type(exc).__name__],
                trace_id=run.run_id,
            )

        await self._emit_progress(run, seq, final)

        run.intent = final.intent
        run.status = final.status
        await self.repos.runs.update_run(run)

        await self._append_thread(workspace_id, thread_id, message, final)
        self.events.close(run.run_id)
        return await self._to_response(final, run)

    async def _emit_progress(self, run: AgentRun, seq: _Counter, final: HarnessState) -> None:
        if final.draft is not None and final.asset is not None:
            await self._emit(run, seq, "draft.created", {"assetId": final.asset.asset_id})
        if final.validation is not None:
            await self._emit(
                run,
                seq,
                "validation.completed",
                {
                    "riskLevel": final.validation.risk_level,
                    "decision": final.validation.final_decision,
                },
            )
        if final.status == "needs_user_input":
            await self._emit(
                run,
                seq,
                "input.required",
                {"missing": [item.field for item in final.missing_items[:5]]},
            )
        if final.status == "needs_approval":
            await self._emit(run, seq, "approval.required", final.approval_payload or {})
        if final.status in ("completed", "needs_approval", "needs_user_input"):
            await self._emit(run, seq, "run.completed", {"status": final.status})

    # --- setup ----------------------------------------------------------
    async def record_setup_step(
        self,
        *,
        workspace_id: str,
        step: str,
        value: Any,
        user_id: str,
    ) -> dict[str, Any]:
        """Persist a setup answer as a confirmed, user-sourced fact.

        These come from the user directly, so they are `confirmed` immediately - the
        highest precedence tier. Nothing inferred can later overwrite them silently.
        """
        predicate = SETUP_PREDICATES.get(step)
        if predicate is None:
            raise ValueError(f"unknown setup step: {step}")

        result = await self.deps.facts.upsert_fact(
            workspace_id=workspace_id,
            subject_id="workspace",
            predicate=predicate,
            object_value=value,
            source=source_from_user(f"user/setup/{step}"),
            confidence=1.0,
            approval_status="confirmed",
        )
        await self.repos.runs.audit(
            workspace_id,
            "setup.recorded",
            {"step": step, "predicate": predicate, "action": result.action, "user_id": user_id},
        )
        profile = await self.deps.brand.build_profile(workspace_id)
        # camelCase like every other response: these dicts are hand-built, so they miss
        # the alias generator that Pydantic response models get for free.
        return {
            "action": result.action,
            "assertionId": result.assertion.assertion_id,
            "readinessScore": profile.readiness_score,
            "missingItems": [item.model_dump(by_alias=True) for item in profile.missing_items],
        }

    # --- the feedback loop ----------------------------------------------
    async def decide(
        self,
        *,
        workspace_id: str,
        asset_id: str,
        decision: FeedbackEventType,
        edited_text: str | None = None,
        reason: str | None = None,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Apply a human decision and feed it back into learning.

        This closes the loop: approve/edit/reject -> feedback event -> preference
        candidates -> promotion -> voice profile -> next draft.
        """
        asset = await self.repos.content.get_asset(workspace_id, asset_id)
        if asset is None:
            raise LookupError(f"asset not found: {asset_id}")

        before_text = asset.edited_text or asset.draft.full_text

        if decision == "approve":
            asset.status = "approved"
        elif decision == "reject":
            asset.status = "rejected"
            asset.review_note = reason or "Bị từ chối"
        elif decision == "edit":
            if not edited_text or not edited_text.strip():
                raise ValueError("edit decision requires edited_text")
            asset.edited_text = edited_text
            asset.status = "approved"
        elif decision == "regenerate":
            asset.status = "draft"

        await self.repos.content.save_asset(asset)
        await self.repos.runs.audit(
            workspace_id,
            f"asset.{decision}",
            {"asset_id": asset_id, "user_id": user_id, "status": asset.status},
        )

        # Analysing an edit and deciding it is a rule are separate steps by design:
        # `record_edit` only observes. Promotion is the gate that stops one edit from
        # becoming a permanent rule, so the caller must invoke it explicitly.
        edit_event = None
        promoted: list[Any] = []
        if decision in ("edit", "pin_as_good") and edited_text and edited_text.strip():
            edit_event = await self.learning.record_edit(
                workspace_id=workspace_id,
                asset_id=asset_id,
                before_text=before_text,
                after_text=edited_text,
                platform=asset.platform,
                reason=reason,
                explicit=decision == "pin_as_good",
            )
            promoted = await self.learning.promote_preferences(
                workspace_id, edit_event.inferred_preferences
            )

        # Texts are omitted here so `record_feedback` does not re-analyse the edit we
        # just recorded above and create a duplicate event.
        event = await self.learning.record_feedback(
            workspace_id=workspace_id,
            asset_id=asset_id,
            event_type=decision,
            reason=reason,
            platform=asset.platform,
            metadata={"edit_event_id": edit_event.event_id} if edit_event else {},
        )

        voice = await self.learning.rebuild_voice_profile(workspace_id)
        active = await self.repos.learning.list_preferences(workspace_id, active_only=True)

        return {
            "assetId": asset_id,
            "status": asset.status,
            "feedbackId": event.feedback_id,
            "voiceProfileVersion": voice.version,
            "learnedThisTurn": [
                {
                    "rule": item.rule_type,
                    "value": item.rule_value,
                    "status": item.status,
                    "scope": item.scope,
                    "observations": item.observation_count,
                }
                for item in promoted
            ],
            "activeRules": [
                {"rule": item.rule_type, "value": item.rule_value, "status": item.status}
                for item in active
            ],
        }

    # --- helpers --------------------------------------------------------
    async def _emit(
        self, run: AgentRun, seq: _Counter, event_name: str, data: dict[str, Any]
    ) -> None:
        event = RunEvent(
            event=event_name,  # type: ignore[arg-type]
            run_id=run.run_id,
            seq=seq.next(),
            data=data,
        )
        await self.repos.runs.append_event(run.workspace_id, event)
        self.events.publish(event)

    async def _append_thread(
        self, workspace_id: str, thread_id: str, message: str, final: HarnessState
    ) -> None:
        history = await self.repos.runs.get_thread(workspace_id, thread_id)
        history.append({"role": "user", "content": message})
        history.append(
            {
                "role": "assistant",
                "content": final.reply_text,
                "run_id": final.run_id,
                "status": final.status,
            }
        )
        # Bound growth; the store is a transcript, not the agent's memory.
        await self.repos.runs.save_thread(workspace_id, thread_id, history[-40:])

    async def _replay(self, run: AgentRun) -> HarnessResponse:
        """Rebuild the original response for a retried request.

        Looks up the asset by run_id: taking the workspace's most recent asset would
        hand back whatever another run produced in the meantime.
        """
        asset = await self.repos.content.find_by_run(run.workspace_id, run.run_id)
        history = await self.repos.runs.get_thread(run.workspace_id, run.thread_id)
        reply = next(
            (
                item.get("content", "")
                for item in reversed(history)
                if item.get("role") == "assistant" and item.get("run_id") == run.run_id
            ),
            "",
        )
        return HarnessResponse(
            run_id=run.run_id,
            thread_id=run.thread_id,
            status=run.status,
            reply=reply,
            asset=asset,
            warnings=["Kết quả lấy lại từ lần gọi trước (idempotency key trùng)."],
            trace_id=run.run_id,
        )

    async def _to_response(self, final: HarnessState, run: AgentRun) -> HarnessResponse:
        node_runs = await self.repos.runs.list_node_runs(final.workspace_id, final.run_id)
        summary = AgentRunSummary(
            run_id=final.run_id,
            workflow_name="chat",
            agent_name=final.intent or "router",
            input_summary=final.user_message[:120],
            output_summary=final.reply_text[:120],
            model=next((item.model for item in node_runs if item.model), "deterministic"),
            latency_ms=sum(item.latency_ms for item in node_runs),
            input_tokens=sum(item.input_tokens for item in node_runs),
            output_tokens=sum(item.output_tokens for item in node_runs),
            cache_hit=any(item.cache_hit for item in node_runs),
            status="failed" if final.status == "failed" else "success",
        )

        return HarnessResponse(
            run_id=final.run_id,
            thread_id=final.thread_id,
            status=final.status,
            reply=final.reply_text,
            widget=final.widget,
            next_actions=[UIAction.model_validate(action) for action in final.ui_actions],
            missing_items=final.missing_items,
            asset=final.asset,
            plan=final.plan,
            citations=final.compiled_context.citations if final.compiled_context else [],
            # `progress` is a free-form dict, so Pydantic cannot alias its inner keys.
            # They are written camelCase by hand to match the rest of the response.
            progress={
                "steps": final.step_count,
                "nodes": [item.node_name for item in node_runs],
                "estimatedTokens": (
                    final.compiled_context.estimated_tokens if final.compiled_context else 0
                ),
                "tokenBudget": (
                    final.compiled_context.token_budget if final.compiled_context else 0
                ),
                "omittedSections": (
                    final.compiled_context.omitted_sections if final.compiled_context else []
                ),
                "contextHash": (
                    final.compiled_context.context_hash if final.compiled_context else ""
                ),
            },
            warnings=final.warnings,
            agent_run=summary,
            trace_id=final.trace_id,
        )


class _Counter:
    def __init__(self) -> None:
        self._value = 0

    def next(self) -> int:
        self._value += 1
        return self._value


__all__ = ["AgenticService", "EventBus", "Intent", "PlatformId", "ContentAsset"]
