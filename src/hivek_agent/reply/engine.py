"""Focused reply decision engine; inbound comments do not enter the content graph."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from hivek_agent.agentic.model_router import ModelRouter
from hivek_agent.config import Settings
from hivek_agent.domain import (
    ReplyDecision,
    ReplyEvidence,
    SocialAccount,
    SocialConversation,
    SocialMessage,
    SocialPublication,
)
from hivek_agent.infrastructure.llm import LLMError, LLMGateway
from hivek_agent.reply.normalization import numbers
from hivek_agent.reply.policy import PolicyVerdict, evaluate_policy
from hivek_agent.reply.retrieval import ReplyCandidate, rank_candidates
from hivek_agent.repositories import Repositories, new_id

_RETRIEVAL_THRESHOLD = 0.18


class GeneratedReply(BaseModel):
    reply_text: str
    used_fact_ids: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    should_handoff: bool = False


class ReplyDecisionEngine:
    def __init__(self, repos: Repositories, llm: LLMGateway, settings: Settings) -> None:
        self.repos = repos
        self.llm = llm
        self.settings = settings
        self.router = ModelRouter(settings)

    async def decide(
        self,
        *,
        workspace_id: str,
        conversation: SocialConversation,
        message: SocialMessage,
        publication: SocialPublication | None,
        account: SocialAccount,
    ) -> ReplyDecision:
        existing = await self.repos.inbox.get_reply_decision_for_message(
            workspace_id, message.message_id
        )
        if existing is not None:
            return existing

        policy = evaluate_policy(message.text)
        if policy.ignore:
            return await self._persist(
                workspace_id,
                conversation,
                message,
                policy,
                action="ignore",
                confidence=0.99,
            )
        if policy.handoff:
            conversation.status = "needs_human"
            conversation.ai_state = "handoff_requested"
            await self.repos.inbox.update_conversation(conversation)
            return await self._persist(
                workspace_id,
                conversation,
                message,
                policy,
                action="human_handoff",
                confidence=0.99,
                suggested_text=(
                    "Chào bạn, mình đã tiếp nhận và chuyển yêu cầu này cho nhân viên phụ trách "
                    "để hỗ trợ chính xác hơn nhé."
                ),
                evidence=[
                    ReplyEvidence(
                        source_type="rule",
                        source_id=f"risk:{policy.intent}",
                        excerpt="Yêu cầu cần người phụ trách xác nhận.",
                        score=1,
                    )
                ],
            )

        if policy.intent == "greeting":
            return await self._persist_candidate(
                workspace_id,
                conversation,
                message,
                account,
                policy,
                text="Chào bạn, cảm ơn bạn đã nhắn cho chúng mình. Mình có thể hỗ trợ gì cho bạn?",
                confidence=0.98,
                evidence=[
                    ReplyEvidence(
                        source_type="rule",
                        source_id="rule:greeting",
                        excerpt="Lời chào xã giao, không chứa dữ kiện kinh doanh.",
                        score=1,
                    )
                ],
            )

        candidates = await self._candidates(workspace_id, publication, conversation.platform)
        ranked = rank_candidates(message.text, candidates)
        if ranked and ranked[0].score >= _RETRIEVAL_THRESHOLD:
            best = ranked[0]
            evidence = [
                ReplyEvidence(
                    source_type=item.candidate.source_type,
                    source_id=item.candidate.source_id,
                    excerpt=item.candidate.reply_text[:280],
                    score=min(1, item.score),
                )
                for item in ranked[:3]
            ]
            confidence = min(0.96, 0.62 + best.score * 0.45)
            return await self._persist_candidate(
                workspace_id,
                conversation,
                message,
                account,
                policy,
                text=best.candidate.reply_text,
                confidence=confidence,
                evidence=evidence,
            )

        generated = await self._generate(message, conversation, candidates)
        if generated is None or generated.should_handoff:
            policy.risk_labels.extend(generated.risk_flags if generated else ["missing_fact"])
            conversation.status = "needs_human"
            conversation.ai_state = "blocked_missing_data"
            await self.repos.inbox.update_conversation(conversation)
            return await self._persist(
                workspace_id,
                conversation,
                message,
                policy,
                action="human_handoff",
                confidence=0.65,
                suggested_text=generated.reply_text if generated else None,
                model_used="gemini" if generated else "deterministic",
            )

        evidence = [
            ReplyEvidence(
                source_type=item.source_type,
                source_id=item.source_id,
                excerpt=item.reply_text[:280],
                score=0.5,
            )
            for item in candidates
            if item.source_id in generated.used_fact_ids
        ]
        unsupported = numbers(generated.reply_text) - {
            number for item in evidence for number in numbers(item.excerpt)
        }
        if unsupported:
            policy.risk_labels.append("unsupported_number")
            conversation.status = "needs_human"
            conversation.ai_state = "blocked_missing_data"
            await self.repos.inbox.update_conversation(conversation)
            return await self._persist(
                workspace_id,
                conversation,
                message,
                policy,
                action="human_handoff",
                confidence=0.99,
                evidence=evidence,
                model_used="gemini",
            )
        return await self._persist_candidate(
            workspace_id,
            conversation,
            message,
            account,
            policy,
            text=generated.reply_text,
            confidence=0.72,
            evidence=evidence,
            model_used="gemini",
        )

    async def _candidates(
        self, workspace_id: str, publication: SocialPublication | None, platform: str
    ) -> list[ReplyCandidate]:
        candidates: list[ReplyCandidate] = []
        if publication:
            for index, reply in enumerate(publication.reply_suggestions):
                candidates.append(
                    ReplyCandidate(
                        candidate_id=f"{publication.publication_id}:{index}",
                        match_text=f"{publication.text} {reply}",
                        reply_text=reply,
                        source_type="post_reply_suggestion",
                        source_id=publication.content_asset_id or publication.publication_id,
                    )
                )
        assertions = await self.repos.knowledge.list_assertions(workspace_id, limit=100)
        for assertion in assertions:
            if assertion.approval_status != "confirmed":
                continue
            rendered = _render_fact(assertion.object_value)
            candidates.append(
                ReplyCandidate(
                    candidate_id=assertion.assertion_id,
                    match_text=f"{assertion.predicate} {rendered}",
                    reply_text=rendered,
                    source_type="confirmed_fact",
                    source_id=assertion.assertion_id,
                )
            )
        for approved in await self.repos.inbox.list_approved_replies(
            workspace_id, platform=platform
        ):
            candidates.append(
                ReplyCandidate(
                    candidate_id=approved.message_id,
                    match_text=approved.text,
                    reply_text=approved.text,
                    source_type="approved_reply",
                    source_id=approved.message_id,
                )
            )
        return candidates

    async def _generate(
        self,
        message: SocialMessage,
        conversation: SocialConversation,
        candidates: list[ReplyCandidate],
    ) -> GeneratedReply | None:
        if not candidates:
            return None
        route = self.router.route("inbound_reply")
        payload = {
            "message": message.text,
            "channel": conversation.channel_type,
            "approvedCandidates": [candidate.model_dump() for candidate in candidates[:8]],
        }
        try:
            generated, _ = await self.llm.complete_structured(
                system=(
                    "Bạn soạn một câu trả lời ngắn cho HIVE-K. Chỉ dùng dữ kiện trong "
                    "approvedCandidates. Thiếu dữ kiện hoặc có rủi ro thì shouldHandoff=true."
                ),
                prompt=json.dumps(payload, ensure_ascii=False, default=str),
                schema=GeneratedReply,
                model=route.model_name,
                temperature=0.2,
                max_output_tokens=route.max_output_tokens,
                fallback_models=route.fallback_chain,
            )
            return generated
        except LLMError:
            return None

    async def _persist_candidate(
        self,
        workspace_id: str,
        conversation: SocialConversation,
        message: SocialMessage,
        account: SocialAccount,
        policy: PolicyVerdict,
        *,
        text: str,
        confidence: float,
        evidence: list[ReplyEvidence],
        model_used: str = "deterministic",
    ) -> ReplyDecision:
        auto = (
            self.settings.auto_reply_mode == "low_risk"
            and account.auto_reply_enabled
            and policy.low_risk
            and policy.intent in self.settings.auto_reply_allowed_intents
            and confidence >= self.settings.auto_reply_min_confidence
            and conversation.handling_mode != "human"
        )
        return await self._persist(
            workspace_id,
            conversation,
            message,
            policy,
            action="auto_reply" if auto else "suggestion",
            confidence=confidence,
            suggested_text=text,
            evidence=evidence,
            model_used=model_used,
        )

    async def _persist(
        self,
        workspace_id: str,
        conversation: SocialConversation,
        message: SocialMessage,
        policy: PolicyVerdict,
        *,
        action: str,
        confidence: float,
        suggested_text: str | None = None,
        evidence: list[ReplyEvidence] | None = None,
        model_used: str = "deterministic",
    ) -> ReplyDecision:
        decision = ReplyDecision(
            decision_id=new_id("decision"),
            workspace_id=workspace_id,
            conversation_id=conversation.conversation_id,
            message_id=message.message_id,
            intent=policy.intent,
            risk_labels=list(dict.fromkeys(policy.risk_labels)),
            confidence=confidence,
            action=action,  # type: ignore[arg-type]
            suggested_text=suggested_text,
            evidence=evidence or [],
            model_used=model_used,
        )
        persisted, _ = await self.repos.inbox.save_reply_decision(decision)
        return persisted


def _render_fact(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
