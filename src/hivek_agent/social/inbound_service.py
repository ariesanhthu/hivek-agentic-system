"""One normalized ingestion path shared by polling and webhooks."""

from __future__ import annotations

from pydantic import BaseModel

from hivek_agent.domain import (
    NormalizedInboundEvent,
    ReplyDecision,
    SocialConversation,
    SocialMessage,
)
from hivek_agent.reply import ReplyDecisionEngine
from hivek_agent.reply.normalization import normalize_text
from hivek_agent.repositories import Repositories, new_id
from hivek_agent.social.reply_service import ReplyService


class IngestResult(BaseModel):
    message: SocialMessage
    conversation: SocialConversation
    inserted: bool
    decision: ReplyDecision | None = None
    auto_reply_sent: bool = False


class InboundService:
    def __init__(
        self,
        repos: Repositories,
        engine: ReplyDecisionEngine,
        replies: ReplyService,
    ) -> None:
        self.repos = repos
        self.engine = engine
        self.replies = replies

    async def ingest(
        self,
        *,
        workspace_id: str,
        account_id: str,
        normalized_event: NormalizedInboundEvent,
        source: str,
    ) -> IngestResult:
        account = await self.repos.social.get_account(workspace_id, account_id)
        if account is None:
            raise LookupError("social account not found")
        if account.provider_account_id != normalized_event.provider_account_id:
            raise ValueError("provider account does not match ingestion account")
        publication = None
        if normalized_event.provider_post_id:
            publication = await self.repos.social.find_publication_by_provider_post(
                workspace_id,
                normalized_event.platform,
                normalized_event.provider_post_id,
            )

        conversation = SocialConversation(
            conversation_id=new_id("conversation"),
            workspace_id=workspace_id,
            social_account_id=account_id,
            platform=normalized_event.platform,
            channel_type=normalized_event.channel_type,
            provider_thread_key=normalized_event.provider_thread_key,
            provider_user_id=normalized_event.sender_id,
            customer_name=normalized_event.sender_name,
            customer_username=normalized_event.sender_name,
            publication_id=publication.publication_id if publication else None,
            source_post_id=normalized_event.provider_post_id,
            source_context=publication.text[:500] if publication else "",
            last_message_at=normalized_event.created_at,
        )
        conversation = await self.repos.inbox.upsert_conversation(conversation)
        message = SocialMessage(
            message_id=new_id("message"),
            workspace_id=workspace_id,
            conversation_id=conversation.conversation_id,
            platform=normalized_event.platform,
            provider_message_id=normalized_event.provider_message_id,
            direction="inbound",
            author_type="customer",
            author_name=normalized_event.sender_name,
            text=normalized_event.text,
            normalized_text=normalize_text(normalized_event.text),
            created_at=normalized_event.created_at,
            delivery_status="delivered",
            metadata={
                "source": source,
                "provider_event_id": normalized_event.provider_event_id,
                "provider_parent_id": normalized_event.provider_parent_id,
                "provider_post_id": normalized_event.provider_post_id,
            },
        )
        message, inserted = await self.repos.inbox.insert_message_once(message)
        if not inserted:
            existing_decision = await self.repos.inbox.get_reply_decision_for_message(
                workspace_id, message.message_id
            )
            return IngestResult(
                message=message,
                conversation=conversation,
                inserted=False,
                decision=existing_decision,
            )

        await self.repos.inbox.mark_inbound(conversation, normalized_event.created_at)
        decision = await self.engine.decide(
            workspace_id=workspace_id,
            conversation=conversation,
            message=message,
            publication=publication,
            account=account,
        )
        auto_sent = False
        if decision.action == "auto_reply" and decision.suggested_text:
            try:
                await self.replies.send_decision(
                    workspace_id=workspace_id,
                    decision_id=decision.decision_id,
                    automated=True,
                )
                auto_sent = True
            except Exception:
                # The inbound message and suggestion are still durable.  Send state is
                # captured by OutboundAction; ingestion must not be rolled back.
                auto_sent = False
        await self.repos.runs.audit(
            workspace_id,
            "social.inbound_ingested",
            {
                "conversation_id": conversation.conversation_id,
                "message_id": message.message_id,
                "source": source,
                "decision_id": decision.decision_id,
            },
        )
        return IngestResult(
            message=message,
            conversation=conversation,
            inserted=True,
            decision=decision,
            auto_reply_sent=auto_sent,
        )
