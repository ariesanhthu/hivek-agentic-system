"""Idempotent provider send path for manual and approved Agent replies."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from hivek_agent.domain import (
    OutboundAction,
    ReplyDecision,
    SocialConversation,
    SocialMessage,
    utc_now,
)
from hivek_agent.infrastructure.social import ProviderAPIError, SocialConnectorFactory
from hivek_agent.repositories import Repositories, new_id
from hivek_agent.social.publication_service import SocialCredentialLoader


class OutboundStateError(RuntimeError):
    pass


class ReplyService:
    def __init__(
        self,
        repos: Repositories,
        connectors: SocialConnectorFactory,
        credentials: SocialCredentialLoader,
    ) -> None:
        self.repos = repos
        self.connectors = connectors
        self.credentials = credentials

    async def send(
        self,
        *,
        workspace_id: str,
        conversation_id: str,
        text: str,
        client_message_id: str | None = None,
        mode: str = "reply",
        decision_id: str | None = None,
        automated: bool = False,
    ) -> tuple[SocialMessage, OutboundAction]:
        text = text.strip()
        if not text:
            raise ValueError("message content must not be empty")
        conversation = await self.repos.inbox.get_conversation(workspace_id, conversation_id)
        if conversation is None:
            raise LookupError("conversation not found")
        account = await self.repos.social.get_account(workspace_id, conversation.social_account_id)
        if account is None:
            raise LookupError("social account not found")

        if mode == "note":
            return await self._save_note(conversation, text, client_message_id=client_message_id)
        if account.status != "connected":
            raise ValueError("social account requires reconnection")

        inbound = await self._target_message(workspace_id, conversation, decision_id)
        if inbound is None:
            raise ValueError("conversation has no inbound message to reply to")
        if conversation.channel_type == "dm":
            if not account.capabilities.reply_messages:
                raise ValueError("account cannot reply to private messages")
            if inbound.created_at < datetime.now(UTC) - timedelta(hours=24):
                raise ValueError("Messenger 24-hour response window has expired")
            action_type = "message"
            target_id = conversation.provider_user_id
        else:
            if not account.capabilities.reply_comments:
                raise ValueError("account cannot reply to public comments")
            action_type = "public_reply"
            target_id = inbound.provider_message_id
        if not target_id:
            raise ValueError("provider reply target is missing")

        key = _idempotency_key(
            conversation.platform, inbound.provider_message_id, text, action_type
        )
        message_id = new_id("message")
        action = OutboundAction(
            action_id=new_id("outbound"),
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            message_id=message_id,
            decision_id=decision_id,
            platform=conversation.platform,
            action_type=action_type,  # type: ignore[arg-type]
            target_id=target_id,
            text=text,
            idempotency_key=key,
        )
        action, created = await self.repos.inbox.save_outbound_action_once(action)
        if not created:
            if action.status == "sent" and action.message_id:
                existing = await self.repos.inbox.get_message(workspace_id, action.message_id)
                if existing is not None:
                    return existing, action
            raise OutboundStateError(f"outbound action already exists with status={action.status}")

        action.status = "sending"
        await self.repos.inbox.update_outbound_action(action)
        connector = self.connectors.for_platform(account.platform)
        token = await self.credentials.token_for(account)
        try:
            if action_type == "message":
                sent = await connector.send_message(
                    account, token, recipient_id=target_id, text=text
                )
            else:
                sent = await connector.send_public_reply(
                    account, token, target_id=target_id, text=text
                )
        except ProviderAPIError as exc:
            # Transport failures, throttling and provider 5xx responses can arrive
            # after a mutation was accepted. Never convert those into a blind retry.
            action.status = (
                "needs_review"
                if exc.status is None or exc.status == 429 or exc.status >= 500
                else "failed"
            )
            action.error = f"provider_error:{exc.code}"
            await self.repos.inbox.update_outbound_action(action)
            raise

        now = utc_now()
        message = SocialMessage(
            message_id=message_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            platform=conversation.platform,
            provider_message_id=sent.provider_message_id,
            direction="outbound",
            author_type="ai_agent" if automated else "current_user",
            author_name="HIVE-K Agent" if automated else "Workspace member",
            text=text,
            normalized_text=" ".join(text.casefold().split()),
            created_at=now,
            delivery_status="sent",
            client_message_id=client_message_id,
            is_automated=automated,
            metadata={"target_provider_message_id": target_id},
        )
        message, _ = await self.repos.inbox.insert_message_once(message)
        action.message_id = message.message_id
        action.provider_request_id = sent.provider_request_id
        action.provider_message_id = sent.provider_message_id
        action.status = "sent"
        await self.repos.inbox.update_outbound_action(action)
        conversation.last_message_at = now
        conversation.unread_count = 0
        conversation.status = "waiting_for_customer"
        await self.repos.inbox.update_conversation(conversation)
        await self.repos.runs.audit(
            workspace_id,
            "social.reply_sent",
            {
                "conversation_id": conversation_id,
                "action_id": action.action_id,
                "platform": conversation.platform,
                "automated": automated,
            },
        )
        return message, action

    async def send_decision(
        self,
        *,
        workspace_id: str,
        decision_id: str,
        edited_text: str | None = None,
        automated: bool = False,
    ) -> tuple[SocialMessage, ReplyDecision, OutboundAction]:
        decision = await self.repos.inbox.get_reply_decision(workspace_id, decision_id)
        if decision is None:
            raise LookupError("reply decision not found")
        if decision.status == "sent":
            action = await self.repos.inbox.get_outbound_action_by_decision(
                workspace_id, decision.decision_id
            )
            if action and action.message_id:
                message = await self.repos.inbox.get_message(workspace_id, action.message_id)
                if message:
                    return message, decision, action
            raise OutboundStateError("reply decision is already sent")
        if decision.status == "rejected":
            raise ValueError("reply decision was rejected")
        text = (edited_text if edited_text is not None else decision.suggested_text) or ""
        if not text.strip():
            raise ValueError("reply decision has no suggested text")
        decision.status = "edited" if edited_text is not None else "approved"
        await self.repos.inbox.update_reply_decision(decision)
        message, action = await self.send(
            workspace_id=workspace_id,
            conversation_id=decision.conversation_id,
            text=text,
            decision_id=decision.decision_id,
            automated=automated,
        )
        decision.status = "sent"
        if edited_text is not None:
            decision.suggested_text = edited_text
        await self.repos.inbox.update_reply_decision(decision)
        return message, decision, action

    async def reject_decision(self, workspace_id: str, decision_id: str) -> ReplyDecision:
        decision = await self.repos.inbox.get_reply_decision(workspace_id, decision_id)
        if decision is None:
            raise LookupError("reply decision not found")
        if decision.status == "sent":
            raise ValueError("sent reply decision cannot be rejected")
        decision.status = "rejected"
        await self.repos.inbox.update_reply_decision(decision)
        return decision

    async def _target_message(
        self,
        workspace_id: str,
        conversation: SocialConversation,
        decision_id: str | None,
    ) -> SocialMessage | None:
        if decision_id:
            decision = await self.repos.inbox.get_reply_decision(workspace_id, decision_id)
            if decision:
                return await self.repos.inbox.get_message(workspace_id, decision.message_id)
        messages = await self.repos.inbox.list_messages(workspace_id, conversation.conversation_id)
        return next((item for item in reversed(messages) if item.direction == "inbound"), None)

    async def _save_note(
        self,
        conversation: SocialConversation,
        text: str,
        *,
        client_message_id: str | None,
    ) -> tuple[SocialMessage, OutboundAction]:
        key = _idempotency_key(
            conversation.platform,
            client_message_id or conversation.conversation_id,
            text,
            "internal_note",
        )
        message_id = new_id("message")
        action = OutboundAction(
            action_id=new_id("outbound"),
            workspace_id=conversation.workspace_id,
            conversation_id=conversation.conversation_id,
            message_id=message_id,
            platform=conversation.platform,
            action_type="internal_note",
            target_id=conversation.conversation_id,
            text=text,
            idempotency_key=key,
            status="sent",
        )
        action, created = await self.repos.inbox.save_outbound_action_once(action)
        if not created and action.message_id:
            existing = await self.repos.inbox.get_message(
                conversation.workspace_id, action.message_id
            )
            if existing:
                return existing, action
        message = SocialMessage(
            message_id=message_id,
            workspace_id=conversation.workspace_id,
            conversation_id=conversation.conversation_id,
            platform=conversation.platform,
            provider_message_id=f"note:{key}",
            direction="internal",
            author_type="internal_note",
            author_name="Workspace member",
            text=text,
            normalized_text=" ".join(text.casefold().split()),
            created_at=utc_now(),
            delivery_status="sent",
            client_message_id=client_message_id,
        )
        message, _ = await self.repos.inbox.insert_message_once(message)
        return message, action


def _idempotency_key(platform: str, target_id: str, text: str, action_type: str) -> str:
    payload = "|".join((platform, target_id, text.strip(), action_type))
    return hashlib.sha256(payload.encode()).hexdigest()
