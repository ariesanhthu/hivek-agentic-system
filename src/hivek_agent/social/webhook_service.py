"""Signature verification and normalized webhook ingestion for Meta/Threads."""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from hivek_agent.config import Settings
from hivek_agent.domain import NormalizedInboundEvent, SocialWebhookEvent, utc_now
from hivek_agent.repositories import Repositories, new_id
from hivek_agent.social.inbound_service import InboundService


class WebhookResult(BaseModel):
    received: int = 0
    inserted: int = 0
    duplicates: int = 0
    ignored: int = 0


class WebhookVerificationError(ValueError):
    pass


class WebhookService:
    def __init__(
        self,
        repos: Repositories,
        inbound: InboundService,
        settings: Settings,
    ) -> None:
        self.repos = repos
        self.inbound = inbound
        self.settings = settings

    def verify_challenge(self, mode: str, token: str, challenge: str) -> str:
        if mode != "subscribe" or not self.settings.webhook_verify_token:
            raise WebhookVerificationError("webhook verification is not configured")
        if not hmac.compare_digest(token, self.settings.webhook_verify_token):
            raise WebhookVerificationError("webhook verification token is invalid")
        return challenge

    def verify_signature(self, provider: str, raw_body: bytes, signature: str | None) -> None:
        if not self.settings.webhook_signature_required and not signature:
            return
        secret = (
            self.settings.threads_app_secret
            if provider == "threads"
            else self.settings.meta_app_secret
        )
        if not secret:
            raise WebhookVerificationError("webhook signature secret is not configured")
        if not signature or not signature.startswith("sha256="):
            raise WebhookVerificationError("webhook signature is missing")
        expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise WebhookVerificationError("webhook signature is invalid")

    async def process(self, provider: str, payload: dict[str, Any]) -> WebhookResult:
        events = _normalize_payload(provider, payload)
        result = WebhookResult(received=len(events))
        for provider_account_id, normalized in events:
            account = await self.repos.social.find_any_account_by_provider(
                normalized.platform, provider_account_id
            )
            if account is None or account.status != "connected":
                result.ignored += 1
                continue
            event = SocialWebhookEvent(
                webhook_event_id=new_id("webhook"),
                workspace_id=account.workspace_id,
                provider=normalized.platform,
                provider_event_id=normalized.provider_event_id,
                account_id=account.account_id,
                payload_hash=hashlib.sha256(
                    f"{provider}|{normalized.provider_event_id}".encode()
                ).hexdigest(),
            )
            if not await self.repos.inbox.insert_webhook_event_once(event):
                result.duplicates += 1
                continue
            try:
                ingested = await self.inbound.ingest(
                    workspace_id=account.workspace_id,
                    account_id=account.account_id,
                    normalized_event=normalized,
                    source="webhook",
                )
                event.status = "processed"
                event.processed_at = utc_now()
                if ingested.inserted:
                    result.inserted += 1
                else:
                    result.duplicates += 1
            except (LookupError, ValueError) as exc:
                event.status = "ignored"
                event.error = type(exc).__name__
                event.processed_at = utc_now()
                result.ignored += 1
            except Exception as exc:
                event.status = "failed"
                event.error = type(exc).__name__
                event.processed_at = utc_now()
                result.ignored += 1
            await self.repos.inbox.update_webhook_event(event)
        return result


def _normalize_payload(
    provider: str, payload: dict[str, Any]
) -> list[tuple[str, NormalizedInboundEvent]]:
    if provider == "threads":
        return _normalize_threads(payload)
    return _normalize_meta(payload)


def _normalize_meta(payload: dict[str, Any]) -> list[tuple[str, NormalizedInboundEvent]]:
    normalized: list[tuple[str, NormalizedInboundEvent]] = []
    for entry in _dicts(payload.get("entry")):
        page_id = str(entry.get("id") or "")
        for item in _dicts(entry.get("messaging")):
            message = item.get("message") if isinstance(item.get("message"), dict) else {}
            if message.get("is_echo") is True:
                continue
            sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
            recipient = item.get("recipient") if isinstance(item.get("recipient"), dict) else {}
            provider_account_id = page_id or str(recipient.get("id") or "")
            message_id = str(message.get("mid") or "")
            sender_id = str(sender.get("id") or "")
            if not provider_account_id or not message_id or not sender_id:
                continue
            normalized.append(
                (
                    provider_account_id,
                    NormalizedInboundEvent(
                        provider_event_id=message_id,
                        provider_message_id=message_id,
                        platform="facebook",
                        channel_type="dm",
                        provider_account_id=provider_account_id,
                        provider_thread_key=f"messenger:{sender_id}",
                        sender_id=sender_id,
                        text=str(message.get("text") or ""),
                        created_at=_timestamp_ms(item.get("timestamp")),
                    ),
                )
            )
        for change in _dicts(entry.get("changes")):
            if change.get("field") not in {"feed", "comments"}:
                continue
            value = change.get("value") if isinstance(change.get("value"), dict) else {}
            if value.get("item") not in {None, "comment"}:
                continue
            comment_id = str(value.get("comment_id") or value.get("id") or "")
            post_id = str(value.get("post_id") or value.get("parent_id") or "")
            sender = value.get("from") if isinstance(value.get("from"), dict) else {}
            sender_id = str(sender.get("id") or value.get("sender_id") or "")
            if not page_id or not comment_id or not post_id:
                continue
            if sender_id and sender_id == page_id:
                continue
            normalized.append(
                (
                    page_id,
                    NormalizedInboundEvent(
                        provider_event_id=comment_id,
                        provider_message_id=comment_id,
                        platform="facebook",
                        channel_type="comment",
                        provider_account_id=page_id,
                        provider_post_id=post_id,
                        provider_parent_id=str(value.get("parent_id") or post_id),
                        provider_thread_key=post_id,
                        sender_id=sender_id or f"facebook-user:{comment_id}",
                        sender_name=str(sender.get("name") or value.get("sender_name") or ""),
                        text=str(value.get("message") or ""),
                        created_at=_parse_time(value.get("created_time")),
                    ),
                )
            )
    return normalized


def _normalize_threads(payload: dict[str, Any]) -> list[tuple[str, NormalizedInboundEvent]]:
    normalized: list[tuple[str, NormalizedInboundEvent]] = []
    for entry in _dicts(payload.get("entry")):
        account_id = str(entry.get("id") or "")
        for change in _dicts(entry.get("changes")):
            value = change.get("value") if isinstance(change.get("value"), dict) else {}
            reply_id = str(value.get("id") or value.get("reply_id") or "")
            root = _node_id(value.get("root_post")) or str(
                value.get("media_id") or value.get("post_id") or ""
            )
            parent = _node_id(value.get("replied_to")) or str(value.get("parent_id") or root)
            username = str(value.get("username") or value.get("sender_name") or "")
            if not account_id or not reply_id or not root:
                continue
            if (
                value.get("is_reply_owned_by_me") is True
                or str(value.get("sender_id") or "") == account_id
            ):
                continue
            normalized.append(
                (
                    account_id,
                    NormalizedInboundEvent(
                        provider_event_id=reply_id,
                        provider_message_id=reply_id,
                        platform="threads",
                        channel_type="public_reply",
                        provider_account_id=account_id,
                        provider_post_id=root,
                        provider_parent_id=parent,
                        provider_thread_key=root,
                        sender_id=str(value.get("sender_id") or username or reply_id),
                        sender_name=username,
                        text=str(value.get("text") or ""),
                        created_at=_parse_time(value.get("timestamp")),
                    ),
                )
            )
    return normalized


def _dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _node_id(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    if isinstance(value, str):
        return value
    return None


def _timestamp_ms(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
    except (TypeError, ValueError, OSError):
        return datetime.now(UTC)


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return _timestamp_ms(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)
