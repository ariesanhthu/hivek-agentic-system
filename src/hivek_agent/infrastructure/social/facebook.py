"""Facebook Page comments and Messenger connector."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from hivek_agent.domain import NormalizedInboundEvent, SocialAccount, SocialPublication
from hivek_agent.infrastructure.social.base import (
    ActivationResult,
    FetchInboundResult,
    HttpSocialConnector,
    ProviderCapabilityError,
    ProviderSendResult,
    versioned_url,
)


class FacebookConnector(HttpSocialConnector):
    platform = "facebook"

    def __init__(self, *args: Any, base_url: str, version: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._base_url = base_url
        self._version = version

    def _url(self, path: str) -> str:
        return versioned_url(self._base_url, self._version, path)

    async def fetch_inbound(
        self,
        publication: SocialPublication,
        account: SocialAccount,
        access_token: str,
        *,
        cursor: str | None,
        limit: int,
    ) -> FetchInboundResult:
        del cursor  # Poll the newest bounded page; repository provider IDs deduplicate.
        params: dict[str, Any] = {
            "fields": (
                "id,message,from,created_time,parent,"
                "comments.limit(50){id,message,from,created_time,parent}"
            ),
            "limit": min(limit, 100),
            "order": "reverse_chronological",
        }
        payload = await self._request_json(
            "GET",
            self._url(f"/{publication.platform_post_id}/comments"),
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
        events: list[NormalizedInboundEvent] = []
        for row, nested_parent_id in _comments_with_replies(payload):
            if not isinstance(row, dict) or not row.get("id"):
                continue
            sender = row.get("from") if isinstance(row.get("from"), dict) else {}
            sender_id = str(sender.get("id") or "")
            if sender_id and sender_id == account.provider_account_id:
                continue
            comment_id = str(row["id"])
            parent_id = (
                _node_id(row.get("parent"))
                or nested_parent_id
                or publication.platform_post_id
            )
            events.append(
                NormalizedInboundEvent(
                    provider_event_id=comment_id,
                    provider_message_id=comment_id,
                    platform="facebook",
                    channel_type="comment",
                    provider_account_id=account.provider_account_id,
                    provider_post_id=publication.platform_post_id,
                    provider_parent_id=parent_id,
                    provider_thread_key=publication.platform_post_id,
                    sender_id=sender_id or f"facebook-user:{comment_id}",
                    sender_name=str(sender.get("name") or ""),
                    text=str(row.get("message") or ""),
                    created_at=_parse_time(row.get("created_time")),
                )
            )
        return FetchInboundResult(
            events=events,
            next_cursor=events[0].provider_message_id if events else None,
        )

    async def send_public_reply(
        self, account: SocialAccount, access_token: str, *, target_id: str, text: str
    ) -> ProviderSendResult:
        del account
        payload = await self._request_json(
            "POST",
            self._url(f"/{target_id}/comments"),
            headers={"Authorization": f"Bearer {access_token}"},
            data={"message": text},
        )
        message_id = str(payload.get("id") or "")
        if not message_id:
            raise ProviderCapabilityError("facebook", code="missing_comment_reply_id")
        return ProviderSendResult(provider_message_id=message_id)

    async def send_message(
        self, account: SocialAccount, access_token: str, *, recipient_id: str, text: str
    ) -> ProviderSendResult:
        payload = await self._request_json(
            "POST",
            self._url(f"/{account.provider_account_id}/messages"),
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "messaging_type": "RESPONSE",
                "recipient": {"id": recipient_id},
                "message": {"text": text},
            },
        )
        message_id = str(payload.get("message_id") or "")
        if not message_id:
            raise ProviderCapabilityError("facebook", code="missing_message_id")
        return ProviderSendResult(
            provider_message_id=message_id,
            provider_request_id=str(payload.get("recipient_id") or "") or None,
        )

    async def activate(self, account: SocialAccount, access_token: str) -> ActivationResult:
        profile = await self._request_json(
            "GET",
            self._url(f"/{account.provider_account_id}"),
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "id,name"},
        )
        provider_id = str(profile.get("id") or "")
        if provider_id != account.provider_account_id:
            raise ProviderCapabilityError("facebook", code="page_account_mismatch")
        subscribed_fields = ["feed"]
        if account.capabilities.read_messages:
            subscribed_fields.extend(
                ["messages", "messaging_postbacks", "message_deliveries", "message_reads"]
            )
        subscribed = await self._request_json(
            "POST",
            self._url(f"/{account.provider_account_id}/subscribed_apps"),
            headers={"Authorization": f"Bearer {access_token}"},
            data={
                "subscribed_fields": ",".join(subscribed_fields),
            },
        )
        return ActivationResult(
            active=True,
            provider_account_id=provider_id,
            display_name=str(profile.get("name") or account.display_name),
            webhook_active=subscribed.get("success") is True,
        )


def _node_id(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    if isinstance(value, str):
        return value
    return None


def _comments_with_replies(payload: dict[str, Any]) -> list[tuple[dict[str, Any], str | None]]:
    rows: list[tuple[dict[str, Any], str | None]] = []
    data = payload.get("data") if isinstance(payload.get("data"), list) else []
    for item in data:
        if not isinstance(item, dict):
            continue
        rows.append((item, None))
        nested = item.get("comments") if isinstance(item.get("comments"), dict) else {}
        replies = nested.get("data") if isinstance(nested.get("data"), list) else []
        for reply in replies:
            if isinstance(reply, dict):
                rows.append((reply, str(item.get("id") or "") or None))
    return rows


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)
