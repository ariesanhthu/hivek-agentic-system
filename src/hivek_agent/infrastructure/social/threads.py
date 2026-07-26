"""Threads Graph API connector.

The host and optional version segment are settings. Provider tokens travel only in
the Authorization header and provider errors are redacted before they leave this layer.
"""

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


class ThreadsConnector(HttpSocialConnector):
    platform = "threads"

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
        del cursor  # Provider IDs + repository indexes are the authoritative dedup gate.
        payload = await self._request_json(
            "GET",
            self._url(f"/{publication.platform_post_id}/replies"),
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "fields": (
                    "id,text,username,timestamp,has_replies,root_post,replied_to,"
                    "is_reply,is_reply_owned_by_me"
                ),
                "reverse": "true",
                "limit": min(limit, 100),
            },
        )
        events: list[NormalizedInboundEvent] = []
        for row in payload.get("data", []):
            if not isinstance(row, dict) or not row.get("id"):
                continue
            # Our own replies are outbound acknowledgements, not new customer input.
            if row.get("is_reply_owned_by_me") is True:
                continue
            reply_id = str(row["id"])
            username = str(row.get("username") or "")
            parent = _node_id(row.get("replied_to")) or publication.platform_post_id
            root = _node_id(row.get("root_post")) or publication.platform_post_id
            events.append(
                NormalizedInboundEvent(
                    provider_event_id=reply_id,
                    provider_message_id=reply_id,
                    platform="threads",
                    channel_type="public_reply",
                    provider_account_id=account.provider_account_id,
                    provider_post_id=root,
                    provider_parent_id=parent,
                    provider_thread_key=root,
                    sender_id=username or f"threads-user:{reply_id}",
                    sender_name=username,
                    text=str(row.get("text") or ""),
                    created_at=_parse_time(row.get("timestamp")),
                )
            )
        return FetchInboundResult(
            events=events,
            next_cursor=events[-1].provider_message_id if events else None,
        )

    async def send_public_reply(
        self, account: SocialAccount, access_token: str, *, target_id: str, text: str
    ) -> ProviderSendResult:
        container = await self._request_json(
            "POST",
            self._url("/me/threads"),
            headers={"Authorization": f"Bearer {access_token}"},
            data={
                "media_type": "TEXT",
                "text": text,
                "reply_to_id": target_id,
            },
        )
        creation_id = str(container.get("id") or "")
        if not creation_id:
            raise ProviderCapabilityError("threads", code="missing_container_id")
        published = await self._request_json(
            "POST",
            self._url("/me/threads_publish"),
            headers={"Authorization": f"Bearer {access_token}"},
            data={"creation_id": creation_id},
        )
        message_id = str(published.get("id") or "")
        if not message_id:
            raise ProviderCapabilityError("threads", code="missing_reply_id")
        return ProviderSendResult(provider_message_id=message_id, provider_request_id=creation_id)

    async def send_message(
        self, account: SocialAccount, access_token: str, *, recipient_id: str, text: str
    ) -> ProviderSendResult:
        del account, access_token, recipient_id, text
        raise ProviderCapabilityError("threads", code="private_messages_not_supported")

    async def activate(self, account: SocialAccount, access_token: str) -> ActivationResult:
        profile = await self._request_json(
            "GET",
            self._url("/me"),
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "id,username"},
        )
        provider_id = str(profile.get("id") or "")
        if not provider_id or provider_id != account.provider_account_id:
            raise ProviderCapabilityError("threads", code="account_mismatch")
        return ActivationResult(
            active=True,
            provider_account_id=provider_id,
            display_name=str(profile.get("username") or account.display_name),
            # Threads subscription setup is app-dashboard specific; polling remains the
            # guaranteed demo bridge until a webhook event is observed.
            webhook_active=account.webhook_status == "active",
        )


def _node_id(value: Any) -> str | None:
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    if isinstance(value, str):
        return value
    return None


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)
