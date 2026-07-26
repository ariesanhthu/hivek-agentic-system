"""Deterministic social connector for local round-trip tests and demo controls."""

from __future__ import annotations

from collections import defaultdict

from hivek_agent.domain import NormalizedInboundEvent, SocialAccount, SocialPublication
from hivek_agent.infrastructure.social.base import (
    ActivationResult,
    FetchInboundResult,
    ProviderSendResult,
)
from hivek_agent.repositories import new_id


class MockSocialConnector:
    platform = "mock"

    def __init__(self) -> None:
        self._events: dict[str, list[NormalizedInboundEvent]] = defaultdict(list)
        self.sent: list[dict[str, str]] = []

    def queue_event(self, publication_id: str, event: NormalizedInboundEvent) -> None:
        self._events[publication_id].append(event)

    async def fetch_inbound(
        self,
        publication: SocialPublication,
        account: SocialAccount,
        access_token: str,
        *,
        cursor: str | None,
        limit: int,
    ) -> FetchInboundResult:
        del account, access_token, cursor
        events = list(self._events.get(publication.publication_id, []))[:limit]
        next_cursor = events[-1].provider_message_id if events else None
        return FetchInboundResult(events=events, next_cursor=next_cursor)

    async def send_public_reply(
        self, account: SocialAccount, access_token: str, *, target_id: str, text: str
    ) -> ProviderSendResult:
        del access_token
        message_id = new_id("mockreply")
        self.sent.append(
            {
                "kind": "public_reply",
                "account_id": account.account_id,
                "target_id": target_id,
                "text": text,
            }
        )
        return ProviderSendResult(provider_message_id=message_id, provider_request_id=message_id)

    async def send_message(
        self, account: SocialAccount, access_token: str, *, recipient_id: str, text: str
    ) -> ProviderSendResult:
        del access_token
        message_id = new_id("mockmessage")
        self.sent.append(
            {
                "kind": "message",
                "account_id": account.account_id,
                "target_id": recipient_id,
                "text": text,
            }
        )
        return ProviderSendResult(provider_message_id=message_id, provider_request_id=message_id)

    async def activate(self, account: SocialAccount, access_token: str) -> ActivationResult:
        del access_token
        return ActivationResult(
            active=True,
            provider_account_id=account.provider_account_id,
            display_name=account.display_name,
            webhook_active=False,
        )
