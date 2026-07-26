"""Provider-neutral connector protocol and safe HTTP helpers."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from hivek_agent.domain import NormalizedInboundEvent, SocialAccount, SocialPublication


class FetchInboundResult(BaseModel):
    events: list[NormalizedInboundEvent] = Field(default_factory=list)
    next_cursor: str | None = None


class ProviderSendResult(BaseModel):
    provider_message_id: str
    provider_request_id: str | None = None


class ActivationResult(BaseModel):
    active: bool
    provider_account_id: str
    display_name: str = ""
    webhook_active: bool = False


class ProviderAPIError(RuntimeError):
    """A redacted provider error safe to put in logs and API responses."""

    def __init__(self, provider: str, *, status: int | None = None, code: str = "unknown") -> None:
        self.provider = provider
        self.status = status
        self.code = code
        suffix = f" status={status}" if status is not None else ""
        super().__init__(f"{provider} API request failed{suffix} code={code}")


class ProviderCapabilityError(ProviderAPIError):
    pass


class SocialConnector(Protocol):
    platform: str

    async def fetch_inbound(
        self,
        publication: SocialPublication,
        account: SocialAccount,
        access_token: str,
        *,
        cursor: str | None,
        limit: int,
    ) -> FetchInboundResult: ...

    async def send_public_reply(
        self, account: SocialAccount, access_token: str, *, target_id: str, text: str
    ) -> ProviderSendResult: ...

    async def send_message(
        self, account: SocialAccount, access_token: str, *, recipient_id: str, text: str
    ) -> ProviderSendResult: ...

    async def activate(self, account: SocialAccount, access_token: str) -> ActivationResult: ...


class HttpSocialConnector:
    platform = "unknown"

    def __init__(self, client: httpx.AsyncClient, *, max_retries: int = 2) -> None:
        self._client = client
        self._max_retries = max(0, max_retries)

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        safe_to_retry = method.upper() in {"GET", "HEAD"}
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                if safe_to_retry and attempt < self._max_retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                raise ProviderAPIError(self.platform, code=type(exc).__name__) from exc

            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.is_success and isinstance(payload, dict):
                return payload

            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            code = str(error.get("code") or error.get("type") or "http_error")
            retryable = response.status_code == 429 or response.status_code >= 500
            if safe_to_retry and retryable and attempt < self._max_retries:
                await asyncio.sleep(0.25 * (2**attempt))
                continue
            raise ProviderAPIError(self.platform, status=response.status_code, code=code)

        raise ProviderAPIError(self.platform, code="retry_exhausted")


def versioned_url(base_url: str, version: str, path: str) -> str:
    pieces = [base_url.rstrip("/")]
    if version.strip():
        pieces.append(version.strip("/"))
    pieces.append(path.lstrip("/"))
    return "/".join(pieces)
