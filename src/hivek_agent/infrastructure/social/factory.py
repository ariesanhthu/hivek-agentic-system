"""Connector assembly and SOCIAL_MODE routing."""

from __future__ import annotations

import httpx

from hivek_agent.config import Settings
from hivek_agent.infrastructure.social.base import SocialConnector
from hivek_agent.infrastructure.social.facebook import FacebookConnector
from hivek_agent.infrastructure.social.mock import MockSocialConnector
from hivek_agent.infrastructure.social.threads import ThreadsConnector


class SocialConnectorFactory:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=settings.provider_http_timeout_seconds)
        self.mock = MockSocialConnector()
        self._threads = ThreadsConnector(
            self._client,
            base_url=settings.threads_api_base_url,
            version=settings.threads_api_version,
            max_retries=settings.provider_max_retries,
        )
        self._facebook = FacebookConnector(
            self._client,
            base_url=settings.meta_graph_api_base_url,
            version=settings.meta_graph_api_version,
            max_retries=settings.provider_max_retries,
        )

    def for_platform(self, platform: str) -> SocialConnector:
        if self.settings.social_mode == "mock":
            return self.mock
        if platform == "threads":
            return self._threads
        if platform in {"facebook", "instagram"}:
            return self._facebook
        raise ValueError(f"unsupported social platform: {platform}")

    async def close(self) -> None:
        await self._client.aclose()
