"""Process-level assembly for social services."""

from __future__ import annotations

from hivek_agent.config import Settings
from hivek_agent.infrastructure.llm import LLMGateway
from hivek_agent.infrastructure.social import SocialConnectorFactory
from hivek_agent.reply import ReplyDecisionEngine
from hivek_agent.repositories import Repositories
from hivek_agent.social.credential_vault import CredentialVault
from hivek_agent.social.inbound_service import InboundService
from hivek_agent.social.publication_service import (
    PublicationService,
    SocialCredentialLoader,
)
from hivek_agent.social.reply_service import ReplyService
from hivek_agent.social.sync_service import SyncService
from hivek_agent.social.webhook_service import WebhookService


class SocialRuntime:
    def __init__(self, repos: Repositories, llm: LLMGateway, settings: Settings) -> None:
        self.settings = settings
        self.repos = repos
        self.vault = CredentialVault(settings.token_encryption_key_base64)
        self.connectors = SocialConnectorFactory(settings)
        self.credentials = SocialCredentialLoader(repos, self.vault, settings)
        self.publications = PublicationService(repos, self.connectors, self.credentials)
        self.reply_engine = ReplyDecisionEngine(repos, llm, settings)
        self.replies = ReplyService(repos, self.connectors, self.credentials)
        self.inbound = InboundService(repos, self.reply_engine, self.replies)
        self.sync = SyncService(repos, self.connectors, self.credentials, self.inbound, settings)
        self.webhooks = WebhookService(repos, self.inbound, settings)

    async def close(self) -> None:
        await self.connectors.close()
