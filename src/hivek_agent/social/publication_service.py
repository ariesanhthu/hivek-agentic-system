"""Account activation and publication registration."""

from __future__ import annotations

from datetime import UTC, datetime

from hivek_agent.config import Settings
from hivek_agent.domain import SocialAccount, SocialPublication
from hivek_agent.infrastructure.social import ProviderAPIError, SocialConnectorFactory
from hivek_agent.repositories import Repositories
from hivek_agent.social.credential_vault import CredentialVault


class SocialAccessError(RuntimeError):
    pass


class SocialCredentialLoader:
    def __init__(self, repos: Repositories, vault: CredentialVault, settings: Settings) -> None:
        self.repos = repos
        self.vault = vault
        self.settings = settings

    async def token_for(self, account: SocialAccount) -> str:
        if self.settings.social_mode == "mock":
            return "mock-access-token"
        credential = await self.repos.social.get_credential(
            account.workspace_id, account.credential_id
        )
        if credential is None or credential.status != "active":
            account.status = "reauthorize_required"
            await self.repos.social.save_account(account)
            raise SocialAccessError("social credential is unavailable")
        if credential.expires_at and credential.expires_at <= datetime.now(UTC):
            credential.status = "expired"
            await self.repos.social.save_credential(credential)
            account.status = "reauthorize_required"
            await self.repos.social.save_account(account)
            raise SocialAccessError("social credential has expired")
        return self.vault.decrypt(credential)


class PublicationService:
    def __init__(
        self,
        repos: Repositories,
        connectors: SocialConnectorFactory,
        credentials: SocialCredentialLoader,
    ) -> None:
        self.repos = repos
        self.connectors = connectors
        self.credentials = credentials

    async def register(self, publication: SocialPublication) -> tuple[SocialPublication, bool]:
        account = await self.repos.social.get_account(
            publication.workspace_id, publication.social_account_id
        )
        if account is None:
            raise LookupError("social account not found")
        if account.platform != publication.platform:
            raise ValueError("publication platform does not match social account")
        created = await self.repos.social.save_publication(publication)
        stored = await self.repos.social.find_publication_by_provider_post(
            publication.workspace_id,
            publication.platform,
            publication.platform_post_id,
        )
        if stored is None:
            raise RuntimeError("publication registration was not persisted")
        await self.repos.runs.audit(
            publication.workspace_id,
            "social.publication_registered",
            {
                "publication_id": stored.publication_id,
                "account_id": stored.social_account_id,
                "platform": stored.platform,
                "created": created,
            },
        )
        return stored, created

    async def activate(self, workspace_id: str, account_id: str) -> SocialAccount:
        account = await self.repos.social.get_account(workspace_id, account_id)
        if account is None:
            raise LookupError("social account not found")
        if account.status == "disconnected":
            raise ValueError("social account is disconnected")
        token = await self.credentials.token_for(account)
        connector = self.connectors.for_platform(account.platform)
        account.webhook_status = "pending"
        await self.repos.social.save_account(account)
        try:
            activated = await connector.activate(account, token)
        except ProviderAPIError:
            account.webhook_status = "error"
            account.webhook_error = "provider_activation_failed"
            await self.repos.social.save_account(account)
            raise
        account.provider_account_id = activated.provider_account_id
        account.display_name = activated.display_name or account.display_name
        account.status = "connected"
        account.webhook_status = "active" if activated.webhook_active else "inactive"
        account.webhook_error = None
        account.last_verified_at = datetime.now(UTC)
        await self.repos.social.save_account(account)
        await self.repos.runs.audit(
            workspace_id,
            "social.account_activated",
            {
                "account_id": account_id,
                "platform": account.platform,
                "webhook_status": account.webhook_status,
            },
        )
        return account
