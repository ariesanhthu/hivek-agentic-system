"""On-demand polling bridge for registered publications."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hivek_agent.config import Settings
from hivek_agent.domain import SocialSyncCursor, utc_now
from hivek_agent.infrastructure.social import ProviderAPIError, SocialConnectorFactory
from hivek_agent.repositories import Repositories
from hivek_agent.social.inbound_service import InboundService
from hivek_agent.social.publication_service import SocialCredentialLoader


class SyncError(BaseModel):
    publication_id: str
    code: str


class SyncResult(BaseModel):
    publications_checked: int = 0
    events_found: int = 0
    messages_inserted: int = 0
    duplicates_ignored: int = 0
    decisions_created: int = 0
    auto_replies_sent: int = 0
    errors: list[SyncError] = Field(default_factory=list)


class SyncService:
    def __init__(
        self,
        repos: Repositories,
        connectors: SocialConnectorFactory,
        credentials: SocialCredentialLoader,
        inbound: InboundService,
        settings: Settings,
    ) -> None:
        self.repos = repos
        self.connectors = connectors
        self.credentials = credentials
        self.inbound = inbound
        self.settings = settings

    async def sync(
        self,
        workspace_id: str,
        *,
        publication_ids: list[str] | None = None,
        limit: int = 50,
    ) -> SyncResult:
        if self.settings.inbound_mode == "webhook":
            raise ValueError("polling is disabled while INBOUND_MODE=webhook")
        publications = await self.repos.social.list_publications(
            workspace_id,
            publication_ids=publication_ids,
            sync_enabled_only=True,
            limit=100,
        )
        result = SyncResult()
        for publication in publications:
            result.publications_checked += 1
            account = await self.repos.social.get_account(
                workspace_id, publication.social_account_id
            )
            if account is None or account.status != "connected":
                result.errors.append(
                    SyncError(publication_id=publication.publication_id, code="account_unavailable")
                )
                continue
            if not account.capabilities.read_comments:
                result.errors.append(
                    SyncError(publication_id=publication.publication_id, code="read_not_allowed")
                )
                continue
            cursor = await self.repos.social.get_sync_cursor(
                workspace_id, publication.publication_id
            )
            connector = self.connectors.for_platform(publication.platform)
            try:
                token = await self.credentials.token_for(account)
                fetched = await connector.fetch_inbound(
                    publication,
                    account,
                    token,
                    cursor=cursor.cursor if cursor else None,
                    limit=limit,
                )
            except ProviderAPIError as exc:
                result.errors.append(
                    SyncError(
                        publication_id=publication.publication_id,
                        code=f"provider:{exc.code}",
                    )
                )
                continue
            except Exception as exc:
                result.errors.append(
                    SyncError(
                        publication_id=publication.publication_id,
                        code=type(exc).__name__,
                    )
                )
                continue

            result.events_found += len(fetched.events)
            last_event = None
            for event in fetched.events:
                ingested = await self.inbound.ingest(
                    workspace_id=workspace_id,
                    account_id=account.account_id,
                    normalized_event=event,
                    source="polling",
                )
                last_event = event
                if ingested.inserted:
                    result.messages_inserted += 1
                    if ingested.decision:
                        result.decisions_created += 1
                    if ingested.auto_reply_sent:
                        result.auto_replies_sent += 1
                else:
                    result.duplicates_ignored += 1
            await self.repos.social.save_sync_cursor(
                SocialSyncCursor(
                    workspace_id=workspace_id,
                    publication_id=publication.publication_id,
                    cursor=fetched.next_cursor or (cursor.cursor if cursor else None),
                    last_provider_message_id=(
                        last_event.provider_message_id
                        if last_event
                        else (cursor.last_provider_message_id if cursor else None)
                    ),
                    last_event_at=(
                        last_event.created_at
                        if last_event
                        else (cursor.last_event_at if cursor else None)
                    ),
                    last_synced_at=utc_now(),
                )
            )
        return result
