from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

from hivek_agent.api.app import create_app
from hivek_agent.config import Settings, get_settings
from hivek_agent.domain import (
    NormalizedInboundEvent,
    SocialAccount,
    SocialCapabilities,
    SocialPublication,
)
from hivek_agent.infrastructure.llm.mock import MockLLM
from hivek_agent.infrastructure.social import ProviderAPIError
from hivek_agent.infrastructure.social.facebook import FacebookConnector
from hivek_agent.infrastructure.social.threads import ThreadsConnector
from hivek_agent.repositories import Repositories
from hivek_agent.social.runtime import SocialRuntime


@pytest.mark.asyncio
async def test_current_meta_mutation_contracts_keep_tokens_out_of_urls() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "access_token" not in str(request.url)
        assert request.headers["Authorization"] == "Bearer provider-token"
        if request.url.path.endswith("/me/threads"):
            return httpx.Response(200, json={"id": "container_1"})
        if request.url.path.endswith("/me/threads_publish"):
            return httpx.Response(200, json={"id": "thread_reply_1"})
        if request.url.path.endswith("/page_1/messages"):
            return httpx.Response(
                200,
                json={"message_id": "message_1", "recipient_id": "customer_1"},
            )
        return httpx.Response(404, json={"error": {"code": "not_found"}})

    account = SocialAccount(
        account_id="social_1",
        workspace_id="demo_workspace",
        platform="threads",
        provider_account_id="page_1",
        credential_id="cred_1",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        threads = ThreadsConnector(
            client,
            base_url="https://graph.threads.net",
            version="",
            max_retries=2,
        )
        sent_thread = await threads.send_public_reply(
            account,
            "provider-token",
            target_id="reply_target_1",
            text="Cảm ơn bạn.",
        )
        assert sent_thread.provider_message_id == "thread_reply_1"

        facebook = FacebookConnector(
            client,
            base_url="https://graph.facebook.com",
            version="v25.0",
            max_retries=2,
        )
        sent_message = await facebook.send_message(
            account,
            "provider-token",
            recipient_id="customer_1",
            text="Cảm ơn bạn.",
        )
        assert sent_message.provider_message_id == "message_1"

    assert [request.url.path for request in requests] == [
        "/me/threads",
        "/me/threads_publish",
        "/v25.0/page_1/messages",
    ]


@pytest.mark.asyncio
async def test_provider_post_is_not_blindly_retried_after_5xx() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": {"code": "temporarily_unavailable"}})

    account = SocialAccount(
        account_id="social_1",
        workspace_id="demo_workspace",
        platform="threads",
        provider_account_id="threads_1",
        credential_id="cred_1",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        connector = ThreadsConnector(
            client,
            base_url="https://graph.threads.net",
            version="",
            max_retries=3,
        )
        with pytest.raises(ProviderAPIError):
            await connector.send_public_reply(
                account,
                "provider-token",
                target_id="reply_target_1",
                text="Cảm ơn bạn.",
            )
    assert attempts == 1


@pytest.mark.asyncio
async def test_mock_publication_to_inbox_to_idempotent_reply(
    repos: Repositories,
) -> None:
    settings = Settings(
        store_backend="memory",
        social_mode="mock",
        inbound_mode="polling",
        auto_reply_mode="suggestion",
        demo_workspace_id="demo_workspace",
    )
    runtime = SocialRuntime(repos, MockLLM(), settings)
    try:
        account = SocialAccount(
            account_id="social_threads",
            workspace_id="demo_workspace",
            platform="threads",
            provider_account_id="mock-threads",
            display_name="@hivek.demo",
            credential_id="cred_threads",
            connection_mode="mock",
            capabilities=SocialCapabilities(
                publish=True,
                read_comments=True,
                reply_comments=True,
            ),
        )
        await repos.social.save_account(account)
        publication = SocialPublication(
            publication_id="pub_demo",
            workspace_id="demo_workspace",
            social_account_id=account.account_id,
            platform="threads",
            platform_post_id="thread_root_1",
            text="Học hiệu quả bắt đầu từ mục tiêu nhỏ.",
            reply_suggestions=["Bắt đầu từ một mục tiêu nhỏ trong 45 phút nhé."],
            published_at=datetime.now(UTC),
        )
        stored, created = await runtime.publications.register(publication)
        assert created is True
        assert stored.platform_post_id == "thread_root_1"

        event = NormalizedInboundEvent(
            provider_event_id="reply_1",
            provider_message_id="reply_1",
            platform="threads",
            channel_type="public_reply",
            provider_account_id=account.provider_account_id,
            provider_post_id=publication.platform_post_id,
            provider_parent_id=publication.platform_post_id,
            provider_thread_key=publication.platform_post_id,
            sender_id="customer_1",
            sender_name="Khách demo",
            text="Xin chào",
            created_at=datetime.now(UTC),
        )
        runtime.connectors.mock.queue_event(publication.publication_id, event)

        first_sync = await runtime.sync.sync("demo_workspace")
        assert first_sync.messages_inserted == 1
        assert first_sync.decisions_created == 1

        conversation = (await repos.inbox.list_conversations("demo_workspace"))[0]
        decision = await repos.inbox.latest_pending_decision(
            "demo_workspace", conversation.conversation_id
        )
        assert decision is not None
        assert decision.action == "suggestion"
        assert decision.suggested_text

        message, sent_decision, action = await runtime.replies.send_decision(
            workspace_id="demo_workspace",
            decision_id=decision.decision_id,
        )
        assert sent_decision.status == "sent"
        assert action.status == "sent"
        assert message.provider_message_id.startswith("mockreply_")

        repeated_message, _, repeated_action = await runtime.replies.send_decision(
            workspace_id="demo_workspace",
            decision_id=decision.decision_id,
        )
        assert repeated_message.message_id == message.message_id
        assert repeated_action.action_id == action.action_id

        second_sync = await runtime.sync.sync("demo_workspace")
        assert second_sync.messages_inserted == 0
        assert second_sync.duplicates_ignored == 1
        assert (
            len(await repos.inbox.list_messages("demo_workspace", conversation.conversation_id))
            == 2
        )
    finally:
        await runtime.close()


def test_social_api_auth_replay_and_webhook_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_DEMO_ACCESS_TOKEN", "demo-secret")
    monkeypatch.setenv("AGENTIC_INTERNAL_SECRET", "internal-secret")
    monkeypatch.setenv("AGENTIC_DEMO_WORKSPACE_ID", "demo_workspace")
    monkeypatch.setenv("DEMO_WORKSPACE_ID", "demo_workspace")
    monkeypatch.setenv("SOCIAL_MODE", "mock")
    monkeypatch.setenv("INBOUND_MODE", "hybrid")
    monkeypatch.setenv("META_APP_SECRET", "meta-secret")
    monkeypatch.setenv("WEBHOOK_SIGNATURE_REQUIRED", "true")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        path = "/v1/workspaces/demo_workspace/social/status"
        assert client.get(path).status_code == 401
        authorized = client.get(
            path,
            headers={
                "Authorization": "Bearer demo-secret",
                "X-Workspace-Id": "demo_workspace",
            },
        )
        assert authorized.status_code == 200
        assert authorized.json()["socialMode"] == "mock"

        stale = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
        internal_headers = {
            "Authorization": "Bearer internal-secret",
            "X-Request-Id": "request-1",
            "X-Timestamp": stale,
        }
        body = {"workspaceId": "demo_workspace", "accountId": "missing"}
        assert (
            client.post(
                "/internal/social/accounts/activate",
                headers=internal_headers,
                json=body,
            ).status_code
            == 401
        )

        internal_headers["X-Timestamp"] = datetime.now(UTC).isoformat()
        assert (
            client.post(
                "/internal/social/accounts/activate",
                headers=internal_headers,
                json=body,
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/internal/social/accounts/activate",
                headers=internal_headers,
                json=body,
            ).status_code
            == 409
        )

        raw = b'{"object":"page","entry":[]}'
        signature = "sha256=" + hmac.new(b"meta-secret", raw, hashlib.sha256).hexdigest()
        webhook = client.post(
            "/webhooks/meta",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
            },
        )
        assert webhook.status_code == 200
        assert webhook.json()["received"] == 1

        bad_webhook = client.post(
            "/webhooks/meta",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": "sha256=bad",
            },
        )
        assert bad_webhook.status_code == 403

    get_settings.cache_clear()
