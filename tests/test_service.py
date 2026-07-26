"""Application service: idempotency, isolation, and the feedback loop.

The service owns the things that cost money or leak data when they are wrong - paying
twice for one draft, starting two runs for one retried request, or letting one
workspace see another's content.
"""

from __future__ import annotations

import pytest

from conftest import THREAD_ID, USER_ID, WORKSPACE_A, WORKSPACE_B
from hivek_agent.agentic.nodes import NodeDeps
from hivek_agent.domain import ContentAsset, HarnessResponse, PostDraft
from hivek_agent.infrastructure.store.base import RUNS
from hivek_agent.infrastructure.store.memory import MemoryStore
from hivek_agent.repositories import Repositories
from hivek_agent.service import AgenticService

WRITE_A_POST = "Viết bài cho Facebook về sản phẩm mới"
SMALLTALK = "Chào bạn"


@pytest.fixture
async def review_asset(repos: Repositories, verbose_text: str) -> ContentAsset:
    """An asset parked at `needs_review`, exactly where the graph leaves one."""
    asset = ContentAsset(
        asset_id="asset_under_review",
        workspace_id=WORKSPACE_A,
        platform="facebook",
        draft=PostDraft(hook="Chào bạn", body=verbose_text, cta="Xem thêm."),
        status="needs_review",
    )
    await repos.content.save_asset(asset)
    return asset


async def _send(
    service: AgenticService,
    *,
    workspace_id: str = WORKSPACE_A,
    message: str = WRITE_A_POST,
    thread_id: str = THREAD_ID,
    idempotency_key: str | None = None,
) -> HarnessResponse:
    return await service.send_message(
        workspace_id=workspace_id,
        user_id=USER_ID,
        thread_id=thread_id,
        message=message,
        idempotency_key=idempotency_key,
    )


# --- idempotency ----------------------------------------------------------


async def test_a_replayed_idempotency_key_returns_the_same_run(
    service: AgenticService, store: MemoryStore
) -> None:
    """A retried POST is the same request, not a new one: the caller gets the original
    run back and no second run is started."""
    first = await _send(service, message=SMALLTALK, idempotency_key="key-1")

    second = await _send(service, message=SMALLTALK, idempotency_key="key-1")

    assert second.run_id == first.run_id
    assert await store.count(RUNS, {"workspace_id": WORKSPACE_A}) == 1


async def test_a_different_idempotency_key_starts_a_new_run(
    service: AgenticService, store: MemoryStore
) -> None:
    """The guard keys on the header; it must not collapse every repeated message."""
    for key in ("key-1", "key-2"):
        await _send(service, message=SMALLTALK, idempotency_key=key)

    assert await store.count(RUNS, {"workspace_id": WORKSPACE_A}) == 2


# --- context hash reuse ---------------------------------------------------


async def test_the_same_context_reuses_the_existing_asset(
    service: AgenticService, seeded_workspace: str
) -> None:
    """Same facts and same prompt version means the same draft. Regenerating would pay
    a model call to reproduce an answer we already have."""
    first = await _send(service, workspace_id=seeded_workspace)

    second = await _send(service, workspace_id=seeded_workspace)

    assert second.asset.asset_id == first.asset.asset_id


async def test_the_same_context_does_not_create_a_second_asset(
    service: AgenticService, seeded_workspace: str, repos: Repositories
) -> None:
    for _ in range(2):
        await _send(service, workspace_id=seeded_workspace)

    assert len(await repos.content.list_assets(seeded_workspace)) == 1


# --- workspace isolation --------------------------------------------------


async def test_another_workspace_sees_none_of_the_first_ones_assets(
    service: AgenticService, seeded_workspace: str, repos: Repositories
) -> None:
    await _send(service, workspace_id=seeded_workspace)

    assert await repos.content.list_assets(WORKSPACE_B) == []


async def test_another_workspace_sees_none_of_the_first_ones_facts(
    seeded_workspace: str, deps: NodeDeps
) -> None:
    """Isolation lives in the repositories, not in each caller: a node never builds the
    query, so it cannot forget the workspace filter."""
    assert await deps.facts.usable_facts(WORKSPACE_B) == {}


async def test_another_workspace_has_no_voice_profile(
    service: AgenticService, review_asset: ContentAsset, repos: Repositories, tightened_text: str
) -> None:
    """A rebuilt voice profile is written per workspace; B must not inherit A's taste."""
    await service.decide(
        workspace_id=WORKSPACE_A,
        asset_id=review_asset.asset_id,
        decision="edit",
        edited_text=tightened_text,
    )

    assert await repos.knowledge.get_voice_profile(WORKSPACE_A) is not None
    assert await repos.knowledge.get_voice_profile(WORKSPACE_B) is None


async def test_a_run_in_another_workspace_starts_from_nothing(
    service: AgenticService, seeded_workspace: str
) -> None:
    """Workspace A is fully set up; B must still be asked for its own brand facts."""
    response = await _send(service, workspace_id=WORKSPACE_B, thread_id="thread-b")

    assert response.status == "needs_user_input"
    assert response.asset is None


# --- the feedback loop ----------------------------------------------------


async def test_decide_edit_writes_a_feedback_event_and_promotes(
    service: AgenticService, review_asset: ContentAsset, repos: Repositories, tightened_text: str
) -> None:
    result = await service.decide(
        workspace_id=WORKSPACE_A,
        asset_id=review_asset.asset_id,
        decision="edit",
        edited_text=tightened_text,
    )

    assert len(await repos.learning.list_feedback(WORKSPACE_A)) == 1
    assert result["learnedThisTurn"] != []


async def test_decide_edit_once_activates_no_rule(
    service: AgenticService, review_asset: ContentAsset, tightened_text: str
) -> None:
    """The same guarantee `test_learning` proves on the service's internals, asserted
    here at the boundary the HTTP layer calls: one edit is observed and stored, but
    nothing it implies is allowed to steer the next draft."""
    result = await service.decide(
        workspace_id=WORKSPACE_A,
        asset_id=review_asset.asset_id,
        decision="edit",
        edited_text=tightened_text,
    )

    assert {item["status"] for item in result["learnedThisTurn"]} == {"candidate"}
    assert result["activeRules"] == []


async def test_decide_approve_marks_the_asset_approved(
    service: AgenticService, review_asset: ContentAsset, repos: Repositories
) -> None:
    await service.decide(
        workspace_id=WORKSPACE_A, asset_id=review_asset.asset_id, decision="approve"
    )

    stored = await repos.content.get_asset(WORKSPACE_A, review_asset.asset_id)
    assert stored is not None and stored.status == "approved"
