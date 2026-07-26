"""The harness graph.

Two things are pinned here. First, routing is deterministic before it is expensive:
keywords decide, and only a genuinely unclear message costs a model call. Second, and
more importantly, the graph has no path to publication - a draft always stops at a
human, and a run without the facts it needs asks instead of guessing.
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import THREAD_ID, USER_ID, WORKSPACE_A
from hivek_agent.agentic.graph import _facts_gate
from hivek_agent.agentic.nodes import HarnessNodes, NodeDeps, classify_intent_keywords
from hivek_agent.agentic.tools import DEFAULT_SCOPES
from hivek_agent.domain import HarnessState
from hivek_agent.repositories import Repositories
from hivek_agent.service import AgenticService

WRITE_A_POST = "Viết bài cho Facebook về sản phẩm mới"


def _state(**overrides: Any) -> HarnessState:
    return HarnessState(
        run_id="run-1",
        workspace_id=WORKSPACE_A,
        user_id=USER_ID,
        thread_id=THREAD_ID,
        **overrides,
    )


# --- keyword routing ------------------------------------------------------


def test_keyword_router_maps_vietnamese_phrases() -> None:
    """Accent-insensitive, and mirrors the client's `resolveIntent` so the UI and the
    backend agree on what a phrase means. Costs nothing: no model is involved."""
    phrases = {
        WRITE_A_POST: "create_post",
        "Lên kế hoạch đăng bài tuần này": "create_content_plan",
        "Thiết lập workspace giúp mình": "setup",
        "Phân tích hiệu quả bài đăng tháng qua": "analyze_performance",
        "Cập nhật giá sản phẩm": "update_knowledge",
    }

    assert {message: classify_intent_keywords(message) for message in phrases} == phrases


def test_keyword_router_returns_none_when_unsure() -> None:
    """None means "escalate", not "smalltalk". Guessing would route a real request to
    the wrong branch; the caller decides whether the message is worth a model call."""
    unclear = ["Hôm nay trời đẹp quá", "", "   "]

    assert [classify_intent_keywords(message) for message in unclear] == [None, None, None]


# --- nodes and gates, called directly -------------------------------------


def test_facts_gate_blocks_when_input_is_required() -> None:
    """A deterministic gate: no model gets to decide it has enough information."""
    assert _facts_gate(_state(status="needs_user_input")) == "blocked"
    assert _facts_gate(_state(status="running")) == "ready"


async def test_authenticate_node_grants_the_default_scopes(deps: NodeDeps) -> None:
    """Nodes know nothing about LangGraph, so they are callable with a bare state."""
    nodes = HarnessNodes(deps)

    result = await nodes.authenticate_and_authorize(_state())

    assert result.request_payload["scopes"] == list(DEFAULT_SCOPES)


async def test_smalltalk_node_completes_without_a_model(deps: NodeDeps) -> None:
    nodes = HarnessNodes(deps)

    result = await nodes.handle_smalltalk(_state(intent="smalltalk"))

    assert result.status == "completed"


# --- end to end, no facts -------------------------------------------------


@pytest.fixture
async def run_without_facts(service: AgenticService) -> Any:
    return await service.send_message(
        workspace_id=WORKSPACE_A, user_id=USER_ID, thread_id=THREAD_ID, message=WRITE_A_POST
    )


async def test_run_without_facts_asks_the_user_with_a_widget(run_without_facts: Any) -> None:
    assert run_without_facts.status == "needs_user_input"
    assert run_without_facts.widget == {"type": "brand-form"}


async def test_run_without_facts_reports_the_gaps_it_searched_for(
    run_without_facts: Any,
) -> None:
    """The blueprint requires the system to prove where it looked before asking."""
    blocking = [
        item.field for item in run_without_facts.missing_items if item.severity == "blocking"
    ]

    assert blocking == ["brand.name", "brand.tone", "brand.channels"]
    assert all(item.searched_sources for item in run_without_facts.missing_items)


async def test_run_without_facts_creates_no_asset(
    run_without_facts: Any, repos: Repositories
) -> None:
    """The gate sits upstream of generation: a blocked run must not have paid for a
    draft, nor left one behind for someone to approve."""
    assert run_without_facts.asset is None
    assert await repos.content.list_assets(WORKSPACE_A) == []


# --- end to end, facts present --------------------------------------------


@pytest.fixture
async def run_with_facts(service: AgenticService, seeded_workspace: str) -> Any:
    return await service.send_message(
        workspace_id=seeded_workspace, user_id=USER_ID, thread_id=THREAD_ID, message=WRITE_A_POST
    )


async def test_run_with_facts_stops_for_approval(run_with_facts: Any) -> None:
    assert run_with_facts.status == "needs_approval"
    assert len(run_with_facts.citations) == 3


async def test_run_with_facts_persists_the_asset_for_review(
    run_with_facts: Any, repos: Repositories
) -> None:
    """Approval must not depend on a live graph: the asset is durable, so the review
    cycle survives a worker restart and works across days."""
    stored = await repos.content.get_asset(WORKSPACE_A, run_with_facts.asset.asset_id)

    assert stored is not None
    assert stored.status == "needs_review"


async def test_run_with_facts_never_auto_publishes(
    run_with_facts: Any, repos: Repositories
) -> None:
    """Everything stops at a human. No node in the graph can reach `approved`,
    `scheduled` or `published` on its own."""
    assets = await repos.content.list_assets(WORKSPACE_A)

    assert [asset.status for asset in assets] == ["needs_review"]
