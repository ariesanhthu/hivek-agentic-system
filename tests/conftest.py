"""Shared fixtures, and the network guard the whole suite depends on.

`Settings` reads the `.env` sitting next to the package, and that file holds a live
Atlas URI and a live Gemini key. Nothing here may touch either. The autouse `test_env`
fixture forces the two provider switches to their zero-infra values and blanks the
credentials, then drops the `get_settings` `lru_cache` so the override is actually
observed rather than a previously cached `Settings` being reused.

Environment variables outrank `.env` in pydantic-settings, so the blanking wins even
though the file is still on disk. `tests/test_store.py` asserts this rather than
trusting it: if that guard ever fails, every other test in the suite is suspect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from hivek_agent.agentic.graph import build_chat_graph
from hivek_agent.agentic.nodes import NodeDeps, source_from_user
from hivek_agent.config import get_settings
from hivek_agent.infrastructure.llm.mock import MockLLM
from hivek_agent.infrastructure.store.memory import MemoryStore
from hivek_agent.repositories import Repositories
from hivek_agent.service import AgenticService

WORKSPACE_A = "ws-alpha"
WORKSPACE_B = "ws-beta"
USER_ID = "user-1"
THREAD_ID = "thread-1"

# The three facts `brand_profile.REQUIRED_FIELDS` marks `blocking`. Without all three
# the harness stops at the knowledge gate, so anything testing later stages needs them.
BLOCKING_BRAND_FACTS: tuple[tuple[str, Any], ...] = (
    ("brand.name", "ACME Coffee"),
    ("brand.tone", "than thien, gan gui"),
    ("brand.channels", ["facebook", "threads"]),
)


@pytest.fixture(autouse=True)
def test_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin every dependency to an in-process fake before any code reads settings.

    The cache is cleared on both sides: before, so this test sees the overrides; after,
    so a Settings built during a test never leaks into whatever runs next.
    """
    monkeypatch.setenv("AI_AGENT_PROVIDER", "mock")
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("MONGODB_URI", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def store(test_env: None) -> AsyncIterator[MemoryStore]:
    """A connected in-memory store with the real unique indexes applied."""
    memory_store = MemoryStore()
    await memory_store.connect()
    yield memory_store
    await memory_store.close()


@pytest.fixture
def repos(store: MemoryStore) -> Repositories:
    return Repositories(store)


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def deps(repos: Repositories, llm: MockLLM) -> NodeDeps:
    return NodeDeps(repos, llm, token_budget=12000)


@pytest.fixture
def graph(deps: NodeDeps) -> Any:
    return build_chat_graph(deps)


@pytest.fixture
def service(deps: NodeDeps, graph: Any, repos: Repositories) -> AgenticService:
    return AgenticService(deps, graph, repos)


@pytest.fixture
async def seeded_workspace(deps: NodeDeps) -> str:
    """Workspace A with every blocking brand fact confirmed by the user."""
    for predicate, value in BLOCKING_BRAND_FACTS:
        await deps.facts.upsert_fact(
            workspace_id=WORKSPACE_A,
            subject_id="workspace",
            predicate=predicate,
            object_value=value,
            source=source_from_user(f"user/setup/{predicate}"),
            confidence=1.0,
            approval_status="confirmed",
        )
    return WORKSPACE_A


@pytest.fixture
def verbose_text() -> str:
    """A long, hype-heavy draft: the 'before' side of a realistic tightening edit."""
    return (
        "Xin chào các bạn, hôm nay chúng tôi rất vui mừng được giới thiệu tới quý khách "
        "hàng một sản phẩm hoàn toàn mới với rất nhiều tính năng vượt trội và hấp dẫn. "
        "Sản phẩm này đã được nghiên cứu kỹ lưỡng trong nhiều năm liền bởi đội ngũ."
    )


@pytest.fixture
def tightened_text() -> str:
    """The 'after' side: short enough that the length signal clears its threshold."""
    return "Sản phẩm mới đã có mặt. Ghé xem thử nhé."
