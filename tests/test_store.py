"""Document store contract.

`MemoryStore` and `MongoStore` are meant to be interchangeable, so the in-memory one
has to reproduce Mongo's semantics for the operator subset the repositories use. These
tests pin that subset, plus the unique indexes that make retried writes idempotent
rather than duplicating.
"""

from __future__ import annotations

from typing import Any

import pytest

from hivek_agent.config import get_settings
from hivek_agent.infrastructure.store.base import (
    NODE_RUNS,
    RUNS,
    DuplicateKeyError,
    StoreError,
    matches,
)
from hivek_agent.infrastructure.store.memory import MemoryStore

WS = "ws-alpha"


def _run(run_id: str, **extra: Any) -> dict[str, Any]:
    return {"run_id": run_id, "workspace_id": WS, "status": "running", **extra}


def _node_run(node_run_id: str, **extra: Any) -> dict[str, Any]:
    return {"node_run_id": node_run_id, "run_id": "run-1", "workspace_id": WS, **extra}


# --- network safety -------------------------------------------------------


def test_the_test_environment_pins_every_dependency_in_process() -> None:
    """The load-bearing guard for the whole suite: a live Atlas URI and a live Gemini
    key sit in the `.env` next to the package, and `Settings` reads that file by
    default. If this fails, every other test in the suite is suspect."""
    settings = get_settings()

    assert settings.resolved_store_backend == "memory"
    assert settings.resolved_llm_provider == "mock"


# --- round trip -----------------------------------------------------------


async def test_insert_find_count_and_delete_round_trip(store: MemoryStore) -> None:
    await store.insert(RUNS, _run("run-1"))
    await store.insert(RUNS, _run("run-2"))
    await store.insert(RUNS, _run("run-3", workspace_id="ws-other"))

    assert await store.find_one(RUNS, {"run_id": "run-1"}) == _run("run-1")
    assert await store.find_one(RUNS, {"run_id": "absent"}) is None
    assert {row["run_id"] for row in await store.find(RUNS, {"workspace_id": WS})} == {
        "run-1",
        "run-2",
    }
    assert await store.count(RUNS, {"workspace_id": WS}) == 2
    assert await store.delete(RUNS, {"workspace_id": WS}) == 2
    assert await store.count(RUNS, {}) == 1


async def test_update_one_replaces_the_matching_document(store: MemoryStore) -> None:
    await store.insert(RUNS, _run("run-1"))

    changed = await store.update_one(RUNS, {"run_id": "run-1"}, _run("run-1", status="completed"))

    assert changed is True
    stored = await store.find_one(RUNS, {"run_id": "run-1"})
    assert stored is not None and stored["status"] == "completed"


async def test_update_one_inserts_only_when_upsert_is_asked_for(store: MemoryStore) -> None:
    missed = await store.update_one(RUNS, {"run_id": "run-1"}, _run("run-1"))
    upserted = await store.update_one(RUNS, {"run_id": "run-1"}, _run("run-1"), upsert=True)

    assert (missed, upserted) == (False, True)
    assert await store.count(RUNS, {}) == 1


# --- unique indexes -------------------------------------------------------


async def test_duplicate_run_id_raises_duplicate_key_error(store: MemoryStore) -> None:
    """`run_id` is uniquely indexed so a retried create cannot fork a second run."""
    await store.insert(RUNS, _run("run-1"))

    with pytest.raises(DuplicateKeyError):
        await store.insert(RUNS, _run("run-1", status="completed"))


# --- query operators ------------------------------------------------------


@pytest.mark.parametrize(
    ("matching", "non_matching"),
    [
        pytest.param(
            {"status": {"$in": ["draft", "approved"]}},
            {"status": {"$in": ["rejected"]}},
            id="$in",
        ),
        pytest.param(
            {"status": {"$nin": ["rejected", "superseded"]}},
            {"status": {"$nin": ["draft"]}},
            id="$nin",
        ),
        pytest.param(
            {"status": {"$ne": "rejected"}},
            {"status": {"$ne": "draft"}},
            id="$ne",
        ),
        pytest.param(
            {"review_note": {"$exists": False}},
            {"review_note": {"$exists": True}},
            id="$exists",
        ),
        pytest.param(
            {"score": {"$gte": 5}},
            {"score": {"$gte": 9}},
            id="$gte",
        ),
        pytest.param(
            {"score": {"$lte": 5}},
            {"score": {"$lte": 1}},
            id="$lte",
        ),
    ],
)
def test_matches_supports_the_documented_query_operators(
    matching: dict[str, Any], non_matching: dict[str, Any]
) -> None:
    document = {"status": "draft", "score": 5}

    assert matches(document, matching) is True
    assert matches(document, non_matching) is False


def test_matches_rejects_an_unsupported_operator() -> None:
    """Silently ignoring `$gt` would let the memory store disagree with Mongo - exactly
    the class of bug that two backends behind one Protocol invites."""
    with pytest.raises(StoreError, match="Unsupported query operator"):
        matches({"score": 5}, {"score": {"$gt": 1}})


# --- sort / limit / skip --------------------------------------------------


async def test_find_applies_sort_then_skip_then_limit(store: MemoryStore) -> None:
    for index, latency in enumerate([40, 10, 30, 20]):
        await store.insert(NODE_RUNS, _node_run(f"nr-{index}", latency_ms=latency))

    rows = await store.find(NODE_RUNS, {}, sort=[("latency_ms", 1)], skip=1, limit=2)

    assert [row["latency_ms"] for row in rows] == [20, 30]


async def test_none_sorts_below_numbers_like_bson(store: MemoryStore) -> None:
    """Null ranks below numbers in BSON, so a missing value sorts first ascending.

    Verified against a real Atlas collection: `[None, 1, 2, 3]` / `[3, 2, 1, None]`.
    This store only earns its place in the test suite by matching Mongo exactly -
    a divergence here means green tests and a broken production query.
    """
    for index, score in enumerate([3, None, 1, 2]):
        await store.insert(NODE_RUNS, _node_run(f"nr-{index}", score=score))

    ascending = await store.find(NODE_RUNS, {}, sort=[("score", 1)])
    descending = await store.find(NODE_RUNS, {}, sort=[("score", -1)])

    assert [row["score"] for row in ascending] == [None, 1, 2, 3]
    assert [row["score"] for row in descending] == [3, 2, 1, None]


async def test_an_upserted_insert_still_honours_the_unique_index(store: MemoryStore) -> None:
    """An upsert that inserts is an insert, and Mongo would apply the unique index.

    Accepting a row Mongo rejects is the one failure this store cannot have.
    """
    await store.update_one(RUNS, {"run_id": "up-1"}, _run("up-1"), upsert=True)

    with pytest.raises(DuplicateKeyError):
        await store.update_one(RUNS, {"run_id": "up-2"}, _run("up-1"), upsert=True)

    assert await store.count(RUNS, {"run_id": "up-1"}) == 1
