"""Fact provenance, precedence and conflict detection.

The three rules `knowledge/facts.py` exists to enforce, each of which is a way the
system could quietly start lying to a user:

  - a confirmed fact is never silently overwritten;
  - two comparably-trusted sources that disagree raise a conflict instead of one
    being picked;
  - history is superseded, never deleted.
"""

from __future__ import annotations

from typing import Any

import pytest

from hivek_agent.domain import SourceRef, SourceType
from hivek_agent.infrastructure.store.base import ASSERTIONS
from hivek_agent.infrastructure.store.memory import MemoryStore
from hivek_agent.knowledge.facts import FactService
from hivek_agent.repositories import Repositories

WS = "ws-alpha"


@pytest.fixture
def facts(repos: Repositories) -> FactService:
    return FactService(repos.knowledge)


def _source(source_id: str, source_type: SourceType, confidence: float = 0.7) -> SourceRef:
    return SourceRef(
        source_id=source_id,
        source_type=source_type,
        confidence=confidence,
        approved=source_type == "user_input",
    )


async def _upsert(facts: FactService, predicate: str, value: Any, source: SourceRef) -> Any:
    return await facts.upsert_fact(
        workspace_id=WS,
        subject_id="workspace",
        predicate=predicate,
        object_value=value,
        source=source,
    )


def _assertion_ids(usable: dict[str, Any]) -> set[str]:
    return {item.assertion_id for item in usable.values()}


# --- precedence -----------------------------------------------------------


async def test_user_input_supersedes_a_website_fact(facts: FactService) -> None:
    """`user_input` outranks `website` by four precedence tiers, so this is not a tie
    and the system is entitled to decide without asking."""
    await _upsert(facts, "brand.name", "ACME Cafe", _source("site/home", "website"))

    result = await _upsert(
        facts, "brand.name", "ACME Coffee", _source("user/chat", "user_input", 0.95)
    )

    assert result.action == "superseded_existing"


async def test_a_superseded_fact_is_marked_not_deleted(
    facts: FactService, repos: Repositories
) -> None:
    """History is the audit trail. A superseded row stays queryable and closed off with
    `valid_to`, so a user can still see what the system used to believe and why."""
    first = await _upsert(facts, "brand.name", "ACME Cafe", _source("site/home", "website"))

    await _upsert(facts, "brand.name", "ACME Coffee", _source("user/chat", "user_input", 0.95))

    old = await repos.knowledge.get_assertion(WS, first.assertion.assertion_id)
    assert old is not None
    assert old.approval_status == "superseded"
    assert old.valid_to is not None


# --- conflicts ------------------------------------------------------------


async def test_a_source_contradicting_a_confirmed_fact_raises_a_conflict(
    facts: FactService,
) -> None:
    first = await _upsert(
        facts, "brand.name", "ACME Coffee", _source("user/chat", "user_input", 0.95)
    )
    await facts.confirm_fact(WS, first.assertion.assertion_id)

    result = await _upsert(facts, "brand.name", "ACME Tea", _source("drive/f1", "drive_file", 0.8))

    assert result.action == "conflict"
    assert result.created_conflict is True


async def test_a_confirmed_fact_survives_a_contradicting_source(
    facts: FactService, repos: Repositories
) -> None:
    """The whole point: a user confirmation is never silently overwritten. The rival is
    parked as a conflict; the confirmed value keeps steering generation until a human
    says otherwise."""
    first = await _upsert(
        facts, "brand.name", "ACME Coffee", _source("user/chat", "user_input", 0.95)
    )
    await facts.confirm_fact(WS, first.assertion.assertion_id)

    await _upsert(facts, "brand.name", "ACME Tea", _source("drive/f1", "drive_file", 0.8))

    survivor = await repos.knowledge.get_assertion(WS, first.assertion.assertion_id)
    assert survivor is not None and survivor.approval_status == "confirmed"


async def test_comparably_trusted_sources_that_disagree_conflict(facts: FactService) -> None:
    """`website` and `drive_file` are one tier apart at equal confidence - inside the
    tie band. Picking a winner here would be a guess dressed up as a fact."""
    await _upsert(facts, "brand.tone", "trang trong", _source("site/about", "website", 0.7))

    result = await _upsert(
        facts, "brand.tone", "than thien", _source("drive/tone", "drive_file", 0.7)
    )

    assert result.action == "conflict"


async def test_a_tie_does_not_silently_supersede_the_incumbent(
    facts: FactService, repos: Repositories
) -> None:
    """The half that would be easy to get wrong: raising a conflict is only correct if
    the existing value is also left alone rather than quietly retired."""
    first = await _upsert(facts, "brand.tone", "trang trong", _source("site/about", "website", 0.7))

    await _upsert(facts, "brand.tone", "than thien", _source("drive/tone", "drive_file", 0.7))

    incumbent = await repos.knowledge.get_assertion(WS, first.assertion.assertion_id)
    assert incumbent is not None and incumbent.approval_status == "candidate"


# --- reinforcement --------------------------------------------------------


async def test_reobserving_the_same_value_reinforces_it(
    facts: FactService, store: MemoryStore
) -> None:
    """Re-reading the same website must strengthen the fact, not stack a duplicate row
    that later reads as two independent sources agreeing."""
    await _upsert(facts, "brand.name", "ACME Coffee", _source("site/home", "website"))

    result = await _upsert(facts, "brand.name", "ACME Coffee", _source("site/home", "website"))

    assert result.action == "reinforced"
    assert await store.count(ASSERTIONS, {"workspace_id": WS}) == 1


async def test_a_list_fact_reasserted_in_a_different_order_is_reinforced(
    facts: FactService,
) -> None:
    """Channels are genuinely multi-valued and a set in spirit. Ordering them
    differently is the same answer, and calling it a contradiction would nag the user to
    resolve a non-conflict."""
    first = await _upsert(
        facts, "brand.channels", ["facebook", "tiktok"], _source("user/chat", "user_input", 0.95)
    )
    assert first.assertion.object_value == ["facebook", "tiktok"]

    result = await _upsert(
        facts, "brand.channels", ["tiktok", "facebook"], _source("user/chat", "user_input", 0.95)
    )

    assert result.action == "reinforced"


# --- usable_facts ---------------------------------------------------------


async def test_usable_facts_excludes_a_superseded_assertion(facts: FactService) -> None:
    first = await _upsert(facts, "brand.name", "Old", _source("site/home", "website"))
    await _upsert(facts, "brand.name", "New", _source("user/chat", "user_input", 0.95))

    usable = await facts.usable_facts(WS)

    assert first.assertion.assertion_id not in _assertion_ids(usable)


async def test_usable_facts_excludes_a_rejected_assertion(facts: FactService) -> None:
    await _upsert(facts, "brand.tone", "Strong", _source("user/chat", "user_input", 0.95))
    rejected = await _upsert(facts, "brand.tone", "Weak", _source("infer/1", "system_inference"))

    usable = await facts.usable_facts(WS)

    assert rejected.action == "ignored_weaker"
    assert rejected.assertion.assertion_id not in _assertion_ids(usable)


async def test_usable_facts_excludes_a_conflicted_assertion(facts: FactService) -> None:
    """A conflicted value must not reach a draft - we do not know whether it is true."""
    await _upsert(facts, "brand.audience", "A", _source("site/a", "website", 0.7))
    conflicted = await _upsert(facts, "brand.audience", "B", _source("drive/b", "drive_file", 0.7))

    usable = await facts.usable_facts(WS)

    assert conflicted.assertion.assertion_id not in _assertion_ids(usable)
