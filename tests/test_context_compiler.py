"""Context compilation.

The compiler is the only component allowed to decide what reaches a model. Two
properties matter: it stays inside its token budget, and when it drops something it
says so. Silent truncation would mean a draft is missing a fact nobody knows was cut.
"""

from __future__ import annotations

from typing import Any

import pytest

from hivek_agent.agentic.context_compiler import ContextCompiler
from hivek_agent.domain import KnowledgeAssertion, PreferenceCandidate, SourceRef

WS = "ws-alpha"

# Small enough that one fact row fits and the second does not (a row costs ~45 tokens
# and the facts section gets 35% of the budget).
TIGHT_BUDGET = 200


@pytest.fixture
def compiler() -> ContextCompiler:
    return ContextCompiler(default_budget=12000)


def _fact(
    assertion_id: str,
    predicate: str,
    value: Any,
    *,
    confidence: float,
    approval_status: str,
) -> KnowledgeAssertion:
    """Fixed `assertion_id` on purpose: it lands in the row the hash is taken over, so
    a generated id would make `context_hash` untestable."""
    return KnowledgeAssertion(
        assertion_id=assertion_id,
        workspace_id=WS,
        subject_id="workspace",
        predicate=predicate,
        object_value=value,
        source=SourceRef(
            source_id="user/chat", source_type="user_input", confidence=0.95, approved=True
        ),
        confidence=confidence,
        approval_status=approval_status,  # type: ignore[arg-type]
    )


@pytest.fixture
def high_priority_fact() -> KnowledgeAssertion:
    return _fact(
        "fact_high", "brand.name", "ACME Coffee", confidence=0.95, approval_status="confirmed"
    )


@pytest.fixture
def low_priority_fact() -> KnowledgeAssertion:
    return _fact(
        "fact_low",
        "brand.audience",
        "Nhan vien van phong 25-35",
        confidence=0.61,
        approval_status="candidate",
    )


def _fact_map(*facts: KnowledgeAssertion) -> dict[str, KnowledgeAssertion]:
    return {fact.key: fact for fact in facts}


def _kept_ids(context: Any) -> list[str]:
    return [row["fact_id"] for row in context.immutable_facts]


def _omitted_facts(context: Any) -> list[str]:
    return [section for section in context.omitted_sections if section.startswith("fact:")]


# --- token budget ---------------------------------------------------------


def test_a_tight_budget_drops_the_lower_priority_fact(
    compiler: ContextCompiler,
    high_priority_fact: KnowledgeAssertion,
    low_priority_fact: KnowledgeAssertion,
) -> None:
    """Facts are ordered required > confirmed > confidence, so the confirmed one keeps
    its place and the weak candidate is the one that goes."""
    context = compiler.compile(
        task="content_compose",
        workspace_id=WS,
        platform="facebook",
        facts=_fact_map(high_priority_fact, low_priority_fact),
        token_budget=TIGHT_BUDGET,
    )

    assert _kept_ids(context) == ["fact_high"]


def test_a_dropped_fact_is_recorded_rather_than_silently_lost(
    compiler: ContextCompiler,
    high_priority_fact: KnowledgeAssertion,
    low_priority_fact: KnowledgeAssertion,
) -> None:
    """`omitted_sections` is surfaced all the way to the API response, so a user can see
    the draft was written without a fact the system knows."""
    context = compiler.compile(
        task="content_compose",
        workspace_id=WS,
        platform="facebook",
        facts=_fact_map(high_priority_fact, low_priority_fact),
        token_budget=TIGHT_BUDGET,
    )

    assert _omitted_facts(context) == ["fact:workspace::brand.audience"]


def test_required_fact_keys_survive_a_budget_they_do_not_fit_in(
    compiler: ContextCompiler,
    high_priority_fact: KnowledgeAssertion,
    low_priority_fact: KnowledgeAssertion,
) -> None:
    """A plan node names the facts its angle depends on. Writing that post without them
    is worse than paying for the extra tokens - so the same budget that dropped the
    weak fact above must now keep it, and not report it as omitted either."""
    context = compiler.compile(
        task="content_compose",
        workspace_id=WS,
        platform="facebook",
        facts=_fact_map(high_priority_fact, low_priority_fact),
        token_budget=TIGHT_BUDGET,
        required_fact_keys=[high_priority_fact.key, low_priority_fact.key],
    )

    assert sorted(_kept_ids(context)) == ["fact_high", "fact_low"]
    assert _omitted_facts(context) == []


# --- preference gating ----------------------------------------------------


@pytest.fixture
def candidate_preference() -> PreferenceCandidate:
    return PreferenceCandidate(
        preference_id="pref_candidate",
        workspace_id=WS,
        rule_type="length",
        rule_value="prefer_shorter",
        scope="platform",
        platform="facebook",
        status="candidate",
        observation_count=1,
        confidence=0.4,
    )


@pytest.fixture
def stable_preference() -> PreferenceCandidate:
    return PreferenceCandidate(
        preference_id="pref_stable",
        workspace_id=WS,
        rule_type="emoji",
        rule_value="reduce_emoji",
        scope="platform",
        platform="facebook",
        status="stable",
        observation_count=4,
        confidence=0.8,
    )


def test_a_candidate_preference_does_not_become_a_brand_rule(
    compiler: ContextCompiler,
    candidate_preference: PreferenceCandidate,
    stable_preference: PreferenceCandidate,
) -> None:
    """One observation is a hypothesis. Letting it steer generation is exactly the
    "system over-fitted to a single edit" failure the lifecycle exists to prevent."""
    context = compiler.compile(
        task="content_compose",
        workspace_id=WS,
        platform="facebook",
        preferences=[candidate_preference, stable_preference],
    )

    assert [rule["value"] for rule in context.brand_rules] == ["reduce_emoji"]


def test_a_candidate_preference_is_reported_as_omitted(
    compiler: ContextCompiler, candidate_preference: PreferenceCandidate
) -> None:
    context = compiler.compile(
        task="content_compose",
        workspace_id=WS,
        platform="facebook",
        preferences=[candidate_preference],
    )

    assert (
        f"preference:{candidate_preference.key}:not_yet_stable" in context.omitted_sections
    )


# --- reproducibility ------------------------------------------------------


def test_context_hash_is_stable_for_identical_input(
    compiler: ContextCompiler, high_priority_fact: KnowledgeAssertion
) -> None:
    """The hash is the asset's idempotency key: an unstable one would pay for a second
    generation on every retry."""
    facts = _fact_map(high_priority_fact)
    kwargs: dict[str, Any] = {
        "task": "content_compose",
        "workspace_id": WS,
        "platform": "facebook",
        "facts": facts,
    }

    first = compiler.compile(**kwargs)
    second = compiler.compile(**kwargs)

    assert first.context_hash == second.context_hash


def test_context_hash_changes_when_a_fact_value_changes(compiler: ContextCompiler) -> None:
    """The other half of the contract: a stale hash would serve an old draft for new
    facts, which is worse than regenerating."""
    original = _fact("fact_1", "brand.name", "ACME", confidence=1.0, approval_status="confirmed")
    edited = _fact("fact_1", "brand.name", "OTHER", confidence=1.0, approval_status="confirmed")

    first = compiler.compile(
        task="content_compose", workspace_id=WS, platform="facebook", facts=_fact_map(original)
    )
    second = compiler.compile(
        task="content_compose", workspace_id=WS, platform="facebook", facts=_fact_map(edited)
    )

    assert first.context_hash != second.context_hash
