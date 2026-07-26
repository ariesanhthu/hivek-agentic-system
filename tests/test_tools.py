"""Tool authorization.

Two rules, both structural rather than advisory: an agent never sees the whole
registry, and nothing with an external side effect is reachable by the model. A tool
that is never listed cannot be called, which is a stronger guarantee than a prompt
asking the model not to call it.
"""

from __future__ import annotations

from typing import get_args

from hivek_agent.agentic.tools import (
    DEFAULT_SCOPES,
    TOOL_REGISTRY,
    authorized_tools,
    requires_approval,
)
from hivek_agent.domain import Intent

ALL_INTENTS = get_args(Intent)

# Every scope any tool declares - deliberately wider than DEFAULT_SCOPES, so the
# side-effect test below cannot pass merely because the caller was under-privileged.
EVERY_SCOPE = sorted(
    {scope for policy in TOOL_REGISTRY.values() for scope in policy.required_scopes}
)


def test_create_post_never_sees_a_gated_tool() -> None:
    """Writing a post has no business queueing a publish or re-syncing Drive."""
    tools = authorized_tools("create_post", list(DEFAULT_SCOPES))

    assert [name for name in tools if name.startswith(("publishing.", "connectors.sync."))] == []


def test_create_post_still_grants_the_tools_it_needs() -> None:
    """Least privilege has to stop short of useless: the negative test above would also
    pass if this returned nothing at all."""
    assert authorized_tools("create_post", list(DEFAULT_SCOPES)) == [
        "knowledge.search.facts",
        "knowledge.search.graph",
        "content.read.assets",
        "content.save_draft",
    ]


def test_no_intent_exposes_an_external_side_effect_tool() -> None:
    """Checked across every intent with every scope granted, so this is a property of
    the intent map rather than an accident of the default scope set. The model may
    queue; only a human sends."""
    exposed = {
        intent: [
            name
            for name in authorized_tools(intent, EVERY_SCOPE)
            if TOOL_REGISTRY[name].risk == "external_side_effect"
        ]
        for intent in ALL_INTENTS
    }

    assert exposed == {intent: [] for intent in ALL_INTENTS}


def test_empty_scopes_authorize_nothing() -> None:
    assert authorized_tools("create_post", []) == []


def test_unknown_tool_requires_approval() -> None:
    """Unknown means dangerous. Defaulting the other way would make every registry typo
    a silent authorization bypass."""
    assert requires_approval("unknown.tool") is True


def test_approval_is_required_for_side_effects_and_not_for_reads() -> None:
    """Both directions matter: a blanket True would be safe but would stall every read."""
    assert requires_approval("publishing.queue.post") is True
    assert requires_approval("knowledge.search.facts") is False
