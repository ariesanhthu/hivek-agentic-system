"""The preference lifecycle: candidate -> repeated -> stable.

This is the core learning guarantee. One edit must never become a permanent rule - a
user who shortens a single post has not asked the system to write short posts forever.
`promote_preferences` re-derives status from evidence on every call, so the rule holds
even if a caller hands it a payload claiming `stable`.
"""

from __future__ import annotations

import pytest

from hivek_agent.domain import FeatureDelta, PreferenceCandidate
from hivek_agent.learning.edit_analysis import LearningService, infer_preferences
from hivek_agent.repositories import Repositories

WS = "ws-alpha"
PLATFORM = "facebook"


@pytest.fixture
def learning(repos: Repositories) -> LearningService:
    return LearningService(repos.learning, repos.knowledge)


async def _observe_edit(
    learning: LearningService,
    before: str,
    after: str,
    *,
    asset_id: str,
    explicit: bool = False,
) -> list[PreferenceCandidate]:
    """One full observe-then-promote cycle, the way `AgenticService.decide` does it."""
    event = await learning.record_edit(
        workspace_id=WS,
        asset_id=asset_id,
        before_text=before,
        after_text=after,
        platform=PLATFORM,
        explicit=explicit,
    )
    return await learning.promote_preferences(WS, event.inferred_preferences)


def _rule(promoted: list[PreferenceCandidate], rule_type: str) -> PreferenceCandidate:
    return next(item for item in promoted if item.rule_type == rule_type)


# --- one edit is not a rule -----------------------------------------------


async def test_a_single_edit_produces_only_candidates(
    learning: LearningService, verbose_text: str, tightened_text: str
) -> None:
    promoted = await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-1")

    assert {item.status for item in promoted} == {"candidate"}


async def test_a_single_edit_activates_no_preference(
    learning: LearningService, repos: Repositories, verbose_text: str, tightened_text: str
) -> None:
    """The load-bearing assertion of this file: after one edit, nothing may steer
    generation. `active_only` is exactly what the context compiler reads."""
    await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-1")

    assert await repos.learning.list_preferences(WS, active_only=True) == []


async def test_a_second_identical_edit_promotes_to_repeated(
    learning: LearningService, verbose_text: str, tightened_text: str
) -> None:
    await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-1")

    promoted = await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-2")

    assert _rule(promoted, "length").status == "repeated"


async def test_a_fourth_edit_promotes_to_stable(
    learning: LearningService, verbose_text: str, tightened_text: str
) -> None:
    """`stable` takes twice the promotion threshold, so the third observation is still
    only `repeated`."""
    statuses = []
    for index in range(1, 5):
        promoted = await _observe_edit(
            learning, verbose_text, tightened_text, asset_id=f"asset-{index}"
        )
        statuses.append(_rule(promoted, "length").status)

    assert statuses == ["candidate", "repeated", "repeated", "stable"]


async def test_an_explicit_pin_is_stable_immediately(
    learning: LearningService, verbose_text: str, tightened_text: str
) -> None:
    """`pin_as_good` is the one signal the user gives on purpose. That is a decision,
    not an inference, so it does not need corroboration."""
    promoted = await _observe_edit(
        learning, verbose_text, tightened_text, asset_id="asset-1", explicit=True
    )

    assert _rule(promoted, "length").status == "stable"


# --- rule scoping ---------------------------------------------------------


def test_a_banned_phrase_is_scoped_globally() -> None:
    """A phrase the brand dislikes is disliked everywhere. Scoping it per platform would
    force the user to re-teach it on every channel before it ever applied."""
    delta = FeatureDelta(removed_phrases=["mua ngay keo lo"])

    rule = _rule(infer_preferences(WS, "asset-1", delta, platform=PLATFORM), "banned_phrase")

    assert rule.scope == "global"
    assert rule.platform is None


def test_a_length_rule_is_scoped_to_the_platform() -> None:
    """Length genuinely differs by channel - TikTok copy is shorter than Facebook copy
    by nature - so this one must not leak across platforms."""
    delta = FeatureDelta(length_delta_ratio=-0.5)

    rule = _rule(infer_preferences(WS, "asset-1", delta, platform=PLATFORM), "length")

    assert rule.scope == "platform"
    assert rule.platform == PLATFORM


def test_an_emoji_rule_is_scoped_to_the_platform() -> None:
    delta = FeatureDelta(emoji_delta=-3)

    assert _rule(infer_preferences(WS, "asset-1", delta, platform=PLATFORM), "emoji").scope == (
        "platform"
    )


# --- voice profile folding ------------------------------------------------


async def test_rebuild_voice_profile_ignores_candidate_rules(
    learning: LearningService, verbose_text: str, tightened_text: str
) -> None:
    """The last line of defence: even with a candidate stored, the profile that reaches
    the composer must not carry it."""
    await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-1")

    profile = await learning.rebuild_voice_profile(WS)

    assert profile.banned_phrases == []


async def test_rebuild_voice_profile_includes_rules_once_active(
    learning: LearningService, verbose_text: str, tightened_text: str
) -> None:
    await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-1")
    await _observe_edit(learning, verbose_text, tightened_text, asset_id="asset-2")

    profile = await learning.rebuild_voice_profile(WS)

    assert profile.banned_phrases != []
