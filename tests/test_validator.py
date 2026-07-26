"""Deterministic validation.

`check_deterministic` is pure and runs before any model, and a draft it fails cannot
be cleared by the LLM judge afterwards. That ordering is the only thing standing
between a hallucinated price and a published post, so these checks are code, and this
file pins them.
"""

from __future__ import annotations

import pytest

from hivek_agent.content.validator import ContentValidator
from hivek_agent.domain import (
    BrandVoiceProfile,
    CompiledContext,
    ContentValidationResult,
    PostDraft,
)

WS = "ws-alpha"
FACT_ID = "fact_1"


@pytest.fixture
def validator() -> ContentValidator:
    """No LLM: `check_deterministic` never needs one, and this proves it."""
    return ContentValidator()


@pytest.fixture
def clean_draft() -> PostDraft:
    return PostDraft(
        hook="Chào bạn",
        body="Hôm nay chúng mình chia sẻ một câu chuyện nhỏ về nghề.",
        cta="Ghé xem thử nhé.",
    )


def _context_with_fact(value: str) -> CompiledContext:
    return CompiledContext(
        task="content_compose",
        workspace_id=WS,
        platform="facebook",
        immutable_facts=[
            {
                "fact_id": FACT_ID,
                "key": "workspace::brand.name",
                "predicate": "brand.name",
                "value": value,
                "status": "confirmed",
                "confidence": 1.0,
                "source_id": "user/chat",
            }
        ],
    )


@pytest.fixture
def context_with_one_fact() -> CompiledContext:
    return _context_with_fact("ACME Coffee")


def _codes(result: ContentValidationResult) -> set[str]:
    return {issue.code for issue in result.issues}


# --- risky claims ---------------------------------------------------------


def test_absolute_guarantee_fires_despite_diacritics(validator: ContentValidator) -> None:
    """The patterns are authored with diacritics but matched against folded text. If the
    folding were skipped on either side, every risky-claim check would silently pass -
    the worst possible failure mode for a guardrail."""
    draft = PostDraft(hook="Cam kết 100% hiệu quả", body="Nội dung thử.", cta="Xem thêm.")

    result = validator.check_deterministic(draft, platform="facebook")

    assert "absolute_guarantee" in _codes(result)
    assert result.risk_level == "red"


def test_superlative_claim_fires_despite_diacritics(validator: ContentValidator) -> None:
    draft = PostDraft(hook="Sản phẩm tốt nhất Việt Nam", body="Nội dung thử.", cta="Xem thêm.")

    result = validator.check_deterministic(draft, platform="facebook")

    assert [issue.severity for issue in result.issues if issue.code == "superlative_claim"] == [
        "error"
    ]
    assert result.risk_level == "red"


# --- brand voice ----------------------------------------------------------


def test_banned_phrase_from_the_voice_profile_is_an_error(validator: ContentValidator) -> None:
    """Learned bans are enforced in code, not left as a prompt suggestion the model may
    ignore."""
    voice = BrandVoiceProfile(workspace_id=WS, banned_phrases=["siêu phẩm"])
    draft = PostDraft(hook="Chào bạn", body="Đây là siêu phẩm của năm.", cta="Xem thêm.")

    result = validator.check_deterministic(draft, platform="facebook", voice=voice)

    assert "banned_phrase" in _codes(result)


def test_ai_tell_phrase_is_only_a_warning(validator: ContentValidator) -> None:
    """Machine-sounding phrasing is a quality signal, not a safety one: it should nudge
    a human to look, not block the draft outright."""
    draft = PostDraft(hook="Trong thế giới ngày nay", body="Nội dung thử nghiệm.", cta="Xem thêm.")

    result = validator.check_deterministic(draft, platform="facebook")

    assert [issue.severity for issue in result.issues if issue.code == "ai_phrasing"] == ["warning"]


# --- factual grounding ----------------------------------------------------


def test_citing_a_fact_outside_the_context_is_an_error(
    validator: ContentValidator, context_with_one_fact: CompiledContext
) -> None:
    """This is what stops a model inventing a price and attaching a plausible-looking
    fact_id to make it look sourced. A supplied id must still pass, or the check would
    be indiscriminate rather than discriminating."""
    invented = PostDraft(
        hook="Chào bạn", body="Nội dung thử.", cta="Xem thêm.", delivered_fact_ids=["fact_999"]
    )
    sourced = invented.model_copy(update={"delivered_fact_ids": [FACT_ID]})

    assert "unknown_fact_reference" in _codes(
        validator.check_deterministic(invented, platform="facebook", context=context_with_one_fact)
    )
    assert "unknown_fact_reference" not in _codes(
        validator.check_deterministic(sourced, platform="facebook", context=context_with_one_fact)
    )


def test_a_number_no_fact_supports_is_flagged(
    validator: ContentValidator, context_with_one_fact: CompiledContext
) -> None:
    """An invented figure is the highest-cost hallucination in commerce copy."""
    draft = PostDraft(
        hook="Chào bạn", body="Giá chỉ 199k thôi.", cta="Xem thêm.", delivered_fact_ids=[FACT_ID]
    )

    result = validator.check_deterministic(
        draft, platform="facebook", context=context_with_one_fact
    )

    assert "unsupported_number" in _codes(result)


def test_a_number_backed_by_a_fact_is_not_flagged(validator: ContentValidator) -> None:
    """The other half of the rule: the check must gate on evidence, not on digits."""
    draft = PostDraft(
        hook="Chào bạn", body="Giá chỉ 199k thôi.", cta="Xem thêm.", delivered_fact_ids=[FACT_ID]
    )

    result = validator.check_deterministic(
        draft, platform="facebook", context=_context_with_fact("199k")
    )

    assert "unsupported_number" not in _codes(result)


# --- platform limits ------------------------------------------------------


def test_text_over_the_platform_limit_is_an_error(validator: ContentValidator) -> None:
    """Threads caps at 500 chars and Facebook at 2000, so the same body must fail on one
    and pass on the other - the limit is per platform, not global."""
    draft = PostDraft(hook="Chào bạn", body="a " * 400, cta="Xem thêm.")

    assert "platform_length_exceeded" in _codes(
        validator.check_deterministic(draft, platform="threads")
    )
    assert "platform_length_exceeded" not in _codes(
        validator.check_deterministic(draft, platform="facebook")
    )


# --- the happy path -------------------------------------------------------


def test_a_clean_draft_is_green_and_approved(
    validator: ContentValidator, clean_draft: PostDraft
) -> None:
    """The guardrails must not fire on ordinary copy, or every post would need review
    and the review signal would become noise."""
    result = validator.check_deterministic(clean_draft, platform="facebook")

    assert result.issues == []
    assert result.risk_level == "green"
    assert result.final_decision == "approve"


# --- purity ---------------------------------------------------------------


def test_check_deterministic_is_pure(
    validator: ContentValidator, clean_draft: PostDraft, context_with_one_fact: CompiledContext
) -> None:
    """Two runs over the same draft must agree, or a retry could flip a verdict and the
    reproducibility trio stored on each asset would be meaningless."""
    first = validator.check_deterministic(
        clean_draft, platform="facebook", context=context_with_one_fact
    )
    second = validator.check_deterministic(
        clean_draft, platform="facebook", context=context_with_one_fact
    )

    assert first == second
