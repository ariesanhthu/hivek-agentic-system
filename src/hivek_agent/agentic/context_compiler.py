"""Context compiler.

The only component allowed to decide what reaches a model. It selects structured
facts and rules, bounded by a token budget, and records what it dropped.

Blueprint constraints implemented here:
  - never inline whole documents (facts go in as structured rows with IDs);
  - keep 3-8 diverse memories, not raw top-k;
  - report `omitted_sections` instead of silently truncating.
"""

from __future__ import annotations

from typing import Any

from hivek_agent.domain import (
    BrandOperatingProfile,
    BrandVoiceProfile,
    CompiledContext,
    ContentAsset,
    KnowledgeAssertion,
    PlatformId,
    PreferenceCandidate,
    SourceRef,
)
from hivek_agent.infrastructure.llm import context_hash, estimate_tokens

# Native norms per platform. Deterministic data, so no model needs to recall them.
PLATFORM_RULES: dict[str, dict[str, Any]] = {
    "facebook": {
        "max_chars": 2000,
        "ideal_chars": 700,
        "hashtag_limit": 5,
        "tone": "Kể chuyện, xuống dòng thoáng, mở đầu bằng tình huống thật.",
        "first_comment_use": "Đặt link hoặc thông tin phụ ở bình luận đầu.",
    },
    "threads": {
        "max_chars": 500,
        "ideal_chars": 280,
        "hashtag_limit": 2,
        "tone": "Ngắn, đối thoại, một ý chính duy nhất.",
        "first_comment_use": "Nối chuỗi bài bằng bình luận đầu.",
    },
    "tiktok": {
        "max_chars": 300,
        "ideal_chars": 150,
        "hashtag_limit": 5,
        "tone": "Hook 2 giây đầu, khẩu ngữ, bám trend âm thanh.",
        "first_comment_use": "Ghim câu hỏi để đẩy tương tác.",
    },
}

MAX_MEMORIES = 8
MIN_MEMORIES = 3

# Rough share of the budget each section may claim before we start omitting.
_SECTION_BUDGET_SHARE = {
    "immutable_facts": 0.35,
    "brand_rules": 0.15,
    "skills": 0.25,
    "relevant_examples": 0.15,
    "negative_memories": 0.10,
}


class ContextCompiler:
    def __init__(self, *, default_budget: int = 12000) -> None:
        self._default_budget = default_budget

    def compile(
        self,
        *,
        task: str,
        workspace_id: str,
        platform: PlatformId | None = None,
        facts: dict[str, KnowledgeAssertion] | None = None,
        brand_profile: BrandOperatingProfile | None = None,
        voice_profile: BrandVoiceProfile | None = None,
        preferences: list[PreferenceCandidate] | None = None,
        approved_examples: list[ContentAsset] | None = None,
        rejected_examples: list[ContentAsset] | None = None,
        skills: list[dict[str, Any]] | None = None,
        required_fact_keys: list[str] | None = None,
        token_budget: int | None = None,
    ) -> CompiledContext:
        budget = token_budget or self._default_budget
        omitted: list[str] = []

        immutable_facts, citations = self._select_facts(
            facts or {}, required_fact_keys or [], budget, omitted
        )
        brand_rules = self._brand_rules(voice_profile, preferences or [], budget, omitted)
        platform_rules = PLATFORM_RULES.get(platform or "", {})
        audience_summary = self._audience(brand_profile)
        examples = self._examples(approved_examples or [], budget, omitted)
        negatives = self._negatives(rejected_examples or [], preferences or [], budget, omitted)
        selected_skills = self._skills(skills or [], budget, omitted)

        context = CompiledContext(
            task=task,
            workspace_id=workspace_id,
            platform=platform,
            immutable_facts=immutable_facts,
            brand_rules=brand_rules,
            audience_summary=audience_summary,
            platform_rules=platform_rules,
            relevant_examples=examples,
            negative_memories=negatives,
            skills=selected_skills,
            citations=citations,
            omitted_sections=omitted,
            token_budget=budget,
        )
        context.estimated_tokens = _estimate(context)
        context.context_hash = context_hash(
            task,
            workspace_id,
            platform,
            immutable_facts,
            brand_rules,
            selected_skills,
            [example.get("id") for example in examples],
        )
        return context

    def _select_facts(
        self,
        facts: dict[str, KnowledgeAssertion],
        required_keys: list[str],
        budget: int,
        omitted: list[str],
    ) -> tuple[list[dict[str, Any]], list[SourceRef]]:
        """Required facts first, then confirmed, then by confidence."""
        ordered = sorted(
            facts.values(),
            key=lambda item: (
                0 if item.key in required_keys else 1,
                0 if item.approval_status == "confirmed" else 1,
                -item.confidence,
            ),
        )

        allowance = int(budget * _SECTION_BUDGET_SHARE["immutable_facts"])
        rows: list[dict[str, Any]] = []
        citations: list[SourceRef] = []
        used = 0

        for assertion in ordered:
            row = {
                "fact_id": assertion.assertion_id,
                "key": assertion.key,
                "predicate": assertion.predicate,
                "value": assertion.object_value,
                "status": assertion.approval_status,
                "confidence": round(assertion.confidence, 2),
                "source_id": assertion.source.source_id,
            }
            cost = estimate_tokens(str(row))
            if used + cost > allowance and assertion.key not in required_keys:
                omitted.append(f"fact:{assertion.key}")
                continue
            rows.append(row)
            citations.append(assertion.source)
            used += cost

        return rows, citations

    def _brand_rules(
        self,
        voice: BrandVoiceProfile | None,
        preferences: list[PreferenceCandidate],
        budget: int,
        omitted: list[str],
    ) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        if voice:
            if voice.tone:
                rules.append({"rule": "tone", "value": voice.tone})
            rules.append(
                {"rule": "sentence_length_range", "value": list(voice.sentence_length_range)}
            )
            if voice.banned_phrases:
                rules.append({"rule": "banned_phrases", "value": voice.banned_phrases[:12]})
            if voice.preferred_openings:
                rules.append({"rule": "preferred_openings", "value": voice.preferred_openings[:5]})
            if voice.avoided_openings:
                rules.append({"rule": "avoided_openings", "value": voice.avoided_openings[:5]})
            if voice.emoji_policy:
                rules.append({"rule": "emoji_policy", "value": voice.emoji_policy})

        # Only learned rules that survived promotion may steer generation.
        for preference in preferences:
            if not preference.is_active:
                omitted.append(f"preference:{preference.key}:not_yet_stable")
                continue
            rules.append(
                {
                    "rule": preference.rule_type,
                    "value": preference.rule_value,
                    "confidence": round(preference.confidence, 2),
                    "learned_from": len(preference.evidence_asset_ids),
                }
            )

        allowance = int(budget * _SECTION_BUDGET_SHARE["brand_rules"])
        return _fit(rules, allowance, omitted, "brand_rule")

    def _audience(self, profile: BrandOperatingProfile | None) -> dict[str, Any]:
        if profile is None:
            return {}
        return {
            "segments": [item.get("description") for item in profile.audiences][:3],
            "products": [item.get("name") for item in profile.products][:5],
            "readiness": profile.readiness_score,
        }

    def _examples(
        self, assets: list[ContentAsset], budget: int, omitted: list[str]
    ) -> list[dict[str, Any]]:
        """Approved posts as positive examples - deduped so we don't show one voice twice."""
        rows: list[dict[str, Any]] = []
        seen_hooks: set[str] = set()
        for asset in assets:
            hook = (asset.draft.hook or "").strip().casefold()[:60]
            if hook and hook in seen_hooks:
                omitted.append(f"example:{asset.asset_id}:near_duplicate")
                continue
            seen_hooks.add(hook)
            rows.append(
                {
                    "id": asset.asset_id,
                    "platform": asset.platform,
                    "hook": asset.draft.hook,
                    "excerpt": (asset.edited_text or asset.draft.full_text)[:240],
                }
            )
            if len(rows) >= MAX_MEMORIES:
                break

        allowance = int(budget * _SECTION_BUDGET_SHARE["relevant_examples"])
        return _fit(rows, allowance, omitted, "example", keep_at_least=min(MIN_MEMORIES, len(rows)))

    def _negatives(
        self,
        rejected: list[ContentAsset],
        preferences: list[PreferenceCandidate],
        budget: int,
        omitted: list[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            {
                "id": asset.asset_id,
                "reason": asset.review_note or "Bị từ chối",
                "excerpt": asset.draft.full_text[:160],
            }
            for asset in rejected[:3]
        ]
        rows.extend(
            {"rule": "avoid", "value": preference.rule_value}
            for preference in preferences
            if preference.rule_type == "banned_phrase" and preference.is_active
        )
        allowance = int(budget * _SECTION_BUDGET_SHARE["negative_memories"])
        return _fit(rows, allowance, omitted, "negative")

    def _skills(
        self, skills: list[dict[str, Any]], budget: int, omitted: list[str]
    ) -> list[dict[str, Any]]:
        allowance = int(budget * _SECTION_BUDGET_SHARE["skills"])
        return _fit(skills, allowance, omitted, "skill")


def _fit(
    rows: list[dict[str, Any]],
    allowance: int,
    omitted: list[str],
    label: str,
    *,
    keep_at_least: int = 0,
) -> list[dict[str, Any]]:
    """Keep rows until the section allowance is spent; record the rest as omitted."""
    kept: list[dict[str, Any]] = []
    used = 0
    for index, row in enumerate(rows):
        cost = estimate_tokens(str(row))
        if used + cost > allowance and index >= keep_at_least:
            omitted.append(f"{label}:{row.get('id') or row.get('rule') or index}")
            continue
        kept.append(row)
        used += cost
    return kept


def _estimate(context: CompiledContext) -> int:
    payload = context.model_dump(
        exclude={"estimated_tokens", "context_hash", "omitted_sections", "token_budget"}
    )
    return estimate_tokens(str(payload))
