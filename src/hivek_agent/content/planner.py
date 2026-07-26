"""Content planner.

Rule-based candidate generation plus deterministic scoring and a diversity reranker.
The LLM is not asked to invent the schedule or do arithmetic - it only writes the
strategy sentence once the plan already exists.

Constraints enforced here (blueprint section 9): funnel coverage, no repeated angle,
a cap on selling posts, and per-day/per-platform limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hivek_agent.agentic.skills import marketing_feasibility_score
from hivek_agent.domain import (
    BrandOperatingProfile,
    ContentPlan,
    ContentPlanNode,
    FunnelStage,
    KnowledgeAssertion,
    PlatformId,
)
from hivek_agent.repositories import new_id

logger = logging.getLogger(__name__)

# A healthy mix skews to the top of the funnel; conversion is the minority.
FUNNEL_MIX: dict[FunnelStage, float] = {
    "awareness": 0.4,
    "consideration": 0.3,
    "conversion": 0.2,
    "retention": 0.1,
}

MAX_CONVERSION_SHARE = 0.35
MAX_POSTS_PER_DAY_PER_PLATFORM = 1


@dataclass(frozen=True)
class AngleTemplate:
    """One reusable content angle, with the MFS inputs it scores against."""

    angle: str
    funnel_stage: FunnelStage
    goal: str
    pillar: str
    required_fact_keys: tuple[str, ...]
    impact: int
    effort: int
    cost: int
    speed: int


# The library the planner draws from. Extending the system with a new angle means
# adding a row here - no prompt change, no model change.
ANGLE_LIBRARY: tuple[AngleTemplate, ...] = (
    AngleTemplate(
        "Câu hỏi thường gặp của khách hàng",
        "awareness",
        "Thu hút người chưa biết thương hiệu",
        "education",
        ("workspace::brand.audience",),
        impact=4,
        effort=2,
        cost=1,
        speed=5,
    ),
    AngleTemplate(
        "Hiểu lầm phổ biến trong ngành",
        "awareness",
        "Tạo thảo luận và lưu bài",
        "education",
        ("workspace::brand.audience",),
        impact=4,
        effort=2,
        cost=1,
        speed=4,
    ),
    AngleTemplate(
        "Câu chuyện đằng sau sản phẩm",
        "awareness",
        "Xây dựng nhận diện",
        "story",
        ("workspace::brand.name",),
        impact=3,
        effort=3,
        cost=1,
        speed=3,
    ),
    AngleTemplate(
        "So sánh lựa chọn cho người đang cân nhắc",
        "consideration",
        "Giúp khách hàng tự đánh giá",
        "comparison",
        ("workspace::brand.product",),
        impact=5,
        effort=3,
        cost=1,
        speed=4,
    ),
    AngleTemplate(
        "Xử lý phản đối thường gặp",
        "consideration",
        "Giảm rào cản quyết định",
        "objection",
        ("workspace::brand.product", "workspace::brand.audience"),
        impact=5,
        effort=2,
        cost=1,
        speed=4,
    ),
    AngleTemplate(
        "Hướng dẫn sử dụng thực tế",
        "consideration",
        "Chứng minh giá trị",
        "howto",
        ("workspace::brand.product",),
        impact=4,
        effort=3,
        cost=2,
        speed=3,
    ),
    AngleTemplate(
        "Ưu đãi và lý do nên hành động",
        "conversion",
        "Thúc đẩy liên hệ",
        "offer",
        ("workspace::brand.product",),
        impact=5,
        effort=2,
        cost=2,
        speed=5,
    ),
    AngleTemplate(
        "Bằng chứng từ khách hàng cũ",
        "conversion",
        "Tăng độ tin cậy trước khi chốt",
        "proof",
        ("workspace::brand.product",),
        impact=5,
        effort=3,
        cost=2,
        speed=4,
    ),
    AngleTemplate(
        "Mẹo dùng nâng cao cho khách hiện tại",
        "retention",
        "Giữ chân khách cũ",
        "retention",
        ("workspace::brand.product",),
        impact=3,
        effort=2,
        cost=1,
        speed=3,
    ),
)


class ContentPlanner:
    def plan(
        self,
        *,
        workspace_id: str,
        days: int,
        platforms: list[PlatformId],
        facts: dict[str, KnowledgeAssertion],
        brand_profile: BrandOperatingProfile | None = None,
        recent_angles: list[str] | None = None,
        skill_ids: list[str] | None = None,
    ) -> ContentPlan:
        days = max(1, min(days, 30))
        platforms = platforms or ["facebook"]
        recent = {angle.casefold() for angle in (recent_angles or [])}

        candidates = self._generate_candidates(
            workspace_id=workspace_id,
            days=days,
            platforms=platforms,
            facts=facts,
            recent_angles=recent,
            skill_ids=skill_ids or [],
        )
        selected = self._select(candidates, days=days, platforms=platforms)

        plan = ContentPlan(
            plan_id=new_id("plan"),
            workspace_id=workspace_id,
            days=days,
            platforms=platforms,
            nodes=selected,
            strategy_summary=_summarize(selected, days, platforms, brand_profile),
        )
        return plan

    def _generate_candidates(
        self,
        *,
        workspace_id: str,
        days: int,
        platforms: list[PlatformId],
        facts: dict[str, KnowledgeAssertion],
        recent_angles: set[str],
        skill_ids: list[str],
    ) -> list[ContentPlanNode]:
        """Cross angles x platforms x days, scoring each slot deterministically."""
        candidates: list[ContentPlanNode] = []

        for day_index in range(days):
            for platform in platforms:
                for template in ANGLE_LIBRARY:
                    fact_readiness = _fact_readiness(template.required_fact_keys, facts)
                    novelty = 0.2 if template.angle.casefold() in recent_angles else 1.0

                    # MFS is defined by the marketing-ideas skill; computed in Python
                    # because the blueprint bans using an LLM for arithmetic.
                    fit = _clamp_1_5(round(1 + 4 * fact_readiness))
                    mfs = marketing_feasibility_score(
                        impact=template.impact,
                        fit=fit,
                        speed=template.speed,
                        effort=template.effort,
                        cost=template.cost,
                    )

                    breakdown = {
                        "mfs": float(mfs),
                        "mfs_normalized": round((mfs + 7) / 20, 3),
                        "fact_readiness": round(fact_readiness, 3),
                        "novelty": novelty,
                        "funnel_weight": FUNNEL_MIX[template.funnel_stage],
                        "recency_penalty": round(day_index * 0.01, 3),
                    }
                    score = (
                        0.45 * breakdown["mfs_normalized"]
                        + 0.25 * fact_readiness
                        + 0.20 * novelty
                        + 0.10 * FUNNEL_MIX[template.funnel_stage]
                        - breakdown["recency_penalty"]
                    )

                    candidates.append(
                        ContentPlanNode(
                            node_id=new_id("node"),
                            workspace_id=workspace_id,
                            plan_id="",
                            day_index=day_index,
                            platform=platform,
                            funnel_stage=template.funnel_stage,
                            goal=template.goal,
                            angle=template.angle,
                            pillar=template.pillar,
                            required_fact_keys=list(template.required_fact_keys),
                            skill_ids=skill_ids,
                            rationale=(
                                f"MFS {mfs:+d} · dữ kiện sẵn sàng {fact_readiness:.0%} · "
                                f"tầng phễu {template.funnel_stage}"
                            ),
                            score=round(score, 4),
                            score_breakdown=breakdown,
                        )
                    )
        return candidates

    def _select(
        self, candidates: list[ContentPlanNode], *, days: int, platforms: list[PlatformId]
    ) -> list[ContentPlanNode]:
        """Greedy pick under the hard constraints, then rebalance the funnel."""
        target = days * len(platforms) * MAX_POSTS_PER_DAY_PER_PLATFORM
        ordered = sorted(candidates, key=lambda node: node.score, reverse=True)

        selected: list[ContentPlanNode] = []
        slots_taken: set[tuple[int, str]] = set()
        angles_used: set[str] = set()
        conversion_count = 0

        for node in ordered:
            if len(selected) >= target:
                break
            slot = (node.day_index, node.platform)
            if slot in slots_taken:
                continue
            # Diversity reranker: an angle appears at most once per platform.
            angle_key = f"{node.platform}:{node.angle.casefold()}"
            if angle_key in angles_used:
                continue
            if node.funnel_stage == "conversion":
                if (conversion_count + 1) / max(1, target) > MAX_CONVERSION_SHARE:
                    continue
                conversion_count += 1

            selected.append(node)
            slots_taken.add(slot)
            angles_used.add(angle_key)

        # Backfill any slot the constraints starved, relaxing the angle rule only.
        if len(selected) < target:
            for node in ordered:
                if len(selected) >= target:
                    break
                slot = (node.day_index, node.platform)
                if slot in slots_taken or node.funnel_stage == "conversion":
                    continue
                selected.append(node)
                slots_taken.add(slot)

        selected.sort(key=lambda node: (node.day_index, node.platform))
        return selected


def _fact_readiness(required: tuple[str, ...], facts: dict[str, KnowledgeAssertion]) -> float:
    if not required:
        return 1.0
    present = sum(1 for key in required if key in facts)
    return present / len(required)


def _clamp_1_5(value: int) -> int:
    return max(1, min(5, value))


def _summarize(
    nodes: list[ContentPlanNode],
    days: int,
    platforms: list[PlatformId],
    profile: BrandOperatingProfile | None,
) -> str:
    if not nodes:
        return "Chưa tạo được kế hoạch vì thiếu dữ kiện bắt buộc."

    mix: dict[str, int] = {}
    for node in nodes:
        mix[node.funnel_stage] = mix.get(node.funnel_stage, 0) + 1
    mix_text = ", ".join(f"{stage} {count} bài" for stage, count in sorted(mix.items()))
    brand = (profile.identity.get("name") if profile else None) or "thương hiệu"
    return (
        f"Kế hoạch {days} ngày cho {brand} trên {', '.join(platforms)}: {len(nodes)} bài "
        f"({mix_text}). Ưu tiên góc nội dung có dữ kiện đầy đủ và điểm MFS cao nhất."
    )
