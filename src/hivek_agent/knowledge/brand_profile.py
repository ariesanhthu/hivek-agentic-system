"""Brand snapshot + gap detection.

Both are rule-based. `readiness_score` is arithmetic over real coverage, not a number
an LLM invented, and the gap detector reports where it already looked before asking
the user anything.
"""

from __future__ import annotations

from typing import Any

from hivek_agent.domain import (
    BrandOperatingProfile,
    KnowledgeAssertion,
    MissingItem,
)
from hivek_agent.knowledge.facts import FactService
from hivek_agent.repositories import KnowledgeRepository

# (predicate, severity, human reason, suggested action)
# Ordered by how much each blocks useful content generation.
REQUIRED_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    (
        "brand.name",
        "blocking",
        "Chưa xác định tên thương hiệu nên không thể viết bài đúng danh xưng.",
        "Nhập tên thương hiệu trong bước thiết lập.",
    ),
    (
        "brand.tone",
        "blocking",
        "Chưa có giọng điệu nên nội dung sẽ không nhất quán.",
        "Chọn giọng điệu ở bước thiết lập thương hiệu.",
    ),
    (
        "brand.channels",
        "blocking",
        "Chưa biết thương hiệu đang dùng kênh nào nên không chọn được định dạng bài.",
        "Chọn các kênh mạng xã hội đang sử dụng.",
    ),
    (
        "brand.audience",
        "quality",
        "Chưa mô tả khách hàng mục tiêu nên bài viết sẽ chung chung.",
        "Mô tả ngắn nhóm khách hàng chính.",
    ),
    (
        "brand.product",
        "quality",
        "Chưa có sản phẩm/dịch vụ cụ thể nên bài viết thiếu nội dung bán hàng.",
        "Thêm ít nhất một sản phẩm hoặc dịch vụ.",
    ),
    (
        "brand.resource_url",
        "optional",
        "Chưa có kho tài nguyên nên không dùng được hình ảnh/guideline sẵn có.",
        "Thêm đường dẫn Drive chứa guideline hoặc hình ảnh.",
    ),
)

_SEVERITY_ORDER = {"blocking": 0, "quality": 1, "optional": 2}


class BrandProfileService:
    def __init__(self, knowledge: KnowledgeRepository, facts: FactService) -> None:
        self._knowledge = knowledge
        self._facts = facts

    async def detect_gaps(self, workspace_id: str) -> list[MissingItem]:
        """List what is missing, and prove we looked before asking."""
        usable = await self._facts.usable_facts(workspace_id)
        all_assertions = await self._knowledge.list_assertions(workspace_id)
        searched = _searched_source_labels(all_assertions)

        gaps: list[MissingItem] = []
        for predicate, severity, reason, action in REQUIRED_FIELDS:
            key = f"workspace::{predicate}"
            if key in usable:
                continue
            gaps.append(
                MissingItem(
                    field=predicate,
                    severity=severity,  # type: ignore[arg-type]
                    reason=reason,
                    searched_sources=searched or ["Chưa có nguồn dữ liệu nào được kết nối."],
                    suggested_action=action,
                    can_infer=severity == "optional",
                )
            )

        # An unresolved conflict blocks just as hard as an absent fact - the value
        # exists but we do not know which one is true.
        for conflict in await self._knowledge.list_conflicts(workspace_id):
            gaps.append(
                MissingItem(
                    field=conflict.key,
                    severity="blocking",
                    reason=f"Có mâu thuẫn dữ liệu: {conflict.reason}",
                    searched_sources=searched,
                    suggested_action="Chọn giá trị đúng để hệ thống ghi nhận.",
                    ui_target={"type": "resolve_conflict", "conflict_id": conflict.conflict_id},
                )
            )

        gaps.sort(key=lambda item: _SEVERITY_ORDER.get(item.severity, 9))
        return gaps

    async def build_profile(self, workspace_id: str) -> BrandOperatingProfile:
        usable = await self._facts.usable_facts(workspace_id)
        gaps = await self.detect_gaps(workspace_id)
        conflicts = await self._knowledge.list_conflicts(workspace_id)

        def value_of(predicate: str) -> Any:
            assertion = usable.get(f"workspace::{predicate}")
            return assertion.object_value if assertion else None

        identity = {
            "name": value_of("brand.name"),
            "tone": value_of("brand.tone"),
            "resource_url": value_of("brand.resource_url"),
        }
        channels = value_of("brand.channels") or []
        if isinstance(channels, str):
            channels = [channels]

        products = [
            {"name": assertion.object_value, "source_id": assertion.source.source_id}
            for key, assertion in usable.items()
            if key.endswith("::brand.product")
        ]
        audiences = [
            {"description": assertion.object_value, "source_id": assertion.source.source_id}
            for key, assertion in usable.items()
            if key.endswith("::brand.audience")
        ]
        approved_claims = [
            {
                "assertion_id": assertion.assertion_id,
                "predicate": assertion.predicate,
                "value": assertion.object_value,
                "source_id": assertion.source.source_id,
            }
            for assertion in usable.values()
            if assertion.approval_status == "confirmed"
        ]

        existing = await self._knowledge.get_brand_profile(workspace_id)
        profile = BrandOperatingProfile(
            workspace_id=workspace_id,
            version=(existing.version + 1) if existing else 1,
            identity=identity,
            products=products,
            audiences=audiences,
            approved_claims=approved_claims,
            channel_roles=[{"platform": channel, "role": "primary"} for channel in channels],
            missing_items=gaps,
            conflicts=[conflict.model_dump() for conflict in conflicts],
            source_coverage=_source_coverage(list(usable.values())),
            readiness_score=compute_readiness_score(usable, gaps, len(conflicts)),
        )
        await self._knowledge.save_brand_profile(profile)
        return profile


def compute_readiness_score(
    usable: dict[str, KnowledgeAssertion],
    gaps: list[MissingItem],
    unresolved_conflicts: int,
) -> float:
    """Weighted coverage, clamped to 0..1.

    Deliberately arithmetic so the number means the same thing every run and can be
    asserted in tests.
    """
    weights = {"blocking": 0.5, "quality": 0.3, "optional": 0.2}
    totals = {"blocking": 0, "quality": 0, "optional": 0}
    for _, severity, _, _ in REQUIRED_FIELDS:
        totals[severity] += 1

    missing = {"blocking": 0, "quality": 0, "optional": 0}
    for gap in gaps:
        if gap.severity in missing and gap.field.startswith("brand."):
            missing[gap.severity] += 1

    score = 0.0
    for severity, weight in weights.items():
        total = totals[severity]
        if total == 0:
            score += weight
            continue
        score += weight * ((total - missing[severity]) / total)

    confirmed = sum(1 for item in usable.values() if item.approval_status == "confirmed")
    if confirmed and usable:
        score += 0.05 * min(1.0, confirmed / len(usable))

    score -= 0.15 * unresolved_conflicts
    return round(max(0.0, min(1.0, score)), 3)


def _searched_source_labels(assertions: list[KnowledgeAssertion]) -> list[str]:
    labels: list[str] = []
    for assertion in assertions:
        label = assertion.source.source_id
        if label not in labels:
            labels.append(label)
    return labels[:8]


def _source_coverage(assertions: list[KnowledgeAssertion]) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for assertion in assertions:
        coverage[assertion.source.source_type] = coverage.get(assertion.source.source_type, 0) + 1
    return coverage
