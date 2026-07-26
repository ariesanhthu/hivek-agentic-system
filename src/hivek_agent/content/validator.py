"""Content validation.

Order matters and is enforced: deterministic checks run first and can fail a draft on
their own. The LLM judge only scores things code cannot measure (naturalness, brand
fit) and is never allowed to clear a draft that failed a hard check.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from pydantic import BaseModel, Field

from hivek_agent.agentic.context_compiler import PLATFORM_RULES
from hivek_agent.domain import (
    BrandVoiceProfile,
    CompiledContext,
    ContentValidationResult,
    PlatformId,
    PostDraft,
    ValidationIssue,
)
from hivek_agent.infrastructure.llm import LLMError, LLMGateway

logger = logging.getLogger(__name__)

# Phrasing that reads as machine-written in Vietnamese marketing copy.
AI_TELL_PHRASES = (
    "trong thế giới ngày nay",
    "trong thời đại số",
    "không thể phủ nhận rằng",
    "hãy cùng khám phá",
    "điều quan trọng cần lưu ý",
    "tóm lại",
    "nhìn chung",
    "đóng vai trò quan trọng",
    "giải pháp toàn diện",
    "đáp ứng mọi nhu cầu",
)

# Claims that need evidence or create legal exposure. Written with diacritics for
# readability; compiled against folded text below, so they must be folded too.
RISKY_CLAIM_PATTERNS = (
    (r"(cam kết|đảm bảo|chắc chắn)\s+(100\s*%|tuyệt đối|hoàn toàn)", "absolute_guarantee"),
    (r"(số\s*1|tốt nhất|hàng đầu|duy nhất|nhất việt nam)", "superlative_claim"),
    (r"(chữa khỏi|trị dứt điểm|khỏi hẳn|hết bệnh)", "medical_claim"),
    (r"(lãi|lợi nhuận|sinh lời)\s+\d+\s*%", "financial_claim"),
    (r"giảm\s+\d+\s*kg", "health_outcome_claim"),
)

_NUMBER_PATTERN = re.compile(r"\b\d[\d.,]*\s*(?:%|k|tr|triệu|nghìn|đ|vnd|usd)?\b", re.IGNORECASE)
_EMOJI_PATTERN = re.compile(
    "[\U0001f300-\U0001f9ff\U0001fa00-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]",
    flags=re.UNICODE,
)
_SENTENCE_SPLIT = re.compile(r"[.!?…]+|\n+")


class SemanticVerdict(BaseModel):
    """Schema the judge model must fill. Scores only - it cannot set the decision."""

    brand_fit_score: float = Field(ge=0, le=1)
    human_likeness_score: float = Field(ge=0, le=1)
    platform_fit_score: float = Field(ge=0, le=1)
    sales_pressure_score: float = Field(ge=0, le=1)
    notes: list[str] = Field(default_factory=list)
    suggested_revision: str = ""


class ContentValidator:
    def __init__(
        self,
        llm: LLMGateway | None = None,
        *,
        judge_model: str = "",
        judge_fallbacks: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._judge_model = judge_model
        self._judge_fallbacks = judge_fallbacks or []

    def check_deterministic(
        self,
        draft: PostDraft,
        *,
        platform: PlatformId,
        context: CompiledContext | None = None,
        voice: BrandVoiceProfile | None = None,
    ) -> ContentValidationResult:
        """Pure function. No I/O, no model - fully testable and always runs."""
        text = draft.full_text
        issues: list[ValidationIssue] = []

        issues.extend(_check_platform_limits(text, draft, platform))
        issues.extend(_check_unsupported_facts(draft, context))
        issues.extend(_check_risky_claims(text))
        issues.extend(_check_banned_phrases(text, voice, context))
        ai_tells = _find_ai_tells(text)
        issues.extend(ai_tells)

        duplication = _duplication_score(text, context)
        if duplication >= 0.8:
            issues.append(
                ValidationIssue(
                    code="near_duplicate",
                    severity="error",
                    message=f"Bài quá giống nội dung đã có (điểm trùng {duplication:.2f}).",
                )
            )
        elif duplication >= 0.6:
            issues.append(
                ValidationIssue(
                    code="high_similarity",
                    severity="warning",
                    message=f"Bài khá giống nội dung cũ (điểm trùng {duplication:.2f}).",
                )
            )

        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity == "warning"]

        return ContentValidationResult(
            risk_level="red" if errors else ("amber" if warnings else "green"),
            factual_consistency_score=_factual_score(draft),
            human_likeness_score=max(0.0, 1.0 - 0.2 * len(ai_tells)),
            platform_fit_score=_platform_fit_score(text, platform),
            sales_pressure_score=_sales_pressure_score(text),
            brand_fit_score=0.5,
            duplication_score=duplication,
            issues=issues,
            final_decision="human_review" if errors else "approve",
            deterministic_only=True,
        )

    async def validate(
        self,
        draft: PostDraft,
        *,
        platform: PlatformId,
        context: CompiledContext | None = None,
        voice: BrandVoiceProfile | None = None,
    ) -> ContentValidationResult:
        result = self.check_deterministic(draft, platform=platform, context=context, voice=voice)

        # A hard failure is final. Asking a model to bless it would let the LLM
        # override a code-level rule, which the blueprint forbids.
        if result.blocking_issues or self._llm is None:
            return result

        try:
            verdict, _ = await self._llm.complete_structured(
                system=_JUDGE_SYSTEM,
                prompt=_judge_prompt(draft, platform, context),
                schema=SemanticVerdict,
                model=self._judge_model,
                temperature=0.0,
                max_output_tokens=768,
                fallback_models=self._judge_fallbacks,
            )
        except LLMError as exc:
            logger.warning("semantic judge unavailable, keeping deterministic verdict: %s", exc)
            result.issues.append(
                ValidationIssue(
                    code="judge_unavailable",
                    severity="info",
                    message="Không chạy được đánh giá ngữ nghĩa; chỉ áp dụng kiểm tra xác định.",
                )
            )
            return result

        result.brand_fit_score = verdict.brand_fit_score
        # Keep the stricter of the two opinions on human-likeness.
        result.human_likeness_score = min(result.human_likeness_score, verdict.human_likeness_score)
        result.platform_fit_score = verdict.platform_fit_score
        result.sales_pressure_score = verdict.sales_pressure_score
        result.suggested_revision = verdict.suggested_revision
        result.deterministic_only = False
        result.issues.extend(
            ValidationIssue(code="judge_note", severity="info", message=note)
            for note in verdict.notes[:5]
        )

        result.final_decision = _decide(result)
        if result.final_decision == "human_review" and result.risk_level == "green":
            result.risk_level = "amber"
        return result


def _decide(result: ContentValidationResult) -> str:
    if result.blocking_issues:
        return "human_review"
    weak = (
        result.brand_fit_score < 0.5
        or result.human_likeness_score < 0.5
        or result.platform_fit_score < 0.5
    )
    if weak or result.sales_pressure_score > 0.75:
        return "revise"
    if any(issue.severity == "warning" for issue in result.issues):
        return "human_review"
    return "approve"


def _check_platform_limits(
    text: str, draft: PostDraft, platform: PlatformId
) -> list[ValidationIssue]:
    rules = PLATFORM_RULES.get(platform)
    if not rules:
        return []

    issues: list[ValidationIssue] = []
    max_chars = int(rules["max_chars"])
    if len(text) > max_chars:
        issues.append(
            ValidationIssue(
                code="platform_length_exceeded",
                severity="error",
                message=f"Bài dài {len(text)} ký tự, vượt giới hạn {max_chars} của {platform}.",
            )
        )
    if len(draft.hashtags) > int(rules["hashtag_limit"]):
        issues.append(
            ValidationIssue(
                code="too_many_hashtags",
                severity="warning",
                message=f"{len(draft.hashtags)} hashtag, nên tối đa {rules['hashtag_limit']}.",
            )
        )
    if not draft.hook.strip():
        issues.append(
            ValidationIssue(
                code="missing_hook", severity="error", message="Bài chưa có hook mở đầu."
            )
        )
    return issues


def _check_unsupported_facts(
    draft: PostDraft, context: CompiledContext | None
) -> list[ValidationIssue]:
    """The draft may only cite facts the compiler actually supplied.

    This is what stops a model from inventing a price and attaching a plausible ID.
    """
    if context is None:
        return []

    available = {row.get("fact_id") for row in context.immutable_facts}
    issues = [
        ValidationIssue(
            code="unknown_fact_reference",
            severity="error",
            message=f"Bài dẫn fact không tồn tại trong ngữ cảnh: {fact_id}",
            evidence=fact_id,
        )
        for fact_id in draft.delivered_fact_ids
        if fact_id not in available
    ]

    if draft.missing_fact_ids:
        issues.append(
            ValidationIssue(
                code="declared_missing_facts",
                severity="warning",
                message="Bài tự khai còn thiếu dữ kiện: " + ", ".join(draft.missing_fact_ids[:5]),
            )
        )

    # A number in the copy that no supplied fact contains is an invented figure.
    fact_values = " ".join(str(row.get("value", "")) for row in context.immutable_facts)
    for number in set(_NUMBER_PATTERN.findall(draft.full_text)):
        normalized = number.strip()
        if len(normalized) < 2:
            continue
        if normalized not in fact_values:
            issues.append(
                ValidationIssue(
                    code="unsupported_number",
                    severity="warning",
                    message=f"Số liệu '{normalized}' không khớp dữ kiện nào được cung cấp.",
                    evidence=normalized,
                )
            )
    return issues


def _check_risky_claims(text: str) -> list[ValidationIssue]:
    lowered = _fold(text)
    return [
        ValidationIssue(
            code=code,
            severity="error",
            message=f"Phát hiện tuyên bố rủi ro ({code}); cần người duyệt.",
            evidence=match.group(0),
        )
        for pattern, code in _COMPILED_RISKY_CLAIMS
        if (match := pattern.search(lowered))
    ]


def _check_banned_phrases(
    text: str, voice: BrandVoiceProfile | None, context: CompiledContext | None
) -> list[ValidationIssue]:
    banned: list[str] = list(voice.banned_phrases) if voice else []
    if context:
        for rule in context.brand_rules:
            if rule.get("rule") == "banned_phrases" and isinstance(rule.get("value"), list):
                banned.extend(rule["value"])
            elif rule.get("rule") == "banned_phrase":
                banned.append(str(rule.get("value", "")))

    lowered = _fold(text)
    seen: set[str] = set()
    issues: list[ValidationIssue] = []
    for phrase in banned:
        folded = _fold(phrase)
        if not folded or folded in seen:
            continue
        seen.add(folded)
        if folded in lowered:
            issues.append(
                ValidationIssue(
                    code="banned_phrase",
                    severity="error",
                    message=f"Dùng cụm bị cấm theo hồ sơ giọng văn: '{phrase}'.",
                    evidence=phrase,
                )
            )
    return issues


def _find_ai_tells(text: str) -> list[ValidationIssue]:
    lowered = _fold(text)
    return [
        ValidationIssue(
            code="ai_phrasing",
            severity="warning",
            message=f"Cụm từ mang giọng máy: '{original}'.",
            evidence=original,
        )
        for folded, original in _FOLDED_AI_TELLS
        if folded in lowered
    ]


def _factual_score(draft: PostDraft) -> float:
    delivered = len(draft.delivered_fact_ids)
    missing = len(draft.missing_fact_ids)
    if delivered + missing == 0:
        return 0.5
    return round(delivered / (delivered + missing), 3)


def _platform_fit_score(text: str, platform: PlatformId) -> float:
    rules = PLATFORM_RULES.get(platform)
    if not rules:
        return 0.5
    ideal = int(rules["ideal_chars"])
    ratio = len(text) / ideal if ideal else 1.0
    # Full marks near the ideal length, decaying either side.
    return round(max(0.0, min(1.0, 1.0 - abs(1.0 - ratio) * 0.6)), 3)


def _sales_pressure_score(text: str) -> float:
    lowered = _fold(text)
    hits = sum(
        lowered.count(token)
        for token in ("mua ngay", "đăng ký ngay", "chốt đơn", "nhanh tay", "cuối cùng", "gấp")
    )
    hits += len(_EMOJI_PATTERN.findall(text)) // 4
    hits += text.count("!") // 3
    return round(min(1.0, hits / 6), 3)


def _duplication_score(text: str, context: CompiledContext | None) -> float:
    """Jaccard over word 5-grams against supplied examples.

    Cheap, deterministic near-duplicate detection; catches lifted sentences without
    needing embeddings.
    """
    if context is None or not context.relevant_examples:
        return 0.0

    candidate = _shingles(text)
    if not candidate:
        return 0.0

    best = 0.0
    for example in context.relevant_examples:
        reference = _shingles(str(example.get("excerpt", "")))
        if not reference:
            continue
        overlap = len(candidate & reference)
        union = len(candidate | reference)
        if union:
            best = max(best, overlap / union)
    return round(best, 3)


def _shingles(text: str, size: int = 5) -> set[str]:
    words = _fold(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _fold(text: str) -> str:
    """Casefold + strip combining marks so 'Cam Kết' and 'cam ket' compare equal."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


# Patterns are matched against folded text, so they are folded once at import. Without
# this the diacritics in the source patterns could never match the folded input and
# every risky-claim check would silently pass.
_COMPILED_RISKY_CLAIMS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(_fold(pattern)), code) for pattern, code in RISKY_CLAIM_PATTERNS
)
_FOLDED_AI_TELLS: tuple[tuple[str, str], ...] = tuple(
    (_fold(phrase), phrase) for phrase in AI_TELL_PHRASES
)


_JUDGE_SYSTEM = (
    "Bạn là bộ đánh giá nội dung của HIVE-K. Bạn CHỈ chấm điểm, không viết lại bài và "
    "không hợp thức hóa dữ kiện thiếu. Nếu không chắc, cho điểm thấp. Trả JSON hợp lệ."
)


def _judge_prompt(draft: PostDraft, platform: PlatformId, context: CompiledContext | None) -> str:
    rules = PLATFORM_RULES.get(platform, {})
    brand_rules = context.brand_rules if context else []
    return (
        f"NỀN TẢNG: {platform}\n"
        f"QUY TẮC NỀN TẢNG: {rules}\n"
        f"QUY TẮC THƯƠNG HIỆU: {brand_rules}\n\n"
        f"BÀI VIẾT:\n{draft.full_text}\n\n"
        "Chấm 0..1 cho: brand_fit_score, human_likeness_score, platform_fit_score, "
        "sales_pressure_score (cao = bán hàng gượng ép). "
        "notes: tối đa 3 nhận xét ngắn. suggested_revision: một câu gợi ý sửa."
    )
