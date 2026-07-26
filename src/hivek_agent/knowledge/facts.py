"""Fact ingestion with provenance, precedence and conflict detection.

The rules this file exists to enforce:
  - a confirmed fact is never silently overwritten;
  - two live assertions on the same claim raise a conflict instead of one winning;
  - history is superseded, never deleted.
"""

from __future__ import annotations

import logging
from typing import Any

from hivek_agent.domain import (
    SOURCE_PRECEDENCE,
    KnowledgeAssertion,
    KnowledgeConflict,
    SourceRef,
    source_rank,
    utc_now_iso,
)
from hivek_agent.repositories import KnowledgeRepository, new_id

logger = logging.getLogger(__name__)

# Below this gap, two sources are "comparably trustworthy" and we refuse to pick a
# winner automatically - the user is asked instead.
_PRECEDENCE_TIE_BAND = 1
_CONFIDENCE_TIE_BAND = 0.15


class FactIngestionResult:
    """Outcome of upserting one assertion, so the caller can explain what happened."""

    def __init__(
        self,
        assertion: KnowledgeAssertion,
        *,
        action: str,
        conflict: KnowledgeConflict | None = None,
        superseded_id: str | None = None,
    ) -> None:
        self.assertion = assertion
        self.action = action  # created | superseded_existing | conflict | ignored_weaker
        self.conflict = conflict
        self.superseded_id = superseded_id

    @property
    def created_conflict(self) -> bool:
        return self.conflict is not None


class FactService:
    def __init__(self, knowledge: KnowledgeRepository) -> None:
        self._knowledge = knowledge

    async def upsert_fact(
        self,
        *,
        workspace_id: str,
        subject_id: str,
        predicate: str,
        object_value: str | float | bool | list[Any] | dict[str, Any],
        source: SourceRef,
        confidence: float | None = None,
        approval_status: str = "candidate",
    ) -> FactIngestionResult:
        """Add a fact, reconciling it against whatever is already known.

        Returns what it decided and why, rather than mutating quietly.
        """
        incoming = KnowledgeAssertion(
            assertion_id=new_id("fact"),
            workspace_id=workspace_id,
            subject_id=subject_id,
            predicate=predicate,
            object_value=object_value,
            source=source,
            confidence=confidence if confidence is not None else source.confidence,
            approval_status=approval_status,  # type: ignore[arg-type]
        )

        existing = await self._knowledge.find_live_by_key(workspace_id, subject_id, predicate)
        agreeing = [item for item in existing if _same_value(item.object_value, object_value)]
        disagreeing = [
            item for item in existing if not _same_value(item.object_value, object_value)
        ]

        # Re-observing a known fact reinforces it instead of duplicating a row.
        if agreeing and not disagreeing:
            winner = agreeing[0]
            winner.confidence = min(1.0, max(winner.confidence, incoming.confidence) + 0.05)
            if source.approved and winner.approval_status == "candidate":
                winner.approval_status = "confirmed"
            await self._knowledge.update_assertion(winner)
            return FactIngestionResult(winner, action="reinforced")

        if not disagreeing:
            await self._knowledge.add_assertion(incoming)
            return FactIngestionResult(incoming, action="created")

        return await self._reconcile(incoming, disagreeing)

    async def _reconcile(
        self, incoming: KnowledgeAssertion, rivals: list[KnowledgeAssertion]
    ) -> FactIngestionResult:
        strongest = max(rivals, key=_strength)

        # A user-confirmed fact outranks any inference. Never overwrite it silently;
        # record the newcomer as a conflict so a human decides.
        if strongest.approval_status == "confirmed" and incoming.approval_status != "confirmed":
            incoming.approval_status = "conflict"
            await self._knowledge.add_assertion(incoming)
            conflict = await self._raise_conflict(
                incoming,
                [strongest],
                reason=(
                    "Nguồn mới mâu thuẫn với dữ kiện đã được người dùng xác nhận. "
                    "Cần xác nhận lại trước khi thay đổi."
                ),
            )
            return FactIngestionResult(incoming, action="conflict", conflict=conflict)

        incoming_strength = _strength(incoming)
        rival_strength = _strength(strongest)
        precedence_gap = incoming_strength[0] - rival_strength[0]
        confidence_gap = incoming_strength[1] - rival_strength[1]

        # Comparable trust on both sides -> do not guess.
        if (
            abs(precedence_gap) <= _PRECEDENCE_TIE_BAND
            and abs(confidence_gap) < _CONFIDENCE_TIE_BAND
        ):
            incoming.approval_status = "conflict"
            await self._knowledge.add_assertion(incoming)
            conflict = await self._raise_conflict(
                incoming,
                rivals,
                reason="Hai nguồn có độ tin cậy tương đương nhưng giá trị khác nhau.",
            )
            return FactIngestionResult(incoming, action="conflict", conflict=conflict)

        if precedence_gap < 0 or (precedence_gap == 0 and confidence_gap < 0):
            incoming.approval_status = "rejected"
            await self._knowledge.add_assertion(incoming)
            return FactIngestionResult(incoming, action="ignored_weaker")

        # Clearly stronger source wins, but the old value is superseded, not deleted.
        incoming.supersedes_id = strongest.assertion_id
        await self._knowledge.add_assertion(incoming)
        for rival in rivals:
            rival.approval_status = "superseded"
            rival.valid_to = utc_now_iso()
            await self._knowledge.update_assertion(rival)
        return FactIngestionResult(
            incoming, action="superseded_existing", superseded_id=strongest.assertion_id
        )

    async def _raise_conflict(
        self, incoming: KnowledgeAssertion, rivals: list[KnowledgeAssertion], *, reason: str
    ) -> KnowledgeConflict:
        conflict = KnowledgeConflict(
            conflict_id=new_id("conflict"),
            workspace_id=incoming.workspace_id,
            key=incoming.key,
            assertion_ids=[incoming.assertion_id, *[rival.assertion_id for rival in rivals]],
            reason=reason,
        )
        await self._knowledge.add_conflict(conflict)
        logger.info("conflict detected workspace=%s key=%s", incoming.workspace_id, incoming.key)
        return conflict

    async def confirm_fact(self, workspace_id: str, assertion_id: str) -> KnowledgeAssertion | None:
        """User confirmation. Supersedes every rival on the same key."""
        assertion = await self._knowledge.get_assertion(workspace_id, assertion_id)
        if assertion is None:
            return None

        assertion.approval_status = "confirmed"
        assertion.confidence = 1.0
        assertion.source.approved = True
        await self._knowledge.update_assertion(assertion)

        rivals = await self._knowledge.find_live_by_key(
            workspace_id, assertion.subject_id, assertion.predicate
        )
        for rival in rivals:
            if rival.assertion_id == assertion_id:
                continue
            rival.approval_status = "superseded"
            rival.valid_to = utc_now_iso()
            await self._knowledge.update_assertion(rival)

        for conflict in await self._knowledge.list_conflicts(workspace_id):
            if assertion_id in conflict.assertion_ids:
                conflict.resolved = True
                conflict.resolved_assertion_id = assertion_id
                await self._knowledge.resolve_conflict(conflict)

        return assertion

    async def usable_facts(self, workspace_id: str) -> dict[str, KnowledgeAssertion]:
        """Best usable assertion per claim key - what the context compiler may read."""
        best: dict[str, KnowledgeAssertion] = {}
        for assertion in await self._knowledge.list_assertions(workspace_id):
            if not assertion.is_usable:
                continue
            current = best.get(assertion.key)
            if current is None or _strength(assertion) > _strength(current):
                best[assertion.key] = assertion
        return best


def _strength(assertion: KnowledgeAssertion) -> tuple[int, float, str]:
    """Comparable strength: source precedence, then confidence, then recency.

    A confirmed fact ranks above every source type, so nothing can outrank a human.
    """
    rank = source_rank(assertion.source.source_type)
    if assertion.approval_status == "confirmed":
        rank = len(SOURCE_PRECEDENCE)
    return (rank, assertion.confidence, assertion.created_at)


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, float | int) and isinstance(right, float | int):
        return abs(float(left) - float(right)) < 1e-9
    if isinstance(left, list) and isinstance(right, list):
        # Multi-valued facts (e.g. channels) are sets in spirit: picking the same
        # platforms in a different order is the same answer, not a new one.
        return _normalize_list(left) == _normalize_list(right)
    return left == right


def _normalize_list(values: list[Any]) -> list[str]:
    return sorted(str(value).strip().casefold() for value in values)
