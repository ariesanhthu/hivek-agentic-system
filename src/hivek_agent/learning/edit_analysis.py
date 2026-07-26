"""Deterministic edit analysis and the preference lifecycle.

Blueprint rule: "Khong dung mo hinh ngon ngu cho phep tinh hoac kiem tra co the viet
bang ma." Every number here is arithmetic over the two strings and every phrase comes
from `difflib`. No LLM call, no network, no randomness - the same before/after pair
always produces the same `FeatureDelta`.

The second rule this module enforces is the memory lifecycle::

    candidate -> repeated -> stable -> deprecated -> rejected

A single edit never becomes a permanent rule. `promote_preferences` is the only writer
of preference status and it *recomputes* status from `observation_count`/`explicit` on
every call, so no caller can hand-craft a `stable` rule out of one observation.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable
from typing import Any, Literal

from hivek_agent.domain import (
    BrandVoiceProfile,
    EditLearningEvent,
    FeatureDelta,
    FeedbackEvent,
    FeedbackEventType,
    MemoryStatus,
    PreferenceCandidate,
)
from hivek_agent.domain.learning import PREFERENCE_PROMOTION_THRESHOLD
from hivek_agent.repositories import KnowledgeRepository, LearningRepository, new_id

__all__ = [
    "LearningService",
    "compute_feature_delta",
    "count_emoji",
    "infer_preferences",
    "split_sentences",
]

# --------------------------------------------------------------------------------------
# Text primitives
# --------------------------------------------------------------------------------------

# Emoji blocks: flags, pictographs, emoticons, transport, supplemental, misc symbols,
# dingbats, symbols-and-arrows, enclosed alphanumerics. Deliberately excludes (c), (R),
# (TM) and plain arrows - those are punctuation in Vietnamese marketing copy, and
# counting them would fire `reduce_emoji` on a post that has no emoji at all.
_EMOJI_RANGES = (
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\U0001f170-\U0001f251"  # enclosed alphanumeric / ideographic supplement
    "\U0001f300-\U0001f5ff"  # misc symbols and pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport and map symbols
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # geometric shapes extended
    "\U0001f800-\U0001f8ff"  # supplemental arrows-c
    "\U0001f900-\U0001f9ff"  # supplemental symbols and pictographs
    "\U0001fa00-\U0001faff"  # symbols and pictographs extended-a
    "\u2600-\u26ff"  # misc symbols
    "\u2700-\u27bf"  # dingbats
    "\u2b00-\u2bff"  # misc symbols and arrows
)

# Escapes, not literals: ZWJ and VS16 are zero-width, so a literal here would be an
# invisible character that any editor or copy-paste could silently drop.
_ZWJ = "\u200d"
_VS16 = "\ufe0f"
_KEYCAP = "\u20e3"
_SKIN_TONES = "\U0001f3fb-\U0001f3ff"

_EMOJI_ATOM = f"[{_EMOJI_RANGES}]"
_EMOJI_TRAIL = f"(?:{_VS16}|[{_SKIN_TONES}])*"

# One match == one emoji *as rendered*: a keycap, a flag pair, or a base glyph with its
# skin-tone/variation modifiers and any ZWJ-joined continuation. Counting raw codepoints
# would score a family emoji as 4 and a flag as 2.
_EMOJI_PATTERN = re.compile(
    f"(?:[0-9#*]{_VS16}?{_KEYCAP})"
    f"|(?:[\U0001f1e6-\U0001f1ff]{{2}})"
    f"|(?:{_EMOJI_ATOM}{_EMOJI_TRAIL}(?:{_ZWJ}{_EMOJI_ATOM}{_EMOJI_TRAIL})*)"
)

_SENTENCE_SPLIT = re.compile(r"[.!?…]+|\n+")
_QUESTION_RUN = re.compile(r"\?+")
_EXCLAMATION_RUN = re.compile(r"!+")
_WORD = re.compile(r"\S+")

_MIN_PHRASE_WORDS = 2
_MAX_PHRASES = 8
_MAX_PHRASE_CHARS = 80


def _nfc(text: str) -> str:
    """Compose Vietnamese diacritics so `len()` and `==` mean what they look like.

    Vietnamese arrives both composed and decomposed depending on the client; without
    this, "ế" as one codepoint would never equal "ế" as two, and length ratios would be
    inflated for decomposed text.
    """
    return unicodedata.normalize("NFC", text or "")


def count_emoji(text: str) -> int:
    """Number of rendered emoji in `text`."""
    return len(_EMOJI_PATTERN.findall(_nfc(text)))


def split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation and newlines. Empty fragments are dropped."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(_nfc(text)) if part.strip()]


def _token_key(token: str) -> str:
    """Comparison key for diffing: case- and punctuation-insensitive.

    Diacritics are kept - "má" and "ma" are different words, and the stored phrase must
    stay exactly as the user wrote it.
    """
    trimmed = token
    while trimmed and unicodedata.category(trimmed[0]).startswith("P"):
        trimmed = trimmed[1:]
    while trimmed and unicodedata.category(trimmed[-1]).startswith("P"):
        trimmed = trimmed[:-1]
    return unicodedata.normalize("NFC", trimmed).casefold()


def _clip(text: str, limit: int) -> str:
    """Truncate on a word boundary so a banned phrase never ends mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    if " " in cut:
        cut = cut[: cut.rindex(" ")].rstrip()
    return cut or text[:limit]


def _append_phrase(sink: list[str], words: list[str]) -> None:
    if len(words) < _MIN_PHRASE_WORDS:
        return
    phrase = _clip(" ".join(words), _MAX_PHRASE_CHARS)
    if phrase and phrase not in sink:
        sink.append(phrase)


def _phrase_diff(before: str, after: str) -> tuple[list[str], list[str]]:
    """Real removed/added word runs, via `difflib` over normalised word lists."""
    before_words = _WORD.findall(before)
    after_words = _WORD.findall(after)
    matcher = difflib.SequenceMatcher(
        a=[_token_key(word) for word in before_words],
        b=[_token_key(word) for word in after_words],
        # autojunk treats any token appearing in >1% of a 200+ element sequence as junk,
        # which on a long post silently drops the most common (most telling) words.
        autojunk=False,
    )
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            _append_phrase(removed, before_words[i1:i2])
        if tag in ("insert", "replace"):
            _append_phrase(added, after_words[j1:j2])
    return removed[:_MAX_PHRASES], added[:_MAX_PHRASES]


def _avg_sentence_words(sentences: list[str]) -> float:
    if not sentences:
        return 0.0
    return sum(len(_WORD.findall(sentence)) for sentence in sentences) / len(sentences)


def _question_ratio(text: str, sentence_count: int) -> float:
    return len(_QUESTION_RUN.findall(text)) / max(sentence_count, 1)


# --------------------------------------------------------------------------------------
# Feature delta
# --------------------------------------------------------------------------------------

_LENGTH_NOTE_THRESHOLD = 0.05
_QUESTION_NOTE_THRESHOLD = 0.1
_EXCLAMATION_NOTE_THRESHOLD = 2
_SHORTER_SENTENCE_FACTOR = 0.8
_LONGER_SENTENCE_FACTOR = 1.25


def _structural_changes(delta: FeatureDelta, before_avg: float, after_avg: float) -> list[str]:
    """Human-readable Vietnamese summary of what the edit did."""
    changes: list[str] = []
    percent = round(abs(delta.length_delta_ratio) * 100)

    if delta.length_delta_ratio <= -_LENGTH_NOTE_THRESHOLD:
        changes.append(f"rút ngắn {percent}%")
    elif delta.length_delta_ratio >= _LENGTH_NOTE_THRESHOLD:
        changes.append(f"viết dài thêm {percent}%")

    if delta.emoji_delta < 0:
        changes.append(f"bỏ {abs(delta.emoji_delta)} emoji")
    elif delta.emoji_delta > 0:
        changes.append(f"thêm {delta.emoji_delta} emoji")

    if before_avg > 0 and after_avg > 0:
        if after_avg <= before_avg * _SHORTER_SENTENCE_FACTOR:
            changes.append("chia đoạn ngắn hơn")
        elif after_avg >= before_avg * _LONGER_SENTENCE_FACTOR:
            changes.append("viết câu dài hơn")

    if delta.question_ratio_delta > _QUESTION_NOTE_THRESHOLD:
        changes.append("thêm câu hỏi")
    elif delta.question_ratio_delta < -_QUESTION_NOTE_THRESHOLD:
        changes.append("bỏ câu hỏi")

    if delta.exclamation_delta <= -_EXCLAMATION_NOTE_THRESHOLD:
        changes.append("giảm câu cảm thán")
    elif delta.exclamation_delta >= _EXCLAMATION_NOTE_THRESHOLD:
        changes.append("thêm câu cảm thán")

    if delta.removed_phrases:
        changes.append(f"bỏ {len(delta.removed_phrases)} cụm từ")
    if delta.added_phrases:
        changes.append(f"thêm {len(delta.added_phrases)} cụm từ")

    return changes


def compute_feature_delta(before: str, after: str) -> FeatureDelta:
    """Diff two drafts into the deterministic features the learner reasons over."""
    before_text = _nfc(before)
    after_text = _nfc(after)

    before_sentences = split_sentences(before_text)
    after_sentences = split_sentences(after_text)

    delta = FeatureDelta(
        length_delta_ratio=round(
            (len(after_text) - len(before_text)) / max(len(before_text), 1), 3
        ),
        sentence_count_delta=len(after_sentences) - len(before_sentences),
        emoji_delta=count_emoji(after_text) - count_emoji(before_text),
        question_ratio_delta=round(
            _question_ratio(after_text, len(after_sentences))
            - _question_ratio(before_text, len(before_sentences)),
            3,
        ),
        exclamation_delta=(
            len(_EXCLAMATION_RUN.findall(after_text)) - len(_EXCLAMATION_RUN.findall(before_text))
        ),
    )
    delta.removed_phrases, delta.added_phrases = _phrase_diff(before_text, after_text)
    delta.structural_changes = _structural_changes(
        delta,
        _avg_sentence_words(before_sentences),
        _avg_sentence_words(after_sentences),
    )
    return delta


# --------------------------------------------------------------------------------------
# Preference inference
# --------------------------------------------------------------------------------------

_LENGTH_RULE_THRESHOLD = 0.25
_LENGTH_FULL_SIGNAL = 0.5
_EMOJI_RULE_THRESHOLD = 2
_EMOJI_FULL_SIGNAL = 6
_QUESTION_RULE_THRESHOLD = 0.1
_QUESTION_FULL_SIGNAL = 0.5
_EXCLAMATION_RULE_THRESHOLD = 2
_EXCLAMATION_FULL_SIGNAL = 5
_BANNED_PHRASE_STRENGTH = 0.25
_MAX_BANNED_PER_EDIT = 3

# A fresh inference is a hypothesis, never a fact: confidence stays in [0.3, 0.6] no
# matter how loud the signal. Only repetition (see `_reinforce`) pushes it higher.
_CONFIDENCE_MIN = 0.3
_CONFIDENCE_SPAN = 0.2
_EXPLICIT_REASON_BONUS = 0.1
_CONFIDENCE_MAX = 0.6

# Repetition closes 30% of the remaining gap to certainty, and never reaches 1.0.
_REINFORCE_RATE = 0.3
_REINFORCE_CEILING = 0.95


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _strength(value: float, full_signal: float) -> float:
    return _clamp01(abs(value) / full_signal)


def _confidence(strength: float, *, reasoned: bool) -> float:
    value = _CONFIDENCE_MIN + _CONFIDENCE_SPAN * _clamp01(strength)
    if reasoned:
        value += _EXPLICIT_REASON_BONUS
    return round(min(value, _CONFIDENCE_MAX), 3)


# Which rules are brand-wide vs format-specific.
#
# A phrase the brand dislikes is disliked everywhere, so scoping it per platform would
# mean re-learning it on each channel (2 edits x N platforms before anything applies).
# Length and emoji genuinely differ by channel - TikTok copy is shorter than Facebook
# copy by nature - so those stay platform-scoped.
_GLOBAL_RULE_TYPES = frozenset({"banned_phrase", "tone"})


def _scope_for(rule_type: str, platform: str | None) -> Literal["global", "platform"]:
    if rule_type in _GLOBAL_RULE_TYPES or platform is None:
        return "global"
    return "platform"


def infer_preferences(
    workspace_id: str,
    asset_id: str,
    delta: FeatureDelta,
    *,
    platform: str | None = None,
    explicit_reason: str | None = None,
) -> list[PreferenceCandidate]:
    """Turn one edit's features into candidate voice rules.

    Rule-based and deterministic. Every result is a `candidate` with a single
    observation - promotion is `promote_preferences`' job, not this function's.
    """
    reasoned = bool(explicit_reason and explicit_reason.strip())

    # (rule_type, rule_value, signal strength in [0, 1])
    signals: list[tuple[str, str, float]] = []

    ratio = delta.length_delta_ratio
    if ratio <= -_LENGTH_RULE_THRESHOLD:
        signals.append(("length", "prefer_shorter", _strength(ratio, _LENGTH_FULL_SIGNAL)))
    elif ratio >= _LENGTH_RULE_THRESHOLD:
        signals.append(("length", "prefer_longer", _strength(ratio, _LENGTH_FULL_SIGNAL)))

    if delta.emoji_delta <= -_EMOJI_RULE_THRESHOLD:
        signals.append(("emoji", "reduce_emoji", _strength(delta.emoji_delta, _EMOJI_FULL_SIGNAL)))
    elif delta.emoji_delta >= _EMOJI_RULE_THRESHOLD:
        signals.append(
            ("emoji", "increase_emoji", _strength(delta.emoji_delta, _EMOJI_FULL_SIGNAL))
        )

    for phrase in delta.removed_phrases[:_MAX_BANNED_PER_EDIT]:
        signals.append(("banned_phrase", phrase, _BANNED_PHRASE_STRENGTH))

    if delta.question_ratio_delta > _QUESTION_RULE_THRESHOLD:
        signals.append(
            (
                "opening",
                "prefer_question_hook",
                _strength(delta.question_ratio_delta, _QUESTION_FULL_SIGNAL),
            )
        )

    if delta.exclamation_delta <= -_EXCLAMATION_RULE_THRESHOLD:
        signals.append(
            ("tone", "reduce_hype", _strength(delta.exclamation_delta, _EXCLAMATION_FULL_SIGNAL))
        )

    return [
        PreferenceCandidate(
            preference_id=new_id("pref"),
            workspace_id=workspace_id,
            rule_type=rule_type,
            rule_value=rule_value,
            scope=_scope_for(rule_type, platform),
            platform=platform if _scope_for(rule_type, platform) == "platform" else None,
            status="candidate",
            observation_count=1,
            confidence=_confidence(strength, reasoned=reasoned),
            evidence_asset_ids=[asset_id],
        )
        for rule_type, rule_value, strength in signals
    ]


# --------------------------------------------------------------------------------------
# Voice profile folding
# --------------------------------------------------------------------------------------

_DEFAULT_SENTENCE_RANGE = (8, 24)
_SHORT_SENTENCE_RANGE = (6, 16)
_LONG_SENTENCE_RANGE = (12, 32)
_DEFAULT_MAX_EMOJI = 3
_LOW_MAX_EMOJI = 1
_HIGH_MAX_EMOJI = 6
_MAX_BANNED_PHRASES = 50

_QUESTION_OPENING = "mở đầu bằng câu hỏi"
_HYPE_OPENING = "mở đầu cường điệu"


def _dedup(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _fold_preferences(preferences: list[PreferenceCandidate]) -> dict[str, Any]:
    """Collapse rules into voice-profile fields, returning only the keys they touch.

    Only touched keys are returned so that a platform override carries the platform's
    rules and nothing else - an override full of defaults would silently mask a global
    rule it never meant to contradict.

    Weakest first, so when two rules disagree (`prefer_shorter` vs `prefer_longer`) the
    better-evidenced one is applied last and wins. `key` breaks ties so the fold is a
    pure function of the rule set, not of store ordering.
    """
    folded: dict[str, Any] = {}
    banned: list[str] = []
    preferred_openings: list[str] = []
    avoided_openings: list[str] = []

    ordered = sorted(
        preferences, key=lambda pref: (pref.observation_count, pref.confidence, pref.key)
    )
    for pref in ordered:
        if pref.rule_type == "banned_phrase":
            banned.append(pref.rule_value)
        elif pref.rule_type == "length":
            if pref.rule_value == "prefer_shorter":
                folded["sentence_length_range"] = list(_SHORT_SENTENCE_RANGE)
            elif pref.rule_value == "prefer_longer":
                folded["sentence_length_range"] = list(_LONG_SENTENCE_RANGE)
        elif pref.rule_type == "emoji":
            if pref.rule_value == "reduce_emoji":
                folded["emoji_policy"] = {"max_per_post": _LOW_MAX_EMOJI}
            elif pref.rule_value == "increase_emoji":
                folded["emoji_policy"] = {"max_per_post": _HIGH_MAX_EMOJI}
        elif pref.rule_type == "opening" and pref.rule_value == "prefer_question_hook":
            preferred_openings.append(_QUESTION_OPENING)
        elif pref.rule_type == "tone" and pref.rule_value == "reduce_hype":
            avoided_openings.append(_HYPE_OPENING)

    if banned:
        folded["banned_phrases"] = _dedup(banned)[:_MAX_BANNED_PHRASES]
    if preferred_openings:
        folded["preferred_openings"] = _dedup(preferred_openings)
    if avoided_openings:
        folded["avoided_openings"] = _dedup(avoided_openings)
    return folded


# --------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------


def _reinforce(existing: float, incoming: float) -> float:
    """Raise confidence on repetition, asymptotically - repetition is not proof."""
    base = max(existing, incoming)
    return round(min(base + (1.0 - base) * _REINFORCE_RATE, _REINFORCE_CEILING), 3)


def _lifecycle_status(preference: PreferenceCandidate) -> MemoryStatus:
    """The single source of truth for preference status.

    Derived from evidence every time, never trusted from the incoming payload: this is
    what makes "one edit is not a rule" a property of the code rather than a convention
    callers are asked to respect.
    """
    if preference.status in ("deprecated", "rejected"):
        # A retired rule does not come back to life just because it was observed again.
        return preference.status
    if preference.explicit:
        # The user pinned it; that is a decision, not an inference.
        return "stable"
    if preference.observation_count >= PREFERENCE_PROMOTION_THRESHOLD * 2:
        return "stable"
    if preference.observation_count >= PREFERENCE_PROMOTION_THRESHOLD:
        return "repeated"
    return "candidate"


def _merge_preference(
    existing: PreferenceCandidate, incoming: PreferenceCandidate
) -> PreferenceCandidate:
    merged = existing.model_copy(deep=True)
    merged.observation_count = existing.observation_count + max(incoming.observation_count, 1)
    merged.evidence_asset_ids = _dedup([*existing.evidence_asset_ids, *incoming.evidence_asset_ids])
    merged.confidence = _reinforce(existing.confidence, incoming.confidence)
    merged.explicit = existing.explicit or incoming.explicit
    return merged


class LearningService:
    """Writes edit events, runs the preference lifecycle, and rebuilds the voice profile."""

    def __init__(self, learning: LearningRepository, knowledge: KnowledgeRepository) -> None:
        self._learning = learning
        self._knowledge = knowledge

    async def record_edit(
        self,
        *,
        workspace_id: str,
        asset_id: str,
        before_text: str,
        after_text: str,
        platform: str | None = None,
        reason: str | None = None,
        explicit: bool = False,
    ) -> EditLearningEvent:
        """Analyse one user edit and store it with the rules it hints at.

        Deliberately does not promote anything: recording what happened and deciding it
        is a rule are separate steps, and only the caller knows if this edit is part of
        a reviewed batch. `explicit=True` marks the inferred rules as user-pinned.
        """
        delta = compute_feature_delta(before_text, after_text)
        candidates = infer_preferences(
            workspace_id,
            asset_id,
            delta,
            platform=platform,
            explicit_reason=reason,
        )
        if explicit:
            for candidate in candidates:
                candidate.explicit = True

        event = EditLearningEvent(
            event_id=new_id("edit"),
            workspace_id=workspace_id,
            asset_id=asset_id,
            before_text=before_text,
            after_text=after_text,
            feature_delta=delta,
            inferred_preferences=candidates,
            explicit_reason=reason,
            confidence=max(
                (candidate.confidence for candidate in candidates), default=_CONFIDENCE_MIN
            ),
        )
        await self._learning.add_edit_event(event)
        return event

    async def promote_preferences(
        self, workspace_id: str, candidates: list[PreferenceCandidate]
    ) -> list[PreferenceCandidate]:
        """Merge candidates into stored preferences and re-derive their status.

        The blueprint's core learning rule lives here: a rule seen once stays
        `candidate` and cannot steer generation. It takes
        `PREFERENCE_PROMOTION_THRESHOLD` observations to reach `repeated`, twice that to
        reach `stable`, or an explicit pin from the user.
        """
        updated: list[PreferenceCandidate] = []
        for candidate in candidates:
            if candidate.workspace_id != workspace_id:
                raise ValueError(
                    f"preference {candidate.preference_id} belongs to workspace "
                    f"{candidate.workspace_id!r}, not {workspace_id!r}"
                )
            existing = await self._learning.get_preference(workspace_id, candidate.key)
            merged = (
                _merge_preference(existing, candidate)
                if existing
                else candidate.model_copy(deep=True)
            )
            merged.status = _lifecycle_status(merged)
            await self._learning.upsert_preference(merged)
            updated.append(merged)
        return updated

    async def rebuild_voice_profile(self, workspace_id: str) -> BrandVoiceProfile:
        """Fold every *active* rule into a fresh version of the voice profile.

        Only `repeated`/`stable` rules are read, so candidates never reach generation.
        Derived fields are rebuilt rather than merged - otherwise a deprecated rule
        would linger in the profile forever. Fields no rule can produce (tone, pronoun
        rules, CTA types) are carried over from the previous version.
        """
        preferences = await self._learning.list_preferences(workspace_id, active_only=True)
        previous = await self._knowledge.get_voice_profile(workspace_id)

        profile = BrandVoiceProfile(
            workspace_id=workspace_id,
            version=previous.version + 1 if previous else 1,
            tone=previous.tone if previous else None,
            pronoun_rules=dict(previous.pronoun_rules) if previous else {},
            preferred_cta_types=list(previous.preferred_cta_types) if previous else [],
        )

        global_prefs = [pref for pref in preferences if pref.scope == "global"]
        by_platform: dict[str, list[PreferenceCandidate]] = {}
        for pref in preferences:
            if pref.scope == "platform" and pref.platform:
                by_platform.setdefault(pref.platform, []).append(pref)

        folded = _fold_preferences(global_prefs)
        sentence_range = folded.get("sentence_length_range", _DEFAULT_SENTENCE_RANGE)
        profile.sentence_length_range = (int(sentence_range[0]), int(sentence_range[1]))
        profile.banned_phrases = folded.get("banned_phrases", [])
        profile.preferred_openings = folded.get("preferred_openings", [])
        profile.avoided_openings = folded.get("avoided_openings", [])
        profile.emoji_policy = {
            "max_per_post": _DEFAULT_MAX_EMOJI,
            **folded.get("emoji_policy", {}),
        }

        profile.platform_overrides = {
            platform: _fold_preferences(platform_prefs)
            for platform, platform_prefs in sorted(by_platform.items())
        }
        # Keyed by `PreferenceCandidate.key`, which already encodes scope and platform,
        # so global and per-platform rules coexist in one flat map.
        profile.confidence_by_rule = {pref.key: pref.confidence for pref in preferences}

        await self._knowledge.save_voice_profile(profile)
        return profile

    async def record_feedback(
        self,
        *,
        workspace_id: str,
        event_type: FeedbackEventType,
        asset_id: str | None = None,
        node_id: str | None = None,
        run_id: str | None = None,
        before_text: str | None = None,
        after_text: str | None = None,
        reason: str | None = None,
        platform: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEvent:
        """Record any user feedback, and learn from it when it carries an edit.

        `pin_as_good` is the one signal the user gives on purpose, so the rules it
        implies are marked explicit and skip straight to `stable` on promotion.
        """
        event = FeedbackEvent(
            feedback_id=new_id("fb"),
            workspace_id=workspace_id,
            asset_id=asset_id,
            node_id=node_id,
            run_id=run_id,
            event_type=event_type,
            before_text=before_text,
            after_text=after_text,
            reason=reason,
            metadata=dict(metadata or {}),
        )

        carries_edit = (
            event_type in ("edit", "pin_as_good")
            and asset_id is not None
            and before_text is not None
            and after_text is not None
        )
        if carries_edit:
            edit_event = await self.record_edit(
                workspace_id=workspace_id,
                asset_id=str(asset_id),
                before_text=str(before_text),
                after_text=str(after_text),
                platform=platform,
                reason=reason,
                explicit=event_type == "pin_as_good",
            )
            # Traceable both ways without duplicating the texts into the feedback doc.
            event.metadata["edit_event_id"] = edit_event.event_id

        await self._learning.add_feedback(event)
        return event
