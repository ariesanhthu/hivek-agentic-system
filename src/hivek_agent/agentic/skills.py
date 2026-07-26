"""Skill registry.

Loads the marketing SKILL.md files that live under `apps/server-ai/SKILL` and hands the
composer a *compact excerpt* of each one. Two blueprint rules shape this module:

* Whole documents never enter a prompt, so `SkillDefinition.guidance` returns only the
  sections a composer acts on and is hard-capped at a character budget.
* Nothing here calls a model. Skill selection is a lookup table and the scoring formulas
  are arithmetic, both of which code does exactly and an LLM does only approximately.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hivek_agent.config import REPO_ROOT, get_settings

logger = logging.getLogger(__name__)

SKILLS_DIR = REPO_ROOT / "apps" / "server-ai" / "SKILL"
SKILL_FILENAME = "SKILL.md"

# task -> skills, in priority order. A constant rather than a model call: the mapping is
# fixed knowledge, and asking an LLM to re-derive it would only add latency and drift.
_TASK_SKILLS: dict[str, tuple[str, ...]] = {
    "content_compose": ("marketing-psychology", "product-marketing-context"),
    "content_plan": ("marketing-ideas", "product-marketing-context"),
    "campaign_ideas": ("marketing-ideas", "marketing-psychology"),
}

# Level-2 headings worth injecting, most useful first. Everything else in these files is
# reference library or worked examples - long, and not what a composer acts on.
_GUIDANCE_SECTIONS: tuple[str, ...] = (
    "required output format",
    "scoring formula",
    "feasibility score",
    "guardrail",
    "selection rules",
    "sections to capture",
)

_SCORE_MIN = 1
_SCORE_MAX = 5
_PLFS_CAP = 15

_ELLIPSIS = " ..."
# Only honour a boundary that keeps most of the budget; otherwise we would throw away
# paid-for context just to land on a prettier cut.
_BOUNDARY_FLOOR = 0.6

_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_NUMBERING_RE = re.compile(r"^\d+[.)]\s*")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
_BLOCK_SEP = "\n\n"


# --- frontmatter -----------------------------------------------------------


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split `---` delimited frontmatter from the markdown body.

    Deliberately not a YAML parser. These files hold flat `key: value` pairs plus one
    2-space-indented `metadata:` block, which is not worth a PyYAML dependency. Anything
    it does not recognise is skipped rather than guessed at, and a file with absent or
    unterminated frontmatter is simply all body.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()

    end = -1
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            end = index
            break
    if end == -1:
        return {}, text.strip()

    meta: dict[str, Any] = {}
    block_key: str | None = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        key, sep, value = raw.strip().partition(":")
        if not sep:
            continue
        key = key.strip()
        value = _unquote(value)
        if raw[:1].isspace():
            # Indented -> belongs to the block opened by the last valueless key.
            block = meta.get(block_key or "")
            if isinstance(block, dict):
                block[key] = value
            continue
        if value:
            meta[key] = value
            block_key = None
        else:
            meta[key] = {}
            block_key = key
    return meta, "\n".join(lines[end + 1 :]).strip()


def _text(value: object, default: str = "") -> str:
    """Frontmatter values are str or (for an opened block) dict; anything else is absent."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


# --- excerpting ------------------------------------------------------------


def _compact(text: str) -> str:
    """Strip horizontal rules and blank runs - pure noise once the doc is an excerpt."""
    text = _RULE_RE.sub("", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _last_sentence_end(window: str) -> int:
    matches = list(_SENTENCE_END_RE.finditer(window))
    return matches[-1].end() if matches else -1


def _truncate_on_boundary(text: str, max_chars: int) -> str:
    """Cut to at most `max_chars`, preferring paragraph > sentence > line > word breaks.

    Marks the cut with an ellipsis so a reader (or a model) can tell the excerpt ended
    early rather than the source document ending there.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(_ELLIPSIS)
    if budget <= 0:
        return _ELLIPSIS.strip()[:max_chars]

    window = text[:budget]
    floor = int(budget * _BOUNDARY_FLOOR)
    for cut in (window.rfind("\n\n"), _last_sentence_end(window), window.rfind("\n")):
        if cut >= floor:
            return text[:cut].rstrip() + _ELLIPSIS
    # No structural break worth taking: fall back to the last word boundary. If even one
    # word will not fit, yield the marker alone - a word fragment is worse than nothing,
    # and a clipped URL or claim reads as real while being false.
    space = window.rfind(" ")
    if space <= 0:
        return _ELLIPSIS.strip()[:max_chars]
    return text[:space].rstrip() + _ELLIPSIS


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Level-2 sections as (normalised title, text including heading and subsections)."""
    matches = list(_HEADING_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        title = _NUMBERING_RE.sub("", match.group("title")).strip().lower()
        sections.append((title, body[match.start() : end].strip()))
    return sections


def _priority_sections(body: str) -> list[str]:
    """Sections named by `_GUIDANCE_SECTIONS`, emitted in that priority order."""
    sections = _split_sections(body)
    picked: list[str] = []
    seen: set[str] = set()
    for pattern in _GUIDANCE_SECTIONS:
        for title, text in sections:
            if pattern in title and title not in seen:
                seen.add(title)
                picked.append(text)
    return picked


def _lede(body: str) -> str:
    """Prose before the first level-2 heading, minus the document title."""
    match = _HEADING_RE.search(body)
    head = body[: match.start()] if match else body
    return "\n".join(line for line in head.splitlines() if not line.startswith("# ")).strip()


def _tasks_for(skill_id: str) -> list[str]:
    return [task for task, skill_ids in _TASK_SKILLS.items() if skill_id in skill_ids]


# --- models ----------------------------------------------------------------


class SkillDefinition(BaseModel):
    """One SKILL.md: its frontmatter, its body, and the tasks it serves."""

    skill_id: str
    name: str
    description: str
    risk: str = "unknown"
    source: str = ""
    version: str = "1.0.0"
    body: str
    path: str
    applies_to: list[str]

    def guidance(self, *, max_chars: int) -> str:
        """Compact excerpt for prompt injection. NEVER the whole body.

        The blueprint forbids pasting whole documents into a prompt, so this keeps the
        one-line description plus the few sections a composer acts on - output format,
        scoring formula, guardrails - and drops the reference libraries and worked
        examples. The result is always at most `max_chars` characters.
        """
        if max_chars <= 0:
            return ""
        blocks = [self.description] if self.description.strip() else []
        blocks.extend(_priority_sections(self.body) or [_lede(self.body)])
        digest = _compact(_BLOCK_SEP.join(block for block in blocks if block.strip()))
        return _truncate_on_boundary(digest, max_chars)


class ScoringRubric(BaseModel):
    """Formula metadata, so a prompt can cite the rubric without loading the document."""

    skill_id: str
    acronym: str
    formula: str
    minimum: int
    maximum: int
    dimensions: tuple[str, ...]


SKILL_SCORING: dict[str, ScoringRubric] = {
    "marketing-ideas": ScoringRubric(
        skill_id="marketing-ideas",
        acronym="MFS",
        formula="(Impact + Fit + Speed) - (Effort + Cost)",
        minimum=-7,
        maximum=13,
        dimensions=("impact", "fit", "speed", "effort", "cost"),
    ),
    "marketing-psychology": ScoringRubric(
        skill_id="marketing-psychology",
        acronym="PLFS",
        formula="(Leverage + Fit + Speed + Ethics) - Implementation Cost",
        minimum=-5,
        maximum=15,
        dimensions=("leverage", "fit", "speed", "ethics", "implementation_cost"),
    ),
}


# --- scoring ---------------------------------------------------------------


def _dimension(name: str, value: int) -> int:
    """Reject out-of-range dimensions instead of clamping them.

    Both rubrics define every dimension on 1..5, so a 0 or a 7 means the caller mis-scored
    rather than meant the boundary. Clamping would bury that in a plausible-looking total.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int in {_SCORE_MIN}..{_SCORE_MAX}, got {value!r}")
    if not _SCORE_MIN <= value <= _SCORE_MAX:
        raise ValueError(f"{name} must be in {_SCORE_MIN}..{_SCORE_MAX}, got {value}")
    return value


def marketing_feasibility_score(impact: int, fit: int, speed: int, effort: int, cost: int) -> int:
    """MFS from SKILL/marketing-ideas: `(Impact + Fit + Speed) - (Effort + Cost)`, -7..+13.

    Effort and cost are inverted by the formula, so a high score means high upside for
    little outlay. Kept in Python because the blueprint bans using a model for arithmetic:
    a planner that asks an LLM to add five integers gets a different total on a retry.
    """
    upside = _dimension("impact", impact) + _dimension("fit", fit) + _dimension("speed", speed)
    outlay = _dimension("effort", effort) + _dimension("cost", cost)
    return upside - outlay


def psychological_leverage_score(
    leverage: int,
    fit: int,
    speed: int,
    ethics: int,
    implementation_cost: int,
) -> int:
    """PLFS from SKILL/marketing-psychology: `(Leverage + Fit + Speed + Ethics) - Cost`.

    Range -5..+15. The skill's own worked example reaches 17 and then says "cap at 15", so
    the cap is applied here rather than left to each caller to remember.
    """
    raw = (
        _dimension("leverage", leverage)
        + _dimension("fit", fit)
        + _dimension("speed", speed)
        + _dimension("ethics", ethics)
    ) - _dimension("implementation_cost", implementation_cost)
    return min(raw, _PLFS_CAP)


# --- registry --------------------------------------------------------------


class SkillRegistry:
    """Reads `<skills_dir>/<skill_id>/SKILL.md` once and serves excerpts from memory."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills_dir = skills_dir or SKILLS_DIR
        self._skills: dict[str, SkillDefinition] = {}
        self._loaded = False

    def load(self) -> None:
        """Populate the registry. Idempotent - a second call is a no-op.

        A missing directory is a warning, not an error: the harness must still boot and be
        demoable when the skill pack has not been checked out.
        """
        if self._loaded:
            return
        self._skills = self._read_all()
        self._loaded = True
        logger.debug("loaded %d skills from %s", len(self._skills), self._skills_dir)

    def get(self, skill_id: str) -> SkillDefinition | None:
        self.load()
        return self._skills.get(skill_id)

    def list_skills(self) -> list[SkillDefinition]:
        self.load()
        return [self._skills[skill_id] for skill_id in sorted(self._skills)]

    def select_for_task(
        self,
        task: str,
        *,
        platform: str | None = None,
        limit: int = 2,
    ) -> list[SkillDefinition]:
        """Skills for a task, most relevant first. Deterministic - never calls a model.

        `platform` narrows the set only for skills declaring a `platform:<name>` tag; the
        bundled marketing skills are all platform-agnostic, so today it filters nothing.
        An unknown task, a non-positive limit, or a skill whose file failed to load all
        yield fewer results rather than an error.
        """
        self.load()
        if limit <= 0:
            return []
        wanted = f"platform:{platform.strip().lower()}" if platform and platform.strip() else None
        selected: list[SkillDefinition] = []
        for skill_id in _TASK_SKILLS.get(task, ()):
            skill = self._skills.get(skill_id)
            if skill is None:
                continue
            scoped = [tag for tag in skill.applies_to if tag.startswith("platform:")]
            if scoped and wanted is not None and wanted not in scoped:
                continue
            selected.append(skill)
            if len(selected) >= limit:
                break
        return selected

    def _read_all(self) -> dict[str, SkillDefinition]:
        directory = self._skills_dir
        if not directory.is_dir():
            logger.warning("skill directory missing at %s; registry is empty", directory)
            return {}
        skills: dict[str, SkillDefinition] = {}
        for child in sorted(directory.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            skill = self._read_skill(child)
            if skill is not None:
                skills[skill.skill_id] = skill
        return skills

    def _read_skill(self, directory: Path) -> SkillDefinition | None:
        path = directory / SKILL_FILENAME
        try:
            # utf-8-sig: these files are authored on Windows and may carry a BOM, which
            # would otherwise hide the opening `---` from the frontmatter parser.
            raw = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning("skipping skill %r: cannot read %s (%s)", directory.name, path, exc)
            return None

        meta, body = _parse_frontmatter(raw)
        skill_id = directory.name
        metadata = meta.get("metadata")
        version = metadata.get("version") if isinstance(metadata, dict) else None
        return SkillDefinition(
            skill_id=skill_id,
            name=_text(meta.get("name"), skill_id),
            description=_text(meta.get("description")),
            risk=_text(meta.get("risk"), "unknown"),
            source=_text(meta.get("source")),
            version=_text(version, "1.0.0"),
            body=body,
            path=str(path),
            applies_to=_tasks_for(skill_id),
        )


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    """Process-wide registry. Skill files are static, so one read per process is enough.

    Honours `SKILLS_DIR` when set: the default path is derived from the package's
    location in the monorepo, which does not survive a container build or a
    standalone install.
    """
    configured = get_settings().skills_dir.strip()
    registry = SkillRegistry(Path(configured) if configured else None)
    registry.load()
    return registry
