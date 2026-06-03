"""Skill discovery and registry for Maestro CLI.

Scans ``.claude/skills/*/SKILL.md`` directories for skill definitions,
parses YAML frontmatter, and exposes a searchable registry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillEntry:
    """A discovered skill with parsed metadata."""

    name: str
    description: str = ""
    argument_hint: str = ""
    disable_model_invocation: bool = False
    path: Path = field(default_factory=lambda: Path("."))
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    recommended_when: str = ""
    recommended_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "argument_hint": self.argument_hint,
            "disable_model_invocation": self.disable_model_invocation,
            "path": str(self.path),
            "tags": self.tags,
            "triggers": self.triggers,
            "recommended_when": self.recommended_when,
            "recommended_chain": self.recommended_chain,
        }


@dataclass
class SkillRecommendation:
    """A scored recommendation for a user query."""

    skill: SkillEntry
    score: int
    matched_triggers: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.to_dict(),
            "score": self.score,
            "matched_triggers": self.matched_triggers,
            "matched_fields": self.matched_fields,
            "rationale": self.rationale,
        }


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./-]*", re.IGNORECASE)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract YAML frontmatter from a SKILL.md file (simple key: value)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    result: dict[str, str] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        colon = line.find(":")
        if colon < 1:
            continue
        key = line[:colon].strip()
        value = line[colon + 1:].strip().strip('"').strip("'")
        result[key] = value
    return result


def _parse_list_field(raw: str) -> list[str]:
    """Parse a frontmatter list encoded as CSV or `[a, b]`."""
    value = raw.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    items: list[str] = []
    for part in value.split(","):
        item = part.strip().strip('"').strip("'")
        if item:
            items.append(item)
    return items


def _parse_chain_field(raw: str) -> list[str]:
    """Parse a recommended chain encoded as `a -> b` or CSV."""
    value = raw.strip()
    if not value:
        return []
    if "->" in value:
        return [
            item.strip()
            for item in value.split("->")
            if item.strip()
        ]
    return _parse_list_field(value)


def _query_keywords(query: str) -> list[str]:
    """Extract normalized query keywords for scoring."""
    return [match.group(0).lower() for match in _QUERY_TOKEN_RE.finditer(query)]


def discover_skills(
    search_dirs: list[Path] | None = None,
) -> list[SkillEntry]:
    """Discover skills from ``.claude/skills/*/SKILL.md`` directories.

    If *search_dirs* is ``None``, searches the current working directory's
    ``.claude/skills/`` subdirectory.
    """
    if search_dirs is None:
        search_dirs = [Path.cwd() / ".claude" / "skills"]

    skills: list[SkillEntry] = []
    seen_names: set[str] = set()

    for skills_dir in search_dirs:
        if not skills_dir.is_dir():
            continue
        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue

            fm = _parse_frontmatter(text)
            name = fm.get("name", skill_dir.name)
            if name in seen_names:
                continue
            seen_names.add(name)

            tags_raw = fm.get("tags", "")
            tags = _parse_list_field(tags_raw) if tags_raw else []
            triggers_raw = fm.get("triggers", "")
            recommended_chain_raw = fm.get("recommended-chain", "")

            skills.append(SkillEntry(
                name=name,
                description=fm.get("description", ""),
                argument_hint=fm.get("argument-hint", ""),
                disable_model_invocation=fm.get("disable-model-invocation", "").lower() == "true",
                path=skill_file,
                tags=tags,
                triggers=_parse_list_field(triggers_raw) if triggers_raw else [],
                recommended_when=fm.get("recommended-when", ""),
                recommended_chain=_parse_chain_field(recommended_chain_raw) if recommended_chain_raw else [],
            ))

    return skills


def search_skills(
    skills: list[SkillEntry],
    query: str,
) -> list[SkillEntry]:
    """Filter skills by keyword match on name, description, or tags."""
    if not query:
        return list(skills)

    keywords = _query_keywords(query)
    scored: list[tuple[int, SkillEntry]] = []

    for skill in skills:
        score = 0
        for kw in keywords:
            if kw in skill.name.lower():
                score += 3
            elif any(kw in trigger.lower() for trigger in skill.triggers):
                score += 2
            elif kw in skill.description.lower():
                score += 2
            elif any(kw in t.lower() for t in skill.tags):
                score += 1
            elif kw in skill.recommended_when.lower():
                score += 1
            elif any(kw in step.lower() for step in skill.recommended_chain):
                score += 1
        if score > 0:
            scored.append((score, skill))

    scored.sort(key=lambda t: (-t[0], t[1].name))
    return [s for _, s in scored]


def recommend_skills(
    skills: list[SkillEntry],
    query: str,
) -> list[SkillRecommendation]:
    """Recommend skills for a query using explicit triggers and keyword overlap."""
    if not query.strip():
        return []

    query_lower = " ".join(query.lower().split())
    keywords = _query_keywords(query)
    recommendations: list[SkillRecommendation] = []

    for skill in skills:
        score = 0
        matched_triggers: list[str] = []
        matched_fields: list[str] = []

        for trigger in skill.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower and trigger_lower in query_lower:
                matched_triggers.append(trigger)
                score += 5
        if matched_triggers:
            matched_fields.append("triggers")

        for kw in keywords:
            if kw in skill.name.lower():
                score += 3
                if "name" not in matched_fields:
                    matched_fields.append("name")
            elif kw in skill.description.lower():
                score += 2
                if "description" not in matched_fields:
                    matched_fields.append("description")
            elif any(kw in tag.lower() for tag in skill.tags):
                score += 1
                if "tags" not in matched_fields:
                    matched_fields.append("tags")
            elif kw in skill.recommended_when.lower():
                score += 1
                if "recommended_when" not in matched_fields:
                    matched_fields.append("recommended_when")
            elif any(kw in step.lower() for step in skill.recommended_chain):
                score += 1
                if "recommended_chain" not in matched_fields:
                    matched_fields.append("recommended_chain")

        if score <= 0:
            continue

        rationale_parts: list[str] = []
        if matched_triggers:
            rationale_parts.append(f"matched trigger(s): {', '.join(matched_triggers)}")
        non_trigger_fields = [field for field in matched_fields if field != "triggers"]
        if non_trigger_fields:
            rationale_parts.append(
                "keyword overlap in " + ", ".join(non_trigger_fields)
            )
        recommendations.append(
            SkillRecommendation(
                skill=skill,
                score=score,
                matched_triggers=matched_triggers,
                matched_fields=matched_fields,
                rationale="; ".join(rationale_parts) if rationale_parts else "keyword overlap",
            )
        )

    recommendations.sort(
        key=lambda item: (-item.score, -len(item.matched_triggers), item.skill.name)
    )
    return recommendations


def format_skills(skills: list[SkillEntry]) -> str:
    """Format skill list for human-readable output."""
    if not skills:
        return "[maestro] No skills found."

    lines = ["[maestro] Available skills:", ""]
    max_name = max(len(s.name) for s in skills) + 2
    for s in skills:
        hint = f"  ({s.argument_hint})" if s.argument_hint else ""
        extras: list[str] = []
        if s.triggers:
            extras.append(f"triggers: {', '.join(s.triggers[:4])}")
        if s.recommended_chain:
            extras.append(f"chain: {' -> '.join(s.recommended_chain)}")
        suffix = f" [{' | '.join(extras)}]" if extras else ""
        lines.append(f"  {s.name:<{max_name}} {s.description}{hint}{suffix}")
    return "\n".join(lines)


def format_skills_json(skills: list[SkillEntry]) -> str:
    """Format skill list as JSON Lines."""
    import json
    return "\n".join(
        json.dumps(s.to_dict(), ensure_ascii=True)
        for s in skills
    )


def format_skill_recommendations(recommendations: list[SkillRecommendation]) -> str:
    """Format recommendations for human-readable CLI output."""
    if not recommendations:
        return "[maestro] No skill recommendations found."

    lines = ["[maestro] Recommended skills:", ""]
    max_name = max(len(item.skill.name) for item in recommendations) + 2
    for item in recommendations:
        skill = item.skill
        hint = f"  ({skill.argument_hint})" if skill.argument_hint else ""
        lines.append(
            f"  {skill.name:<{max_name}} score={item.score:<2} {skill.description}{hint}"
        )
        if item.rationale:
            lines.append(f"    why: {item.rationale}")
        if skill.recommended_when:
            lines.append(f"    when: {skill.recommended_when}")
        if skill.recommended_chain:
            lines.append(f"    chain: {' -> '.join(skill.recommended_chain)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_skill_recommendations_json(recommendations: list[SkillRecommendation]) -> str:
    """Format recommendations as JSON Lines."""
    import json

    return "\n".join(
        json.dumps(item.to_dict(), ensure_ascii=True)
        for item in recommendations
    )
