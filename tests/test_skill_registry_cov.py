"""Coverage tests for skill_registry parsing, discovery, search, and formatting.

Drives the under-covered branches: frontmatter edge cases (no match, comment/
blank lines, lines without a usable colon), empty list/chain parsing, the
chain CSV fallback, discover_skills defaulting/skip/OSError paths, the
recommended_when / recommended_chain scoring branches in both search_skills and
recommend_skills, the zero-score skip, and the format_skills chain extras line.
"""
from __future__ import annotations

from pathlib import Path

from maestro_cli.skill_registry import (
    SkillEntry,
    _parse_chain_field,
    _parse_frontmatter,
    _parse_list_field,
    discover_skills,
    format_skills,
    recommend_skills,
    search_skills,
)


# --- _parse_frontmatter ---------------------------------------------------


def test_parse_frontmatter_no_match_returns_empty() -> None:
    # Text without leading --- frontmatter block -> no regex match -> {}.
    assert _parse_frontmatter("just body text, no frontmatter") == {}


def test_parse_frontmatter_skips_blank_and_comment_lines() -> None:
    # Blank line and a comment line both hit the `continue` branch.
    text = "---\n\n# a comment\nname: alpha\n---\n\nbody"
    fm = _parse_frontmatter(text)
    assert fm == {"name": "alpha"}


def test_parse_frontmatter_skips_line_without_usable_colon() -> None:
    # A line starting with ':' has colon index 0 (< 1) -> skipped; a plain
    # word with no colon also skipped. The valid line still parses.
    text = "---\n:leadingcolon\nnocolonhere\ndescription: real value\n---\n\nbody"
    fm = _parse_frontmatter(text)
    assert fm == {"description": "real value"}


# --- _parse_list_field ----------------------------------------------------


def test_parse_list_field_empty_returns_empty() -> None:
    assert _parse_list_field("   ") == []


def test_parse_list_field_bracketed_csv() -> None:
    assert _parse_list_field("[a, b, c]") == ["a", "b", "c"]


# --- _parse_chain_field ---------------------------------------------------


def test_parse_chain_field_empty_returns_empty() -> None:
    assert _parse_chain_field("") == []


def test_parse_chain_field_csv_fallback_when_no_arrow() -> None:
    # No '->' present, so it falls through to _parse_list_field (CSV).
    assert _parse_chain_field("validate, run") == ["validate", "run"]


def test_parse_chain_field_arrow_form() -> None:
    assert _parse_chain_field("validate -> run -> report") == [
        "validate",
        "run",
        "report",
    ]


# --- discover_skills ------------------------------------------------------


def _write_skill(base: Path, dir_name: str, body: str) -> Path:
    skill_dir = base / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(body, encoding="utf-8")
    return skill_file


def test_discover_skills_default_search_dir(tmp_path: Path, monkeypatch) -> None:
    # search_dirs is None -> defaults to cwd/.claude/skills. Point cwd at tmp.
    skills_root = tmp_path / ".claude" / "skills"
    _write_skill(
        skills_root,
        "make-plan",
        "---\nname: make-plan\ndescription: scaffold a plan\n---\n\nbody",
    )
    monkeypatch.chdir(tmp_path)

    skills = discover_skills()
    assert [s.name for s in skills] == ["make-plan"]
    assert skills[0].description == "scaffold a plan"


def test_discover_skills_skips_non_directory_entries(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    # A loose file directly inside the skills dir is not a skill directory.
    (skills_root / "stray.txt").write_text("ignore me", encoding="utf-8")
    _write_skill(skills_root, "real-skill", "---\nname: real-skill\n---\n\nbody")

    skills = discover_skills([skills_root])
    assert [s.name for s in skills] == ["real-skill"]


def test_discover_skills_skips_missing_skill_md(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    (skills_root / "empty-dir").mkdir(parents=True)  # no SKILL.md inside
    _write_skill(skills_root, "has-skill", "---\nname: has-skill\n---\n\nbody")

    skills = discover_skills([skills_root])
    assert [s.name for s in skills] == ["has-skill"]


def test_discover_skills_continues_on_read_error(tmp_path: Path, monkeypatch) -> None:
    # read_text raising OSError must be swallowed and the skill skipped.
    skills_root = tmp_path / "skills"
    _write_skill(skills_root, "bad-skill", "---\nname: bad-skill\n---\n\nbody")

    real_read_text = Path.read_text

    def _boom(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "SKILL.md":
            raise OSError("simulated read failure")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _boom)

    skills = discover_skills([skills_root])
    assert skills == []


def test_discover_skills_skips_non_directory_search_dir(tmp_path: Path) -> None:
    # A search dir that isn't a directory is skipped entirely.
    missing = tmp_path / "does-not-exist"
    assert discover_skills([missing]) == []


def test_discover_skills_full_metadata_and_chain(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    body = (
        "---\n"
        "name: deploy-flow\n"
        "description: deploy the app\n"
        "argument-hint: <env>\n"
        "disable-model-invocation: true\n"
        "tags: [ci, deploy]\n"
        "triggers: push, release\n"
        "recommended-when: when shipping\n"
        "recommended-chain: build -> test -> ship\n"
        "---\n\nbody"
    )
    _write_skill(skills_root, "deploy-flow", body)

    skills = discover_skills([skills_root])
    assert len(skills) == 1
    entry = skills[0]
    assert entry.disable_model_invocation is True
    assert entry.tags == ["ci", "deploy"]
    assert entry.triggers == ["push", "release"]
    assert entry.recommended_when == "when shipping"
    assert entry.recommended_chain == ["build", "test", "ship"]


# --- search_skills (recommended_when / recommended_chain branches) --------


def test_search_skills_matches_recommended_when() -> None:
    skill = SkillEntry(
        name="alpha",
        description="",
        recommended_when="useful for deployment scenarios",
    )
    results = search_skills([skill], "deployment")
    assert results == [skill]


def test_search_skills_matches_recommended_chain_step() -> None:
    skill = SkillEntry(
        name="beta",
        description="",
        recommended_chain=["bootstrap", "finalize"],
    )
    results = search_skills([skill], "finalize")
    assert results == [skill]


def test_search_skills_empty_query_returns_all() -> None:
    skill = SkillEntry(name="gamma")
    assert search_skills([skill], "") == [skill]


# --- recommend_skills (tags / recommended_when / recommended_chain) -------


def test_recommend_skills_keyword_overlap_in_tags() -> None:
    skill = SkillEntry(name="alpha", description="", tags=["security"])
    recs = recommend_skills([skill], "security")
    assert len(recs) == 1
    assert "tags" in recs[0].matched_fields
    assert recs[0].score == 1


def test_recommend_skills_keyword_overlap_in_recommended_when() -> None:
    skill = SkillEntry(
        name="beta",
        description="",
        recommended_when="apply during migration",
    )
    recs = recommend_skills([skill], "migration")
    assert len(recs) == 1
    assert "recommended_when" in recs[0].matched_fields


def test_recommend_skills_keyword_overlap_in_recommended_chain() -> None:
    skill = SkillEntry(
        name="cappa",
        description="",
        recommended_chain=["plan", "execute"],
    )
    recs = recommend_skills([skill], "execute")
    assert len(recs) == 1
    assert "recommended_chain" in recs[0].matched_fields


def test_recommend_skills_skips_zero_score() -> None:
    # No overlap anywhere -> score stays 0 -> skill is filtered out.
    skill = SkillEntry(
        name="alpha",
        description="nothing relevant here",
        tags=["unrelated"],
    )
    assert recommend_skills([skill], "zzzznomatch") == []


def test_recommend_skills_empty_query_returns_empty() -> None:
    skill = SkillEntry(name="alpha", description="something")
    assert recommend_skills([skill], "   ") == []


# --- format_skills (recommended_chain extras line) ------------------------


def test_format_skills_includes_chain_extras() -> None:
    skill = SkillEntry(
        name="deploy",
        description="deploy app",
        recommended_chain=["build", "ship"],
    )
    out = format_skills([skill])
    assert "chain: build -> ship" in out


def test_format_skills_empty_message() -> None:
    assert "No skills found" in format_skills([])
