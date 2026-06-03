from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.scheduler import _select_tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _make_task(
    task_id: str,
    tags: list[str] | None = None,
    depends_on: list[str] | None = None,
    command: str = "echo ok",
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        description=f"task {task_id}",
        tags=tags or [],
        depends_on=depends_on or [],
        command=command,
    )


def _make_plan(tasks: list[TaskSpec]) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test-plan",
        defaults=PlanDefaults(),
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# Loader: tags parsing
# ---------------------------------------------------------------------------


class TestTagsParsing:
    def test_tags_parsed_from_yaml(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo ok
    tags: [unit, fast]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].tags == ["unit", "fast"]

    def test_tags_empty_default(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo ok
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].tags == []

    def test_tags_string_coerced_to_list(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo ok
    tags: unit
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].tags == ["unit"]

    def test_tags_whitespace_triggers_warning(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo ok
    tags: ["slow test"]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert any("whitespace" in w.lower() or "slow test" in w for w in plan.validation_warnings)

    def test_tags_in_matrix_expansion(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo ok
    tags: [ci, unit]
    matrix:
      env: [dev, prod]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert len(plan.tasks) == 2
        for task in plan.tasks:
            assert task.tags == ["ci", "unit"]

    def test_tags_multiple_tasks(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    command: echo a
    tags: [unit]
  - id: b
    command: echo b
    tags: [integration]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].tags == ["unit"]
        assert plan.tasks[1].tags == ["integration"]


# ---------------------------------------------------------------------------
# Scheduler: _select_tasks tag filtering
# ---------------------------------------------------------------------------


class TestSelectTasksTagFiltering:
    def test_filter_by_tags_include(self) -> None:
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["integration"]),
            _make_task("c", tags=["unit", "fast"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags={"unit"})
        ids = {t.id for t in result}
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_filter_by_tags_exclude(self) -> None:
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["slow"]),
            _make_task("c", tags=["unit"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, skip_tags={"slow"})
        ids = {t.id for t in result}
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_filter_by_tags_or_logic(self) -> None:
        """Multiple tags in --tags → OR logic: task with ANY tag is included."""
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["integration"]),
            _make_task("c", tags=["slow"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags={"unit", "integration"})
        ids = {t.id for t in result}
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    def test_filter_tags_and_only_combined(self) -> None:
        """--tags and --only combined: only the intersection of both filters."""
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["unit"]),
            _make_task("c", tags=["unit"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only={"a", "b"}, skip=None, tags={"unit"})
        ids = {t.id for t in result}
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    def test_filter_tags_preserves_dependencies(self) -> None:
        """Tagged task's dependency is auto-included even if it has no matching tag."""
        tasks = [
            _make_task("setup", tags=[]),
            _make_task("test", tags=["unit"], depends_on=["setup"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags={"unit"})
        ids = {t.id for t in result}
        assert "test" in ids
        assert "setup" in ids

    def test_filter_tags_transitive_dependencies(self) -> None:
        """Multi-hop transitive deps are all included."""
        tasks = [
            _make_task("root", tags=[]),
            _make_task("mid", tags=[], depends_on=["root"]),
            _make_task("leaf", tags=["ci"], depends_on=["mid"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags={"ci"})
        ids = {t.id for t in result}
        assert "leaf" in ids
        assert "mid" in ids
        assert "root" in ids

    def test_no_tags_returns_all(self) -> None:
        """No tag filter → all tasks returned."""
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["slow"]),
            _make_task("c"),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags=None, skip_tags=None)
        assert {t.id for t in result} == {"a", "b", "c"}

    def test_skip_tags_removes_untagged_tasks_untouched(self) -> None:
        """--skip-tags only removes tasks that HAVE the skipped tag."""
        tasks = [
            _make_task("a", tags=["slow"]),
            _make_task("b"),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, skip_tags={"slow"})
        ids = {t.id for t in result}
        assert "b" in ids
        assert "a" not in ids

    def test_tags_empty_match_returns_no_tasks(self) -> None:
        """Tag that matches nothing → empty selection (no transitive deps either)."""
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["slow"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags={"nonexistent"})
        assert result == []

    def test_skip_tags_combined_with_include_tags(self) -> None:
        """--tags and --skip-tags can be combined: include tag-a but exclude tag-b."""
        tasks = [
            _make_task("a", tags=["unit"]),
            _make_task("b", tags=["unit", "slow"]),
            _make_task("c", tags=["integration"]),
        ]
        plan = _make_plan(tasks)
        result = _select_tasks(plan, only=None, skip=None, tags={"unit"}, skip_tags={"slow"})
        ids = {t.id for t in result}
        assert "a" in ids
        assert "b" not in ids
        assert "c" not in ids


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestTagsSerialization:
    def test_tags_in_to_dict(self) -> None:
        task = _make_task("a", tags=["unit", "fast"])
        d = task.to_dict()
        assert d["tags"] == ["unit", "fast"]

    def test_tags_empty_in_to_dict(self) -> None:
        task = _make_task("a")
        d = task.to_dict()
        assert d["tags"] == []

    def test_tags_roundtrip_yaml(self, tmp_path: Path) -> None:
        """YAML → load → to_dict preserves tags exactly."""
        yaml = """\
version: 1
name: roundtrip
tasks:
  - id: task-a
    command: echo ok
    tags: [alpha, beta, gamma]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        task_dict = plan.tasks[0].to_dict()
        assert task_dict["tags"] == ["alpha", "beta", "gamma"]

    def test_tags_roundtrip_matrix(self, tmp_path: Path) -> None:
        """Matrix expansion → all expanded tasks preserve tags in to_dict."""
        yaml = """\
version: 1
name: matrix-tags
tasks:
  - id: build
    command: echo ok
    tags: [ci]
    matrix:
      env: [staging, production]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        for task in plan.tasks:
            assert task.to_dict()["tags"] == ["ci"]
