from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.errors import E018, PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.runners import _load_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, content: str, filename: str = "plan.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _make_plan(task: TaskSpec, source_path: Path | None = None) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test-plan",
        defaults=PlanDefaults(),
        tasks=[task],
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Matrix expansion (via load_plan)
# ---------------------------------------------------------------------------


class TestMatrixExpansion:
    def test_simple_matrix_expands_to_two_tasks(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: [ubuntu, windows]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert len(plan.tasks) == 2
        ids = {t.id for t in plan.tasks}
        assert "build.os-ubuntu" in ids
        assert "build.os-windows" in ids

    def test_two_key_matrix_expands_cartesian_product(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: test
    command: echo test
    matrix:
      os: [ubuntu, win]
      py: ["3.11", "3.12"]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert len(plan.tasks) == 4
        ids = {t.id for t in plan.tasks}
        assert "test.os-ubuntu.py-3.11" in ids
        assert "test.os-ubuntu.py-3.12" in ids
        assert "test.os-win.py-3.11" in ids
        assert "test.os-win.py-3.12" in ids

    def test_base_task_removed_after_expansion(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      env: [dev, prod]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert all(t.id != "build" for t in plan.tasks)

    def test_expanded_task_ids_use_dot_dash_format(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: [ubuntu]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].id == "build.os-ubuntu"

    def test_matrix_values_set_on_expanded_tasks(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      env: [dev, prod]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        by_id = {t.id: t for t in plan.tasks}
        assert by_id["build.env-dev"].matrix_values == {"env": "dev"}
        assert by_id["build.env-prod"].matrix_values == {"env": "prod"}

    def test_matrix_parent_set_on_expanded_tasks(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: [ubuntu, windows]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        for task in plan.tasks:
            assert task.matrix_parent == "build"

    def test_expanded_task_inherits_depends_on(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: setup
    command: echo setup
  - id: build
    command: echo build
    depends_on: [setup]
    matrix:
      os: [ubuntu, windows]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        matrix_tasks = [t for t in plan.tasks if t.matrix_parent == "build"]
        for task in matrix_tasks:
            assert "setup" in task.depends_on

    def test_expanded_task_inherits_workspace_index_exclude(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    workspace_index_exclude: [.git/**, node_modules/**]
    matrix:
      os: [ubuntu, windows]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        matrix_tasks = [t for t in plan.tasks if t.matrix_parent == "build"]
        for task in matrix_tasks:
            assert task.workspace_index_exclude == [".git/**", "node_modules/**"]

    def test_depends_on_matrix_task_expands_to_all_instances(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: [ubuntu, windows]
  - id: report
    command: echo report
    depends_on: [build]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        report = next(t for t in plan.tasks if t.id == "report")
        assert "build.os-ubuntu" in report.depends_on
        assert "build.os-windows" in report.depends_on
        assert "build" not in report.depends_on

    def test_context_from_matrix_task_expands_to_all_instances(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: [ubuntu, windows]
  - id: report
    command: echo report
    depends_on: [build]
    context_from: [build]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        report = next(t for t in plan.tasks if t.id == "report")
        assert "build.os-ubuntu" in report.context_from
        assert "build.os-windows" in report.context_from
        assert "build" not in report.context_from

    def test_yaml_float_values_coerced_to_string(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      python: [3.11, 3.12]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        versions = {t.matrix_values["python"] for t in plan.tasks}  # type: ignore[index]
        assert "3.11" in versions
        assert "3.12" in versions
        for v in versions:
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# Matrix + judge field propagation (v0.10.0 regression guard)
# ---------------------------------------------------------------------------


class TestMatrixJudgePropagation:
    def test_judge_method_and_aggregation_preserved_in_matrix_expansion(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: eval
    command: echo ok
    matrix:
      env: [staging, prod]
    judge:
      criteria: ["Output is correct"]
      method: g_eval
      aggregation: weighted_mean
      pass_threshold: 0.8
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert len(plan.tasks) == 2
        for task in plan.tasks:
            assert task.judge is not None
            assert task.judge.method == "g_eval"
            assert task.judge.aggregation == "weighted_mean"
            assert task.judge.pass_threshold == 0.8

    def test_judge_preset_preserved_in_matrix_expansion(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: audit
    command: echo ok
    matrix:
      target: [api, db]
    judge:
      preset: code_quality
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert len(plan.tasks) == 2
        for task in plan.tasks:
            assert task.judge is not None
            assert task.judge.preset == "code_quality"


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------


class TestMatrixValidation:
    def test_empty_matrix_raises_error(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix: {}
"""
        with pytest.raises(PlanValidationError, match=E018):
            load_plan(_write_plan(tmp_path, yaml))

    def test_matrix_key_with_empty_list_raises_error(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: []
"""
        with pytest.raises(PlanValidationError, match=E018):
            load_plan(_write_plan(tmp_path, yaml))

    def test_matrix_not_a_dict_raises_error(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix: [ubuntu, windows]
"""
        with pytest.raises(PlanValidationError, match=E018):
            load_plan(_write_plan(tmp_path, yaml))

    def test_matrix_key_value_not_a_list_raises_error(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: build
    command: echo build
    matrix:
      os: ubuntu
"""
        with pytest.raises(PlanValidationError, match=E018):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# Matrix template variables
# ---------------------------------------------------------------------------


class TestMatrixTemplateVariables:
    def test_matrix_vars_resolved_in_prompt(self, tmp_path: Path) -> None:
        task = TaskSpec(
            id="build.os-ubuntu",
            engine="claude",
            prompt="Build for {{ matrix.os }}",
            matrix_values={"os": "ubuntu"},
        )
        plan = _make_plan(task, source_path=tmp_path / "plan.yaml")
        (tmp_path / "plan.yaml").write_text("version: 1\nname: test\n", encoding="utf-8")

        resolved = _load_prompt(plan, task, upstream_results=None, context_synthesis="")

        assert "ubuntu" in resolved
        assert "{{ matrix.os }}" not in resolved

    def test_matrix_vars_resolved_in_prompt_multiple_keys(self, tmp_path: Path) -> None:
        task = TaskSpec(
            id="build.os-ubuntu.py-3.11",
            engine="claude",
            prompt="OS={{ matrix.os }} PY={{ matrix.py }}",
            matrix_values={"os": "ubuntu", "py": "3.11"},
        )
        (tmp_path / "plan.yaml").write_text("version: 1\nname: test\n", encoding="utf-8")
        plan = _make_plan(task, source_path=tmp_path / "plan.yaml")

        resolved = _load_prompt(plan, task, upstream_results=None, context_synthesis="")

        assert "OS=ubuntu" in resolved
        assert "PY=3.11" in resolved
