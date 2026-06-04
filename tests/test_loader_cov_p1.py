from __future__ import annotations

"""Coverage-focused tests for src/maestro_cli/loader.py validation branches.

Each test drives the real loader (or a real module-level helper) through a
specific currently-uncovered error branch by crafting plan YAML or direct
helper inputs. All assertions use substring matching against the rendered
PlanValidationError message (which is prefixed with ``[<code>]``).
"""

from pathlib import Path

import pytest

import maestro_cli.loader as loader
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import (
    _check_unknown_criteria_fields,
    _migrate_plan,
    _to_float_or_none,
    _to_pct_or_none,
    _to_workspace_assertions,
    load_plan,
)


def _write_plan(tmp_path: Path, content: str, filename: str = "plan.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _to_workspace_assertions
# ---------------------------------------------------------------------------


class TestWorkspaceAssertions:
    def test_assert_not_a_list_raises(self, tmp_path: Path) -> None:
        # task.assert must be a list -> .
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    command: echo hi
    assert: "not-a-list"
"""
        with pytest.raises(PlanValidationError, match="must be a list"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_direct_non_list_value(self) -> None:
        with pytest.raises(PlanValidationError, match="must be a list"):
            _to_workspace_assertions("oops", "tasks[0].assert")


# ---------------------------------------------------------------------------
# _check_unknown_criteria_fields ( -- known type is None)
# ---------------------------------------------------------------------------


class TestUnknownCriteriaFields:
    def test_unknown_assertion_type_returns_silently(self) -> None:
        # An assertion_type not registered in _KNOWN_CRITERIA_FIELDS returns
        # without raising -- the "unknown type already caught upstream" path.
        # No exception expected; the call simply returns None.
        result = _check_unknown_criteria_fields(
            {"type": "totally-made-up", "value": "x"},
            "totally-made-up",
            "tasks[0].judge.criteria[0]",
        )
        assert result is None


# ---------------------------------------------------------------------------
# _to_judge_spec json-schema typed-criterion branches
# ---------------------------------------------------------------------------


class TestJudgeJsonSchemaCriteria:
    def test_schema_must_be_dict(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria:
        - type: json-schema
          schema: "not-a-dict"
      pass_threshold: 0.5
"""
        with pytest.raises(PlanValidationError, match="schema must be a dict"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_schema_file_must_be_string(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria:
        - type: json-schema
          schema_file: [1, 2, 3]
      pass_threshold: 0.5
"""
        with pytest.raises(PlanValidationError, match="schema_file must be a string"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# _to_judge_spec rubric typed-criterion branches
# ---------------------------------------------------------------------------


class TestJudgeRubricCriteria:
    def test_rubric_level_not_object(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria:
        - type: rubric
          name: clarity
          levels:
            - "not-an-object"
      pass_threshold: 0.5
"""
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_rubric_level_missing_score_and_description(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria:
        - type: rubric
          name: clarity
          levels:
            - {foo: bar}
      pass_threshold: 0.5
"""
        with pytest.raises(PlanValidationError, match="must contain 'score' and 'description'"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_rubric_min_score_not_int(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria:
        - type: rubric
          name: clarity
          min_score: "high"
          levels:
            - {score: 3, description: ok}
      pass_threshold: 0.5
"""
        with pytest.raises(PlanValidationError, match="min_score must be an integer"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_rubric_weight_not_number(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria:
        - type: rubric
          name: clarity
          weight: "heavy"
          levels:
            - {score: 3, description: ok}
      pass_threshold: 0.5
"""
        with pytest.raises(PlanValidationError, match="weight must be a number"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# _to_judge_spec model + timeout branches
# ---------------------------------------------------------------------------


class TestJudgeModelAndTimeout:
    def test_judge_model_empty_string(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria: ["is it good"]
      pass_threshold: 0.5
      model: "   "
"""
        with pytest.raises(PlanValidationError, match="model must be a non-empty string"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_judge_timeout_not_int(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    judge:
      criteria: ["is it good"]
      pass_threshold: 0.5
      timeout_sec: "soon"
"""
        with pytest.raises(PlanValidationError, match="timeout_sec must be a positive integer"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# _to_float_or_none / _to_pct_or_none parse failure
# ---------------------------------------------------------------------------


class TestFloatAndPctParsing:
    def test_max_cost_usd_not_a_number(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
max_cost_usd: "lots"
tasks:
  - id: a
    command: echo hi
"""
        with pytest.raises(PlanValidationError, match="max_cost_usd must be a number"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_float_or_none_typeerror_direct(self) -> None:
        # float([]) raises TypeError -> .
        with pytest.raises(PlanValidationError, match="must be a number"):
            _to_float_or_none([], "max_cost_usd")

    def test_budget_warning_pct_not_a_number(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
budget_warning_pct: "half"
tasks:
  - id: a
    command: echo hi
"""
        with pytest.raises(PlanValidationError, match="budget_warning_pct must be a number"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_pct_or_none_typeerror_direct(self) -> None:
        # float({}) raises TypeError -> .
        with pytest.raises(PlanValidationError, match="must be a number"):
            _to_pct_or_none({}, "budget_warning_pct")


# ---------------------------------------------------------------------------
# _to_trajectory_guard branches
# ---------------------------------------------------------------------------


class TestTrajectoryGuard:
    def test_trajectory_guard_not_object(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    trajectory_guard: "nope"
"""
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_trajectory_guard_max_retries_without_progress_too_low(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    trajectory_guard:
      max_retries_without_progress: 0
"""
        with pytest.raises(
            PlanValidationError, match="max_retries_without_progress must be >= 1"
        ):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# _to_population_spec branch
# ---------------------------------------------------------------------------


class TestPopulationSpec:
    def test_population_not_object(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    population: "nope"
"""
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# _to_council_spec branches
# ---------------------------------------------------------------------------


class TestCouncilSpec:
    def test_council_not_object(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    council: "nope"
"""
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_council_participants_not_list(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    council:
      participants: "not-a-list"
"""
        with pytest.raises(PlanValidationError, match="participants must be a non-empty list"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_council_participant_not_object(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    council:
      participants:
        - "just-a-string"
        - {engine: claude, model: sonnet}
"""
        with pytest.raises(PlanValidationError, match=r"participants\[0\] must be an object"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_council_invalid_topology(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    council:
      participants:
        - {engine: claude, model: sonnet, role: a}
        - {engine: claude, model: haiku, role: b}
      topology: ring
"""
        with pytest.raises(PlanValidationError, match="topology must be one of"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_council_consensus_threshold_out_of_range(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    council:
      participants:
        - {engine: claude, model: sonnet, role: a}
        - {engine: claude, model: haiku, role: b}
      consensus_threshold: 1.5
"""
        with pytest.raises(PlanValidationError, match="consensus_threshold must be"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# Matrix second-pass context_from append
# ---------------------------------------------------------------------------


class TestMatrixContextFromRewrite:
    def test_matrix_context_from_non_matrix_upstream(self, tmp_path: Path) -> None:
        # The matrix task references a NON-matrix upstream via context_from.
        # During the second-pass rewrite, that id is not in matrix_expansions,
        # so it is appended as-is .
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    command: echo a
  - id: m
    command: echo m
    depends_on: [a]
    context_from: [a]
    matrix:
      os: [ubuntu, windows]
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        children = [t for t in plan.tasks if t.matrix_parent == "m"]
        assert len(children) == 2
        for child in children:
            assert child.context_from == ["a"]


# ---------------------------------------------------------------------------
# _migrate_plan fallthrough return
# ---------------------------------------------------------------------------


class TestMigratePlanFallthrough:
    def test_migrate_returns_raw_when_supported_older_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drive the fallthrough return at the end of _migrate_plan: a version
        # below the current one that is still in the supported set skips both
        # the "too new" raise and the "unsupported" raise and returns raw.
        monkeypatch.setattr(loader, "_CURRENT_SCHEMA_VERSION", 1)
        monkeypatch.setattr(loader, "_SUPPORTED_SCHEMA_VERSIONS", {0, 1})
        raw: dict[str, object] = {"version": 0, "name": "x", "tasks": []}
        result = _migrate_plan(raw, 0)
        assert result is raw


# ---------------------------------------------------------------------------
# _resolve_imports: imports None coercion
# ---------------------------------------------------------------------------


class TestImportsNone:
    def test_imports_explicit_null(self, tmp_path: Path) -> None:
        # imports: null -> coerced to [] -> plan loads fine.
        yaml = """\
version: 1
name: test
imports: null
tasks:
  - id: a
    command: echo hi
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert [t.id for t in plan.tasks] == ["a"]


# ---------------------------------------------------------------------------
# _resolve_imports: per-import error branches
#
# ---------------------------------------------------------------------------


def _write_main_with_import(tmp_path: Path, frag_name: str) -> Path:
    main_yaml = f"""\
version: 1
name: main
imports:
  - path: {frag_name}
    prefix: lib
tasks:
  - id: top
    command: echo top
"""
    return _write_plan(tmp_path, main_yaml, "main.yaml")


class TestImportErrorBranches:
    def test_overrides_none_coerced(self, tmp_path: Path) -> None:
        # overrides: null -> coerced to {} ; import resolves cleanly.
        _write_plan(
            tmp_path,
            """\
tasks:
  - id: work
    command: echo work
""",
            "frag.yaml",
        )
        main_yaml = """\
version: 1
name: main
imports:
  - path: frag.yaml
    prefix: lib
    overrides: null
tasks:
  - id: top
    command: echo top
"""
        plan = load_plan(_write_plan(tmp_path, main_yaml, "main.yaml"))
        ids = {t.id for t in plan.tasks}
        assert "lib/work" in ids

    def test_imported_task_not_object(self, tmp_path: Path) -> None:
        _write_plan(
            tmp_path,
            """\
tasks:
  - "just-a-string"
""",
            "frag.yaml",
        )
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(_write_main_with_import(tmp_path, "frag.yaml"))

    def test_imported_task_empty_id(self, tmp_path: Path) -> None:
        _write_plan(
            tmp_path,
            """\
tasks:
  - id: "   "
    command: echo x
""",
            "frag.yaml",
        )
        with pytest.raises(PlanValidationError, match="has empty 'id'"):
            load_plan(_write_main_with_import(tmp_path, "frag.yaml"))

    def test_imported_task_invalid_depends_on(self, tmp_path: Path) -> None:
        _write_plan(
            tmp_path,
            """\
tasks:
  - id: work
    command: echo work
    depends_on: {bad: mapping}
""",
            "frag.yaml",
        )
        with pytest.raises(PlanValidationError, match="invalid depends_on"):
            load_plan(_write_main_with_import(tmp_path, "frag.yaml"))

    def test_imported_task_invalid_context_from(self, tmp_path: Path) -> None:
        _write_plan(
            tmp_path,
            """\
tasks:
  - id: work
    command: echo work
    context_from: {bad: mapping}
""",
            "frag.yaml",
        )
        with pytest.raises(PlanValidationError, match="invalid context_from"):
            load_plan(_write_main_with_import(tmp_path, "frag.yaml"))


# ---------------------------------------------------------------------------
# output_redact wrong type
# ---------------------------------------------------------------------------


class TestOutputRedact:
    def test_output_redact_wrong_type(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: do it
    output_redact: 12345
"""
        with pytest.raises(
            PlanValidationError, match="output_redact must be a list of regex patterns"
        ):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# pre_command list with non-string entries
# ---------------------------------------------------------------------------


class TestPreCommandList:
    def test_pre_command_list_non_string_entries(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test
tasks:
  - id: a
    command: echo hi
    pre_command: ["echo", 123]
"""
        with pytest.raises(
            PlanValidationError, match="pre_command list must contain only strings"
        ):
            load_plan(_write_plan(tmp_path, yaml))
