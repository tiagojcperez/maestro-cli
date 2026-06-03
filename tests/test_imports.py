from __future__ import annotations

import textwrap

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAIN_HEADER = textwrap.dedent("""\
    version: 1
    name: test-plan
""")

_FRAGMENT_TASK = textwrap.dedent("""\
    tasks:
      - id: step-a
        command: echo a
      - id: step-b
        command: echo b
        depends_on: [step-a]
""")


def _write_plan(tmp_path, content: str, name: str = "plan.yaml"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _write_fragment(tmp_path, content: str, name: str = "fragment.yaml"):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Basic import loading
# ---------------------------------------------------------------------------


class TestBasicImport:
    def test_simple_import(self, tmp_path) -> None:
        """A single import loads tasks into the plan without errors."""
        _write_fragment(tmp_path, _FRAGMENT_TASK)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: lib
            tasks:
              - id: main-task
                command: echo main
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        ids = [t.id for t in plan.tasks]
        assert "lib/step-a" in ids
        assert "lib/step-b" in ids
        assert "main-task" in ids

    def test_import_prefixes_ids(self, tmp_path) -> None:
        """Imported task IDs are rewritten to prefix/id format."""
        _write_fragment(tmp_path, _FRAGMENT_TASK)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: tools
            tasks:
              - id: done
                command: echo done
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        ids = {t.id for t in plan.tasks}
        assert "tools/step-a" in ids
        assert "tools/step-b" in ids
        # original IDs must not appear
        assert "step-a" not in ids
        assert "step-b" not in ids

    def test_import_prefixes_depends_on(self, tmp_path) -> None:
        """Internal depends_on references inside a fragment are prefixed."""
        _write_fragment(tmp_path, _FRAGMENT_TASK)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: lib
            tasks:
              - id: final
                command: echo final
                depends_on: [lib/step-b]
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        step_b = next(t for t in plan.tasks if t.id == "lib/step-b")
        assert step_b.depends_on == ["lib/step-a"]

    def test_import_prefixes_context_from(self, tmp_path) -> None:
        """context_from references inside a fragment are prefixed."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: producer
                command: echo data
              - id: consumer
                command: echo use
                depends_on: [producer]
                context_from: [producer]
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: mod
            tasks:
              - id: done
                command: echo done
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        consumer = next(t for t in plan.tasks if t.id == "mod/consumer")
        assert consumer.context_from == ["mod/producer"]

    def test_import_preserves_task_fields(self, tmp_path) -> None:
        """Non-ID fields (engine, model, prompt) are preserved after import."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: review
                engine: claude
                model: haiku
                prompt: Review the code
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: qa
            tasks:
              - id: done
                command: echo done
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        review = next(t for t in plan.tasks if t.id == "qa/review")
        assert review.engine == "claude"
        assert review.model == "haiku"
        assert review.prompt == "Review the code"


# ---------------------------------------------------------------------------
# Multiple imports
# ---------------------------------------------------------------------------


class TestMultipleImports:
    def test_multiple_imports(self, tmp_path) -> None:
        """Two imports with different prefixes both load cleanly."""
        frag_a = textwrap.dedent("""\
            tasks:
              - id: do-it
                command: echo a
        """)
        frag_b = textwrap.dedent("""\
            tasks:
              - id: do-it
                command: echo b
        """)
        _write_fragment(tmp_path, frag_a, "frag_a.yaml")
        _write_fragment(tmp_path, frag_b, "frag_b.yaml")
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: frag_a.yaml
                prefix: alpha
              - path: frag_b.yaml
                prefix: beta
            tasks:
              - id: main
                command: echo main
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        ids = {t.id for t in plan.tasks}
        assert "alpha/do-it" in ids
        assert "beta/do-it" in ids

    def test_import_order(self, tmp_path) -> None:
        """Imported tasks appear before plan-defined tasks in the task list."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: first
                command: echo first
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: lib
            tasks:
              - id: last
                command: echo last
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        ids = [t.id for t in plan.tasks]
        assert ids.index("lib/first") < ids.index("last")


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


class TestImportOverrides:
    def test_import_overrides_env(self, tmp_path) -> None:
        """Override env is merged with the imported task's existing env."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: worker
                command: echo work
                env:
                  BASE_VAR: base_value
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: svc
                overrides:
                  env:
                    EXTRA_VAR: extra_value
            tasks:
              - id: done
                command: echo done
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        worker = next(t for t in plan.tasks if t.id == "svc/worker")
        assert worker.env.get("BASE_VAR") == "base_value"
        assert worker.env.get("EXTRA_VAR") == "extra_value"

    def test_import_overrides_scalar(self, tmp_path) -> None:
        """Scalar overrides replace the imported task's field value."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: runner
                command: echo original
                timeout_sec: 30
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: svc
                overrides:
                  timeout_sec: 120
            tasks:
              - id: done
                command: echo done
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        runner = next(t for t in plan.tasks if t.id == "svc/runner")
        assert runner.timeout_sec == 120

    def test_import_no_overrides(self, tmp_path) -> None:
        """Import without overrides does not alter any task fields."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: task-x
                command: echo x
                timeout_sec: 60
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: lib
            tasks:
              - id: done
                command: echo done
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        task_x = next(t for t in plan.tasks if t.id == "lib/task-x")
        assert task_x.timeout_sec == 60


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------


class TestCrossReferences:
    def test_plan_task_depends_on_imported(self, tmp_path) -> None:
        """A plan-level task can declare depends_on for an imported task ID."""
        _write_fragment(tmp_path, _FRAGMENT_TASK)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: lib
            tasks:
              - id: final
                command: echo final
                depends_on: [lib/step-b]
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        final = next(t for t in plan.tasks if t.id == "final")
        assert "lib/step-b" in final.depends_on

    def test_imported_task_in_dag(self, tmp_path) -> None:
        """A plan with imported tasks passes DAG cycle validation."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: a
                command: echo a
              - id: b
                command: echo b
                depends_on: [a]
              - id: c
                command: echo c
                depends_on: [b]
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: chain
            tasks:
              - id: done
                command: echo done
                depends_on: [chain/c]
        """)
        # Must not raise
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        assert any(t.id == "chain/c" for t in plan.tasks)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_circular_import_E025(self, tmp_path) -> None:
        """A→B→A circular import raises E025."""
        frag_b = tmp_path / "frag_b.yaml"
        frag_a = tmp_path / "frag_a.yaml"
        # frag_b imports frag_a
        frag_b.write_text(
            textwrap.dedent("""\
                imports:
                  - path: frag_a.yaml
                    prefix: inner
                tasks:
                  - id: b-task
                    command: echo b
            """),
            encoding="utf-8",
        )
        # frag_a imports frag_b
        frag_a.write_text(
            textwrap.dedent("""\
                imports:
                  - path: frag_b.yaml
                    prefix: loop
                tasks:
                  - id: a-task
                    command: echo a
            """),
            encoding="utf-8",
        )
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: frag_a.yaml
                prefix: start
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E025"):
            load_plan(_write_plan(tmp_path, plan_yaml))

    def test_missing_import_file_E026(self, tmp_path) -> None:
        """A missing import file path raises E026."""
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: nonexistent.yaml
                prefix: ghost
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(_write_plan(tmp_path, plan_yaml))

    def test_duplicate_prefix_E027(self, tmp_path) -> None:
        """Using the same prefix twice in one plan raises E027."""
        _write_fragment(tmp_path, _FRAGMENT_TASK, "frag1.yaml")
        _write_fragment(tmp_path, _FRAGMENT_TASK, "frag2.yaml")
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: frag1.yaml
                prefix: lib
              - path: frag2.yaml
                prefix: lib
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E027"):
            load_plan(_write_plan(tmp_path, plan_yaml))

    def test_invalid_prefix_E028(self, tmp_path) -> None:
        """An uppercase-containing prefix raises E028."""
        _write_fragment(tmp_path, _FRAGMENT_TASK)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: BadPrefix
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E028"):
            load_plan(_write_plan(tmp_path, plan_yaml))

    def test_import_missing_path_field(self, tmp_path) -> None:
        """An import entry without 'path' raises E026."""
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - prefix: lib
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(_write_plan(tmp_path, plan_yaml))

    def test_import_missing_prefix_field(self, tmp_path) -> None:
        """An import entry without 'prefix' raises E026."""
        _write_fragment(tmp_path, _FRAGMENT_TASK)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(_write_plan(tmp_path, plan_yaml))

    def test_import_max_depth(self, tmp_path) -> None:
        """Nesting imports beyond _IMPORT_MAX_DEPTH (5) raises E025.

        The depth check fires only when a fragment at depth > 5 *also* has
        imports (the function returns early for import-free fragments).
        Chain: plan(0) → f1(1) → f2(2) → f3(3) → f4(4) → f5(5) → f6(6, has import) → BOOM.
        """
        # f7 is the leaf — no imports
        deepest = textwrap.dedent("""\
            tasks:
              - id: deep-task
                command: echo deep
        """)
        (tmp_path / "f7.yaml").write_text(deepest, encoding="utf-8")

        # f6 imports f7 — when _resolve_imports is called for f6 at depth=6
        # it won't return early (f6 has imports), so depth check 6 > 5 fires.
        for level in range(6, 0, -1):
            child = f"f{level + 1}.yaml"
            content = textwrap.dedent(f"""\
                imports:
                  - path: {child}
                    prefix: lvl{level}
                tasks:
                  - id: t{level}
                    command: echo {level}
            """)
            (tmp_path / f"f{level}.yaml").write_text(content, encoding="utf-8")

        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: f1.yaml
                prefix: root
            tasks:
              - id: main
                command: echo main
        """)
        with pytest.raises(PlanValidationError, match="E025"):
            load_plan(_write_plan(tmp_path, plan_yaml))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_import_empty_tasks(self, tmp_path) -> None:
        """A fragment with an empty tasks list loads without error (adds no tasks)."""
        _write_fragment(tmp_path, "tasks: []\n")
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: empty
            tasks:
              - id: main
                command: echo main
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        ids = [t.id for t in plan.tasks]
        assert ids == ["main"]

    def test_import_wildcard_context_from(self, tmp_path) -> None:
        """A '*' in context_from is not prefixed and passes through unchanged."""
        fragment = textwrap.dedent("""\
            tasks:
              - id: aggregator
                command: echo agg
                context_from: ["*"]
        """)
        _write_fragment(tmp_path, fragment)
        plan_yaml = _MAIN_HEADER + textwrap.dedent("""\
            imports:
              - path: fragment.yaml
                prefix: lib
            tasks:
              - id: main
                command: echo main
        """)
        plan = load_plan(_write_plan(tmp_path, plan_yaml))
        aggregator = next(t for t in plan.tasks if t.id == "lib/aggregator")
        assert aggregator.context_from == ["*"]
