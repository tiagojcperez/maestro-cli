"""Tests for RLM-inspired context enhancement features.

Feature 3: Deterministic structured context extraction (zero LLM cost).
Feature 1: context_mode: summarized (haiku LLM call per upstream).
Feature 2: context_mode: map_reduce (map + reduce via haiku).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro_cli.errors import E021, PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    CONTEXT_MODES,
    PlanDefaults,
    PlanSpec,
    StructuredContext,
    TaskResult,
    TaskSpec,
)
from maestro_cli.runners import (
    _load_prompt,
    _run_map_reduce,
    _run_summarization,
    build_command,
)
from maestro_cli.utils import (
    build_reduce_prompt,
    build_summarization_prompt,
    extract_structured_context,
    format_structured_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(UTC)


def _make_result(
    task_id: str = "upstream",
    status: str = "success",
    exit_code: int = 0,
    stdout_tail: str = "",
    log_path: Path | None = None,
    structured: StructuredContext | None = None,
    cost_usd: float | None = None,
) -> TaskResult:
    lp = log_path or Path("/tmp/fake.log")
    return TaskResult(
        task_id=task_id,
        status=status,
        exit_code=exit_code,
        started_at=_NOW,
        finished_at=_NOW,
        duration_sec=1.5,
        command="echo test",
        log_path=lp,
        result_path=lp.with_suffix(".result.json"),
        message="ok",
        stdout_tail=stdout_tail,
        cost_usd=cost_usd,
        structured_context=structured,
    )


def _make_structured(
    task_id: str = "upstream",
    files: list[str] | None = None,
    errors: list[str] | None = None,
    summary: str = "",
) -> StructuredContext:
    return StructuredContext(
        task_id=task_id,
        status="success",
        exit_code=0,
        duration_sec=1.5,
        files_changed=files or [],
        errors=errors or [],
        summary=summary,
    )


def _write_plan_yaml(tmp_path: Path, tasks_yaml: str) -> Path:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(
        f"version: 1\nname: test-plan\ntasks:\n{tasks_yaml}",
        encoding="utf-8",
    )
    return plan_yaml


# ===========================================================================
# Feature 3: Structured context extraction (deterministic)
# ===========================================================================


class TestExtractStructuredContext:
    """Tests for extract_structured_context() in utils.py."""

    def test_empty_log_produces_empty_context(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert ctx.task_id == "t1"
        assert ctx.files_changed == []
        assert ctx.errors == []
        assert ctx.warnings == []
        assert ctx.result_text == ""

    def test_extracts_files_from_git_status(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text(
            " M src/main.py\n A tests/test_new.py\n?? untracked.txt\n",
            encoding="utf-8",
        )
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert "src/main.py" in ctx.files_changed
        assert "tests/test_new.py" in ctx.files_changed
        assert "untracked.txt" in ctx.files_changed

    def test_extracts_files_from_git_diff(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text(
            "--- a/old_file.py\n+++ b/new_file.py\n",
            encoding="utf-8",
        )
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert "old_file.py" in ctx.files_changed
        assert "new_file.py" in ctx.files_changed

    def test_extracts_result_text_from_json(self, tmp_path: Path) -> None:
        result_json = json.dumps({"type": "result", "result": "All tests passed"})
        log = tmp_path / "task.log"
        log.write_text(f"some output\n{result_json}\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert ctx.result_text == "All tests passed"

    def test_extracts_errors_from_stderr(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text(
            "[stderr] Error: module not found\n[stderr] warning: deprecated\n",
            encoding="utf-8",
        )
        ctx = extract_structured_context(log, "t1", "failed", 1, 1.0, None)
        assert any("module not found" in e for e in ctx.errors)
        assert any("deprecated" in w for w in ctx.warnings)

    def test_extracts_errors_from_error_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text(
            "Traceback (most recent call last):\nsome code line\n",
            encoding="utf-8",
        )
        ctx = extract_structured_context(log, "t1", "failed", 1, 1.0, None)
        assert any("Traceback" in e for e in ctx.errors)

    def test_deduplicates_files(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text(
            " M src/main.py\n M src/main.py\n+++ b/src/main.py\n",
            encoding="utf-8",
        )
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert ctx.files_changed.count("src/main.py") == 1

    def test_caps_file_list(self, tmp_path: Path) -> None:
        lines = [f" M file_{i}.py" for i in range(200)]
        log = tmp_path / "task.log"
        log.write_text("\n".join(lines), encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert len(ctx.files_changed) <= 100

    def test_missing_log_returns_empty_context(self, tmp_path: Path) -> None:
        log = tmp_path / "nonexistent.log"
        ctx = extract_structured_context(log, "t1", "failed", 1, 1.0, None)
        assert ctx.files_changed == []
        assert ctx.errors == []

    def test_result_text_truncated(self, tmp_path: Path) -> None:
        long_result = "x" * 1000
        result_json = json.dumps({"type": "result", "result": long_result})
        log = tmp_path / "task.log"
        log.write_text(result_json, encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert len(ctx.result_text) <= 500


class TestFormatStructuredContext:
    """Tests for format_structured_context()."""

    def test_formats_basic_context(self) -> None:
        ctx = _make_structured(
            files=["src/main.py"],
            errors=["Error: compile failed"],
        )
        text = format_structured_context(ctx)
        assert "upstream" in text
        assert "src/main.py" in text
        assert "compile failed" in text

    def test_empty_context_produces_minimal_output(self) -> None:
        ctx = _make_structured()
        text = format_structured_context(ctx)
        assert "upstream" in text
        assert "Files changed" not in text


class TestStructuredContextToDict:
    """Tests for StructuredContext.to_dict() serialization."""

    def test_serializes_all_fields(self) -> None:
        ctx = _make_structured(files=["a.py"], errors=["err1"])
        d = ctx.to_dict()
        assert d["task_id"] == "upstream"
        assert d["files_changed"] == ["a.py"]
        assert d["errors"] == ["err1"]
        assert d["summary"] == ""


class TestTaskResultWithStructuredContext:
    """Tests for TaskResult.structured_context integration."""

    def test_to_dict_includes_structured_context(self) -> None:
        sc = _make_structured(files=["x.py"])
        result = _make_result(structured=sc)
        d = result.to_dict()
        assert d["structured_context"] is not None
        assert d["structured_context"]["files_changed"] == ["x.py"]

    def test_to_dict_none_structured_context(self) -> None:
        result = _make_result()
        d = result.to_dict()
        assert d["structured_context"] is None


# ===========================================================================
# Feature 3: Template variables from structured context
# ===========================================================================


class TestStructuredTemplateVariables:
    """Tests for new template variables in _load_prompt()."""

    def test_files_changed_variable(self) -> None:
        sc = _make_structured("task-a", files=["src/a.py", "src/b.py"])
        result = _make_result("task-a", structured=sc)
        task = TaskSpec(
            id="downstream",
            engine="claude",
            prompt="Files: {{ task-a.files_changed }}",
            depends_on=["task-a"],
            context_from=["task-a"],
        )
        plan = PlanSpec(version=1, name="test", tasks=[task])
        prompt = _load_prompt(plan, task, {"task-a": result})
        assert "src/a.py" in prompt
        assert "src/b.py" in prompt

    def test_errors_variable(self) -> None:
        sc = _make_structured("task-a", errors=["compile failed"])
        result = _make_result("task-a", structured=sc)
        task = TaskSpec(
            id="downstream",
            engine="claude",
            prompt="Errors: {{ task-a.errors }}",
            depends_on=["task-a"],
            context_from=["task-a"],
        )
        plan = PlanSpec(version=1, name="test", tasks=[task])
        prompt = _load_prompt(plan, task, {"task-a": result})
        assert "compile failed" in prompt

    def test_summary_variable_without_summary(self) -> None:
        sc = _make_structured("task-a")
        result = _make_result("task-a", structured=sc)
        task = TaskSpec(
            id="downstream",
            engine="claude",
            prompt="Summary: {{ task-a.summary }}",
            depends_on=["task-a"],
            context_from=["task-a"],
        )
        plan = PlanSpec(version=1, name="test", tasks=[task])
        prompt = _load_prompt(plan, task, {"task-a": result})
        assert "(no summary)" in prompt

    def test_summary_variable_with_summary(self) -> None:
        sc = _make_structured("task-a")
        sc.summary = "Implemented user auth with JWT"
        result = _make_result("task-a", structured=sc)
        task = TaskSpec(
            id="downstream",
            engine="claude",
            prompt="Summary: {{ task-a.summary }}",
            depends_on=["task-a"],
            context_from=["task-a"],
        )
        plan = PlanSpec(version=1, name="test", tasks=[task])
        prompt = _load_prompt(plan, task, {"task-a": result})
        assert "Implemented user auth with JWT" in prompt

    def test_upstream_synthesis_variable(self) -> None:
        task = TaskSpec(
            id="downstream",
            engine="claude",
            prompt="Synthesis: {{ upstream_synthesis }}",
        )
        plan = PlanSpec(version=1, name="test", tasks=[task])
        prompt = _load_prompt(plan, task, context_synthesis="All tasks OK")
        assert "All tasks OK" in prompt

    def test_no_structured_context_leaves_variables_unchanged(self) -> None:
        result = _make_result("task-a")
        assert result.structured_context is None
        task = TaskSpec(
            id="downstream",
            engine="claude",
            prompt="{{ task-a.files_changed }}",
            depends_on=["task-a"],
            context_from=["task-a"],
        )
        plan = PlanSpec(version=1, name="test", tasks=[task])
        prompt = _load_prompt(plan, task, {"task-a": result})
        assert "{{ task-a.files_changed }}" in prompt


# ===========================================================================
# Loader validation: context_mode
# ===========================================================================


class TestContextModeValidation:
    """Tests for context_mode parsing and validation in loader.py."""

    def test_valid_context_modes_accepted(self, tmp_path: Path) -> None:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        # council mode requires a dedicated council block — tested in test_council.py
        for mode in CONTEXT_MODES - {"council"}:
            workspace_root_line = (
                f"workspace_root: {workspace_root.as_posix()}\n"
                if mode in {"recursive", "codebase_map"}
                else ""
            )
            yaml_text = (
                f"version: 1\nname: test-plan\n{workspace_root_line}tasks:\n"
                f"  - id: a\n    command: echo a\n"
                f"  - id: b\n    command: echo b\n    depends_on: [a]\n"
                f"    context_from: [a]\n    context_mode: {mode}\n"
            )
            p = tmp_path / f"plan_{mode}.yaml"
            p.write_text(yaml_text, encoding="utf-8")
            plan = load_plan(p)
            b = next(t for t in plan.tasks if t.id == "b")
            assert b.context_mode == mode

    def test_invalid_context_mode_raises(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: test-plan\ntasks:\n"
            "  - id: a\n    command: echo a\n"
            "  - id: b\n    command: echo b\n    depends_on: [a]\n"
            "    context_from: [a]\n    context_mode: invalid\n"
        )
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_mode"):
            load_plan(p)

    def test_summarized_without_context_from_raises(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: test-plan\ntasks:\n"
            "  - id: a\n    command: echo a\n    context_mode: summarized\n"
        )
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="requires.*non-empty context_from"):
            load_plan(p)

    def test_map_reduce_without_context_from_raises(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: test-plan\ntasks:\n"
            "  - id: a\n    command: echo a\n    context_mode: map_reduce\n"
        )
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="requires.*non-empty context_from"):
            load_plan(p)

    def test_recursive_without_workspace_root_raises_e021(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: test-plan\ntasks:\n"
            "  - id: a\n    command: echo a\n"
            "  - id: b\n    command: echo b\n    depends_on: [a]\n"
            "    context_from: [a]\n    context_mode: recursive\n"
        )
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=rf"\[{E021}\]"):
            load_plan(p)

    def test_raw_without_context_from_is_fine(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: test-plan\ntasks:\n"
            "  - id: a\n    command: echo a\n    context_mode: raw\n"
        )
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(p)
        assert plan.tasks[0].context_mode == "raw"

    def test_default_context_mode_is_raw(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: test-plan\ntasks:\n"
            "  - id: a\n    command: echo a\n"
        )
        p = tmp_path / "plan.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(p)
        assert plan.tasks[0].context_mode == "raw"


# ===========================================================================
# Feature 1: Summarization prompts and LLM calls
# ===========================================================================


class TestBuildSummarizationPrompt:
    """Tests for build_summarization_prompt()."""

    def test_includes_task_id_and_status(self) -> None:
        sc = _make_structured("my-task")
        prompt = build_summarization_prompt("my-task", "output line 1\n", sc)
        assert "my-task" in prompt
        assert "success" in prompt

    def test_includes_files_changed(self) -> None:
        sc = _make_structured("t", files=["a.py", "b.py"])
        prompt = build_summarization_prompt("t", "", sc)
        assert "a.py" in prompt
        assert "b.py" in prompt

    def test_includes_stdout_tail(self) -> None:
        sc = _make_structured("t")
        prompt = build_summarization_prompt("t", "hello world\n", sc)
        assert "hello world" in prompt

    def test_structured_fields_in_prompt(self) -> None:
        """Prompt requests 9-section structured format with scratchpad."""
        sc = _make_structured("t")
        prompt = build_summarization_prompt("t", "output\n", sc)
        assert "<analysis>" in prompt
        assert "**1. Primary Request:**" in prompt
        assert "**9. Next Steps:**" in prompt


class TestBuildReducePrompt:
    """Tests for build_reduce_prompt()."""

    def test_includes_all_summaries(self) -> None:
        summaries = {"task-a": "Summary A", "task-b": "Summary B"}
        prompt = build_reduce_prompt(summaries)
        assert "task-a" in prompt
        assert "Summary A" in prompt
        assert "task-b" in prompt
        assert "Summary B" in prompt


class TestRunSummarization:
    """Tests for _run_summarization() with mocked subprocess."""

    @pytest.fixture(autouse=True)
    def _reset_circuit_breaker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset the summarization circuit breaker before each test."""
        monkeypatch.setattr("maestro_cli.runners._summarization_consecutive_failures", 0)

    def test_success_returns_summary(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="Summary text\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        result = _run_summarization("t", "output", sc, tmp_path)
        assert result == "Summary text"

    def test_failure_returns_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        result = _run_summarization("t", "output", sc, tmp_path)
        assert "failed" in result

    def test_timeout_returns_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60)

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        result = _run_summarization("t", "output", sc, tmp_path)
        assert "timed out" in result

    def test_exception_returns_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            raise FileNotFoundError("claude not found")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        result = _run_summarization("t", "output", sc, tmp_path)
        assert "error" in result

    def test_uses_haiku_model(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        captured_cmd: list[str] = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        _run_summarization("t", "output", sc, tmp_path)
        assert "--model" in captured_cmd
        idx = captured_cmd.index("--model")
        assert captured_cmd[idx + 1] == "haiku"

    def test_empty_stdout_returns_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        result = _run_summarization("t", "output", sc, tmp_path)
        assert "failed" in result

    def test_scratchpad_stripped_from_output(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Analysis scratchpad block is stripped from LLM output."""
        raw_output = (
            "<analysis>\nThe task modified two files.\n</analysis>\n"
            "**1. Primary Request:** Implement feature X.\n"
            "**9. Next Steps:** Run tests."
        )

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=raw_output, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        result = _run_summarization("t", "output", sc, tmp_path)
        assert "<analysis>" not in result
        assert "Primary Request" in result
        assert "Next Steps" in result

    def test_circuit_breaker_trips_after_threshold(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """After N consecutive failures, falls back to mechanical extraction."""
        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        # Fail 3 times (threshold)
        for _ in range(3):
            _run_summarization("t", "## Heading\nDetail line", sc, tmp_path)
        assert call_count == 3
        # 4th call should NOT invoke subprocess (circuit breaker open)
        result = _run_summarization("t", "## Heading\nDetail line", sc, tmp_path)
        assert call_count == 3  # no new subprocess call
        assert "Heading" in result  # L1 mechanical extraction

    def test_circuit_breaker_resets_on_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A successful call resets the circuit breaker counter."""
        import maestro_cli.runners as _r

        monkeypatch.setattr(_r, "_summarization_consecutive_failures", 2)

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        sc = _make_structured("t")
        _run_summarization("t", "output", sc, tmp_path)
        assert _r._summarization_consecutive_failures == 0


# ===========================================================================
# Feature 2: Map/reduce
# ===========================================================================


class TestRunMapReduce:
    """Tests for _run_map_reduce() with mocked subprocess."""

    def test_returns_synthesis(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="Synthesis result\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        sc_a = _make_structured("a")
        sc_a.summary = "Summary A"
        sc_b = _make_structured("b")
        sc_b.summary = "Summary B"

        upstream = {
            "a": _make_result("a", structured=sc_a),
            "b": _make_result("b", structured=sc_b),
        }
        result = _run_map_reduce(upstream, tmp_path)
        assert "Synthesis result" in result

    def test_empty_upstream_returns_fallback(self, tmp_path: Path) -> None:
        result = _run_map_reduce({}, tmp_path)
        assert "no upstream" in result

    def test_no_summaries_returns_fallback(self, tmp_path: Path) -> None:
        upstream = {
            "a": _make_result("a", structured=_make_structured("a")),
        }
        result = _run_map_reduce(upstream, tmp_path)
        assert "no upstream" in result

    def test_reduce_failure_returns_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        sc = _make_structured("a")
        sc.summary = "Summary"
        upstream = {"a": _make_result("a", structured=sc)}
        result = _run_map_reduce(upstream, tmp_path)
        assert "failed" in result


# ===========================================================================
# Scaffold: quality gates use context_mode
# ===========================================================================


class TestScaffoldContextMode:
    """Tests that scaffold generates quality gates with context_mode."""

    def test_quality_gates_have_context_mode(self) -> None:
        from maestro_cli.scaffold import _generate_quality_gates

        gates = _generate_quality_gates(["a", "b", "c"], None, "p", "g")
        review = next(g for g in gates if g["id"] == "code-review")
        qa = next(g for g in gates if g["id"] == "qa-verification")

        assert review["context_mode"] == "map_reduce"  # >= 3 impl tasks
        assert qa["context_mode"] == "summarized"

    def test_code_review_uses_summarized_for_few_tasks(self) -> None:
        from maestro_cli.scaffold import _generate_quality_gates

        gates = _generate_quality_gates(["a", "b"], None, "p", "g")
        review = next(g for g in gates if g["id"] == "code-review")
        assert review["context_mode"] == "summarized"  # < 3 impl tasks

    def test_code_review_prompt_references_upstream_synthesis(self) -> None:
        from maestro_cli.scaffold import _generate_quality_gates

        gates = _generate_quality_gates(["a", "b", "c"], None, "p", "g")
        review = next(g for g in gates if g["id"] == "code-review")
        assert "upstream_synthesis" in review["prompt"]

    def test_qa_prompt_references_summaries(self) -> None:
        from maestro_cli.scaffold import _generate_quality_gates

        gates = _generate_quality_gates(["task-a", "task-b"], None, "p", "g")
        qa = next(g for g in gates if g["id"] == "qa-verification")
        assert "task-a.summary" in qa["prompt"]
        assert "task-b.summary" in qa["prompt"]
