from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import EngineDefaults, PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.runners import _resolve_context_model, _run_summarization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(**kwargs) -> TaskSpec:  # type: ignore[type-arg]
    return TaskSpec(id="t1", engine="claude", prompt="do something", **kwargs)


def _plan(*tasks: TaskSpec, **kwargs) -> PlanSpec:  # type: ignore[type-arg]
    return PlanSpec(name="test", tasks=list(tasks), **kwargs)


# ---------------------------------------------------------------------------
# Feature 3: context_model field resolution
# ---------------------------------------------------------------------------


class TestResolveContextModel:
    def test_task_level_override(self) -> None:
        task = _task(context_model="sonnet")
        plan = _plan(task)
        assert _resolve_context_model(task, plan) == "sonnet"

    def test_engine_defaults_override(self) -> None:
        task = _task(context_model=None)
        plan = _plan(task)
        plan.defaults.claude.context_model = "flash"
        assert _resolve_context_model(task, plan) == "flash"

    def test_task_beats_engine_defaults(self) -> None:
        task = _task(context_model="opus")
        plan = _plan(task)
        plan.defaults.claude.context_model = "flash"
        assert _resolve_context_model(task, plan) == "opus"

    def test_fallback_to_haiku(self) -> None:
        task = _task(context_model=None)
        plan = _plan(task)
        assert _resolve_context_model(task, plan) == "haiku"

    def test_non_claude_engine_uses_its_defaults(self) -> None:
        task = TaskSpec(id="t1", engine="gemini", prompt="do something", context_model=None)
        plan = _plan(task)
        plan.defaults.gemini.context_model = "flash-lite"
        assert _resolve_context_model(task, plan) == "flash-lite"

    def test_no_engine_defaults_returns_haiku(self) -> None:
        task = TaskSpec(id="t1", engine="codex", prompt="do something", context_model=None)
        plan = _plan(task)
        assert _resolve_context_model(task, plan) == "haiku"


class TestTaskSpecContextModelField:
    def test_field_exists_with_none_default(self) -> None:
        task = TaskSpec(id="t1", command="echo hi")
        assert task.context_model is None

    def test_field_accepts_string(self) -> None:
        task = TaskSpec(id="t1", command="echo hi", context_model="sonnet")
        assert task.context_model == "sonnet"

    def test_to_dict_includes_context_model(self) -> None:
        task = TaskSpec(id="t1", command="echo hi", context_model="flash")
        d = task.to_dict()
        assert "context_model" in d
        assert d["context_model"] == "flash"

    def test_to_dict_context_model_none(self) -> None:
        task = TaskSpec(id="t1", command="echo hi")
        d = task.to_dict()
        assert d["context_model"] is None


class TestEngineDefaultsContextModelField:
    def test_field_exists_with_none_default(self) -> None:
        ed = EngineDefaults()
        assert ed.context_model is None

    def test_field_accepts_string(self) -> None:
        ed = EngineDefaults(context_model="flash-lite")
        assert ed.context_model == "flash-lite"


class TestLoaderContextModel:
    def test_parse_context_model_in_task(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do something"
                context_model: sonnet
        """)
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(yaml_content, encoding="utf-8")
        plan = load_plan(str(plan_file))
        assert plan.tasks[0].context_model == "sonnet"

    def test_parse_context_model_none_when_absent(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: ["echo", "hi"]
        """)
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(yaml_content, encoding="utf-8")
        plan = load_plan(str(plan_file))
        assert plan.tasks[0].context_model is None

    def test_parse_engine_defaults_context_model(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            name: test
            defaults:
              claude:
                model: sonnet
                context_model: haiku
            tasks:
              - id: t1
                engine: claude
                prompt: "do something"
        """)
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(yaml_content, encoding="utf-8")
        plan = load_plan(str(plan_file))
        assert plan.defaults.claude.context_model == "haiku"


class TestRunSummarizationUsesModel:
    def _make_sc(self):  # type: ignore[no-untyped-def]
        from maestro_cli.models import StructuredContext
        return StructuredContext(task_id="t1", status="success", exit_code=0, duration_sec=1.0)

    def test_model_param_passed_to_subprocess(self, tmp_path: Path) -> None:
        sc = self._make_sc()
        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Summary text"
            mock_run.return_value = mock_proc

            _run_summarization("t1", "some output", sc, tmp_path, model="sonnet")

        assert mock_run.called
        cmd_used = mock_run.call_args[0][0]
        assert "--model" in cmd_used
        idx = cmd_used.index("--model")
        assert cmd_used[idx + 1] == "sonnet"

    def test_default_model_is_haiku(self, tmp_path: Path) -> None:
        sc = self._make_sc()
        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Summary text"
            mock_run.return_value = mock_proc

            _run_summarization("t1", "some output", sc, tmp_path)

        cmd_used = mock_run.call_args[0][0]
        idx = cmd_used.index("--model")
        assert cmd_used[idx + 1] == "haiku"
