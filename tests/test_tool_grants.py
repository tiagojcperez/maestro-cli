from __future__ import annotations

"""Tests for parameter-scoped tool grants (v2.5.4).

Covers ``check_tool_grants`` post-hoc verification, the native
``--allowedTools`` specifier pass-through for Claude, ``on_grant_violation``
loader validation (E073), post-hoc enforcement inside ``execute_task``, and
the ``has_scoped_tools`` policy field.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    EngineDefaults,
    PlanDefaults,
    PlanSpec,
    PolicySpec,
    TaskSpec,
)
from maestro_cli.policy import compile_policy
from maestro_cli.runners import build_command, check_tool_grants, execute_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(tmp_path: Path) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test",
        max_parallel=1,
        fail_fast=False,
        run_dir=str(tmp_path / "runs"),
        defaults=PlanDefaults(codex=EngineDefaults(), claude=EngineDefaults()),
        tasks=[],
    )


class _DummyProc:
    """Stand-in for a ``subprocess.Popen`` object — never actually used."""


def _patch_popen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "maestro_cli.runners.subprocess.Popen",
        lambda *a, **kw: _DummyProc(),
    )


def _patch_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "maestro_cli.runners.build_command",
        lambda _plan, _task, _workdir, **kwargs: (["claude", "--print", "p"], False),
    )


def _patch_stream(
    monkeypatch: pytest.MonkeyPatch,
    results: list[tuple[int, str, str]],
    *,
    feed_lines: list[str] | None = None,
) -> None:
    def _fake_stream(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
        cb = kwargs.get("line_callback")
        if feed_lines and cb is not None:
            for line in feed_lines:
                cb(line)
        return results.pop(0)

    monkeypatch.setattr("maestro_cli.runners._stream_process", _fake_stream)


def _tool_use_line(name: str, tool_input: dict[str, Any]) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": name, "input": tool_input},
            ]
        },
    })


# ---------------------------------------------------------------------------
# check_tool_grants — pure post-hoc verification
# ---------------------------------------------------------------------------


class TestCheckToolGrants:
    def _task(self, allowed: list[str] | None) -> TaskSpec:
        return TaskSpec(id="t1", engine="claude", allowed_tools=allowed)

    def test_no_allowed_tools_returns_empty(self) -> None:
        task = self._task(None)
        assert check_tool_grants(task, [("Bash", {"command": "rm -rf /"})]) == []

    def test_bare_grant_covers_any_arguments(self) -> None:
        task = self._task(["Bash"])
        assert check_tool_grants(task, [("Bash", {"command": "anything goes"})]) == []

    def test_scoped_grant_matching_call_passes(self) -> None:
        task = self._task(["Read", "Bash(git *)"])
        calls = [("Bash", {"command": "git status"})]
        assert check_tool_grants(task, calls) == []

    def test_scoped_grant_non_matching_call_violates(self) -> None:
        task = self._task(["Read", "Bash(git *)"])
        calls = [("Bash", {"command": "curl http://evil.example"})]
        violations = check_tool_grants(task, calls)
        assert len(violations) == 1
        assert "Bash" in violations[0]
        assert "git *" in violations[0]

    def test_bare_grant_wins_over_scoped_grant(self) -> None:
        task = self._task(["Bash", "Bash(git *)"])
        calls = [("Bash", {"command": "npm install"})]
        assert check_tool_grants(task, calls) == []

    def test_ungranted_known_tool_violates(self) -> None:
        task = self._task(["Read"])
        violations = check_tool_grants(task, [("Write", {"file_path": "x.py"})])
        assert len(violations) == 1
        assert "Write" in violations[0]
        assert "outside allowed_tools" in violations[0]

    def test_unknown_tool_is_skipped(self) -> None:
        # Tools outside CLAUDE_TOOLS (internal/MCP) mirror CLI behaviour: not blocked.
        task = self._task(["Read"])
        assert check_tool_grants(task, [("Task", {"prompt": "spawn"})]) == []
        assert check_tool_grants(task, [("mcp__srv__tool", {"x": 1})]) == []

    def test_multiple_patterns_second_matches(self) -> None:
        task = self._task(["Bash(git *)", "Bash(py -m pytest*)"])
        calls = [("Bash", {"command": "py -m pytest tests/ -q"})]
        assert check_tool_grants(task, calls) == []

    def test_path_grant_matches_windows_separators(self) -> None:
        task = self._task(["Write(src/*)"])
        calls = [("Write", {"file_path": "src\\maestro_cli\\runners.py"})]
        assert check_tool_grants(task, calls) == []

    def test_path_grant_outside_scope_violates(self) -> None:
        task = self._task(["Write(src/*)"])
        violations = check_tool_grants(task, [("Write", {"file_path": "docs/x.md"})])
        assert len(violations) == 1
        assert "Write" in violations[0]

    def test_unmapped_tool_falls_back_to_json_input(self) -> None:
        # TodoWrite has no primary-arg mapping: pattern matches serialized input.
        task = self._task(["TodoWrite(*release*)"])
        calls = [("TodoWrite", {"todos": [{"content": "cut release"}]})]
        assert check_tool_grants(task, calls) == []

    def test_category_expansion_enforces_scoped_member(self) -> None:
        # git-only expands to Read, Glob, Grep, Bash(git *) for claude.
        task = self._task(["git-only"])
        assert check_tool_grants(task, [("Bash", {"command": "git diff"})]) == []
        violations = check_tool_grants(task, [("Bash", {"command": "rm -rf x"})])
        assert len(violations) == 1

    def test_missing_primary_arg_falls_back_to_json(self) -> None:
        # A Bash call without "command" should not crash; the serialized
        # input is matched instead.
        task = self._task(["Bash(git *)"])
        violations = check_tool_grants(task, [("Bash", {})])
        assert len(violations) == 1

    def test_non_serializable_input_falls_back_to_str(self) -> None:
        # json.dumps fails on exotic values; str() fallback keeps matching.
        task = self._task(["TodoWrite(*frozenset*)"])
        calls: list[tuple[str, dict[str, Any]]] = [
            ("TodoWrite", {"items": frozenset({"a"})}),
        ]
        assert check_tool_grants(task, calls) == []


# ---------------------------------------------------------------------------
# Claude command: native --allowedTools specifiers
# ---------------------------------------------------------------------------


class TestClaudeScopedAllowedToolsFlag:
    def test_scoped_grant_emits_allowed_tools_flag(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, "Bash(git *)"]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Bash(git *)"
        # Bash has a grant — it must NOT be in the disallow list.
        d_idx = cmd.index("--disallowedTools")
        disallowed = set(cmd[d_idx + 1].split(","))
        assert "Bash" not in disallowed
        assert "Write" in disallowed

    def test_no_scoped_grants_no_allowed_tools_flag(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, Grep]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert "--allowedTools" not in cmd

    def test_multiple_scoped_grants_sorted_and_joined(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: ["Write(src/*)", "Bash(git *)"]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Bash(git *),Write(src/*)"


# ---------------------------------------------------------------------------
# Loader: on_grant_violation parsing + E073
# ---------------------------------------------------------------------------


class TestOnGrantViolationLoader:
    def test_default_is_warn(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read]
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].on_grant_violation == "warn"

    def test_parse_fail_action(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: ["Bash(git *)"]
    on_grant_violation: fail
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].on_grant_violation == "fail"

    def test_invalid_value_raises_e073(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read]
    on_grant_violation: explode
""")
        with pytest.raises(PlanValidationError, match="E073") as exc_info:
            load_plan(plan_file)
        assert "on_grant_violation" in str(exc_info.value)

    def test_without_allowed_tools_raises_e073(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    on_grant_violation: fail
""")
        with pytest.raises(PlanValidationError, match="E073") as exc_info:
            load_plan(plan_file)
        assert "requires allowed_tools" in str(exc_info.value)


# ---------------------------------------------------------------------------
# execute_task: post-hoc enforcement
# ---------------------------------------------------------------------------


class TestPostHocGrantEnforcement:
    def test_violation_recorded_warn_keeps_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Read", "Bash(git *)"],
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[_tool_use_line("Bash", {"command": "curl http://x"})],
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan, task, run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )
        assert result.status == "success"
        assert len(result.grant_violations) == 1
        assert "Bash" in result.grant_violations[0]
        violation_events = [p for n, p in events if n == "tool_grant_violation"]
        assert len(violation_events) == 1
        assert violation_events[0]["task_id"] == "t"
        assert violation_events[0]["action"] == "warn"

    def test_violation_with_fail_action_fails_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Bash(git *)"],
            on_grant_violation="fail",
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[_tool_use_line("Bash", {"command": "rm -rf build"})],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "[tool grants]" in result.message
        assert result.grant_violations

    def test_compliant_calls_produce_no_violations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Read", "Bash(git *)"],
            on_grant_violation="fail",
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[
                _tool_use_line("Read", {"file_path": "src/x.py"}),
                _tool_use_line("Bash", {"command": "git log --oneline"}),
            ],
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan, task, run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )
        assert result.status == "success"
        assert result.grant_violations == []
        assert not [n for n, _ in events if n == "tool_grant_violation"]

    def test_collection_works_without_event_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Bash(git *)"],
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[_tool_use_line("Bash", {"command": "curl http://x"})],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert len(result.grant_violations) == 1

    def test_no_grants_no_collection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", model="haiku", prompt="do it")
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[_tool_use_line("Bash", {"command": "anything"})],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert result.grant_violations == []

    def test_violation_event_callback_exception_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Bash(git *)"],
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[_tool_use_line("Bash", {"command": "curl http://x"})],
        )

        def _boom(name: str, payload: dict[str, object]) -> None:
            if name == "tool_grant_violation":
                raise RuntimeError("callback boom")

        result = execute_task(plan, task, run_path, event_callback=_boom)
        # Callback raised; execute_task swallowed it and still recorded.
        assert result.status == "success"
        assert len(result.grant_violations) == 1

    def test_non_dict_tool_input_collected_as_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Bash(git *)"],
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        # input is a string, not a dict — collected as {} and checked
        # against the serialized-input fallback (no primary arg → violation).
        line = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": "raw-string"},
                ]
            },
        })
        _patch_stream(monkeypatch, [(0, "ok", "")], feed_lines=[line])

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert len(result.grant_violations) == 1

    def test_violations_serialized_in_result_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allowed_tools=["Bash(git *)"],
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[_tool_use_line("Bash", {"command": "curl http://x"})],
        )

        result = execute_task(plan, task, run_path)
        d = result.to_dict()
        assert "grant_violations" in d
        assert len(d["grant_violations"]) == 1


# ---------------------------------------------------------------------------
# Policy engine: has_scoped_tools
# ---------------------------------------------------------------------------


class TestHasScopedToolsPolicy:
    def _make_plan(self) -> PlanSpec:
        return PlanSpec(version=1, name="test-plan", tasks=[], defaults=PlanDefaults())

    def test_true_when_scoped_grant_present(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read", "Bash(git *)"])
        policy = PolicySpec(
            name="check", rule="task.has_scoped_tools == True", action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, self._make_plan(), None) is True

    def test_false_when_only_bare_grants(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read", "Bash"])
        policy = PolicySpec(
            name="check", rule="task.has_scoped_tools == True", action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, self._make_plan(), None) is False

    def test_false_when_no_allowed_tools(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=None)
        policy = PolicySpec(
            name="check", rule="task.has_scoped_tools == True", action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, self._make_plan(), None) is False
