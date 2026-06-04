"""Coverage tests for runners.py — targeted uncovered lines (batch p3).

Each test drives the REAL runner function with crafted inputs and mocks only
external boundaries (executable resolution, subprocess). Engine/LLM/network/git
calls are never made for real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import TaskExecutionError
from maestro_cli.models import (
    EngineDefaults,
    MCPServerSpec,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
)
from maestro_cli import plugins as plugins_mod
from maestro_cli.plugins import EngineCommandContext, EnginePlugin

from maestro_cli.runners import (
    _build_claude_command,
    _build_mcp_config,
    _build_plugin_command,
    _extract_cost_from_json_payload,
    _kill_process_tree,
    _load_qwen_pricing_table,
    _resolve_engine_plugin_for_task,
    _resolve_model_for_pricing,
    _run_map_reduce,
    _run_task_assertions,
    build_command,
    kill_all_active,
)


def _plan() -> PlanSpec:
    return PlanSpec(version=1, name="p", defaults=PlanDefaults(), tasks=[])


def _ctx(task: TaskSpec, workdir: Path, prompt: str = "do it") -> EngineCommandContext:
    return EngineCommandContext(
        plan=_plan(),
        task=task,
        workdir=workdir,
        prompt_text=prompt,
    )


# ---------------------------------------------------------------------------
# _build_mcp_config — (no resolved servers -> return None)
# ---------------------------------------------------------------------------


class TestBuildMcpConfig:
    def test_returns_none_when_no_server_matches_tool(self, tmp_path: Path) -> None:
        # task.mcp_tools is non-empty and plan.mcp_servers is non-empty, but the
        # requested tool name does not match any declared server. The resolver
        # returns an empty list, so mcp_config["mcpServers"] stays empty -> None.
        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(),
            tasks=[],
            mcp_servers=[MCPServerSpec(name="github", command=["npx", "gh"])],
        )
        task = TaskSpec(id="t", engine="claude", prompt="x", mcp_tools=["nonexistent"])
        result = _build_mcp_config(plan, task, tmp_path)
        assert result is None
        # No config file should be written for an empty server set.
        assert list(tmp_path.glob(".mcp-config-*.json")) == []

    def test_writes_config_when_server_matches(self, tmp_path: Path) -> None:
        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(),
            tasks=[],
            mcp_servers=[
                MCPServerSpec(name="github", command=["npx", "gh"], url="http://x", env={"K": "v"}),
            ],
        )
        task = TaskSpec(id="t", engine="claude", prompt="x", mcp_tools=["github"])
        result = _build_mcp_config(plan, task, tmp_path)
        assert result is not None
        assert result.exists()
        assert "github" in result.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _build_claude_command — (--agent), (--mcp-config)
# ---------------------------------------------------------------------------


class TestBuildClaudeCommand:
    def test_agent_flag_added(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        task = TaskSpec(id="t", engine="claude", agent="qa-engineer", prompt="x")
        cmd, shell = _build_claude_command(_ctx(task, tmp_path))
        assert "--agent" in cmd
        assert cmd[cmd.index("--agent") + 1] == "qa-engineer"
        assert shell is False

    def test_mcp_config_flag_added(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(),
            tasks=[],
            mcp_servers=[MCPServerSpec(name="github", command=["npx", "gh"])],
        )
        task = TaskSpec(id="t", engine="claude", prompt="x", mcp_tools=["github"])
        ctx = EngineCommandContext(plan=plan, task=task, workdir=tmp_path, prompt_text="x")
        cmd, _shell = _build_claude_command(ctx)
        assert "--mcp-config" in cmd
        idx = cmd.index("--mcp-config")
        config_path = Path(cmd[idx + 1])
        assert config_path.exists()


# ---------------------------------------------------------------------------
# _resolve_engine_plugin_for_task — (PluginResolutionError wrap)
# ---------------------------------------------------------------------------


class TestResolveEnginePluginForTask:
    def test_unknown_engine_raises_task_execution_error(self) -> None:
        # engine literal not enforced at runtime; an unknown engine name triggers
        # PluginResolutionError inside get_engine_plugin, wrapped as E102.
        task = TaskSpec(id="t", engine="bogus-engine", prompt="x")  # type: ignore[arg-type]
        with pytest.raises(TaskExecutionError) as excinfo:
            _resolve_engine_plugin_for_task(task)
        assert "bogus-engine" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _build_plugin_command — (re-raise TaskExecutionError)
# ---------------------------------------------------------------------------


class TestBuildPluginCommand:
    def test_task_execution_error_propagates(self, tmp_path: Path) -> None:
        def _raise(_ctx: EngineCommandContext) -> tuple[list[str], bool]:
            raise TaskExecutionError("boom from builder")

        plugin = EnginePlugin(name="raising", build_command=_raise)
        task = TaskSpec(id="t", engine="raising", prompt="x")  # type: ignore[arg-type]
        ctx = _ctx(task, tmp_path)
        with pytest.raises(TaskExecutionError) as excinfo:
            _build_plugin_command(plugin, ctx)
        # Re-raised verbatim (not wrapped with the engine builder-failed prefix).
        assert "boom from builder" in str(excinfo.value)

    def test_generic_exception_is_wrapped(self, tmp_path: Path) -> None:
        def _raise(_ctx: EngineCommandContext) -> tuple[list[str], bool]:
            raise ValueError("internal")

        plugin = EnginePlugin(name="wrapper", build_command=_raise)
        task = TaskSpec(id="t", engine="wrapper", prompt="x")  # type: ignore[arg-type]
        ctx = _ctx(task, tmp_path)
        with pytest.raises(TaskExecutionError) as excinfo:
            _build_plugin_command(plugin, ctx)
        msg = str(excinfo.value)
        assert "command builder" in msg
        assert "ValueError" in msg


# ---------------------------------------------------------------------------
# build_command — (auto routing), 3462 (no engine raise)
# ---------------------------------------------------------------------------


class TestBuildCommandAutoRouting:
    def test_auto_model_resolves_to_concrete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(claude=EngineDefaults()),
            tasks=[],
        )
        task = TaskSpec(id="t", engine="claude", model="auto", prompt="do it")
        cmd, shell = build_command(plan, task, tmp_path)
        assert shell is False
        assert "--model" in cmd
        resolved = cmd[cmd.index("--model") + 1]
        # routing returns a concrete claude tier, never the literal "auto"
        assert resolved != "auto"
        assert resolved in {"haiku", "sonnet", "opus"}

    def test_no_engine_raises(self, tmp_path: Path) -> None:
        # No command and no engine -> build_command reaches the E103 raise.
        task = TaskSpec(id="t", prompt="x")
        with pytest.raises(TaskExecutionError) as excinfo:
            build_command(_plan(), task, tmp_path)
        assert "has no engine" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _run_task_assertions — (custom message on failure)
# ---------------------------------------------------------------------------


class TestRunTaskAssertions:
    def test_failure_with_custom_message(self, tmp_path: Path) -> None:
        # Unsupported assertion type fails; a non-empty custom "message" is returned.
        assertions: list[dict[str, Any]] = [
            {"type": "definitely_unknown_type", "message": "  Custom failure note  "}
        ]
        ok, reasoning, custom = _run_task_assertions(assertions, tmp_path)
        assert ok is False
        assert "FAIL" in reasoning
        assert custom == "Custom failure note"

    def test_failure_without_custom_message(self, tmp_path: Path) -> None:
        assertions: list[dict[str, Any]] = [{"type": "definitely_unknown_type"}]
        ok, _reasoning, custom = _run_task_assertions(assertions, tmp_path)
        assert ok is False
        assert custom is not None
        assert "failed" in custom

    def test_all_pass_returns_none_message(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("hello world", encoding="utf-8")
        assertions: list[dict[str, Any]] = [
            {"type": "file_contains", "path": "out.txt", "pattern": "hello"}
        ]
        ok, _reasoning, custom = _run_task_assertions(assertions, tmp_path)
        assert ok is True
        assert custom is None


# ---------------------------------------------------------------------------
# _kill_process_tree — (taskkill failure -> proc.kill)
# _kill_all_active — (swallow exceptions)
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self) -> None:
        self.pid = 4242
        self.killed = False

    def kill(self) -> None:
        self.killed = True


class TestKillProcessTree:
    def test_kills_directly_when_taskkill_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If subprocess.run (the taskkill call on Windows) raises, the function
        # must fall back to proc.kill(). On POSIX the else-branch also calls
        # proc.kill(), so kill() is invoked regardless of OS.
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise OSError("taskkill exploded")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", _boom)
        proc = _FakeProc()
        _kill_process_tree(proc)  # type: ignore[arg-type]
        assert proc.killed is True


class TestKillAllActive:
    def test_swallows_kill_exceptions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.runners import _active_procs, _active_procs_lock

        def _boom(_proc: Any) -> None:
            raise RuntimeError("kill failed hard")

        monkeypatch.setattr("maestro_cli.runners._kill_process_tree", _boom)
        proc = _FakeProc()
        with _active_procs_lock:
            _active_procs["cov-p3"] = proc  # type: ignore[assignment]
        try:
            # Must not raise even though _kill_process_tree raises for each proc.
            kill_all_active()
        finally:
            with _active_procs_lock:
                _active_procs.pop("cov-p3", None)


# ---------------------------------------------------------------------------
# _extract_cost_from_json_payload — (modelUsage value not a dict)
# ---------------------------------------------------------------------------


class TestExtractCostFromJsonPayload:
    def test_modelusage_skips_non_dict_entries(self) -> None:
        payload = {
            "modelUsage": {
                "m1": "not-a-dict",  # skipped at the non-dict continue
                "m2": {"costUSD": 0.25},
                "m3": {"costUSD": 0.75},
            }
        }
        cost = _extract_cost_from_json_payload(payload)
        assert cost == pytest.approx(1.0)

    def test_modelusage_all_non_dict_falls_through(self) -> None:
        # All entries non-dict -> model_costs stays empty -> falls through to the
        # recursive search (no cost anywhere) -> None.
        payload = {"modelUsage": {"m1": "x", "m2": 5}}
        assert _extract_cost_from_json_payload(payload) is None


# ---------------------------------------------------------------------------
# _load_qwen_pricing_table —
# ---------------------------------------------------------------------------


class TestLoadQwenPricingTable:
    def test_returns_normalized_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MAESTRO_QWEN_PRICING_JSON", raising=False)
        table = _load_qwen_pricing_table()
        assert isinstance(table, dict)
        # Every entry is a 3-tuple of floats (input, cached, output).
        for rates in table.values():
            assert isinstance(rates, tuple)
            assert len(rates) == 3
        assert len(table) >= 1


# ---------------------------------------------------------------------------
# _resolve_model_for_pricing — (no resolve_pricing_model hook)
# ---------------------------------------------------------------------------


class TestResolveModelForPricing:
    def test_falls_back_to_resolve_model_and_none(self) -> None:
        # Register a temporary builtin plugin WITHOUT a resolve_pricing_model
        # hook so the fallback branch (plugin.resolve_model / None) executes.
        plugin = EnginePlugin(
            name="cov-p3-engine",
            build_command=lambda ctx: (["x"], False),
            model_aliases={"short": "long-model-name"},
            resolve_pricing_model=None,
        )
        plugins_mod.register_builtin_engine(plugin)
        try:
            # task_model truthy -> resolve_model maps the alias.
            resolved = _resolve_model_for_pricing("cov-p3-engine", "short", [])
            assert resolved == "long-model-name"
            # passthrough for an unknown alias
            assert _resolve_model_for_pricing("cov-p3-engine", "bare", []) == "bare"
            # task_model falsy -> returns None.
            assert _resolve_model_for_pricing("cov-p3-engine", None, []) is None
        finally:
            plugins_mod._builtin_plugins.pop("cov-p3-engine", None)


# ---------------------------------------------------------------------------
# _run_map_reduce — (subprocess raises generic Exception)
# ---------------------------------------------------------------------------


class TestRunMapReduce:
    def test_reduce_generic_exception_returns_error_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli.models import StructuredContext, TaskResult

        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("reduce subprocess crashed")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", _boom)

        result_a = TaskResult(task_id="a", status="success")
        result_a.structured_context = StructuredContext(
            task_id="a", status="success", exit_code=0, duration_sec=1.0, summary="Summary of A"
        )
        result_b = TaskResult(task_id="b", status="success")
        result_b.structured_context = StructuredContext(
            task_id="b", status="success", exit_code=0, duration_sec=1.0, summary="Summary of B"
        )

        out = _run_map_reduce({"a": result_a, "b": result_b}, tmp_path)
        assert out.startswith("[reduce error:")
        assert "reduce subprocess crashed" in out

    def test_no_summaries_returns_sentinel(self, tmp_path: Path) -> None:
        from maestro_cli.models import TaskResult

        result = TaskResult(task_id="a", status="success")
        out = _run_map_reduce({"a": result}, tmp_path)
        assert "no upstream tasks had summaries" in out
