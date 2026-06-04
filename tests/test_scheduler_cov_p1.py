from __future__ import annotations

import types
from pathlib import Path

import pytest

from maestro_cli.models import (
    PlanDefaults,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.plugins import DoctorProbe, EnginePlugin, PluginResolutionError
from maestro_cli.scheduler import (
    _apply_context_budget,
    _compute_task_hash_safe,
    _enable_win_ansi,
    _estimate_tokens,
    _estimate_workspace_timeout,
    _extract_test_summary,
    _filter_tail_by_intent,
    _load_task_prompt_text,
    _new_cached_result,
    _preflight_checks,
    _write_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(
    tasks: list[TaskSpec],
    name: str = "cov-plan",
    source_path: Path | None = None,
    workspace_root: str | None = None,
    defaults: PlanDefaults | None = None,
) -> PlanSpec:
    return PlanSpec(
        version=1,
        name=name,
        fail_fast=True,
        max_parallel=4,
        defaults=defaults or PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
        workspace_root=workspace_root,
    )


def _result(task_id: str, stdout_tail: str = "", status: str = "success") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,
        exit_code=0,
        stdout_tail=stdout_tail,
        duration_sec=1.0,
    )


def _engine_plugin(name: str, executable: str | None = None, install_hint: str | None = None) -> EnginePlugin:
    return EnginePlugin(
        name=name,
        build_command=lambda ctx: ([name, ctx.prompt_text], False),
        doctor_probe=DoctorProbe(executable=executable or name, install_hint=install_hint),
    )


# ---------------------------------------------------------------------------
# _enable_win_ansi
# ---------------------------------------------------------------------------

class TestEnableWinAnsi:
    def test_non_windows_returns_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On a POSIX-style os.name the function returns immediately."""
        # We replace scheduler.os with a namespace reporting a non-nt name so the
        # guard at the top of the function returns without touching ctypes. This
        # avoids monkeypatching the real os.name (which would corrupt pathlib).
        import maestro_cli.scheduler as sched

        fake_os = types.SimpleNamespace(name="posix")
        monkeypatch.setattr(sched, "os", fake_os)
        assert _enable_win_ansi() is None

    def test_windows_branch_swallows_exception(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When os.name == 'nt' a failure in the ctypes path is swallowed.

        We do NOT monkeypatch the real os.name (that mutates pathlib's behaviour).
        Instead we swap scheduler.os for a namespace whose .name == 'nt' — the only
        attribute the function reads — and force the Windows console call to raise
        so the bare ``except`` handler is exercised on every platform.
        """
        import ctypes

        import maestro_cli.scheduler as sched

        fake_os = types.SimpleNamespace(name="nt")
        monkeypatch.setattr(sched, "os", fake_os)

        class _RaisingWindll:
            def __getattr__(self, _name: str) -> object:
                raise OSError("simulated windows console failure")

        # Replace ctypes.windll so the function's `getattr(ctypes, "windll")` path
        # raises; the function must catch it and return None without propagating.
        monkeypatch.setattr(ctypes, "windll", _RaisingWindll(), raising=False)
        assert _enable_win_ansi() is None


# ---------------------------------------------------------------------------
# _preflight_checks plugin / probe branches
# ---------------------------------------------------------------------------

class TestPreflightProbeBranches:
    def test_plugin_resolution_error_becomes_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A PluginResolutionError is re-raised as ValueError."""
        plan = _make_plan(
            [TaskSpec(id="t1", engine="mystery", prompt="go", command=None)],
            source_path=tmp_path / "plan.yaml",
        )

        def _boom(_name: str) -> EnginePlugin:
            raise PluginResolutionError("no such engine 'mystery'")

        monkeypatch.setattr("maestro_cli.scheduler.get_engine_plugin", _boom)

        with pytest.raises(ValueError, match="mystery"):
            _preflight_checks(plan, plan.tasks, dry_run=False)

    def test_engine_missing_no_probe_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Probe whose executable == engine and no install_hint -> generic error (438-439)."""
        plan = _make_plan(
            [TaskSpec(id="t1", engine="ghost", prompt="go", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        # Probe executable equals engine name, no install hint.
        monkeypatch.setattr(
            "maestro_cli.scheduler.get_engine_plugin",
            lambda name: _engine_plugin(name, executable="ghost", install_hint=None),
        )
        monkeypatch.setattr("maestro_cli.scheduler.shutil.which", lambda _name: None)

        with pytest.raises(ValueError, match=r"Engine 'ghost' not found on PATH"):
            _preflight_checks(plan, plan.tasks, dry_run=False)

    def test_engine_missing_distinct_executable_with_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executable differs from engine + install_hint present."""
        plan = _make_plan(
            [TaskSpec(id="t1", engine="acme", prompt="go", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        monkeypatch.setattr(
            "maestro_cli.scheduler.get_engine_plugin",
            lambda name: _engine_plugin(
                name, executable="acme-cli", install_hint="pip install acme",
            ),
        )
        monkeypatch.setattr("maestro_cli.scheduler.shutil.which", lambda _name: None)

        with pytest.raises(ValueError, match="acme-cli") as exc:
            _preflight_checks(plan, plan.tasks, dry_run=False)
        assert "pip install acme" in str(exc.value)

    def test_engine_missing_distinct_executable_no_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Executable differs from engine, no install_hint -> fallback msg."""
        plan = _make_plan(
            [TaskSpec(id="t1", engine="acme", prompt="go", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        monkeypatch.setattr(
            "maestro_cli.scheduler.get_engine_plugin",
            lambda name: _engine_plugin(name, executable="acme-cli", install_hint=None),
        )
        monkeypatch.setattr("maestro_cli.scheduler.shutil.which", lambda _name: None)

        with pytest.raises(ValueError, match="acme-cli") as exc:
            _preflight_checks(plan, plan.tasks, dry_run=False)
        assert "Install it or check your PATH" in str(exc.value)


# ---------------------------------------------------------------------------
# _extract_test_summary
# ---------------------------------------------------------------------------

class TestExtractTestSummary:
    def test_empty_input_returns_none(self) -> None:
        assert _extract_test_summary("") is None

    def test_no_match_returns_none(self) -> None:
        assert _extract_test_summary("nothing test-like here at all") is None

    def test_pytest_all_categories(self) -> None:
        """Exercise failed/errors/xfailed/xpassed branches (595, 599, 601, 603)."""
        tail = (
            "10 passed, 2 failed, 3 skipped, 1 errors, 4 xfailed, "
            "5 xpassed, 6 warnings in 1.23s"
        )
        out = _extract_test_summary(tail)
        assert out is not None
        assert "(pytest)" in out
        assert "10 passed" in out
        assert "2 failed" in out
        assert "1 errors" in out
        assert "4 xfailed" in out
        assert "5 xpassed" in out
        assert "6 warnings" in out

    def test_jest_with_skipped(self) -> None:
        """Jest summary including a skipped count."""
        tail = "Tests: 1 failed, 2 skipped, 80 passed, 83 total"
        out = _extract_test_summary(tail)
        assert out is not None
        assert "(jest)" in out
        assert "1 failed" in out
        assert "2 skipped" in out
        assert "80 passed" in out
        assert "83 total" in out

    def test_mocha_passing_failing_pending(self) -> None:
        """Mocha branch with failing + pending counts."""
        tail = "94 passing\n  2 failing\n  5 pending"
        out = _extract_test_summary(tail)
        assert out is not None
        assert "(mocha)" in out
        assert "94 passing" in out
        assert "2 failing" in out
        assert "5 pending" in out

    def test_mocha_only_passing(self) -> None:
        """Mocha branch with only a passing count."""
        tail = "12 passing"
        out = _extract_test_summary(tail)
        assert out is not None
        assert "(mocha)" in out
        assert "12 passing" in out


# ---------------------------------------------------------------------------
# _write_summary wave-skip continue
# ---------------------------------------------------------------------------

class TestWriteSummaryWaveSkip:
    def test_wave_with_no_results_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A computed wave referencing tasks absent from results triggers the continue (772)."""
        from datetime import UTC, datetime

        from maestro_cli.models import PlanRunResult

        tasks = [
            TaskSpec(id="a", command="echo a"),
            TaskSpec(id="b", command="echo b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "plan.yaml")

        run_path = tmp_path / "run"
        run_path.mkdir()

        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-cov",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results={
                "a": _result("a", stdout_tail="ok"),
                "b": _result("b", stdout_tail="ok"),
            },
        )

        # Force the wave computation to include a phantom wave that has tasks not
        # present in task_results, so the wave_results list is empty -> continue.
        monkeypatch.setattr(
            "maestro_cli.scheduler._compute_waves",
            lambda _plan, _rr: [["a", "b"], ["ghost-1", "ghost-2"]],
        )

        _write_summary(run_result, plan, run_path)
        summary_text = (run_path / "run_summary.md").read_text(encoding="utf-8")
        # The phantom wave produced no entry; real tasks are still present.
        assert "ghost-1" not in summary_text


# ---------------------------------------------------------------------------
# _load_task_prompt_text OSError fallbacks
# ---------------------------------------------------------------------------

class TestLoadTaskPromptText:
    def test_prompt_file_oserror_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OSError while reading prompt_file yields '' ."""
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("hello", encoding="utf-8")
        plan = _make_plan(
            [TaskSpec(id="t1", engine="claude", prompt_file="prompt.txt", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        task = plan.tasks[0]

        original_read = Path.read_text

        def _boom(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "prompt.txt":
                raise OSError("disk gone")
            return original_read(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _boom)
        assert _load_task_prompt_text(plan, task) == ""

    def test_prompt_md_file_oserror_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OSError while reading prompt_md_file yields '' ."""
        md_path = tmp_path / "prompts.md"
        md_path.write_text("## Heading\n\n```text\nhi\n```\n", encoding="utf-8")
        plan = _make_plan(
            [
                TaskSpec(
                    id="t1",
                    engine="claude",
                    prompt_md_file="prompts.md",
                    prompt_md_heading="Heading",
                    command=None,
                )
            ],
            source_path=tmp_path / "plan.yaml",
        )
        task = plan.tasks[0]

        original_read = Path.read_text

        def _boom(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "prompts.md":
                raise OSError("disk gone")
            return original_read(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _boom)
        assert _load_task_prompt_text(plan, task) == ""

    def test_inline_prompt_passthrough(self, tmp_path: Path) -> None:
        plan = _make_plan(
            [TaskSpec(id="t1", engine="claude", prompt="inline body", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        assert _load_task_prompt_text(plan, plan.tasks[0]) == "inline body"


# ---------------------------------------------------------------------------
# _estimate_workspace_timeout no-prompt + OSError stat
# ---------------------------------------------------------------------------

class TestEstimateWorkspaceTimeout:
    def test_no_prompt_text_returns_none(self, tmp_path: Path) -> None:
        """Engine task with no resolvable prompt text -> None."""
        plan = _make_plan(
            [TaskSpec(id="t1", engine="claude", prompt="", command=None)],
            source_path=tmp_path / "plan.yaml",
            workspace_root=str(tmp_path),
        )
        # prompt is empty string and there is no prompt_file/md, so prompt_text == ""
        assert _estimate_workspace_timeout(plan, plan.tasks[0]) is None

    def test_stat_oserror_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stat OSError on a referenced path is ignored."""
        plan = _make_plan(
            [
                TaskSpec(
                    id="t1",
                    engine="claude",
                    prompt="Please modify src/app/module.py thoroughly",
                    command=None,
                )
            ],
            source_path=tmp_path / "plan.yaml",
            workspace_root=str(tmp_path),
        )
        task = plan.tasks[0]

        def _boom(self: Path) -> object:
            raise OSError("no stat")

        monkeypatch.setattr(Path, "stat", _boom)
        # All referenced paths raise on stat -> total_bytes stays 0 -> None.
        assert _estimate_workspace_timeout(plan, task) is None


# ---------------------------------------------------------------------------
# _filter_tail_by_intent empty-sections early return
# ---------------------------------------------------------------------------

class TestFilterTailByIntentEmptySections:
    def test_whitespace_only_tail_returns_original(self) -> None:
        """A truthy-but-blank tail splits to no sections."""
        tail = "\n   \n\t\n"  # truthy, but no non-empty lines
        out_tail, score, matched = _filter_tail_by_intent(tail, {"api"})
        assert out_tail == tail
        assert score == 0
        assert matched == []


# ---------------------------------------------------------------------------
# _apply_context_budget keep-original branch
# ---------------------------------------------------------------------------

class TestApplyContextBudgetKeepOriginal:
    def test_intent_filter_keeps_full_match_unchanged(self) -> None:
        """When intent filtering brings total under budget, an upstream whose tail
        fully matches the intent is kept unchanged."""
        # Upstream "keep" fully matches the intent keywords (single section).
        keep_text = "alpha beta gamma delta"
        # Upstream "drop" has a relevant section plus a large irrelevant section
        # so filtering shortens it and brings the prepared total under budget.
        drop_text = (
            "alpha beta gamma delta\n\n"
            + ("zulu yankee xray whiskey " * 40)
        )
        upstream = {
            "keep": _result("keep", stdout_tail=keep_text),
            "drop": _result("drop", stdout_tail=drop_text),
        }
        # Budget below the raw total but above the filtered total.
        raw_total = _estimate_tokens(keep_text) + _estimate_tokens(drop_text)
        budget = _estimate_tokens(keep_text) + 30
        assert budget < raw_total

        result, records, meta = _apply_context_budget(
            upstream,
            budget_tokens=budget,
            intent_keywords={"alpha", "beta", "gamma", "delta"},
        )
        # "keep" fully matched -> not shortened -> original object preserved.
        assert result["keep"] is upstream["keep"]
        # "drop" should be present in the result map regardless.
        assert "drop" in result

    def test_within_budget_with_intent_returns_meta(self) -> None:
        """Total already under budget but intent keywords present -> meta path."""
        upstream = {"a": _result("a", stdout_tail="alpha beta")}
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=10_000, intent_keywords={"alpha"},
        )
        assert result is upstream
        assert records == []


# ---------------------------------------------------------------------------
# _compute_task_hash_safe exception path
# ---------------------------------------------------------------------------

class TestComputeTaskHashSafe:
    def test_exception_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = _make_plan(
            [TaskSpec(id="t1", command="echo hi")],
            source_path=tmp_path / "plan.yaml",
        )
        task = plan.tasks[0]

        def _boom(*_a: object, **_k: object) -> str:
            raise RuntimeError("hash failure")

        monkeypatch.setattr("maestro_cli.scheduler.compute_task_hash", _boom)
        assert _compute_task_hash_safe(task, plan, {}) is None

    def test_success_returns_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = _make_plan(
            [TaskSpec(id="t1", command="echo hi")],
            source_path=tmp_path / "plan.yaml",
        )
        task = plan.tasks[0]
        monkeypatch.setattr(
            "maestro_cli.scheduler.compute_task_hash",
            lambda *_a, **_k: "deadbeef",
        )
        assert _compute_task_hash_safe(task, plan, {}) == "deadbeef"


# ---------------------------------------------------------------------------
# _new_cached_result produced_contract reconstruction
# ---------------------------------------------------------------------------

class TestNewCachedResultContract:
    def test_produced_contract_restored(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        cache_dir = tmp_path / "cache"
        task_hash = "0123456789abcdef"

        result = _new_cached_result(
            task_id="cached-task",
            run_path=run_path,
            cached={
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                "command": "echo cached",
                "stdout_tail": "cached out",
                "produced_contract": {
                    "producer_task_id": "cached-task",
                    "contract_type": "sql-schema",
                    "summary": "users table",
                    "body": "CREATE TABLE users (...);",
                    "content_hash": "abc123",
                    "metadata": {"table_count": 1},
                },
            },
            task_hash=task_hash,
            cache_dir=cache_dir,
        )

        assert result.produced_contract is not None
        assert result.produced_contract.contract_type == "sql-schema"
        assert result.produced_contract.summary == "users table"
        assert result.produced_contract.metadata == {"table_count": 1}
