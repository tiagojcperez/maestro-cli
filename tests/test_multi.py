from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.errors import TaskExecutionError
from maestro_cli.models import MultiPlanResult, PlanRunResult, TaskResult
from maestro_cli.multi import (
    _aggregate_results,
    _new_plan_result,
    _run_parallel,
    _run_sequential,
    _write_multi_summary,
    run_multi_plan,
)


def _make_plan_run_result(
    *,
    plan_name: str = "test-plan",
    success: bool = True,
    total_cost_usd: float | None = None,
    total_tokens: int | None = None,
    budget_exceeded: bool = False,
) -> PlanRunResult:
    now = datetime.now(UTC)
    return PlanRunResult(
        plan_name=plan_name,
        run_id=f"run-{plan_name}",
        run_path=Path("/tmp/runs"),
        started_at=now,
        finished_at=now,
        success=success,
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        budget_exceeded=budget_exceeded,
    )


def _make_plan_spec(name: str = "test-plan") -> MagicMock:
    spec = MagicMock()
    spec.name = name
    return spec


class TestMulti:
    def test_aggregate_results_empty(self) -> None:
        result = _aggregate_results([])
        assert result.success is True
        assert result.total_cost_usd is None
        assert result.total_tokens is None
        assert result.budget_exceeded is False
        assert result.plan_results == []

    def test_aggregate_results_single(self) -> None:
        r = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=1.5, total_tokens=100)
        result = _aggregate_results([r])
        assert result.success is True
        assert result.total_cost_usd == pytest.approx(1.5)
        assert result.total_tokens == 100
        assert len(result.plan_results) == 1

    def test_aggregate_results_mixed(self) -> None:
        r1 = _make_plan_run_result(plan_name="plan-a", success=True)
        r2 = _make_plan_run_result(plan_name="plan-b", success=False)
        result = _aggregate_results([r1, r2])
        assert result.success is False

    def test_aggregate_results_costs(self) -> None:
        r1 = _make_plan_run_result(total_cost_usd=1.0, total_tokens=100)
        r2 = _make_plan_run_result(total_cost_usd=2.5, total_tokens=300)
        result = _aggregate_results([r1, r2])
        assert result.total_cost_usd == pytest.approx(3.5)
        assert result.total_tokens == 400

    def test_run_sequential_single_plan(self, tmp_path: Path) -> None:
        plan = _make_plan_spec("single-plan")
        run_result = _make_plan_run_result(plan_name="single-plan", success=True, total_cost_usd=0.5)

        with patch("maestro_cli.multi.run_plan", return_value=run_result) as mock_run:
            result = _run_sequential(
                [plan],
                [str(tmp_path / "plan.yaml")],
            )

        mock_run.assert_called_once()
        assert result.success is True
        assert len(result.plan_results) == 1

    def test_run_sequential_budget_exceeded(self, tmp_path: Path) -> None:
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")

        # First plan costs $2 — exhausts the $1.50 budget
        run_result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=2.0)

        with patch("maestro_cli.multi.run_plan", return_value=run_result_a) as mock_run:
            result = _run_sequential(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
                max_cost_usd=1.50,
            )

        # run_plan only called once (second plan skipped)
        mock_run.assert_called_once()
        assert result.budget_exceeded is True
        assert len(result.plan_results) == 2
        # Second plan result should indicate it was skipped
        assert result.plan_results[1].success is False

    def test_multi_plan_result_dataclass(self) -> None:
        r = MultiPlanResult()
        assert r.plan_results == []
        assert r.total_cost_usd is None
        assert r.total_tokens is None
        assert r.budget_exceeded is False
        assert r.success is True

        d = r.to_dict()
        assert "plan_results" in d
        assert "success" in d
        assert "budget_exceeded" in d
        assert "total_cost_usd" in d
        assert "total_tokens" in d
        assert "started_at" in d
        assert "finished_at" in d

    def test_write_multi_summary(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="alpha",
            run_id="run-alpha",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=True,
            total_cost_usd=0.10,
            total_tokens=500,
        )
        multi = MultiPlanResult(
            plan_results=[r1],
            total_cost_usd=0.10,
            total_tokens=500,
            success=True,
            started_at=now,
            finished_at=now,
        )

        _write_multi_summary(multi, tmp_path)

        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "Multi-Plan Run Summary" in summary
        assert "alpha" in summary
        assert "$0.10" in summary
        assert "500" in summary

    def test_run_multi_plan_dispatches_sequential(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )

        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._run_parallel") as mock_par,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], parallel=False)

        mock_seq.assert_called_once()
        mock_par.assert_not_called()

    def test_aggregate_results_all_none_costs(self) -> None:
        r1 = _make_plan_run_result(plan_name="plan-a", total_cost_usd=None, total_tokens=None)
        r2 = _make_plan_run_result(plan_name="plan-b", total_cost_usd=None, total_tokens=None)
        result = _aggregate_results([r1, r2])
        assert result.total_cost_usd is None
        assert result.total_tokens is None

    def test_aggregate_results_mixed_none_and_numeric_costs(self) -> None:
        r1 = _make_plan_run_result(plan_name="plan-a", total_cost_usd=None, total_tokens=None)
        r2 = _make_plan_run_result(plan_name="plan-b", total_cost_usd=1.0, total_tokens=200)
        result = _aggregate_results([r1, r2])
        assert result.total_cost_usd == pytest.approx(1.0)
        assert result.total_tokens == 200

    def test_run_sequential_exception_captured_as_failed(self, tmp_path: Path) -> None:
        plan = _make_plan_spec("failing-plan")

        with patch("maestro_cli.multi.run_plan", side_effect=TaskExecutionError("boom")):
            result = _run_sequential(
                [plan],
                [str(tmp_path / "plan.yaml")],
            )

        assert len(result.plan_results) == 1
        assert result.plan_results[0].success is False
        assert "boom" in result.plan_results[0].task_results[next(iter(result.plan_results[0].task_results))].message

    def test_run_parallel_basic_execution(self, tmp_path: Path) -> None:
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=0.5, total_tokens=100)
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=1.0, total_tokens=200)

        call_count = 0

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return result_a if plan.name == "plan-a" else result_b

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
            )

        assert call_count == 2
        assert result.success is True
        assert result.total_cost_usd == pytest.approx(1.5)
        assert result.total_tokens == 300
        assert len(result.plan_results) == 2

    def test_run_multi_plan_dispatches_parallel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )

        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._run_parallel") as mock_par,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_par.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], parallel=True)

        mock_par.assert_called_once()
        mock_seq.assert_not_called()

    def test_aggregate_results_budget_exceeded_from_any_plan(self) -> None:
        r1 = _make_plan_run_result(plan_name="plan-a", success=True, budget_exceeded=False)
        r2 = _make_plan_run_result(plan_name="plan-b", success=True, budget_exceeded=True)
        result = _aggregate_results([r1, r2])
        assert result.budget_exceeded is True

    def test_aggregate_results_started_finished_from_min_max(self) -> None:
        now = datetime.now(UTC)
        early = PlanRunResult(
            plan_name="early",
            run_id="run-early",
            run_path=Path("/tmp/runs"),
            started_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
            finished_at=datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC),
            success=True,
        )
        late = PlanRunResult(
            plan_name="late",
            run_id="run-late",
            run_path=Path("/tmp/runs"),
            started_at=datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
            finished_at=datetime(2024, 1, 1, 11, 30, 0, tzinfo=UTC),
            success=True,
        )
        result = _aggregate_results([early, late])
        assert result.started_at == datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        assert result.finished_at == datetime(2024, 1, 1, 11, 30, 0, tzinfo=UTC)

    def test_run_parallel_exception_captured_as_failed(self, tmp_path: Path) -> None:
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=0.5)

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            if plan.name == "plan-a":
                raise TaskExecutionError("parallel boom")
            return result_b

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
            )

        assert len(result.plan_results) == 2
        # First plan should be failed, second should succeed
        failed = next(r for r in result.plan_results if r.plan_name == "plan-a")
        assert failed.success is False
        assert result.success is False

    def test_write_multi_summary_budget_exceeded_in_status(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="beta",
            run_id="run-beta",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            total_cost_usd=None,
            total_tokens=None,
            budget_exceeded=True,
        )
        multi = MultiPlanResult(
            plan_results=[r1],
            total_cost_usd=None,
            total_tokens=None,
            success=False,
            budget_exceeded=True,
            started_at=now,
            finished_at=now,
        )

        _write_multi_summary(multi, tmp_path)

        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "budget exceeded" in summary
        assert "---" in summary  # None cost/tokens rendered as "---"

    def test_run_multi_plan_load_error_creates_failed_result(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.errors import PlanValidationError

        plan_yaml = tmp_path / "bad.yaml"
        plan_yaml.write_text("invalid: yaml: content", encoding="utf-8")

        with (
            patch("maestro_cli.multi.load_plan", side_effect=PlanValidationError("bad schema")),
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_seq.return_value = MultiPlanResult(plan_results=[])
            result = run_multi_plan([str(plan_yaml)])

        captured = capsys.readouterr()
        assert "failed to load plan" in captured.out
        assert result.success is False

    def test_run_sequential_multiple_plans_all_succeed(self, tmp_path: Path) -> None:
        plans = [_make_plan_spec(f"plan-{i}") for i in range(3)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(3)]
        results = [_make_plan_run_result(plan_name=f"plan-{i}", success=True, total_cost_usd=0.1) for i in range(3)]

        call_order: list[str] = []

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            call_order.append(plan.name)
            return results[int(plan.name.split("-")[1])]

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_sequential(plans, plan_paths)

        assert len(call_order) == 3
        assert call_order == ["plan-0", "plan-1", "plan-2"]
        assert result.success is True
        assert result.total_cost_usd == pytest.approx(0.3)

    def test_run_multi_plan_summary_write_failure_prints_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )

        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary", side_effect=OSError("disk full")),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            result = run_multi_plan([str(plan_yaml)])

        captured = capsys.readouterr()
        assert "failed to write multi summary" in captured.out
        # Run still succeeds despite summary write failure
        assert result is not None

    def test_run_parallel_budget_exceeded(self, tmp_path: Path) -> None:
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=1.0)
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=1.0)

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            return result_a if plan.name == "plan-a" else result_b

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
                max_cost_usd=1.5,
            )

        # Total cost $2 > $1.5 budget → budget_exceeded
        assert result.budget_exceeded is True
        assert result.total_cost_usd == pytest.approx(2.0)

    def test_new_plan_result_task_id_format(self, tmp_path: Path) -> None:
        result = _new_plan_result(
            plan_name="my plan",
            run_path=tmp_path,
            success=False,
            message="load failed",
            status="failed",
        )
        # task_id must be "{sanitized_name}:multi"
        task_ids = list(result.task_results.keys())
        assert len(task_ids) == 1
        assert task_ids[0].endswith(":multi")
        assert result.success is False
        assert result.total_cost_usd is None
        assert result.total_tokens is None
        assert result.budget_exceeded is False

    def test_run_multi_plan_invalid_path_creates_failed_result(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # resolve_path returning None triggers the "invalid plan path" branch
        with (
            patch("maestro_cli.multi.resolve_path", return_value=None),
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_seq.return_value = MultiPlanResult(plan_results=[])
            result = run_multi_plan(["some/bad/path.yaml"])

        captured = capsys.readouterr()
        assert "invalid plan path" in captured.out
        assert result.success is False

    def test_run_multi_plan_max_cost_triggers_budget_exceeded(self, tmp_path: Path) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )
        # run_sequential returns a result with cost above max_cost_usd
        expensive_result = _make_plan_run_result(plan_name="demo", success=True, total_cost_usd=5.0)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(
                plan_results=[expensive_result],
                total_cost_usd=5.0,
                success=True,
            )

            result = run_multi_plan([str(plan_yaml)], max_cost_usd=2.0)

        assert result.budget_exceeded is True

    def test_run_sequential_budget_exactly_exhausted_skips_second(self, tmp_path: Path) -> None:
        """remaining_budget reaches exactly 0.0 after first plan → second plan is skipped."""
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        # First plan costs exactly the full budget
        run_result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=1.5)

        with patch("maestro_cli.multi.run_plan", return_value=run_result_a) as mock_run:
            result = _run_sequential(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
                max_cost_usd=1.5,
            )

        mock_run.assert_called_once()
        assert result.budget_exceeded is True
        assert len(result.plan_results) == 2
        # Second result must be a synthetic skipped entry
        second = result.plan_results[1]
        assert second.success is False
        task_msg = next(iter(second.task_results.values())).message
        assert "skipped" in task_msg

    def test_run_parallel_results_ordered_by_original_index(self, tmp_path: Path) -> None:
        """_run_parallel must return results in original plan order, not completion order."""
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        plan_c = _make_plan_spec("plan-c")
        result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=0.1)
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=0.2)
        result_c = _make_plan_run_result(plan_name="plan-c", success=True, total_cost_usd=0.3)

        name_to_result = {"plan-a": result_a, "plan-b": result_b, "plan-c": result_c}

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            return name_to_result[plan.name]

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan_a, plan_b, plan_c],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"), str(tmp_path / "c.yaml")],
            )

        assert len(result.plan_results) == 3
        assert result.plan_results[0].plan_name == "plan-a"
        assert result.plan_results[1].plan_name == "plan-b"
        assert result.plan_results[2].plan_name == "plan-c"

    def test_run_sequential_oserror_captured_as_failed(self, tmp_path: Path) -> None:
        plan = _make_plan_spec("oserror-plan")

        with patch("maestro_cli.multi.run_plan", side_effect=OSError("disk error")):
            result = _run_sequential(
                [plan],
                [str(tmp_path / "plan.yaml")],
            )

        assert len(result.plan_results) == 1
        assert result.plan_results[0].success is False
        task_msg = next(iter(result.plan_results[0].task_results.values())).message
        assert "disk error" in task_msg

    def test_run_parallel_budget_warning_printed_once(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Budget warning is printed exactly once even when multiple plans exceed budget."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(3)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(3)]

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            return _make_plan_run_result(plan_name=plan.name, success=True, total_cost_usd=2.0)

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                plans,
                plan_paths,
                max_cost_usd=1.0,
            )

        captured = capsys.readouterr()
        assert result.budget_exceeded is True
        # Warning should appear exactly once
        assert captured.out.count("budget exceeded") == 1

    def test_new_plan_result_message_stored_in_task_result(self, tmp_path: Path) -> None:
        """_new_plan_result stores the message in the single TaskResult entry."""
        result = _new_plan_result(
            plan_name="failed-plan",
            run_path=tmp_path,
            success=False,
            message="plan load failed: bad schema",
            status="failed",
        )

        assert len(result.task_results) == 1
        task = next(iter(result.task_results.values()))
        assert task.message == "plan load failed: bad schema"
        assert task.status == "failed"
        assert result.plan_name == "failed-plan"

    def test_run_sequential_plan_validation_error_captured(self, tmp_path: Path) -> None:
        """PlanValidationError raised by run_plan is captured as a failed result."""
        from maestro_cli.errors import PlanValidationError

        plan = _make_plan_spec("validation-error-plan")

        with patch("maestro_cli.multi.run_plan", side_effect=PlanValidationError("schema violation")):
            result = _run_sequential(
                [plan],
                [str(tmp_path / "plan.yaml")],
            )

        assert len(result.plan_results) == 1
        assert result.plan_results[0].success is False
        task_msg = next(iter(result.plan_results[0].task_results.values())).message
        assert "schema violation" in task_msg

    def test_run_multi_plan_event_callback_forwarded(self, tmp_path: Path) -> None:
        """event_callback is forwarded to _run_sequential."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)
        callback = MagicMock()

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], parallel=False, event_callback=callback)

        _call_kwargs = mock_seq.call_args.kwargs
        assert _call_kwargs["event_callback"] is callback

    def test_run_multi_plan_mixed_preload_failure_and_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Preload failures and successful runs are combined in the final result."""
        from maestro_cli.errors import PlanValidationError

        good_yaml = tmp_path / "good.yaml"
        good_yaml.write_text(
            "version: 1\nname: good\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml", encoding="utf-8")

        good_result = _make_plan_run_result(plan_name="good", success=True, total_cost_usd=0.5)

        def fake_load(path):  # type: ignore[no-untyped-def]
            if "bad" in str(path):
                raise PlanValidationError("bad schema")
            return _make_plan_spec("good")

        with (
            patch("maestro_cli.multi.load_plan", side_effect=fake_load),
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_seq.return_value = MultiPlanResult(
                plan_results=[good_result],
                total_cost_usd=0.5,
                success=True,
            )
            result = run_multi_plan([str(bad_yaml), str(good_yaml)])

        # Final result has both: preload failure + successful run
        assert result.success is False  # one failed → overall fails
        captured = capsys.readouterr()
        assert "failed to load plan" in captured.out

    def test_run_sequential_budget_zero_skips_all_plans(self, tmp_path: Path) -> None:
        """When max_cost_usd=0.0, remaining_budget starts at 0 → all plans skipped immediately."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(2)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(2)]

        with patch("maestro_cli.multi.run_plan") as mock_run:
            result = _run_sequential(plans, plan_paths, max_cost_usd=0.0)

        mock_run.assert_not_called()
        assert result.budget_exceeded is True
        assert len(result.plan_results) == 2
        for r in result.plan_results:
            assert r.success is False
            task_msg = next(iter(r.task_results.values())).message
            assert "skipped" in task_msg

    def test_run_multi_plan_no_plans(self, tmp_path: Path) -> None:
        """Empty plan_paths list → no plans run, result is success with no plan_results."""
        with patch("maestro_cli.multi._write_multi_summary"):
            result = run_multi_plan([])

        assert result.plan_results == []
        assert result.success is True
        assert result.total_cost_usd is None

    def test_run_parallel_value_error_captured_as_failed(self, tmp_path: Path) -> None:
        """ValueError raised by run_plan in parallel worker is captured as a failed result."""
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        result_b = _make_plan_run_result(plan_name="plan-b", success=True)

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            if plan.name == "plan-a":
                raise ValueError("unexpected value")
            return result_b

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
            )

        assert len(result.plan_results) == 2
        failed = next(r for r in result.plan_results if r.plan_name == "plan-a")
        assert failed.success is False
        task_msg = next(iter(failed.task_results.values())).message
        assert "unexpected value" in task_msg

    def test_run_sequential_no_budget_runs_all_plans(self, tmp_path: Path) -> None:
        """When max_cost_usd is None, all plans run regardless of accumulated cost."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(3)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(3)]
        # Each plan costs $100 — would exceed any finite budget
        results = [
            _make_plan_run_result(plan_name=f"plan-{i}", success=True, total_cost_usd=100.0)
            for i in range(3)
        ]

        call_order: list[str] = []

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            call_order.append(plan.name)
            return results[int(plan.name.split("-")[1])]

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_sequential(plans, plan_paths, max_cost_usd=None)

        assert len(call_order) == 3
        assert result.budget_exceeded is False
        assert result.total_cost_usd == pytest.approx(300.0)

    def test_run_multi_plan_all_preload_failures_skip_runner(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When every plan fails to load, _run_sequential is never called and result is all-failed."""
        from maestro_cli.errors import PlanValidationError

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: yaml", encoding="utf-8")
        bad_yaml2 = tmp_path / "bad2.yaml"
        bad_yaml2.write_text("also invalid", encoding="utf-8")

        with (
            patch("maestro_cli.multi.load_plan", side_effect=PlanValidationError("no good")),
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._run_parallel") as mock_par,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            result = run_multi_plan([str(bad_yaml), str(bad_yaml2)])

        mock_seq.assert_not_called()
        mock_par.assert_not_called()
        assert result.success is False
        assert len(result.plan_results) == 2
        captured = capsys.readouterr()
        assert captured.out.count("failed to load plan") == 2

    def test_write_multi_summary_per_row_none_cost_shows_dashes(self, tmp_path: Path) -> None:
        """Individual plan row with None cost/tokens renders '---' in those cells."""
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="cheap",
            run_id="run-cheap",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=True,
            total_cost_usd=0.05,
            total_tokens=50,
        )
        r2 = PlanRunResult(
            plan_name="unknown-cost",
            run_id="run-unknown",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=True,
            total_cost_usd=None,
            total_tokens=None,
        )
        multi = MultiPlanResult(
            plan_results=[r1, r2],
            total_cost_usd=0.05,
            total_tokens=50,
            success=True,
            started_at=now,
            finished_at=now,
        )

        _write_multi_summary(multi, tmp_path)

        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "unknown-cost" in summary
        assert "$0.05" in summary
        # The None-cost plan row must have "---" (not crash)
        assert "---" in summary

    def test_run_parallel_plan_validation_error_captured(self, tmp_path: Path) -> None:
        """PlanValidationError raised by run_plan in parallel worker is captured as failed."""
        from maestro_cli.errors import PlanValidationError

        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=0.3)

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            if plan.name == "plan-a":
                raise PlanValidationError("validation error in parallel")
            return result_b

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
            )

        assert len(result.plan_results) == 2
        failed = next(r for r in result.plan_results if r.plan_name == "plan-a")
        assert failed.success is False
        task_msg = next(iter(failed.task_results.values())).message
        assert "validation error in parallel" in task_msg
        # plan-b still succeeds
        succeeded = next(r for r in result.plan_results if r.plan_name == "plan-b")
        assert succeeded.success is True

    def test_run_sequential_none_cost_plan_does_not_deplete_budget(self, tmp_path: Path) -> None:
        """A plan returning None cost is treated as 0.0 — budget stays intact."""
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        # plan-a has no cost (dry run, shell command, etc.)
        result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=None)
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=0.5)

        results_map = {"plan-a": result_a, "plan-b": result_b}

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            return results_map[plan.name]

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan) as mock_run:
            result = _run_sequential(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
                max_cost_usd=1.0,
            )

        # Both plans must run — None cost didn't eat the $1.00 budget
        assert mock_run.call_count == 2
        assert result.budget_exceeded is False
        assert result.total_cost_usd == pytest.approx(0.5)

    def test_write_multi_summary_success_with_budget_exceeded_status(self, tmp_path: Path) -> None:
        """A plan row with success=True and budget_exceeded=True shows 'success (budget exceeded)'."""
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="partial",
            run_id="run-partial",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=True,
            total_cost_usd=2.0,
            total_tokens=200,
            budget_exceeded=True,
        )
        multi = MultiPlanResult(
            plan_results=[r1],
            total_cost_usd=2.0,
            total_tokens=200,
            success=True,
            budget_exceeded=True,
            started_at=now,
            finished_at=now,
        )

        _write_multi_summary(multi, tmp_path)

        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "success (budget exceeded)" in summary

    def test_write_multi_summary_total_row_mixed_plans(self, tmp_path: Path) -> None:
        """TOTAL row in summary reflects aggregated cost/tokens across multiple plans."""
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="alpha",
            run_id="run-alpha",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=True,
            total_cost_usd=0.30,
            total_tokens=300,
        )
        r2 = PlanRunResult(
            plan_name="beta",
            run_id="run-beta",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            total_cost_usd=0.70,
            total_tokens=700,
        )
        multi = MultiPlanResult(
            plan_results=[r1, r2],
            total_cost_usd=1.00,
            total_tokens=1000,
            success=False,
            started_at=now,
            finished_at=now,
        )

        _write_multi_summary(multi, tmp_path)

        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "TOTAL" in summary
        assert "$1.00" in summary
        assert "1,000" in summary
        assert "alpha" in summary
        assert "beta" in summary
        # failed plan should show "failed" status
        assert "failed" in summary

    def test_run_parallel_no_budget_all_succeed(self, tmp_path: Path) -> None:
        """With max_cost_usd=None, parallel runner never sets budget_exceeded regardless of cost."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(3)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(3)]

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            return _make_plan_run_result(plan_name=plan.name, success=True, total_cost_usd=999.0)

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(plans, plan_paths, max_cost_usd=None)

        assert result.budget_exceeded is False
        assert result.success is True
        assert len(result.plan_results) == 3
        assert result.total_cost_usd == pytest.approx(2997.0)

    def test_run_multi_plan_oserror_on_load_creates_failed_result(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """OSError raised by load_plan is captured as a preload failure result."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text("version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n", encoding="utf-8")

        with (
            patch("maestro_cli.multi.load_plan", side_effect=OSError("permission denied")),
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_seq.return_value = MultiPlanResult(plan_results=[])
            result = run_multi_plan([str(plan_yaml)])

        captured = capsys.readouterr()
        assert "failed to load plan" in captured.out
        assert "permission denied" in captured.out
        assert result.success is False

    def test_run_sequential_value_error_captured_as_failed(self, tmp_path: Path) -> None:
        """ValueError raised by run_plan in sequential runner is captured as a failed result."""
        plan = _make_plan_spec("value-error-plan")

        with patch("maestro_cli.multi.run_plan", side_effect=ValueError("bad value")):
            result = _run_sequential(
                [plan],
                [str(tmp_path / "plan.yaml")],
            )

        assert len(result.plan_results) == 1
        assert result.plan_results[0].success is False
        task_msg = next(iter(result.plan_results[0].task_results.values())).message
        assert "bad value" in task_msg

    def test_run_sequential_single_plan_exceeds_budget(self, tmp_path: Path) -> None:
        """Single plan that exceeds the budget sets budget_exceeded=True (no plans left to skip)."""
        plan_a = _make_plan_spec("plan-a")
        result_a = _make_plan_run_result(plan_name="plan-a", success=True, total_cost_usd=2.0)

        with patch("maestro_cli.multi.run_plan", return_value=result_a) as mock_run:
            result = _run_sequential(
                [plan_a],
                [str(tmp_path / "a.yaml")],
                max_cost_usd=0.5,
            )

        # Plan still runs (budget check is BEFORE running, and at start it was still > 0)
        mock_run.assert_called_once()
        assert result.budget_exceeded is True
        assert len(result.plan_results) == 1
        assert result.plan_results[0].success is True

    def test_run_multi_plan_event_callback_forwarded_to_parallel(self, tmp_path: Path) -> None:
        """event_callback is forwarded to _run_parallel when parallel=True."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo hi\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)
        callback = MagicMock()

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_parallel") as mock_par,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_par.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], parallel=True, event_callback=callback)

        _call_kwargs = mock_par.call_args.kwargs
        assert _call_kwargs["event_callback"] is callback

    def test_write_multi_summary_empty_plan_results(self, tmp_path: Path) -> None:
        """Summary with no plan rows still produces valid markdown with TOTAL row."""
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        multi = MultiPlanResult(
            plan_results=[],
            total_cost_usd=None,
            total_tokens=None,
            success=True,
            started_at=now,
            finished_at=now,
        )

        _write_multi_summary(multi, tmp_path)

        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "Multi-Plan Run Summary" in summary
        assert "TOTAL" in summary
        assert "Plans | 0" in summary


# ---------------------------------------------------------------------------
# Additional tests: run_multi_plan integration
# ---------------------------------------------------------------------------


class TestRunMultiPlanAdditional:
    """Additional tests for run_multi_plan top-level function."""

    def test_single_plan_degenerates_to_normal_run(self, tmp_path: Path) -> None:
        """Single plan passed to run_multi_plan invokes _run_sequential with one plan."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: solo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        result_solo = _make_plan_run_result(plan_name="solo", success=True, total_cost_usd=0.1, total_tokens=50)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("solo")
            mock_seq.return_value = MultiPlanResult(
                plan_results=[result_solo], total_cost_usd=0.1, total_tokens=50, success=True,
            )
            result = run_multi_plan([str(plan_yaml)])

        mock_seq.assert_called_once()
        assert len(result.plan_results) == 1
        assert result.success is True

    def test_execution_profile_forwarded_to_sequential(self, tmp_path: Path) -> None:
        """execution_profile parameter is forwarded to _run_sequential."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], execution_profile="yolo")

        assert mock_seq.call_args.kwargs["execution_profile"] == "yolo"

    def test_execution_profile_forwarded_to_parallel(self, tmp_path: Path) -> None:
        """execution_profile parameter is forwarded to _run_parallel."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_parallel") as mock_par,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_par.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], parallel=True, execution_profile="safe")

        assert mock_par.call_args.kwargs["execution_profile"] == "safe"

    def test_dry_run_forwarded(self, tmp_path: Path) -> None:
        """dry_run parameter is forwarded correctly."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], dry_run=True)

        assert mock_seq.call_args.kwargs["dry_run"] is True

    def test_auto_approve_forwarded(self, tmp_path: Path) -> None:
        """auto_approve parameter is forwarded correctly."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            run_multi_plan([str(plan_yaml)], auto_approve=True)

        assert mock_seq.call_args.kwargs["auto_approve"] is True

    def test_max_cost_forwarded_to_sequential(self, tmp_path: Path) -> None:
        """max_cost_usd is forwarded to _run_sequential."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True, total_cost_usd=0.5)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(
                plan_results=[fake_result], total_cost_usd=0.5, success=True,
            )

            run_multi_plan([str(plan_yaml)], max_cost_usd=5.0)

        assert mock_seq.call_args.kwargs["max_cost_usd"] == 5.0

    def test_started_at_and_finished_at_set(self, tmp_path: Path) -> None:
        """Final result has started_at and finished_at set."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: demo\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        fake_result = _make_plan_run_result(plan_name="demo", success=True)

        with (
            patch("maestro_cli.multi.load_plan") as mock_load,
            patch("maestro_cli.multi._run_sequential") as mock_seq,
            patch("maestro_cli.multi._write_multi_summary"),
        ):
            mock_load.return_value = _make_plan_spec("demo")
            mock_seq.return_value = MultiPlanResult(plan_results=[fake_result])

            result = run_multi_plan([str(plan_yaml)])

        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at


# ---------------------------------------------------------------------------
# Additional tests: _aggregate_results edge cases
# ---------------------------------------------------------------------------


class TestAggregateResultsAdditional:
    """Additional tests for _aggregate_results."""

    def test_aggregate_preserves_order(self) -> None:
        """Plan results are preserved in the same order they were passed."""
        results = [
            _make_plan_run_result(plan_name=f"plan-{i}", success=True, total_cost_usd=float(i))
            for i in range(5)
        ]
        aggregated = _aggregate_results(results)
        assert [r.plan_name for r in aggregated.plan_results] == [f"plan-{i}" for i in range(5)]

    def test_aggregate_single_failure_overall_fails(self) -> None:
        """Even one failed plan makes overall success False."""
        results = [
            _make_plan_run_result(plan_name="good-1", success=True),
            _make_plan_run_result(plan_name="good-2", success=True),
            _make_plan_run_result(plan_name="bad", success=False),
            _make_plan_run_result(plan_name="good-3", success=True),
        ]
        aggregated = _aggregate_results(results)
        assert aggregated.success is False

    def test_aggregate_all_success(self) -> None:
        """All successful plans means overall success."""
        results = [
            _make_plan_run_result(plan_name=f"plan-{i}", success=True)
            for i in range(3)
        ]
        aggregated = _aggregate_results(results)
        assert aggregated.success is True

    def test_aggregate_tokens_accumulation(self) -> None:
        """Token counts accumulate across plans."""
        results = [
            _make_plan_run_result(plan_name="p1", total_tokens=100),
            _make_plan_run_result(plan_name="p2", total_tokens=250),
            _make_plan_run_result(plan_name="p3", total_tokens=50),
        ]
        aggregated = _aggregate_results(results)
        assert aggregated.total_tokens == 400

    def test_aggregate_partial_none_tokens(self) -> None:
        """Plans with None tokens don't break token accumulation."""
        results = [
            _make_plan_run_result(plan_name="p1", total_tokens=100),
            _make_plan_run_result(plan_name="p2", total_tokens=None),
            _make_plan_run_result(plan_name="p3", total_tokens=200),
        ]
        aggregated = _aggregate_results(results)
        assert aggregated.total_tokens == 300

    def test_aggregate_partial_none_costs(self) -> None:
        """Plans with None costs don't break cost accumulation."""
        results = [
            _make_plan_run_result(plan_name="p1", total_cost_usd=1.0),
            _make_plan_run_result(plan_name="p2", total_cost_usd=None),
            _make_plan_run_result(plan_name="p3", total_cost_usd=2.0),
        ]
        aggregated = _aggregate_results(results)
        assert aggregated.total_cost_usd == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Additional tests: _run_sequential
# ---------------------------------------------------------------------------


class TestRunSequentialAdditional:
    """Additional tests for _run_sequential."""

    def test_first_plan_fails_second_still_runs(self, tmp_path: Path) -> None:
        """No fail_fast between plans: first plan failure doesn't skip second."""
        plan_a = _make_plan_spec("plan-a")
        plan_b = _make_plan_spec("plan-b")
        result_a = _make_plan_run_result(plan_name="plan-a", success=False, total_cost_usd=0.5)
        result_b = _make_plan_run_result(plan_name="plan-b", success=True, total_cost_usd=0.3)

        call_order: list[str] = []

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            call_order.append(plan.name)
            return result_a if plan.name == "plan-a" else result_b

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_sequential(
                [plan_a, plan_b],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
            )

        # Both plans ran
        assert call_order == ["plan-a", "plan-b"]
        assert len(result.plan_results) == 2
        assert result.success is False  # overall fails because plan-a failed
        assert result.plan_results[0].success is False
        assert result.plan_results[1].success is True

    def test_sequential_cost_accumulation(self, tmp_path: Path) -> None:
        """Costs accumulate correctly across sequential plans."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(3)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(3)]
        costs = [0.1, 0.2, 0.3]

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            idx = int(plan.name.split("-")[1])
            return _make_plan_run_result(
                plan_name=plan.name, success=True, total_cost_usd=costs[idx]
            )

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_sequential(plans, plan_paths)

        assert result.total_cost_usd == pytest.approx(0.6)

    def test_sequential_budget_multiple_skips(self, tmp_path: Path) -> None:
        """When budget exceeded after first plan, all remaining plans get skipped."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(4)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(4)]
        result_a = _make_plan_run_result(plan_name="plan-0", success=True, total_cost_usd=10.0)

        with patch("maestro_cli.multi.run_plan", return_value=result_a) as mock_run:
            result = _run_sequential(plans, plan_paths, max_cost_usd=5.0)

        mock_run.assert_called_once()
        assert len(result.plan_results) == 4
        assert result.budget_exceeded is True
        # Plans 1-3 should all be skipped
        for i in range(1, 4):
            task_msg = next(iter(result.plan_results[i].task_results.values())).message
            assert "skipped" in task_msg

    def test_sequential_event_callback_forwarded(self, tmp_path: Path) -> None:
        """event_callback parameter is forwarded to run_plan."""
        plan = _make_plan_spec("plan-a")
        result = _make_plan_run_result(plan_name="plan-a", success=True)
        callback = MagicMock()

        with patch("maestro_cli.multi.run_plan", return_value=result) as mock_run:
            _run_sequential(
                [plan], [str(tmp_path / "a.yaml")],
                event_callback=callback,
            )

        assert mock_run.call_args.kwargs["event_callback"] is callback

    def test_sequential_execution_profile_forwarded(self, tmp_path: Path) -> None:
        """execution_profile is forwarded to each run_plan call."""
        plan = _make_plan_spec("plan-a")
        result = _make_plan_run_result(plan_name="plan-a", success=True)

        with patch("maestro_cli.multi.run_plan", return_value=result) as mock_run:
            _run_sequential(
                [plan], [str(tmp_path / "a.yaml")],
                execution_profile="safe",
            )

        assert mock_run.call_args.kwargs["execution_profile"] == "safe"


# ---------------------------------------------------------------------------
# Additional tests: _run_parallel
# ---------------------------------------------------------------------------


class TestRunParallelAdditional:
    """Additional tests for _run_parallel."""

    def test_parallel_cost_accumulation(self, tmp_path: Path) -> None:
        """Costs accumulate correctly across parallel plans."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(3)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(3)]
        costs = [0.5, 1.0, 1.5]

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            idx = int(plan.name.split("-")[1])
            return _make_plan_run_result(
                plan_name=plan.name, success=True, total_cost_usd=costs[idx]
            )

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(plans, plan_paths)

        assert result.total_cost_usd == pytest.approx(3.0)

    def test_parallel_all_fail(self, tmp_path: Path) -> None:
        """All plans failing gives overall success=False."""
        plans = [_make_plan_spec(f"plan-{i}") for i in range(2)]
        plan_paths = [str(tmp_path / f"p{i}.yaml") for i in range(2)]

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            raise TaskExecutionError(f"{plan.name} failed")

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(plans, plan_paths)

        assert result.success is False
        assert len(result.plan_results) == 2
        assert all(not r.success for r in result.plan_results)

    def test_parallel_oserror_captured(self, tmp_path: Path) -> None:
        """OSError from run_plan in parallel is captured as failed result."""
        plan = _make_plan_spec("plan-a")
        result_ok = _make_plan_run_result(plan_name="plan-b", success=True)

        def fake_run_plan(plan, **kwargs):  # type: ignore[no-untyped-def]
            if plan.name == "plan-a":
                raise OSError("disk failure")
            return result_ok

        with patch("maestro_cli.multi.run_plan", side_effect=fake_run_plan):
            result = _run_parallel(
                [plan, _make_plan_spec("plan-b")],
                [str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")],
            )

        assert len(result.plan_results) == 2
        failed = next(r for r in result.plan_results if r.plan_name == "plan-a")
        assert failed.success is False
        task_msg = next(iter(failed.task_results.values())).message
        assert "disk failure" in task_msg

    def test_parallel_execution_profile_forwarded(self, tmp_path: Path) -> None:
        """execution_profile is forwarded to each run_plan call."""
        plan = _make_plan_spec("plan-a")
        result = _make_plan_run_result(plan_name="plan-a", success=True)

        with patch("maestro_cli.multi.run_plan", return_value=result) as mock_run:
            _run_parallel(
                [plan], [str(tmp_path / "a.yaml")],
                execution_profile="yolo",
            )

        assert mock_run.call_args.kwargs["execution_profile"] == "yolo"

    def test_parallel_single_plan(self, tmp_path: Path) -> None:
        """Single plan in parallel mode works correctly."""
        plan = _make_plan_spec("solo")
        result = _make_plan_run_result(plan_name="solo", success=True, total_cost_usd=0.5)

        with patch("maestro_cli.multi.run_plan", return_value=result):
            multi = _run_parallel(
                [plan], [str(tmp_path / "solo.yaml")],
            )

        assert len(multi.plan_results) == 1
        assert multi.success is True
        assert multi.total_cost_usd == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Additional tests: _write_multi_summary edge cases
# ---------------------------------------------------------------------------


class TestWriteMultiSummaryAdditional:
    """Additional tests for _write_multi_summary."""

    def test_summary_duration_calculated(self, tmp_path: Path) -> None:
        """Summary includes correct duration calculation."""
        start = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        end = datetime(2024, 6, 1, 10, 5, 30, tzinfo=UTC)
        r1 = PlanRunResult(
            plan_name="alpha", run_id="run-alpha", run_path=tmp_path,
            started_at=start, finished_at=end, success=True,
        )
        multi = MultiPlanResult(
            plan_results=[r1], success=True, started_at=start, finished_at=end,
        )
        _write_multi_summary(multi, tmp_path)
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "330.0" in summary  # 5 min 30 sec = 330 seconds

    def test_summary_failed_status_text(self, tmp_path: Path) -> None:
        """Failed plan shows 'failed' in status column."""
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="broken", run_id="run-broken", run_path=tmp_path,
            started_at=now, finished_at=now, success=False,
            total_cost_usd=0.5, total_tokens=100,
        )
        multi = MultiPlanResult(
            plan_results=[r1], success=False,
            total_cost_usd=0.5, total_tokens=100,
            started_at=now, finished_at=now,
        )
        _write_multi_summary(multi, tmp_path)
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "| broken | failed |" in summary

    def test_summary_multiple_plans_all_fields(self, tmp_path: Path) -> None:
        """Summary with multiple plans includes plan names and TOTAL row."""
        now = datetime.now(UTC)
        plans = []
        for name, cost, tokens, success in [
            ("alpha", 0.10, 100, True),
            ("beta", 0.20, 200, True),
            ("gamma", 0.30, 300, False),
        ]:
            plans.append(PlanRunResult(
                plan_name=name, run_id=f"run-{name}", run_path=tmp_path,
                started_at=now, finished_at=now, success=success,
                total_cost_usd=cost, total_tokens=tokens,
            ))
        multi = MultiPlanResult(
            plan_results=plans, success=False,
            total_cost_usd=0.60, total_tokens=600,
            started_at=now, finished_at=now,
        )
        _write_multi_summary(multi, tmp_path)
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "alpha" in summary
        assert "beta" in summary
        assert "gamma" in summary
        assert "TOTAL" in summary
        assert "Plans | 3" in summary
        assert "$0.60" in summary
        assert "600" in summary

    def test_summary_creates_output_dir(self, tmp_path: Path) -> None:
        """_write_multi_summary creates output_dir if it doesn't exist."""
        now = datetime.now(UTC)
        output_dir = tmp_path / "new_dir"
        multi = MultiPlanResult(
            plan_results=[], success=True,
            started_at=now, finished_at=now,
        )
        _write_multi_summary(multi, output_dir)
        assert (output_dir / "summary.md").exists()


# ---------------------------------------------------------------------------
# Additional tests: _new_plan_result
# ---------------------------------------------------------------------------


class TestNewPlanResultAdditional:
    """Additional tests for _new_plan_result."""

    def test_new_plan_result_skipped_status(self, tmp_path: Path) -> None:
        """Skipped status is stored correctly."""
        result = _new_plan_result(
            plan_name="skipped-plan", run_path=tmp_path,
            success=False, message="skipped due to budget", status="skipped",
        )
        task = next(iter(result.task_results.values()))
        assert task.status == "skipped"
        assert task.message == "skipped due to budget"

    def test_new_plan_result_timestamps_set(self, tmp_path: Path) -> None:
        """started_at and finished_at are set to current time."""
        result = _new_plan_result(
            plan_name="test", run_path=tmp_path,
            success=True, message="ok", status="success",
        )
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_new_plan_result_run_id_format(self, tmp_path: Path) -> None:
        """run_id starts with 'multi-' prefix."""
        result = _new_plan_result(
            plan_name="my-plan", run_path=tmp_path,
            success=False, message="error", status="failed",
        )
        assert result.run_id.startswith("multi-")

    def test_new_plan_result_special_chars_in_name(self, tmp_path: Path) -> None:
        """Plan names with special characters are sanitized."""
        result = _new_plan_result(
            plan_name="my plan/with:specials", run_path=tmp_path,
            success=False, message="error", status="failed",
        )
        # task_id and run_id should be sanitized
        task_id = next(iter(result.task_results.keys()))
        assert ":" in task_id  # the ":multi" suffix
        assert result.run_id.startswith("multi-")


# ---------------------------------------------------------------------------
# Additional tests: MultiPlanResult dataclass
# ---------------------------------------------------------------------------


class TestMultiPlanResultAdditional:
    """Additional tests for MultiPlanResult dataclass."""

    def test_to_dict_with_plan_results(self) -> None:
        """to_dict includes serialized plan results."""
        now = datetime.now(UTC)
        r1 = PlanRunResult(
            plan_name="alpha", run_id="run-alpha", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=True,
            total_cost_usd=0.5, total_tokens=100,
        )
        multi = MultiPlanResult(
            plan_results=[r1], total_cost_usd=0.5, total_tokens=100,
            success=True, started_at=now, finished_at=now,
        )
        d = multi.to_dict()
        assert len(d["plan_results"]) == 1
        assert d["plan_results"][0]["plan_name"] == "alpha"
        assert d["total_cost_usd"] == 0.5
        assert d["total_tokens"] == 100
        assert d["success"] is True

    def test_to_dict_timestamps_are_iso_strings(self) -> None:
        """to_dict serializes timestamps as ISO format strings."""
        now = datetime.now(UTC)
        multi = MultiPlanResult(started_at=now, finished_at=now)
        d = multi.to_dict()
        assert isinstance(d["started_at"], str)
        assert isinstance(d["finished_at"], str)
        # Should be parseable back
        datetime.fromisoformat(d["started_at"])
        datetime.fromisoformat(d["finished_at"])

    def test_to_dict_empty_result(self) -> None:
        """Default MultiPlanResult serializes correctly."""
        multi = MultiPlanResult()
        d = multi.to_dict()
        assert d["plan_results"] == []
        assert d["total_cost_usd"] is None
        assert d["total_tokens"] is None
        assert d["budget_exceeded"] is False
        assert d["success"] is True
