from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from maestro_cli import watch as watch_mod
from maestro_cli.models import (
    PlanRunResult,
    PlanSpec,
    SteppingStone,
    TaskResult,
    TaskSpec,
    WatchIteration,
    WatchSpec,
)
from maestro_cli.watch import (
    _STEPPING_STONES_MAX,
    _apply_stepping_stone,
    _build_blame_context,
    _compact_stepping_stones,
    _extract_metric,
    _load_best_stepping_stone,
    _run_consolidation,
    _save_stepping_stone,
    _stepping_stones_dir,
    watch,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_run_result(
    tmp_path: Path,
    *,
    task_results: dict[str, TaskResult],
    success: bool = True,
    cost: float = 0.0,
) -> PlanRunResult:
    run_path = tmp_path / "run"
    run_path.mkdir(parents=True, exist_ok=True)
    return PlanRunResult(
        plan_name="t",
        run_id="r1",
        run_path=run_path,
        started_at=datetime.now(),
        finished_at=datetime.now(),
        success=success,
        task_results=task_results,
        total_cost_usd=cost,
    )


def _task_result(
    task_id: str,
    *,
    status: str = "success",
    log_path: Path | None = None,
    stdout_tail: str = "",
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        exit_code=0,
        duration_sec=1.0,
        stdout_tail=stdout_tail,
        log_path=log_path or Path("missing.log"),
    )


# ---------------------------------------------------------------------------
# _extract_metric early-return branches
# ---------------------------------------------------------------------------


def test_extract_metric_no_tasks_returns_none(tmp_path: Path) -> None:
    """When the plan has no tasks, _extract_metric returns None immediately."""
    plan = PlanSpec(name="empty", tasks=[])
    spec = WatchSpec(metric="score", metric_source="manifest")
    result = _make_run_result(tmp_path, task_results={"a": _task_result("a")})
    assert _extract_metric(result, spec, plan, result.run_path) is None


def test_extract_metric_manifest_handles_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure inside the manifest extractor is swallowed and yields None."""
    plan = PlanSpec(name="p", tasks=[TaskSpec(id="a")])
    spec = WatchSpec(metric="score", metric_source="manifest")
    result = _make_run_result(tmp_path, task_results={"a": _task_result("a")})

    def _boom(_res: PlanRunResult) -> float:
        raise ValueError("manifest exploded")

    monkeypatch.setattr(watch_mod, "_extract_manifest_metric", _boom)
    assert _extract_metric(result, spec, plan, result.run_path) is None


def test_extract_metric_verify_command_missing_section_returns_none(
    tmp_path: Path,
) -> None:
    """verify_command source: log exists & pattern set but section header absent."""
    log_path = tmp_path / "task.log"
    # Log file exists but contains no [verify_command] section header.
    log_path.write_text("just some unrelated output\nno section here\n", encoding="utf-8")
    plan = PlanSpec(name="p", tasks=[TaskSpec(id="a")])
    spec = WatchSpec(
        metric="score",
        metric_source="verify_command",
        metric_pattern=r"score: ([0-9.]+)",
        metric_task="a",
    )
    result = _make_run_result(
        tmp_path,
        task_results={"a": _task_result("a", log_path=log_path)},
    )
    assert _extract_metric(result, spec, plan, result.run_path) is None


# ---------------------------------------------------------------------------
# _build_blame_context: blame_run raises -> except Exception
# ---------------------------------------------------------------------------


def test_build_blame_context_blame_run_raises_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If blame_run raises, blame_json stays empty but manifest is still parsed."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    manifest = run_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "exit_code": 1,
                        "duration_sec": 2.5,
                        "message": "boom",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def _raise(_p: Path) -> Any:
        raise RuntimeError("blame failed")

    monkeypatch.setattr(watch_mod, "blame_run", _raise)
    blame_json, manifest_summary = _build_blame_context(run_path)
    assert blame_json == ""
    assert "t1" in manifest_summary
    assert "failed" in manifest_summary


# ---------------------------------------------------------------------------
# _save_stepping_stone: malformed / unreadable lessons file
# ---------------------------------------------------------------------------


def test_save_stepping_stone_skips_malformed_lesson_lines(tmp_path: Path) -> None:
    """Non-JSON lines in the lessons file are skipped (json.JSONDecodeError path)."""
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: p\n", encoding="utf-8")
    lessons_path = tmp_path / "lessons.jsonl"
    lessons_path.write_text(
        '{"lesson": "good"}\nnot-json-at-all\n\n{"lesson": "also good"}\n',
        encoding="utf-8",
    )
    stone = _save_stepping_stone(
        plan_path,
        "p",
        metric_value=3.0,
        metric_name="score",
        iteration=1,
        git_commit=None,
        lessons_path=lessons_path,
        watch_run_path=str(tmp_path / "wr"),
        total_cost_usd=0.0,
        archive_source_dir=tmp_path,
    )
    assert stone is not None
    # Only the two valid JSON lines survive; the garbage line is dropped.
    assert len(stone.lessons) == 2


def test_save_stepping_stone_lessons_read_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError while reading the lessons file is swallowed."""
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: p\n", encoding="utf-8")
    lessons_path = tmp_path / "lessons.jsonl"
    lessons_path.write_text('{"lesson": "x"}\n', encoding="utf-8")

    real_read_text = Path.read_text

    def _fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == lessons_path:
            raise OSError("cannot read lessons")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    stone = _save_stepping_stone(
        plan_path,
        "p",
        metric_value=2.0,
        metric_name="score",
        iteration=1,
        git_commit=None,
        lessons_path=lessons_path,
        watch_run_path=str(tmp_path / "wr"),
        total_cost_usd=0.0,
        archive_source_dir=tmp_path,
    )
    assert stone is not None
    # Read failed, so no lessons were collected.
    assert stone.lessons == []


# ---------------------------------------------------------------------------
# _compact_stepping_stones branches (1790-1791, 1799, 1811, 1826-1827)
# ---------------------------------------------------------------------------


def test_compact_stepping_stones_read_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError reading the stones file returns early without raising."""
    stones_path = tmp_path / "stones.jsonl"
    stones_path.write_text("x\n", encoding="utf-8")

    def _fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        raise OSError("read failed")

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    # Should not raise.
    _compact_stepping_stones(stones_path, "score")


def _write_stones(stones_path: Path, count: int, metric_name: str = "score") -> None:
    lines: list[str] = []
    for i in range(count):
        lines.append(
            json.dumps(
                {
                    "metric_name": metric_name,
                    "metric_value": float(i),
                    "plan_yaml": "version: 1\nname: p\n",
                }
            )
        )
    stones_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compact_stepping_stones_skips_blank_lines_and_keeps_within_limit(
    tmp_path: Path,
) -> None:
    """Blank lines are skipped; when matching count <= limit, returns without rewrite."""
    stones_path = tmp_path / "stones.jsonl"
    # Over the line limit overall (so we pass the len(lines) gate), but matching the
    # target metric stays within the per-metric cap thanks to many other-metric rows.
    over = _STEPPING_STONES_MAX + 5
    lines: list[str] = []
    # Insert blank lines that must be skipped.
    for i in range(over):
        lines.append("")
        lines.append(
            json.dumps(
                {
                    "metric_name": "other_metric",
                    "metric_value": float(i),
                }
            )
        )
    # Only a few rows for the queried metric -> matching <= limit.
    for i in range(3):
        lines.append(json.dumps({"metric_name": "score", "metric_value": float(i)}))
    stones_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    before = stones_path.read_text(encoding="utf-8")
    _compact_stepping_stones(stones_path, "score")
    # Early return path: file untouched.
    assert stones_path.read_text(encoding="utf-8") == before


def test_compact_stepping_stones_write_oserror_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When write_text fails, the OSError is swallowed."""
    stones_path = tmp_path / "stones.jsonl"
    _write_stones(stones_path, _STEPPING_STONES_MAX + 5, metric_name="score")

    real_write_text = Path.write_text

    def _fake_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        if self == stones_path:
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)
    # Must not raise even though the rewrite fails.
    _compact_stepping_stones(stones_path, "score")


# ---------------------------------------------------------------------------
# _load_best_stepping_stone branches (1846, 1864-1865)
# ---------------------------------------------------------------------------


def test_load_best_stepping_stone_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines in the stones file are skipped."""
    stones_dir = _stepping_stones_dir(tmp_path, "p")
    stones_dir.mkdir(parents=True, exist_ok=True)
    stones_path = stones_dir / "stones.jsonl"
    stones_path.write_text(
        "\n"
        + json.dumps({"metric_name": "score", "metric_value": 1.0}) + "\n"
        + "\n"
        + json.dumps({"metric_name": "score", "metric_value": 5.0}) + "\n"
        + "\n",
        encoding="utf-8",
    )
    best = _load_best_stepping_stone("p", tmp_path, "score", higher_is_better=True)
    assert best is not None
    assert best.metric_value == 5.0


def test_load_best_stepping_stone_read_oserror_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError while reading the stones file returns None."""
    stones_dir = _stepping_stones_dir(tmp_path, "p")
    stones_dir.mkdir(parents=True, exist_ok=True)
    stones_path = stones_dir / "stones.jsonl"
    stones_path.write_text(
        json.dumps({"metric_name": "score", "metric_value": 1.0}) + "\n",
        encoding="utf-8",
    )

    def _fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == stones_path:
            raise OSError("read failed")
        return ""

    monkeypatch.setattr(Path, "read_text", _fake_read_text)
    assert _load_best_stepping_stone("p", tmp_path, "score") is None


# ---------------------------------------------------------------------------
# _apply_stepping_stone: restore-backup OSError swallow
# ---------------------------------------------------------------------------


def test_apply_stepping_stone_restore_backup_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the restored plan is invalid AND restoring the backup fails, swallow OSError."""
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: original\ntasks:\n  - id: a\n    command: echo hi\n", encoding="utf-8")
    backup = plan_path.read_text(encoding="utf-8")
    assert backup  # non-empty backup so the restore branch is reached

    stone = SteppingStone(
        plan_name="bad",
        plan_hash="h",
        metric_value=1.0,
        metric_name="score",
        iteration=1,
        plan_yaml="this: is: not: valid: yaml: at all: [",
    )

    # Make load_plan reject the restored YAML, forcing the restore-backup path.
    def _bad_load(_p: Path) -> Any:
        raise ValueError("invalid plan")

    monkeypatch.setattr(watch_mod, "load_plan", _bad_load)

    real_write_text = Path.write_text
    state = {"calls": 0}

    def _fake_write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self == plan_path:
            state["calls"] += 1
            # First call writes the (bad) stone YAML successfully; the second call is
            # the backup restore, which we force to fail.
            if state["calls"] >= 2:
                raise OSError("cannot restore backup")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)

    ok = _apply_stepping_stone(stone, plan_path)
    assert ok is False
    # The restore was attempted (second write call happened).
    assert state["calls"] >= 2


# ---------------------------------------------------------------------------
# _run_consolidation firewall pass-2 + instructionality (2013-2022, 2028)
# ---------------------------------------------------------------------------


def _patch_consolidation_subprocess(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _mock_run)


def test_run_consolidation_firewall_pass2_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With firewall_model set and a 'block' verdict, output is suppressed to ''."""
    _patch_consolidation_subprocess(monkeypatch, "some consolidated strategy text\n")

    import maestro_cli.runners as runners_mod

    class _Decision:
        verdict = "block"

    def _fake_pass2(model: str, kind: str, text: str, *, workdir: Any) -> _Decision:
        return _Decision()

    monkeypatch.setattr(runners_mod, "_run_firewall_pass2", _fake_pass2)

    spec = WatchSpec(
        metric="score",
        metric_source="manifest",
        consolidate_model="haiku",
    )
    plan = PlanSpec(name="p", firewall_model="haiku", tasks=[TaskSpec(id="a")])
    out = _run_consolidation(spec, "history text", tmp_path, plan=plan)
    assert out == ""


def test_run_consolidation_firewall_pass2_exception_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If pass-2 raises, fail open to the pass-1 sanitized output (except branch)."""
    _patch_consolidation_subprocess(monkeypatch, "harmless summary line\n")

    import maestro_cli.runners as runners_mod

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("firewall unavailable")

    monkeypatch.setattr(runners_mod, "_run_firewall_pass2", _boom)

    spec = WatchSpec(
        metric="score",
        metric_source="manifest",
        consolidate_model="haiku",
    )
    plan = PlanSpec(name="p", firewall_model="haiku", tasks=[TaskSpec(id="a")])
    out = _run_consolidation(spec, "history text", tmp_path, plan=plan)
    assert "harmless summary" in out


def test_run_consolidation_firewall_pass2_allow_then_instructionality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pass-2 allows; high instructionality triggers a second strip pass."""
    _patch_consolidation_subprocess(monkeypatch, "you must always do exactly this\n")

    import maestro_cli.runners as runners_mod

    class _Decision:
        verdict = "allow"

    monkeypatch.setattr(
        runners_mod, "_run_firewall_pass2", lambda *a, **k: _Decision()
    )
    # Force a high instructionality score so the extra strip pass executes.
    # compute_instructionality is imported locally from .memory inside the
    # function under test, so patch it at its definition module.
    import maestro_cli.memory as memory_mod

    monkeypatch.setattr(memory_mod, "compute_instructionality", lambda _t: 0.9)

    spec = WatchSpec(
        metric="score",
        metric_source="manifest",
        consolidate_model="haiku",
    )
    plan = PlanSpec(name="p", firewall_model="haiku", tasks=[TaskSpec(id="a")])
    out = _run_consolidation(spec, "history text", tmp_path, plan=plan)
    # Output is still a string (sanitized twice); not None.
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# watch(): consolidation agent call inside the main loop
# ---------------------------------------------------------------------------


def test_watch_loop_invokes_consolidation_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the consolidation-agent branch in the main watch() loop.

    Resume from a run dir that already has two experiments so state.iterations is
    non-empty and total_iterations is a multiple of consolidate_every, satisfying
    the consolidation guard and reaching the _run_consolidation call.
    """
    # A real, loadable plan whose watch block enables consolidation.
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        "version: 1\n"
        "name: cons-watch\n"
        "workspace_root: .\n"
        "watch:\n"
        "  metric: tasks_passed\n"
        "  metric_source: manifest\n"
        "  metric_direction: higher_is_better\n"
        "  warmup_iterations: 0\n"
        "  plateau_threshold: 99\n"
        "  max_iterations: 3\n"
        "  consolidate_model: haiku\n"
        "  consolidate_every: 1\n"
        "tasks:\n"
        "  - id: a\n"
        "    command: echo hi\n",
        encoding="utf-8",
    )

    # Pre-seed a resume directory with two experiment records.
    resume_dir = tmp_path / "watch_resume"
    resume_dir.mkdir()
    exp_path = resume_dir / "experiments.jsonl"
    rows = [
        WatchIteration(iteration=1, metric_value=1.0, best_metric=1.0, improved=True).to_dict(),
        WatchIteration(iteration=2, metric_value=1.0, best_metric=1.0, improved=True).to_dict(),
    ]
    exp_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    # Mock run_plan so we never touch real engines / scheduler.
    def _fake_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
        run_path = tmp_path / "innerrun"
        run_path.mkdir(exist_ok=True)
        return PlanRunResult(
            plan_name=plan_arg.name,
            run_id="inner",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"a": _task_result("a", status="success")},
            total_cost_usd=0.0,
        )

    monkeypatch.setattr(watch_mod, "run_plan", _fake_run_plan)

    # Track that the consolidation agent was actually invoked.
    called = {"n": 0}

    def _fake_consolidation(*args: Any, **kwargs: Any) -> str:
        called["n"] += 1
        return "consolidated-strategy"

    monkeypatch.setattr(watch_mod, "_run_consolidation", _fake_consolidation)
    # No real git.
    monkeypatch.setattr(watch_mod, "_git_commit_changes", lambda *a, **k: None)
    monkeypatch.setattr(watch_mod, "_git_rollback", lambda *a, **k: True)

    state = watch(plan_path, resume_from=resume_dir)

    assert called["n"] >= 1
    # The loop ran one more iteration (iteration 3) and ended at max_iterations.
    assert state.total_iterations >= 3
