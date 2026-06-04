from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    LessonRecord,
    PlanRunResult,
    PlanSpec,
    SessionSnapshot,
    TaskResult,
    TaskSpec,
    WatchIteration,
    WatchSpec,
    WatchState,
)
from maestro_cli.watch import (
    _build_recent_iteration_outputs,
    _build_session_memory_snapshot,
    _extract_fix_summary,
    _load_lessons,
    _maybe_extract_session_memory,
    _one_line_excerpt,
    _save_stepping_stone,
    _watch_improve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PLAN_YAML = (
    "version: 1\n"
    "name: imp-target\n"
    "workspace_root: .\n"
    "tasks:\n"
    "  - id: t1\n"
    "    command: echo ok\n"
)


def _improve_plan(tmp_path: Path, run_dir: str = ".maestro-runs") -> tuple[Path, PlanSpec]:
    """Write a valid improve-mode plan file and return (path, loaded-ish spec).

    We build the PlanSpec by hand (rather than loading) so callers can tweak
    run_dir freely while keeping a real plan file on disk for load_plan().
    """
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(_VALID_PLAN_YAML, encoding="utf-8")
    spec = WatchSpec(
        metric="tasks_passed",
        metric_source="manifest",
        metric_direction="higher_is_better",
        mode="improve",
        max_iterations=5,
        plateau_threshold=5,
        warmup_iterations=0,
    )
    plan = PlanSpec(
        name="imp-target",
        source_path=plan_path,
        workspace_root=".",
        run_dir=run_dir,
        tasks=[TaskSpec(id="t1", command="echo ok")],
        watch=spec,
    )
    return plan_path, plan


def _make_iteration(
    iteration: int,
    *,
    improved: bool = False,
    action: str = "keep",
    metric_value: float | None = None,
    best_metric: float | None = None,
    error: str | None = None,
    fix_summary: str | None = None,
    manifest_excerpt: str | None = None,
    blame_excerpt: str | None = None,
    consolidated_excerpt: str | None = None,
) -> WatchIteration:
    return WatchIteration(
        iteration=iteration,
        metric_value=metric_value,
        best_metric=best_metric,
        improved=improved,
        action=action,
        error=error,
        fix_summary=fix_summary,
        manifest_excerpt=manifest_excerpt,
        blame_excerpt=blame_excerpt,
        consolidated_excerpt=consolidated_excerpt,
        timestamp=datetime.now().isoformat(),
    )


# ---------------------------------------------------------------------------
# _extract_fix_summary — no FIX: line returns None (match is None branch)
# ---------------------------------------------------------------------------

class TestExtractFixSummary:
    def test_no_fix_line_returns_none(self) -> None:
        # Non-empty log without any "FIX:" line drives the match-is-None branch.
        assert _extract_fix_summary("agent did some work\nbut no marker here") is None


# ---------------------------------------------------------------------------
# _one_line_excerpt — blank, whitespace-only, short, truncated
# ---------------------------------------------------------------------------

class TestOneLineExcerpt:
    def test_blank_input_returns_none(self) -> None:
        # Falsy/empty text returns None immediately.
        assert _one_line_excerpt("") is None
        assert _one_line_excerpt(None) is None

    def test_whitespace_only_collapses_to_none(self) -> None:
        # text.split() on whitespace-only yields no tokens -> collapsed is empty.
        assert _one_line_excerpt("   \n\t  ") is None

    def test_short_text_passes_through(self) -> None:
        # Within max_chars: returned collapsed verbatim.
        assert _one_line_excerpt("a   b\n c") == "a b c"

    def test_long_text_is_truncated_with_ellipsis(self) -> None:
        out = _one_line_excerpt("x " * 200, max_chars=20)
        assert out is not None
        assert out.endswith("...")
        assert len(out) <= 20


# ---------------------------------------------------------------------------
# _build_recent_iteration_outputs — section-less items + all-empty fallback
# ---------------------------------------------------------------------------

class TestBuildRecentIterationOutputs:
    def test_all_items_without_sections_yields_empty_text(self) -> None:
        # Items carry no fix/manifest/blame/consolidated excerpt -> every loop
        # iteration hits `continue`, then the no-blocks fallback returns the
        # empty-outputs sentinel.
        iterations = [_make_iteration(1), _make_iteration(2)]
        out = _build_recent_iteration_outputs(iterations)
        assert "No recent verbatim iteration outputs" in out

    def test_mixed_items_skip_empty_and_render_populated(self) -> None:
        # One blank item (continue) and one populated item (rendered block).
        iterations = [
            _make_iteration(1),
            _make_iteration(2, action="keep", fix_summary="raised timeout"),
        ]
        out = _build_recent_iteration_outputs(iterations)
        assert "FIX summary: raised timeout" in out
        assert "### Iteration 2" in out


# ---------------------------------------------------------------------------
# _build_session_memory_snapshot — archived-empty None, success/failure/blocker
# fallbacks, lessons rendering, and empty-snapshot None.
# ---------------------------------------------------------------------------

class TestBuildSessionMemorySnapshot:
    def test_no_archived_returns_none(self) -> None:
        # Enough iterations to pass the trigger, but tail count equals length so
        # `archived` is empty -> returns None.
        iterations = [_make_iteration(i) for i in range(1, 9)]
        snap = _build_session_memory_snapshot(
            iterations,
            [],
            plan_name="p",
            watch_run_path="/run",
            metric_name="tasks_passed",
            metric_direction="higher_is_better",
            plateau_count=0,
            plateau_threshold=5,
            consolidate_model=None,
            consolidated_summary="",
            recent_tail_count=8,
        )
        assert snap is None

    def test_no_successes_no_failures_no_blockers_uses_fallbacks(self) -> None:
        # All archived iterations are baselines (not improved, action 'baseline')
        # so successes/failures stay empty and no errors -> blocker fallback too.
        iterations = [
            _make_iteration(i, improved=False, action="baseline")
            for i in range(1, 12)
        ]
        snap = _build_session_memory_snapshot(
            iterations,
            [],
            plan_name="p",
            watch_run_path="/run",
            metric_name="tasks_passed",
            metric_direction="higher_is_better",
            plateau_count=0,
            plateau_threshold=5,
            consolidate_model=None,
            consolidated_summary="",
        )
        assert snap is not None
        text = snap.snapshot_text
        assert "- None yet." in text  # no successes
        assert "No repeated failures recorded yet." in text  # no failures
        assert "No clear blockers in archived iterations." in text  # no blockers
        assert "No durable lessons yet." in text  # empty lessons list

    def test_blockers_from_error_and_lessons_rendered(self) -> None:
        # An archived iteration carrying an error feeds the blockers list, and a
        # provided lesson exercises the lessons-render branch. archived = all
        # but the last 3 iterations, so the error iteration must land in the
        # earlier (archived) portion.
        iterations: list[WatchIteration] = []
        iterations.append(_make_iteration(1, improved=False, action="rollback",
                                          metric_value=2.0, error="boom failure"))
        iterations.extend(
            _make_iteration(i, improved=True, action="keep", metric_value=float(i))
            for i in range(2, 12)
        )
        lessons = [LessonRecord(iteration=3, task_id="t1", category="successful_fix",
                                lesson="bump timeout", confidence=0.9)]
        snap = _build_session_memory_snapshot(
            iterations,
            lessons,
            plan_name="p",
            watch_run_path="/run",
            metric_name="tasks_passed",
            metric_direction="higher_is_better",
            plateau_count=1,
            plateau_threshold=5,
            consolidate_model="haiku",
            consolidated_summary="overall: keep raising timeouts",
        )
        assert snap is not None
        text = snap.snapshot_text
        assert "boom failure" in text  # blocker from error
        assert "bump timeout" in text  # lesson rendered
        assert "90%" in text  # lesson confidence formatting

    def test_empty_snapshot_text_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the excerpt truncation to yield a blank snapshot so the
        # `if not snapshot_text: return None` branch fires.
        import maestro_cli.watch as watch_mod

        monkeypatch.setattr(watch_mod, "_truncate_iteration_excerpt",
                            lambda *a, **k: None)
        iterations = [_make_iteration(i, improved=True, action="keep", metric_value=float(i))
                      for i in range(1, 12)]
        snap = _build_session_memory_snapshot(
            iterations,
            [],
            plan_name="p",
            watch_run_path="/run",
            metric_name="tasks_passed",
            metric_direction="higher_is_better",
            plateau_count=0,
            plateau_threshold=5,
            consolidate_model=None,
            consolidated_summary="",
        )
        assert snap is None


# ---------------------------------------------------------------------------
# _maybe_extract_session_memory — latest snapshot already at/ahead of new range
# ---------------------------------------------------------------------------

class TestMaybeExtractSessionMemory:
    def test_returns_existing_latest_when_already_ahead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Build enough iterations to produce a snapshot, then make the stored
        # latest snapshot's iteration_to >= the new snapshot's -> the function
        # returns the existing latest without storing again.
        import maestro_cli.memory as memory_mod

        existing = SessionSnapshot(
            id=7,
            plan_name="p",
            watch_run_path=str(tmp_path / "run"),
            iteration_from=1,
            iteration_to=999,  # far ahead of any new snapshot
            snapshot_text="already-stored",
        )
        store_called: list[bool] = []

        monkeypatch.setattr(
            memory_mod, "load_latest_session_snapshot",
            lambda *a, **k: existing,
        )
        monkeypatch.setattr(
            memory_mod, "store_session_snapshot",
            lambda *a, **k: store_called.append(True),
        )
        monkeypatch.setattr(
            memory_mod, "prune_session_snapshots",
            lambda *a, **k: None,
        )

        iterations = [_make_iteration(i, improved=True, action="keep", metric_value=float(i))
                      for i in range(1, 12)]
        result = _maybe_extract_session_memory(
            plan_name="p",
            source_dir=tmp_path,
            watch_run_path=tmp_path / "run",
            iterations=iterations,
            lessons=[],
            metric_name="tasks_passed",
            metric_direction="higher_is_better",
            plateau_count=0,
            plateau_threshold=5,
            consolidate_model=None,
            consolidated_summary="",
        )
        assert result is existing
        assert store_called == []  # store path skipped


# ---------------------------------------------------------------------------
# _load_lessons — blank-line skip, time-decay exception swallow, OSError -> []
# ---------------------------------------------------------------------------

class TestLoadLessons:
    def test_blank_lines_skipped_and_valid_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        record = LessonRecord(
            iteration=1, task_id="t1", category="successful_fix",
            lesson="fixed it", confidence=0.8,
            timestamp=datetime.now().isoformat(),
        )
        path.write_text(
            "\n   \n" + json.dumps(record.to_dict()) + "\n",
            encoding="utf-8",
        )
        lessons = _load_lessons(path)
        assert len(lessons) == 1
        assert lessons[0].lesson == "fixed it"

    def test_invalid_timestamp_is_swallowed(self, tmp_path: Path) -> None:
        # A non-ISO timestamp makes datetime.fromisoformat raise -> the
        # ValueError/TypeError handler keeps the original confidence.
        path = tmp_path / "lessons.jsonl"
        data = {
            "iteration": 2,
            "task_id": "t2",
            "category": "failed_attempt",
            "lesson": "no decay",
            "confidence": 0.6,
            "timestamp": "not-a-real-timestamp",
        }
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        lessons = _load_lessons(path)
        assert len(lessons) == 1
        # Decay skipped, confidence unchanged.
        assert lessons[0].confidence == pytest.approx(0.6)

    def test_valid_timestamp_applies_decay(self, tmp_path: Path) -> None:
        # Sanity: a parseable old timestamp decays confidence below original.
        old = (datetime.now() - timedelta(days=60)).isoformat()
        path = tmp_path / "lessons.jsonl"
        data = {
            "iteration": 3, "task_id": "t3", "category": "failed_attempt",
            "lesson": "old", "confidence": 0.8, "timestamp": old,
        }
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        lessons = _load_lessons(path)
        assert lessons[0].confidence < 0.8

    def test_read_oserror_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # File exists, but read_text raises OSError -> the outer handler returns [].
        path = tmp_path / "lessons.jsonl"
        path.write_text("ignored\n", encoding="utf-8")

        real_read_text = Path.read_text

        def _boom(self: Path, *a: Any, **k: Any) -> str:
            if self == path:
                raise OSError("disk gone")
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _boom)
        assert _load_lessons(path) == []


# ---------------------------------------------------------------------------
# _watch_improve — unresolvable run dir, stepping-stone application, cancel break
# ---------------------------------------------------------------------------

class TestWatchImproveEarlyBranches:
    def test_unresolvable_run_dir_raises(self, tmp_path: Path) -> None:
        # run_dir="" makes resolve_path return None -> ValueError before any run.
        plan_path, plan = _improve_plan(tmp_path, run_dir="")
        spec = plan.watch
        assert spec is not None
        with pytest.raises(ValueError, match="run directory"):
            _watch_improve(plan_path, plan, spec)

    def test_applies_prior_stepping_stone_then_dry_run_returns(
        self, tmp_path: Path,
    ) -> None:
        # Pre-seed a stepping stone whose plan_yaml is a valid plan. With
        # stepping_stones enabled, no resume, and an empty state, _watch_improve
        # loads + applies it, reloads the plan, then dry_run exits cleanly.
        plan_path, plan = _improve_plan(tmp_path)
        spec = WatchSpec(
            metric="tasks_passed",
            metric_source="manifest",
            metric_direction="higher_is_better",
            mode="improve",
            max_iterations=5,
            plateau_threshold=5,
            warmup_iterations=0,
            stepping_stones=True,
        )

        # The stone carries a *different but valid* plan YAML so we can confirm
        # the file gets rewritten by _apply_stepping_stone.
        stone_yaml = _VALID_PLAN_YAML.replace("echo ok", "echo improved")
        stone_path = tmp_path / "stone_plan.yaml"
        stone_path.write_text(stone_yaml, encoding="utf-8")
        saved = _save_stepping_stone(
            plan_path=stone_path,
            plan_name=plan.name,
            metric_value=3.0,
            metric_name="tasks_passed",
            iteration=4,
            git_commit="deadbeef",
            lessons_path=None,
            watch_run_path=str(tmp_path / "prior_run"),
            total_cost_usd=1.0,
            archive_source_dir=plan.source_dir,
        )
        assert saved is not None

        events: list[tuple[str, dict[str, object]]] = []
        state = _watch_improve(
            plan_path, plan, spec,
            dry_run=True,
            event_callback=lambda n, p: events.append((n, p)),
        )

        names = [n for n, _ in events]
        assert "stepping_stone_applied" in names
        assert "watch_start" in names
        assert "watch_complete" in names
        # Stone metric_value seeded best_metric before the dry-run exit.
        assert state.best_metric == 3.0
        # The plan file on disk now reflects the applied stone.
        assert "echo improved" in plan_path.read_text(encoding="utf-8")

    def test_cancel_event_set_breaks_with_interrupted_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A pre-set cancel_event trips the first loop check before run_plan,
        # marking the state interrupted. run_plan is stubbed to fail loudly if
        # ever reached.
        plan_path, plan = _improve_plan(tmp_path)
        spec = plan.watch
        assert spec is not None

        def _boom_run_plan(*a: Any, **k: Any) -> PlanRunResult:
            raise AssertionError("run_plan should not be called after cancel")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _boom_run_plan)

        cancel = threading.Event()
        cancel.set()
        events: list[tuple[str, dict[str, object]]] = []
        state = _watch_improve(
            plan_path, plan, spec,
            dry_run=False,
            cancel_event=cancel,
            event_callback=lambda n, p: events.append((n, p)),
        )
        assert state.status == "interrupted"
        assert "watch_complete" in [n for n, _ in events]
