"""Coverage-focused tests for maestro_cli.web.routes_api (cluster B).

These tests drive specific uncovered branches of the REST API:
- _build_collaboration: int last_progress_pct passthrough; owner completed_count bump
- _enrich_run_payload: non-dict task_results normalization to {}
- validate_plan: temp-file write failure cleanup branch
- browse_files: OSError handling for both inner and outer iterdir loops
- start_run background thread: thread-match fallback + exception handler

External boundaries (load_plan, run_plan, the run-dir filesystem, threading)
are mocked or driven via crafted inputs. No engine CLI, network, or git.
"""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from maestro_cli.models import PlanRunResult, PlanSpec, TaskSpec
from maestro_cli.web import create_app
from maestro_cli.web.routes_api import (
    _build_collaboration,
    _enrich_run_payload,
)
from maestro_cli.web.state import (
    RunState,
    _active_runs,
    _lock,
    register_run,
    set_project_root,
    set_project_roots,
)

_VALID_PLAN_YAML = """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan_spec(
    name: str = "test-plan",
    task_ids: list[str] | None = None,
    source_path: Path | None = None,
    run_dir: str = ".maestro-runs",
) -> PlanSpec:
    if task_ids is None:
        task_ids = ["t1"]
    tasks = [TaskSpec(id=tid, command="echo hi") for tid in task_ids]
    return PlanSpec(
        version=1,
        name=name,
        tasks=tasks,
        source_path=source_path,
        run_dir=run_dir,
    )


def _make_run_state(
    run_id: str = "rs-001",
    plan_name: str = "test-plan",
    task_ids: list[str] | None = None,
    run_path: Path | None = None,
    finished: bool = False,
    task_graph: dict[str, dict[str, object]] | None = None,
) -> RunState:
    if task_ids is None:
        task_ids = ["t1"]
    thread = MagicMock(spec=threading.Thread)
    thread.is_alive.return_value = not finished
    return RunState(
        run_id=run_id,
        plan_name=plan_name,
        task_ids=task_ids,
        run_path=run_path or Path("."),
        started_at=datetime.now(UTC),
        thread=thread,
        task_graph=task_graph or {},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_active_runs() -> None:
    set_project_roots([Path(".")])
    with _lock:
        _active_runs.clear()
    yield  # type: ignore[misc]
    set_project_roots([Path(".")])
    with _lock:
        _active_runs.clear()


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# _build_collaboration — int last_progress_pct passthrough (line ~411)
# ===========================================================================

class TestCollaborationProgress:
    def test_int_last_progress_pct_passed_through(self, tmp_path: Path) -> None:
        """An int last_progress_pct on a task result is carried into the entry."""
        task_results = {
            "t1": {"status": "running", "last_progress_pct": 42},
        }
        task_graph = {"t1": {"id": "t1", "depends_on": []}}

        collab = _build_collaboration(
            tmp_path,
            ["t1"],
            task_results,
            task_graph,
            include_activity=False,
        )
        assert collab["tasks"]["t1"]["last_progress_pct"] == 42

    def test_non_int_last_progress_pct_ignored(self, tmp_path: Path) -> None:
        """A non-int (e.g. float/str) last_progress_pct is left as None."""
        task_results = {
            "t1": {"status": "running", "last_progress_pct": "halfway"},
        }
        task_graph = {"t1": {"id": "t1", "depends_on": []}}

        collab = _build_collaboration(
            tmp_path,
            ["t1"],
            task_results,
            task_graph,
            include_activity=False,
        )
        assert collab["tasks"]["t1"]["last_progress_pct"] is None

    def test_int_progress_via_run_detail_endpoint(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """The active-run detail route surfaces int progress through enrichment."""
        result_file = tmp_path / "t1.result.json"
        result_file.write_text(
            json.dumps({"task_id": "t1", "status": "running", "last_progress_pct": 73}),
            encoding="utf-8",
        )

        state = _make_run_state(
            run_id="prog-run",
            run_path=tmp_path,
            task_ids=["t1"],
            finished=False,
            task_graph={"t1": {"id": "t1", "depends_on": []}},
        )
        register_run(state)

        resp = client.get("/api/runs/prog-run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collaboration"]["tasks"]["t1"]["last_progress_pct"] == 73


# ===========================================================================
# _build_collaboration — owner completed_count bump (line ~436)
# ===========================================================================

class TestCollaborationOwnerCompleted:
    def test_owned_success_task_increments_completed_count(
        self, tmp_path: Path,
    ) -> None:
        """An owned task with a success-like status bumps completed_count."""
        task_results = {"t1": {"status": "success"}}
        task_graph = {
            "t1": {"id": "t1", "agent": "qa-engineer", "depends_on": []},
        }

        collab = _build_collaboration(
            tmp_path,
            ["t1"],
            task_results,
            task_graph,
            include_activity=False,
        )
        owners = {o["label"]: o for o in collab["owners"]}
        assert owners["qa-engineer"]["task_count"] == 1
        assert owners["qa-engineer"]["completed_count"] == 1
        # success is terminal, so it is not counted as active
        assert owners["qa-engineer"]["active_count"] == 0

    def test_owned_dry_run_task_counts_as_completed(self, tmp_path: Path) -> None:
        """dry_run is also a success-like status that bumps completed_count."""
        task_results = {"t1": {"status": "dry_run"}}
        task_graph = {
            "t1": {"id": "t1", "agent": "architect", "depends_on": []},
        }

        collab = _build_collaboration(
            tmp_path,
            ["t1"],
            task_results,
            task_graph,
            include_activity=False,
        )
        owners = {o["label"]: o for o in collab["owners"]}
        assert owners["architect"]["completed_count"] == 1

    def test_owned_completed_via_run_detail_endpoint(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """End-to-end: completed owned task is reflected in the owner summary."""
        result_file = tmp_path / "t1.result.json"
        result_file.write_text(
            json.dumps({"task_id": "t1", "status": "success"}),
            encoding="utf-8",
        )
        state = _make_run_state(
            run_id="owner-run",
            run_path=tmp_path,
            task_ids=["t1"],
            finished=False,
            task_graph={"t1": {"id": "t1", "agent": "qa-engineer", "depends_on": []}},
        )
        register_run(state)

        resp = client.get("/api/runs/owner-run")
        assert resp.status_code == 200
        data = resp.json()
        owners = {o["label"]: o for o in data["collaboration"]["owners"]}
        assert owners["qa-engineer"]["completed_count"] == 1


# ===========================================================================
# _enrich_run_payload — non-dict task_results normalized to {} (lines ~506-507)
# ===========================================================================

class TestEnrichPayloadNonDictTaskResults:
    def test_list_task_results_replaced_with_empty_dict(self, tmp_path: Path) -> None:
        """A payload with a list task_results is normalized to an empty dict."""
        payload: dict[str, Any] = {"task_results": ["not", "a", "dict"]}
        enriched = _enrich_run_payload(payload, tmp_path)
        assert enriched["task_results"] == {}

    def test_missing_task_results_replaced_with_empty_dict(
        self, tmp_path: Path,
    ) -> None:
        """A payload with no task_results key gets an empty dict installed."""
        payload: dict[str, Any] = {"plan_name": "p"}
        enriched = _enrich_run_payload(payload, tmp_path)
        assert enriched["task_results"] == {}

    def test_non_dict_task_results_via_historical_detail_route(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """The historical-run detail route normalizes a manifest's bad task_results."""
        run_dir = tmp_path / ".maestro-runs" / "bad-tr-run"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "bad-tr-plan",
            "run_id": "bad-tr-run",
            "success": True,
            # task_results is a string instead of a dict — exercises the guard
            "task_results": "oops-not-a-dict",
            "task_graph": {},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        set_project_root(tmp_path)

        resp = client.get("/api/runs/bad-tr-run")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_results"] == {}
        assert data["plan_name"] == "bad-tr-plan"


# ===========================================================================
# validate_plan — temp-file write failure cleanup (lines ~619-621)
# ===========================================================================

class TestValidateTempFileWriteFailure:
    def test_write_failure_unlinks_temp_and_raises(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """If writing the temp file fails, the temp file is unlinked and the error propagates."""
        real_tmp = tmp_path / "boom.yaml"
        # Create the file so the unlink path has something to remove.
        real_tmp.write_text("", encoding="utf-8")

        class _FakeTmp:
            name = str(real_tmp)

            def write(self, _data: str) -> int:
                raise OSError("disk full")

            def close(self) -> None:  # pragma: no cover - not reached
                pass

        def _fake_named_temp(*_args: Any, **_kwargs: Any) -> _FakeTmp:
            return _FakeTmp()

        # validate_plan does `import tempfile` then NamedTemporaryFile(...)
        monkeypatch.setattr(
            "tempfile.NamedTemporaryFile", _fake_named_temp,
        )

        resp = client.post("/api/plans/validate", json={"yaml_content": _VALID_PLAN_YAML})
        # The raised exception bubbles to a 500 (TestClient does not raise).
        assert resp.status_code == 500
        # The temp file should have been removed by the cleanup branch.
        assert not real_tmp.exists()


# ===========================================================================
# browse_files — OSError handling (lines ~679-682)
# ===========================================================================

class TestBrowseFilesOSError:
    def test_inner_iterdir_oserror_is_swallowed(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """A subdirectory that raises OSError on iterdir is skipped, others kept."""
        # A top-level yaml file that should still be returned.
        (tmp_path / "top.yaml").write_text(_VALID_PLAN_YAML, encoding="utf-8")
        # A good subdir with a yaml file.
        good = tmp_path / "good"
        good.mkdir()
        (good / "child.yaml").write_text(_VALID_PLAN_YAML, encoding="utf-8")
        # A subdir whose iteration will raise OSError.
        bad = tmp_path / "bad"
        bad.mkdir()

        set_project_root(tmp_path)

        real_iterdir = Path.iterdir

        def _patched_iterdir(self: Path) -> Any:
            if self.name == "bad":
                raise OSError("permission denied")
            return real_iterdir(self)

        # Patch the bound method used by browse_files' inner loop.
        import maestro_cli.web.routes_api as routes_api  # noqa: F401

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "iterdir", _patched_iterdir)
            resp = client.get("/api/files/browse")

        assert resp.status_code == 200
        data = resp.json()
        names = {entry["name"] for entry in data}
        assert "top.yaml" in names
        assert "child.yaml" in names

    def test_outer_iterdir_oserror_returns_empty(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """If the root iterdir itself raises OSError, browse returns an empty list."""
        set_project_root(tmp_path)

        root_resolved = tmp_path.resolve()

        def _patched_iterdir(self: Path) -> Any:
            if self.resolve() == root_resolved:
                raise OSError("root unreadable")
            return iter(())

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "iterdir", _patched_iterdir)
            resp = client.get("/api/files/browse")

        assert resp.status_code == 200
        assert resp.json() == []


# ===========================================================================
# start_run — background thread fallback + exception handler (lines ~734-745)
# ===========================================================================

class TestStartRunThreadFallback:
    @pytest.fixture(autouse=True)
    def _confine_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.get_project_root", lambda: tmp_path,
        )

    def test_thread_match_fallback_binds_result(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When get_run can't find the result by name/id, the thread-match loop binds it.

        We register a RunState (under a different run_id than the result's
        run_path.name / run_id) whose thread is the running worker thread. The
        _run body then falls through to the list_active_runs() loop and matches
        by `candidate.thread is threading.current_thread()`.
        """
        from time import monotonic, sleep

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        run_path = tmp_path / "runs" / "mismatch-run"
        run_path.mkdir(parents=True)

        now = datetime.now(UTC)
        result = PlanRunResult(
            plan_name="test-plan",
            run_id="result-only-id",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            execution_profile="yolo",
            task_results={},
        )

        def _mock_run_plan(*_a: Any, **_kw: Any) -> PlanRunResult:
            # Register a state keyed by an id that does NOT match the result's
            # run_path.name ("mismatch-run") or run_id ("result-only-id"),
            # but whose thread IS the current worker thread.
            state = RunState(
                run_id="some-other-key",
                plan_name="test-plan",
                task_ids=["t1"],
                run_path=run_path,
                started_at=now,
                thread=threading.current_thread(),
                execution_profile="plan",
            )
            register_run(state)
            return result

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", _mock_run_plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.post("/api/runs", json={"plan_path": str(plan_file)})
        assert resp.status_code == 200

        # The bound state should now carry the result (set by the fallback loop).
        deadline = monotonic() + 1.0
        bound = None
        from maestro_cli.web.state import get_run as _get_run
        while monotonic() < deadline:
            bound = _get_run("some-other-key")
            if bound is not None and bound.result is not None:
                break
            sleep(0.02)
        assert bound is not None
        assert bound.result is result
        assert bound.execution_profile == "yolo"

    def test_run_plan_exception_recorded_on_thread_state(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If run_plan raises, the worker stores the error on the thread's RunState."""
        from time import monotonic, sleep

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        (tmp_path / "runs").mkdir(parents=True)
        captured: dict[str, Any] = {}

        def _boom_run_plan(*_a: Any, **_kw: Any) -> PlanRunResult:
            # Register a state whose thread is this worker so the except branch
            # can find it and record the error.
            state = RunState(
                run_id="err-thread-key",
                plan_name="test-plan",
                task_ids=["t1"],
                run_path=tmp_path / "runs",
                started_at=datetime.now(UTC),
                thread=threading.current_thread(),
            )
            register_run(state)
            captured["state"] = state
            raise RuntimeError("scheduler exploded")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", _boom_run_plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.post("/api/runs", json={"plan_path": str(plan_file)})
        # Endpoint itself still responds 200; the failure lives on the run state.
        assert resp.status_code == 200

        deadline = monotonic() + 1.0
        state = captured["state"]
        while monotonic() < deadline:
            if state.error:
                break
            sleep(0.02)
        assert state.error is not None
        assert "scheduler exploded" in state.error
