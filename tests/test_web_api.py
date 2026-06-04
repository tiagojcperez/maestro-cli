from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from maestro_cli.errors import PlanValidationError
from maestro_cli.models import PlanDefaults, PlanRunResult, PlanSpec, TaskResult, TaskSpec
from maestro_cli.web import create_app
from maestro_cli.web.state import (
    RunState,
    _active_runs,
    _lock,
    register_run,
    remove_run,
    set_project_root,
    set_project_roots,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_VALID_PLAN_YAML = """\
version: 1
name: test-plan
max_parallel: 2
fail_fast: true
tasks:
  - id: t1
    command: "echo hello"
  - id: t2
    depends_on: [t1]
    command: "echo world"
"""

_INVALID_YAML = """\
version: 1
name: test
tasks: "not a list"
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
    """Build a minimal PlanSpec for mocking."""
    if task_ids is None:
        task_ids = ["t1", "t2"]
    tasks = [TaskSpec(id=tid, command="echo hi") for tid in task_ids]
    return PlanSpec(
        version=1,
        name=name,
        tasks=tasks,
        source_path=source_path,
        run_dir=run_dir,
    )


def _make_plan_run_result(
    plan_name: str = "test-plan",
    run_id: str = "20260226_test-plan",
    run_path: Path | None = None,
    success: bool = True,
    task_graph: dict[str, dict[str, Any]] | None = None,
) -> PlanRunResult:
    """Build a PlanRunResult for mocking."""
    now = datetime.now(UTC)
    return PlanRunResult(
        plan_name=plan_name,
        run_id=run_id,
        run_path=run_path or Path("."),
        started_at=now,
        finished_at=now,
        success=success,
        task_graph=task_graph or {},
    )


def _make_task_result(
    task_id: str = "t1",
    status: str = "success",
    exit_code: int = 0,
) -> TaskResult:
    """Build a TaskResult for mocking."""
    now = datetime.now(UTC)
    return TaskResult(
        task_id=task_id,
        status=status,
        exit_code=exit_code,
        started_at=now,
        finished_at=now,
        duration_sec=1.0,
        command="echo hi",
        log_path=Path("t1.log"),
        result_path=Path("t1.result.json"),
    )


def _make_run_state(
    run_id: str = "test-run-001",
    plan_name: str = "test-plan",
    task_ids: list[str] | None = None,
    run_path: Path | None = None,
    finished: bool = True,
    execution_profile: str = "plan",
    dry_run: bool = False,
    result: PlanRunResult | None = None,
    task_graph: dict[str, dict[str, object]] | None = None,
) -> RunState:
    """Build a RunState with a controllable thread alive status."""
    if task_ids is None:
        task_ids = ["t1", "t2"]
    thread = MagicMock(spec=threading.Thread)
    thread.is_alive.return_value = not finished
    return RunState(
        run_id=run_id,
        plan_name=plan_name,
        task_ids=task_ids,
        run_path=run_path or Path("."),
        started_at=datetime.now(UTC),
        thread=thread,
        execution_profile=execution_profile,
        dry_run=dry_run,
        result=result,
        task_graph=task_graph or {},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_active_runs() -> None:
    """Ensure global run state is clean for every test."""
    set_project_roots([Path(".")])
    with _lock:
        _active_runs.clear()
    yield  # type: ignore[misc]
    set_project_roots([Path(".")])
    with _lock:
        _active_runs.clear()


@pytest.fixture
def client() -> TestClient:
    """Return a Starlette TestClient wrapping the Maestro web app."""
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# GET /api/health
# ===========================================================================

class TestHealth:
    def test_returns_ok_status(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_returns_version(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        data = resp.json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert data["version"]  # non-empty

    def test_version_matches_package(self, client: TestClient) -> None:
        from maestro_cli import __version__
        resp = client.get("/api/health")
        assert resp.json()["version"] == __version__


# ===========================================================================
# POST /api/plans/validate
# ===========================================================================

class TestValidate:
    def test_valid_yaml_content(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = _make_plan_spec()
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.load_plan",
            lambda _path: plan,
        )
        resp = client.post("/api/plans/validate", json={"yaml_content": _VALID_PLAN_YAML})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["plan"]["name"] == "test-plan"
        assert data["plan"]["tasks"] == 2
        assert data["plan"]["task_ids"] == ["t1", "t2"]

    def test_valid_yaml_returns_max_parallel(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = _make_plan_spec()
        plan.max_parallel = 4
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.load_plan",
            lambda _path: plan,
        )
        resp = client.post("/api/plans/validate", json={"yaml_content": _VALID_PLAN_YAML})
        assert resp.json()["plan"]["max_parallel"] == 4

    def test_valid_yaml_returns_fail_fast(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = _make_plan_spec()
        plan.fail_fast = False
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.load_plan",
            lambda _path: plan,
        )
        resp = client.post("/api/plans/validate", json={"yaml_content": _VALID_PLAN_YAML})
        assert resp.json()["plan"]["fail_fast"] is False

    def test_valid_path(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec()
        monkeypatch.setattr("maestro_cli.web.routes_api.get_project_root", lambda: tmp_path)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.load_plan",
            lambda _path: plan,
        )
        resp = client.post("/api/plans/validate", json={"path": str(plan_file)})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    def test_path_outside_project_root_rejected(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A path outside the project root is rejected (no arbitrary file reads)."""
        monkeypatch.setattr("maestro_cli.web.routes_api.get_project_root", lambda: tmp_path)
        resp = client.post("/api/plans/validate", json={"path": "/etc/passwd"})
        assert resp.status_code == 400
        assert "project root" in resp.json()["detail"].lower()

    def test_invalid_yaml_content(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_path: Any) -> None:
            raise PlanValidationError("'tasks' must be a list")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _raise)
        resp = client.post("/api/plans/validate", json={"yaml_content": _INVALID_YAML})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert "tasks" in data["error"]

    def test_missing_both_fields_returns_400(self, client: TestClient) -> None:
        resp = client.post("/api/plans/validate", json={})
        assert resp.status_code == 400
        assert "yaml_content" in resp.json()["detail"].lower() or "path" in resp.json()["detail"].lower()

    def test_missing_both_fields_explicit_none(self, client: TestClient) -> None:
        resp = client.post("/api/plans/validate", json={"yaml_content": None, "path": None})
        assert resp.status_code == 400

    def test_path_that_does_not_exist(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        missing = tmp_path / "nonexistent.yaml"

        def _raise(_path: Any) -> None:
            raise PlanValidationError(f"Plan file not found: {missing}")

        monkeypatch.setattr("maestro_cli.web.routes_api.get_project_root", lambda: tmp_path)
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _raise)
        resp = client.post("/api/plans/validate", json={"path": str(missing)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert "not found" in data["error"].lower()

    def test_yaml_content_creates_temp_file(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When yaml_content is provided, a temporary file is created and passed to load_plan."""
        captured_path: list[Path] = []

        def _capture_path(path: Any) -> PlanSpec:
            captured_path.append(Path(path))
            return _make_plan_spec()

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _capture_path)
        resp = client.post("/api/plans/validate", json={"yaml_content": _VALID_PLAN_YAML})
        assert resp.status_code == 200
        assert len(captured_path) == 1
        # The temp file should have been cleaned up after the request
        assert not captured_path[0].exists()

    def test_path_preferred_over_yaml_content(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both path and yaml_content are provided, path takes precedence."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        captured_path: list[str] = []

        def _capture(path: Any) -> PlanSpec:
            captured_path.append(str(path))
            return _make_plan_spec()

        monkeypatch.setattr("maestro_cli.web.routes_api.get_project_root", lambda: tmp_path)
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _capture)
        resp = client.post("/api/plans/validate", json={
            "path": str(plan_file),
            "yaml_content": _VALID_PLAN_YAML,
        })
        assert resp.status_code == 200
        # The path should be the file path, not a temp file
        assert captured_path[0] == str(plan_file)


# ===========================================================================
# POST /api/runs
# ===========================================================================

class TestStartRun:
    def test_start_run_returns_plan_info(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        # Create the run dir so the endpoint can discover it
        run_dir = tmp_path / "runs" / "20260226_test-plan"
        run_dir.mkdir(parents=True)

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", lambda *a, **kw: _make_plan_run_result())
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _base, _rel: tmp_path / "runs")

        resp = client.post("/api/runs", json={"plan_path": str(plan_file)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_name"] == "test-plan"
        assert "t1" in data["tasks"]
        assert "t2" in data["tasks"]
        assert "run_id" in data

    def test_start_run_with_dry_run(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        run_dir = tmp_path / "runs" / "20260226_dry"
        run_dir.mkdir(parents=True)

        captured_kwargs: dict[str, Any] = {}

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            captured_kwargs.update(kwargs)
            return _make_plan_run_result()

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "runs")

        resp = client.post("/api/runs", json={
            "plan_path": str(plan_file),
            "dry_run": True,
        })
        assert resp.status_code == 200
        assert captured_kwargs.get("dry_run") is True

    def test_start_run_invalid_plan_returns_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_path: Any) -> None:
            raise PlanValidationError("bad plan")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _raise)
        resp = client.post("/api/runs", json={"plan_path": "/nonexistent.yaml"})
        assert resp.status_code == 400
        assert "bad plan" in resp.json()["detail"]

    def test_start_run_passes_execution_profile(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        run_dir = tmp_path / "runs" / "20260226_prof"
        run_dir.mkdir(parents=True)

        captured_kwargs: dict[str, Any] = {}

        def _mock_run(*args: Any, **kwargs: Any) -> PlanRunResult:
            captured_kwargs.update(kwargs)
            return _make_plan_run_result()

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", _mock_run)
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "runs")

        resp = client.post("/api/runs", json={
            "plan_path": str(plan_file),
            "execution_profile": "yolo",
        })
        assert resp.status_code == 200
        assert captured_kwargs.get("execution_profile") == "yolo"

    def test_start_run_passes_only_and_skip(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        run_dir = tmp_path / "runs" / "20260226_filt"
        run_dir.mkdir(parents=True)

        captured_kwargs: dict[str, Any] = {}

        def _mock_run(*args: Any, **kwargs: Any) -> PlanRunResult:
            captured_kwargs.update(kwargs)
            return _make_plan_run_result()

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", _mock_run)
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "runs")

        resp = client.post("/api/runs", json={
            "plan_path": str(plan_file),
            "only": ["t1"],
            "skip": ["t2"],
        })
        assert resp.status_code == 200
        assert captured_kwargs.get("only") == {"t1"}
        assert captured_kwargs.get("skip") == {"t2"}

    def test_start_run_binds_result_by_run_path_name(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Result from background thread is attached to active RunState."""
        from time import monotonic, sleep

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        run_name = "20260226_120000_abcd12_test-plan"
        run_path = tmp_path / "runs" / run_name
        run_path.mkdir(parents=True)

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "runs")

        now = datetime.now(UTC)
        result = PlanRunResult(
            plan_name="test-plan",
            run_id="20260226_120000_abcd12",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            execution_profile="yolo",
            task_results={},
        )
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", lambda *a, **kw: result)

        start_resp = client.post("/api/runs", json={
            "plan_path": str(plan_file),
            "execution_profile": "yolo",
        })
        assert start_resp.status_code == 200
        run_id = start_resp.json()["run_id"]
        assert run_id == run_name

        deadline = monotonic() + 1.0
        data: dict[str, Any] = {}
        while monotonic() < deadline:
            detail_resp = client.get(f"/api/runs/{run_id}")
            assert detail_resp.status_code == 200
            data = detail_resp.json()
            if data.get("success") is True:
                break
            sleep(0.05)

        assert data["success"] is True
        assert data["execution_profile"] == "yolo"

    def test_start_run_no_run_dir_exists(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the run directory doesn't exist yet, run_id defaults to 'unknown'."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "nonexistent"))

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", lambda *a, **kw: _make_plan_run_result())
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "nonexistent")

        resp = client.post("/api/runs", json={"plan_path": str(plan_file)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "unknown"


# ===========================================================================
# GET /api/runs
# ===========================================================================

class TestListRuns:
    def test_empty_state_no_filesystem(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no active runs and no .maestro-runs dir, returns empty list."""
        # Ensure the default .maestro-runs does not resolve to anything real
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.Path",
            lambda p: Path(p) if p != ".maestro-runs" else MagicMock(exists=lambda: False),
        )
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        # Should be a list (possibly empty or with entries from the real cwd)
        assert isinstance(resp.json(), list)

    def test_active_runs_included(self, client: TestClient) -> None:
        """Active runs registered in state are returned."""
        state = _make_run_state(run_id="active-001", finished=False)
        register_run(state)

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["run_id"] for r in data]
        assert "active-001" in ids

    def test_active_run_shows_active_true(self, client: TestClient) -> None:
        state = _make_run_state(run_id="active-002", finished=False)
        register_run(state)

        resp = client.get("/api/runs")
        data = resp.json()
        active = [r for r in data if r["run_id"] == "active-002"]
        assert len(active) == 1
        assert active[0]["active"] is True

    def test_active_run_includes_execution_profile(
        self, client: TestClient,
    ) -> None:
        state = _make_run_state(
            run_id="active-profile",
            finished=False,
            execution_profile="yolo",
        )
        register_run(state)

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        run = [r for r in data if r["run_id"] == "active-profile"]
        assert len(run) == 1
        assert run[0]["execution_profile"] == "yolo"

    def test_active_run_includes_dry_run_flag(
        self, client: TestClient,
    ) -> None:
        state = _make_run_state(
            run_id="active-dry-run",
            finished=False,
            dry_run=True,
        )
        register_run(state)

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        run = [r for r in data if r["run_id"] == "active-dry-run"]
        assert len(run) == 1
        assert run[0]["dry_run"] is True

    def test_active_run_includes_collaboration_summary(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        state = _make_run_state(
            run_id="active-collab",
            run_path=tmp_path,
            task_ids=["plan", "apply"],
            finished=False,
            task_graph={
                "plan": {"id": "plan", "agent": "architect", "depends_on": []},
                "apply": {"id": "apply", "agent": "python-developer", "depends_on": ["plan"]},
            },
        )
        register_run(state)

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        run = [r for r in data if r["run_id"] == "active-collab"][0]
        assert run["collaboration_summary"]["owner_count"] == 2
        assert run["collaboration_summary"]["blocked_count"] == 1

    def test_historical_runs_from_filesystem(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Historical runs (with manifests) on disk are discovered."""
        # Create a fake run directory with a manifest
        run_dir = tmp_path / "runs" / "20260226_historical"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "hist-plan",
            "started_at": "2026-02-26T10:00:00+00:00",
            "success": True,
            "task_results": {"t1": {}, "t2": {}},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["run_id"] for r in data]
        assert "20260226_historical" in ids

    def test_historical_run_metadata(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = tmp_path / "runs" / "20260226_meta"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "meta-plan",
            "started_at": "2026-02-26T12:00:00+00:00",
            "success": False,
            "task_results": {"t1": {}},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        data = resp.json()
        run = [r for r in data if r["run_id"] == "20260226_meta"][0]
        assert run["plan_name"] == "meta-plan"
        assert run["success"] is False
        assert run["active"] is False
        assert run["task_count"] == 1

    def test_historical_run_includes_dry_run_from_manifest(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = tmp_path / "runs" / "20260226_dry_hist"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "dry-plan",
            "started_at": "2026-02-26T12:00:00+00:00",
            "success": True,
            "dry_run": True,
            "execution_profile": "safe",
            "task_results": {"t1": {"status": "dry_run"}},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        assert resp.status_code == 200
        data = resp.json()
        run = [r for r in data if r["run_id"] == "20260226_dry_hist"][0]
        assert run["dry_run"] is True
        assert run["execution_profile"] == "safe"

    def test_historical_run_includes_duration_and_cost(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = tmp_path / "runs" / "20260226_metrics_hist"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "metrics-plan",
            "started_at": "2026-02-26T12:00:00+00:00",
            "finished_at": "2026-02-26T12:01:30+00:00",
            "success": True,
            "execution_profile": "plan",
            "task_results": {
                "t1": {"status": "success", "cost_usd": 1.25},
                "t2": {"status": "success", "cost_usd": 0.50},
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        assert resp.status_code == 200
        data = resp.json()
        run = [r for r in data if r["run_id"] == "20260226_metrics_hist"][0]
        assert run["duration_sec"] == 90.0
        assert run["total_cost_usd"] == 1.75

    def test_active_run_not_duplicated_as_historical(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An active run whose dir also has a manifest should not appear twice."""
        run_dir = tmp_path / "runs" / "active-nodupe"
        run_dir.mkdir(parents=True)
        manifest = {"plan_name": "dup-plan", "started_at": "", "success": True, "task_results": {}}
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        state = _make_run_state(run_id="active-nodupe", run_path=run_dir, finished=False)
        register_run(state)

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        data = resp.json()
        matches = [r for r in data if r["run_id"] == "active-nodupe"]
        assert len(matches) == 1

    def test_dirs_without_manifest_skipped(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Directories without run_manifest.json are not included."""
        (tmp_path / "runs" / "no-manifest").mkdir(parents=True)

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        data = resp.json()
        ids = [r["run_id"] for r in data]
        assert "no-manifest" not in ids

    def test_corrupt_manifest_skipped(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Directories with corrupt JSON manifest are silently skipped."""
        run_dir = tmp_path / "runs" / "corrupt-json"
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text("not json", encoding="utf-8")

        plan = _make_plan_spec(source_path=tmp_path / "plan.yaml", run_dir="runs")
        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "runs",
        )

        resp = client.get("/api/runs", params={"plan_path": str(tmp_path / "plan.yaml")})
        data = resp.json()
        ids = [r["run_id"] for r in data]
        assert "corrupt-json" not in ids

    def test_plan_path_load_failure_returns_empty_historical(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the plan_path cannot be loaded, historical runs are skipped gracefully."""
        def _raise(_path: Any) -> None:
            raise PlanValidationError("bad plan")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _raise)
        resp = client.get("/api/runs", params={"plan_path": "/bad.yaml"})
        assert resp.status_code == 200
        # Should still return a list (possibly empty)
        assert isinstance(resp.json(), list)

    def test_historical_runs_discovered_in_deep_nested_run_root(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        run_dir = (
            tmp_path
            / "clients"
            / "alpha"
            / "workspace"
            / ".maestro-runs"
            / "20260227_deep-nested"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "plan_name": "deep-plan",
                "started_at": "2026-02-27T16:00:00+00:00",
                "finished_at": "2026-02-27T16:01:00+00:00",
                "success": True,
                "task_results": {"t1": {"status": "success"}},
            }),
            encoding="utf-8",
        )

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["run_id"] for r in data]
        assert "20260227_deep-nested" in ids

    def test_historical_runs_aggregated_from_multiple_project_roots(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        root_a = tmp_path / "repo-a"
        root_b = tmp_path / "repo-b"
        run_a = root_a / ".maestro-runs" / "20260227_a"
        run_b = root_b / ".maestro-runs" / "20260227_b"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)
        (run_a / "run_manifest.json").write_text(
            json.dumps({
                "plan_name": "plan-a",
                "started_at": "2026-02-27T16:00:00+00:00",
                "finished_at": "2026-02-27T16:01:00+00:00",
                "success": True,
                "task_results": {},
            }),
            encoding="utf-8",
        )
        (run_b / "run_manifest.json").write_text(
            json.dumps({
                "plan_name": "plan-b",
                "started_at": "2026-02-27T16:02:00+00:00",
                "finished_at": "2026-02-27T16:03:00+00:00",
                "success": True,
                "task_results": {},
            }),
            encoding="utf-8",
        )
        set_project_roots([root_a, root_b])

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        ids = [r["run_id"] for r in data]
        assert "20260227_a" in ids
        assert "20260227_b" in ids


# ===========================================================================
# GET /api/runs/roots
# ===========================================================================

class TestRunRoots:
    def test_returns_discovered_run_roots(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        root_a = tmp_path / "plans" / ".maestro-runs"
        root_b = tmp_path / "clients" / "alpha" / "workspace" / ".maestro-runs"
        root_a.mkdir(parents=True)
        root_b.mkdir(parents=True)

        resp = client.get("/api/runs/roots")
        assert resp.status_code == 200
        data = resp.json()

        assert data["project_root"] == str(tmp_path)
        assert data["project_roots"] == [str(tmp_path)]
        assert data["count"] == 2
        assert str(root_a) in data["run_roots"]
        assert str(root_b) in data["run_roots"]

    def test_returns_empty_when_no_run_roots(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)

        resp = client.get("/api/runs/roots")
        assert resp.status_code == 200
        data = resp.json()

        assert data["project_root"] == str(tmp_path)
        assert data["project_roots"] == [str(tmp_path)]
        assert data["count"] == 0
        assert data["run_roots"] == []

    def test_returns_multi_project_roots(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        root_a = tmp_path / "repo-a"
        root_b = tmp_path / "repo-b"
        (root_a / "plans" / ".maestro-runs").mkdir(parents=True)
        (root_b / "docs" / "plans" / ".maestro-runs").mkdir(parents=True)
        set_project_roots([root_a, root_b])

        resp = client.get("/api/runs/roots")
        assert resp.status_code == 200
        data = resp.json()

        assert data["project_root"] == str(root_a)
        assert data["project_roots"] == [str(root_a), str(root_b)]
        assert data["count"] == 2


# ===========================================================================
# GET /api/runs/{run_id}
# ===========================================================================

class TestGetRunDetail:
    def test_active_run_with_result(self, client: TestClient, tmp_path: Path) -> None:
        result = _make_plan_run_result(
            run_path=tmp_path,
            task_graph={
                "plan": {"id": "plan", "description": "Analyse", "depends_on": [], "agent": "architect"},
            },
        )
        state = _make_run_state(
            run_id="detail-001",
            run_path=tmp_path,
            finished=True,
            result=result,
        )
        register_run(state)

        resp = client.get("/api/runs/detail-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_name"] == "test-plan"
        assert data["success"] is True
        assert data["task_graph"]["plan"]["agent"] == "architect"
        assert data["collaboration_summary"]["owner_count"] == 1

    def test_active_run_without_result_returns_summary(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """An active run that hasn't finished yet returns partial info."""
        state = _make_run_state(
            run_id="detail-002",
            run_path=tmp_path,
            finished=False,
            task_graph={
                "t1": {"id": "t1", "agent": "architect", "depends_on": []},
                "t2": {"id": "t2", "agent": "python-developer", "depends_on": ["t1"]},
            },
        )
        register_run(state)

        resp = client.get("/api/runs/detail-002")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "detail-002"
        assert data["active"] is True
        assert data["execution_profile"] == "plan"
        assert "task_results" in data
        assert data["collaboration_summary"]["blocked_count"] == 1
        assert data["collaboration"]["tasks"]["t2"]["blocked_by"][0]["task_id"] == "t1"

    def test_finished_active_run_without_result_uses_manifest(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        manifest = {
            "plan_name": "finished-plan",
            "run_id": "detail-finished",
            "success": True,
            "execution_profile": "safe",
            "task_results": {},
        }
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        state = _make_run_state(
            run_id="detail-finished",
            run_path=tmp_path,
            finished=True,
            result=None,
        )
        register_run(state)

        resp = client.get("/api/runs/detail-finished")
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_name"] == "finished-plan"
        assert data["execution_profile"] == "safe"

    def test_active_run_picks_up_task_result_files(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Task .result.json files are read for in-progress runs."""
        task_result = {"task_id": "t1", "status": "success", "exit_code": 0}
        (tmp_path / "t1.result.json").write_text(
            json.dumps(task_result), encoding="utf-8",
        )

        state = _make_run_state(
            run_id="detail-003",
            run_path=tmp_path,
            task_ids=["t1", "t2"],
            finished=False,
        )
        register_run(state)

        resp = client.get("/api/runs/detail-003")
        data = resp.json()
        assert "t1" in data["task_results"]
        assert data["task_results"]["t1"]["status"] == "success"
        # t2 has no result file yet
        assert "t2" not in data["task_results"]

    def test_historical_run_from_filesystem(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A completed run found on disk is returned."""
        run_dir = tmp_path / ".maestro-runs" / "hist-run-001"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "old-plan",
            "run_id": "hist-run-001",
            "success": True,
            "task_graph": {
                "lint": {"id": "lint", "agent": "qa-engineer", "depends_on": []},
            },
            "task_results": {},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )

        set_project_root(tmp_path)

        resp = client.get("/api/runs/hist-run-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan_name"] == "old-plan"
        assert data["task_ids"] == ["lint"]
        assert data["collaboration_summary"]["owner_count"] == 1

    def test_not_found_returns_404(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_path = Path

        def _patched_path(p: str) -> Path:
            if p in (".maestro-runs", "examples/.maestro-runs"):
                return tmp_fake / p if False else original_path(p)  # noqa: SIM210
            return original_path(p)

        # Make both search paths not exist
        def _fake_path(p: str) -> MagicMock:
            m = MagicMock()
            m.exists.return_value = False
            m.__truediv__ = lambda self, other: MagicMock(exists=lambda: False)
            return m

        monkeypatch.setattr("maestro_cli.web.routes_api.Path", _fake_path)

        resp = client.get("/api/runs/nonexistent-run")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_corrupt_task_result_json_skipped(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Corrupt .result.json files are silently ignored."""
        (tmp_path / "t1.result.json").write_text("{{bad", encoding="utf-8")

        state = _make_run_state(
            run_id="detail-corrupt",
            run_path=tmp_path,
            task_ids=["t1"],
            finished=False,
        )
        register_run(state)

        resp = client.get("/api/runs/detail-corrupt")
        assert resp.status_code == 200
        data = resp.json()
        assert "t1" not in data["task_results"]


# ===========================================================================
# GET /api/runs/{run_id}/tasks/{task_id}/log
# ===========================================================================

class TestGetTaskLog:
    def test_existing_log_from_active_run(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        log_content = "line 1\nline 2\nline 3\n"
        (tmp_path / "t1.log").write_text(log_content, encoding="utf-8")

        state = _make_run_state(run_id="log-001", run_path=tmp_path)
        register_run(state)

        resp = client.get("/api/runs/log-001/tasks/t1/log")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "t1"
        assert "line 1" in data["content"]
        assert "line 3" in data["content"]

    def test_missing_log_in_active_run_returns_404(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        state = _make_run_state(run_id="log-002", run_path=tmp_path)
        register_run(state)

        resp = client.get("/api/runs/log-002/tasks/nonexistent/log")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_log_from_filesystem(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Log from a historical run found on disk."""
        run_dir = tmp_path / ".maestro-runs" / "log-hist"
        run_dir.mkdir(parents=True)
        (run_dir / "t1.log").write_text("historical log output\n", encoding="utf-8")

        set_project_root(tmp_path)

        resp = client.get("/api/runs/log-hist/tasks/t1/log")
        assert resp.status_code == 200
        assert "historical log output" in resp.json()["content"]

    def test_no_run_no_filesystem_returns_404(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When there is no active run and no filesystem match, returns 404."""
        # Change to a clean tmp directory so .maestro-runs / examples/ won't exist
        set_project_root(tmp_path)

        resp = client.get("/api/runs/no-such-run/tasks/t1/log")
        assert resp.status_code == 404

    def test_log_content_with_special_chars(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Log files with unicode/special characters are returned correctly."""
        content = "Resultado: sucesso\nAccent: cafe\nEmoji: ok\n"
        (tmp_path / "t1.log").write_text(content, encoding="utf-8")

        state = _make_run_state(run_id="log-special", run_path=tmp_path)
        register_run(state)

        resp = client.get("/api/runs/log-special/tasks/t1/log")
        assert resp.status_code == 200
        assert "sucesso" in resp.json()["content"]

    def test_empty_log_file(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """An empty log file is returned with empty content string."""
        (tmp_path / "t1.log").write_text("", encoding="utf-8")

        state = _make_run_state(run_id="log-empty", run_path=tmp_path)
        register_run(state)

        resp = client.get("/api/runs/log-empty/tasks/t1/log")
        assert resp.status_code == 200
        assert resp.json()["content"] == ""


# ===========================================================================
# DELETE /api/runs/{run_id}
# ===========================================================================

class TestDeleteRun:
    def test_delete_finished_active_run(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        run_path = tmp_path / "run-to-delete"
        run_path.mkdir()
        (run_path / "t1.log").write_text("log", encoding="utf-8")

        state = _make_run_state(run_id="del-001", run_path=run_path, finished=True)
        register_run(state)

        resp = client.delete("/api/runs/del-001")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # Directory should have been removed
        assert not run_path.exists()

    def test_delete_active_run_returns_409(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        state = _make_run_state(run_id="del-active", run_path=tmp_path, finished=False)
        register_run(state)

        resp = client.delete("/api/runs/del-active")
        assert resp.status_code == 409
        assert "active" in resp.json()["detail"].lower()

    def test_delete_historical_from_filesystem(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = tmp_path / ".maestro-runs" / "del-hist"
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

        set_project_root(tmp_path)

        resp = client.delete("/api/runs/del-hist")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert not run_dir.exists()

    def test_delete_not_found_returns_404(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        set_project_root(tmp_path)

        resp = client.delete("/api/runs/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_delete_removes_from_state(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        run_path = tmp_path / "del-state-check"
        run_path.mkdir()

        state = _make_run_state(run_id="del-state", run_path=run_path, finished=True)
        register_run(state)

        from maestro_cli.web.state import get_run
        assert get_run("del-state") is not None

        client.delete("/api/runs/del-state")
        assert get_run("del-state") is None

    def test_delete_finished_run_path_does_not_exist(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Deleting a finished run whose path was already cleaned up should still succeed."""
        nonexistent = tmp_path / "already-gone"
        state = _make_run_state(run_id="del-gone", run_path=nonexistent, finished=True)
        register_run(state)

        resp = client.delete("/api/runs/del-gone")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


# ===========================================================================
# POST /api/cleanup
# ===========================================================================

class TestCleanup:
    def test_cleanup_calls_cleanup_runs(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        deleted_paths = [runs_dir / "old-run-1", runs_dir / "old-run-2"]

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: runs_dir,
        )
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.cleanup_runs",
            lambda *a, **kw: deleted_paths,
        )

        resp = client.post("/api/cleanup", json={
            "plan_path": str(plan_file),
            "keep": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["deleted"]) == 2

    def test_cleanup_with_dry_run(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        captured_kwargs: dict[str, Any] = {}

        def _mock_cleanup(*args: Any, **kwargs: Any) -> list[Path]:
            captured_kwargs.update(kwargs)
            return []

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: runs_dir)
        monkeypatch.setattr("maestro_cli.web.routes_api.cleanup_runs", _mock_cleanup)

        resp = client.post("/api/cleanup", json={
            "plan_path": str(plan_file),
            "dry_run": True,
        })
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True
        assert captured_kwargs.get("dry_run") is True

    def test_cleanup_invalid_plan_returns_400(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_path: Any) -> None:
            raise PlanValidationError("bad plan")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _raise)
        resp = client.post("/api/cleanup", json={"plan_path": "/nonexistent.yaml"})
        assert resp.status_code == 400
        assert "bad plan" in resp.json()["detail"]

    def test_cleanup_no_run_dir_returns_empty(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "nonexistent"))

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: None,
        )

        resp = client.post("/api/cleanup", json={"plan_path": str(plan_file)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["deleted"] == []

    def test_cleanup_passes_keep_and_older_than(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        captured_kwargs: dict[str, Any] = {}

        def _mock_cleanup(*args: Any, **kwargs: Any) -> list[Path]:
            captured_kwargs.update(kwargs)
            return []

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: runs_dir)
        monkeypatch.setattr("maestro_cli.web.routes_api.cleanup_runs", _mock_cleanup)

        resp = client.post("/api/cleanup", json={
            "plan_path": str(plan_file),
            "keep": 3,
            "older_than_days": 30,
        })
        assert resp.status_code == 200
        assert captured_kwargs["keep"] == 3
        assert captured_kwargs["older_than_days"] == 30

    def test_cleanup_resolve_path_returns_nonexistent_dir(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When resolve_path returns a path that does not exist, return empty."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir="runs")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _b, _r: tmp_path / "does-not-exist",
        )

        resp = client.post("/api/cleanup", json={"plan_path": str(plan_file)})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ===========================================================================
# Edge cases and cross-cutting concerns
# ===========================================================================

class TestEdgeCases:
    def test_cors_headers_present(self, client: TestClient) -> None:
        """CORS middleware should add the appropriate headers."""
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS preflight should succeed
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_root_redirects_to_static(self, client: TestClient) -> None:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert "/static/index.html" in resp.headers["location"]

    def test_unknown_api_route_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/nonexistent")
        assert resp.status_code in (404, 405)

    def test_validate_wrong_method_returns_405(self, client: TestClient) -> None:
        resp = client.get("/api/plans/validate")
        assert resp.status_code == 405

    def test_cleanup_wrong_method_returns_405(self, client: TestClient) -> None:
        resp = client.get("/api/cleanup")
        assert resp.status_code == 405

    def test_start_run_missing_both_plan_path_and_content_returns_400(
        self, client: TestClient,
    ) -> None:
        """Neither plan_path nor yaml_content — should return 400."""
        resp = client.post("/api/runs", json={})
        assert resp.status_code == 400
        assert "plan_path" in resp.json()["detail"].lower() or "yaml_content" in resp.json()["detail"].lower()


# ===========================================================================
# GET /api/runs/stats
# ===========================================================================

class TestRunsStats:
    """Tests for the /api/runs/stats endpoint."""

    def test_no_runs_returns_zeros(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When there are no runs, stats should return zeroes."""
        set_project_root(tmp_path)
        (tmp_path / ".maestro-runs").mkdir()

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0
        assert data["success_count"] == 0
        assert data["failed_count"] == 0
        assert data["total_cost_usd"] is None
        assert data["avg_duration_sec"] == 0.0
        assert data["recent_runs"] == []
        assert data["cost_by_run"] == []
        assert data["cost_by_model"] == []
        assert data["status_distribution"] == {}

    def test_discovers_stats_from_deep_nested_run_root(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        run_dir = (
            tmp_path
            / "projects"
            / "team-a"
            / "delivery"
            / ".maestro-runs"
            / "20260227_stats_deep"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({
                "plan_name": "stats-deep-plan",
                "run_id": "20260227_stats_deep",
                "success": True,
                "started_at": "2026-02-27T16:00:00",
                "finished_at": "2026-02-27T16:02:00",
                "task_results": {
                    "t1": {"status": "success", "duration_sec": 120},
                },
            }),
            encoding="utf-8",
        )

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 1
        assert data["success_count"] == 1
        assert data["failed_count"] == 0

    def test_stats_aggregated_from_multiple_project_roots(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        root_a = tmp_path / "repo-a"
        root_b = tmp_path / "repo-b"
        run_a = root_a / ".maestro-runs" / "run-a"
        run_b = root_b / ".maestro-runs" / "run-b"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)
        (run_a / "run_manifest.json").write_text(
            json.dumps({
                "plan_name": "plan-a",
                "run_id": "run-a",
                "success": True,
                "started_at": "2026-02-27T16:00:00",
                "finished_at": "2026-02-27T16:01:00",
                "task_results": {"t1": {"status": "success"}},
            }),
            encoding="utf-8",
        )
        (run_b / "run_manifest.json").write_text(
            json.dumps({
                "plan_name": "plan-b",
                "run_id": "run-b",
                "success": False,
                "started_at": "2026-02-27T16:10:00",
                "finished_at": "2026-02-27T16:12:00",
                "task_results": {"t1": {"status": "failed"}},
            }),
            encoding="utf-8",
        )
        set_project_roots([root_a, root_b])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 2
        assert data["success_count"] == 1
        assert data["failed_count"] == 1

    def test_aggregates_multiple_runs(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stats should aggregate metrics across multiple run manifests."""
        set_project_root(tmp_path)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        # Run 1: success, with cost
        run1 = run_root / "20260226_run1"
        run1.mkdir()
        (run1 / "run_manifest.json").write_text(json.dumps({
            "plan_name": "plan-a",
            "run_id": "20260226_run1",
            "success": True,
            "started_at": "2026-02-26T10:00:00",
            "finished_at": "2026-02-26T10:05:00",
            "total_cost_usd": 5.50,
            "task_results": {
                "t1": {"status": "success", "duration_sec": 120},
                "t2": {"status": "success", "duration_sec": 180},
            },
        }), encoding="utf-8")

        # Run 2: failed, no cost
        run2 = run_root / "20260226_run2"
        run2.mkdir()
        (run2 / "run_manifest.json").write_text(json.dumps({
            "plan_name": "plan-b",
            "run_id": "20260226_run2",
            "success": False,
            "started_at": "2026-02-26T11:00:00",
            "finished_at": "2026-02-26T11:02:00",
            "task_results": {
                "t1": {"status": "failed", "duration_sec": 60},
            },
        }), encoding="utf-8")

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_runs"] == 2
        assert data["success_count"] == 1
        assert data["failed_count"] == 1
        assert data["total_cost_usd"] == 5.50
        assert data["avg_duration_sec"] > 0
        assert len(data["recent_runs"]) == 2
        assert len(data["cost_by_run"]) == 1  # only run1 has cost

        # Status distribution
        dist = data["status_distribution"]
        assert dist.get("success", 0) == 2
        assert dist.get("failed", 0) == 1

    def test_corrupt_manifest_skipped(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corrupt manifest files should be silently skipped."""
        set_project_root(tmp_path)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        bad_run = run_root / "bad_run"
        bad_run.mkdir()
        (bad_run / "run_manifest.json").write_text("not valid json", encoding="utf-8")

        good_run = run_root / "good_run"
        good_run.mkdir()
        (good_run / "run_manifest.json").write_text(json.dumps({
            "plan_name": "good",
            "run_id": "good_run",
            "success": True,
            "started_at": "2026-02-26T10:00:00",
            "finished_at": "2026-02-26T10:01:00",
            "task_results": {"t1": {"status": "success"}},
        }), encoding="utf-8")

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        assert resp.json()["total_runs"] == 1

    def test_recent_runs_limited_to_20(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """recent_runs should return at most 20 entries."""
        set_project_root(tmp_path)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        for i in range(25):
            d = run_root / f"run_{i:03d}"
            d.mkdir()
            (d / "run_manifest.json").write_text(json.dumps({
                "plan_name": f"plan-{i}",
                "run_id": f"run_{i:03d}",
                "success": True,
                "started_at": f"2026-02-26T{10 + i // 60:02d}:{i % 60:02d}:00",
                "finished_at": f"2026-02-26T{10 + i // 60:02d}:{i % 60:02d}:30",
                "task_results": {},
            }), encoding="utf-8")

        resp = client.get("/api/runs/stats")
        data = resp.json()
        assert data["total_runs"] == 25
        assert len(data["recent_runs"]) == 20

    def test_cost_by_run_only_includes_runs_with_cost(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cost_by_run should only include runs that have total_cost_usd."""
        set_project_root(tmp_path)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        # Run with cost
        r1 = run_root / "cost_run"
        r1.mkdir()
        (r1 / "run_manifest.json").write_text(json.dumps({
            "plan_name": "has-cost",
            "run_id": "cost_run",
            "success": True,
            "started_at": "2026-02-26T10:00:00",
            "finished_at": "2026-02-26T10:01:00",
            "total_cost_usd": 3.14,
            "task_results": {},
        }), encoding="utf-8")

        # Run without cost
        r2 = run_root / "no_cost_run"
        r2.mkdir()
        (r2 / "run_manifest.json").write_text(json.dumps({
            "plan_name": "no-cost",
            "run_id": "no_cost_run",
            "success": True,
            "started_at": "2026-02-26T10:02:00",
            "finished_at": "2026-02-26T10:03:00",
            "task_results": {},
        }), encoding="utf-8")

        resp = client.get("/api/runs/stats")
        data = resp.json()
        assert data["total_runs"] == 2
        assert len(data["cost_by_run"]) == 1
        assert data["cost_by_run"][0]["total_cost_usd"] == 3.14

    def test_returns_structure_fields(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Response should contain all expected top-level fields."""
        set_project_root(tmp_path)
        (tmp_path / ".maestro-runs").mkdir()

        resp = client.get("/api/runs/stats")
        data = resp.json()

        expected_keys = {
            "total_runs", "success_count", "failed_count",
            "total_cost_usd", "avg_duration_sec",
            "recent_runs", "cost_by_run", "cost_by_model", "status_distribution",
            "total_tokens", "avg_tokens_per_run", "tokens_by_model",
        }
        assert set(data.keys()) == expected_keys

    def test_cost_by_model_aggregates_from_task_results(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        set_project_root(tmp_path)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        run1 = run_root / "run_model_1"
        run1.mkdir()
        (run1 / "run_manifest.json").write_text(json.dumps({
            "plan_name": "model-test-a",
            "run_id": "run_model_1",
            "success": True,
            "started_at": "2026-02-26T10:00:00",
            "finished_at": "2026-02-26T10:01:00",
            "task_results": {
                "t1": {
                    "status": "success",
                    "cost_usd": 1.25,
                    "command": "node codex.js exec --json -m gpt-5.3-codex prompt",
                },
                "t2": {
                    "status": "success",
                    "cost_usd": 0.75,
                    "command": "claude --print --output-format stream-json --model sonnet prompt",
                },
            },
        }), encoding="utf-8")

        run2 = run_root / "run_model_2"
        run2.mkdir()
        (run2 / "run_manifest.json").write_text(json.dumps({
            "plan_name": "model-test-b",
            "run_id": "run_model_2",
            "success": True,
            "started_at": "2026-02-26T10:02:00",
            "finished_at": "2026-02-26T10:03:00",
            "task_results": {
                "t3": {
                    "status": "success",
                    "cost_usd": 0.50,
                    "command": "node codex.js exec --json -m gpt-5.3-codex prompt",
                },
            },
        }), encoding="utf-8")

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        rows = data["cost_by_model"]

        assert len(rows) == 2
        assert rows[0]["model"] == "gpt-5.3-codex"
        assert rows[0]["total_cost_usd"] == 1.75
        assert rows[0]["task_count"] == 2
        assert rows[0]["avg_cost_usd"] == 0.875

        assert rows[1]["model"] == "sonnet"
        assert rows[1]["total_cost_usd"] == 0.75
        assert rows[1]["task_count"] == 1

    def test_no_run_dir_returns_empty_stats(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When .maestro-runs doesn't exist, return empty stats."""
        set_project_root(tmp_path)
        # Don't create .maestro-runs — it shouldn't exist

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0


# ===========================================================================
# GET /api/plans/examples
# ===========================================================================


class TestListExamplePlans:
    def test_returns_yaml_files_from_examples_dir(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        (examples_dir / "demo.yaml").write_text("version: 1", encoding="utf-8")
        (examples_dir / "advanced.yml").write_text("version: 1", encoding="utf-8")
        (examples_dir / "readme.md").write_text("not a plan", encoding="utf-8")

        resp = client.get("/api/plans/examples")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["name"] for e in data]
        assert "demo" in names
        assert "advanced" in names
        assert "readme" not in names

    def test_returns_yaml_files_from_plans_dir(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        (plans_dir / "my-plan.yaml").write_text("version: 1", encoding="utf-8")

        resp = client.get("/api/plans/examples")
        assert resp.status_code == 200
        data = resp.json()
        assert any(e["name"] == "my-plan" for e in data)

    def test_returns_both_examples_and_plans(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        (tmp_path / "examples").mkdir()
        (tmp_path / "examples" / "a.yaml").write_text("v: 1", encoding="utf-8")
        (tmp_path / "plans").mkdir()
        (tmp_path / "plans" / "b.yaml").write_text("v: 1", encoding="utf-8")

        resp = client.get("/api/plans/examples")
        data = resp.json()
        names = [e["name"] for e in data]
        assert "a" in names
        assert "b" in names

    def test_empty_when_no_directories(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        resp = client.get("/api/plans/examples")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_path_uses_forward_slashes(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()
        (examples_dir / "test.yaml").write_text("v: 1", encoding="utf-8")

        resp = client.get("/api/plans/examples")
        data = resp.json()
        assert len(data) == 1
        assert "\\" not in data[0]["path"]
        assert "examples/test.yaml" == data[0]["path"]


# ===========================================================================
# POST /api/plans/validate — task_details
# ===========================================================================


class TestValidateTaskDetails:
    def test_valid_plan_includes_task_details(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = PlanSpec(
            version=1,
            name="detail-test",
            tasks=[
                TaskSpec(id="t1", command="echo hi", description="First task"),
                TaskSpec(
                    id="t2",
                    engine="claude",
                    model="sonnet",
                    prompt="do stuff",
                    depends_on=["t1"],
                    allow_failure=True,
                ),
            ],
        )
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.load_plan", lambda _: plan,
        )
        resp = client.post("/api/plans/validate", json={"yaml_content": "v: 1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        details = data["plan"]["task_details"]
        assert len(details) == 2

        # First task: shell command
        assert details[0]["id"] == "t1"
        assert details[0]["description"] == "First task"
        assert details[0]["has_command"] is True
        assert details[0]["engine"] is None
        assert details[0]["depends_on"] == []

        # Second task: engine task with deps
        assert details[1]["id"] == "t2"
        assert details[1]["engine"] == "claude"
        assert details[1]["model"] == "sonnet"
        assert details[1]["has_command"] is False
        assert details[1]["depends_on"] == ["t1"]
        assert details[1]["allow_failure"] is True

    def test_invalid_plan_has_no_task_details(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_path: Any) -> None:
            raise PlanValidationError("bad plan")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _raise)
        resp = client.post("/api/plans/validate", json={"yaml_content": "v: 1"})
        data = resp.json()
        assert data["valid"] is False
        assert "task_details" not in data


# ===========================================================================
# GET /api/files/browse
# ===========================================================================


class TestBrowseFiles:
    def test_finds_yaml_in_root(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        (tmp_path / "plan.yaml").write_text("v: 1", encoding="utf-8")
        (tmp_path / "other.txt").write_text("nope", encoding="utf-8")

        resp = client.get("/api/files/browse")
        assert resp.status_code == 200
        data = resp.json()
        names = [f["name"] for f in data]
        assert "plan.yaml" in names
        assert "other.txt" not in names

    def test_finds_yaml_in_subdirs(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        sub = tmp_path / "mydir"
        sub.mkdir()
        (sub / "deep.yml").write_text("v: 1", encoding="utf-8")

        resp = client.get("/api/files/browse")
        data = resp.json()
        assert any(f["path"] == "mydir/deep.yml" for f in data)
        assert data[0]["dir"] == "mydir"

    def test_skips_hidden_dirs(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.yaml").write_text("v: 1", encoding="utf-8")

        resp = client.get("/api/files/browse")
        data = resp.json()
        assert len(data) == 0

    def test_empty_project(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        resp = client.get("/api/files/browse")
        assert resp.json() == []

    def test_paths_use_forward_slashes(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        set_project_root(tmp_path)
        sub = tmp_path / "plans"
        sub.mkdir()
        (sub / "test.yaml").write_text("v: 1", encoding="utf-8")

        resp = client.get("/api/files/browse")
        data = resp.json()
        assert "\\" not in data[0]["path"]


# ===========================================================================
# POST /api/runs — yaml_content support
# ===========================================================================


class TestStartRunYamlContent:
    def test_yaml_content_creates_temp_and_runs(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When yaml_content is provided instead of plan_path, the run should work."""
        plan = _make_plan_spec(source_path=tmp_path / "tmp.yaml", run_dir=str(tmp_path / "runs"))
        run_dir = tmp_path / "runs" / "20260226_content"
        run_dir.mkdir(parents=True)

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", lambda *a, **kw: _make_plan_run_result())
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "runs")

        resp = client.post("/api/runs", json={"yaml_content": _VALID_PLAN_YAML})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["plan_name"] == "test-plan"

    def test_missing_both_returns_400(
        self, client: TestClient,
    ) -> None:
        """When neither plan_path nor yaml_content is provided, return 400."""
        resp = client.post("/api/runs", json={"dry_run": True})
        assert resp.status_code == 400

    def test_plan_path_preferred_over_yaml_content(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both are provided, plan_path takes precedence."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
        plan = _make_plan_spec(source_path=plan_file, run_dir=str(tmp_path / "runs"))

        run_dir = tmp_path / "runs" / "20260226_both"
        run_dir.mkdir(parents=True)

        captured_path: list[str] = []

        def _capture(path: Any) -> PlanSpec:
            captured_path.append(str(path))
            return plan

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _capture)
        monkeypatch.setattr("maestro_cli.web.routes_api.run_plan", lambda *a, **kw: _make_plan_run_result())
        monkeypatch.setattr("maestro_cli.web.routes_api.resolve_path", lambda _b, _r: tmp_path / "runs")

        resp = client.post("/api/runs", json={
            "plan_path": str(plan_file),
            "yaml_content": _VALID_PLAN_YAML,
        })
        assert resp.status_code == 200
        # Should use plan_path, not yaml_content (temp file)
        assert captured_path[0] == str(plan_file)
