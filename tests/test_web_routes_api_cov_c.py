"""Coverage tests for src/maestro_cli/web/routes_api.py list/stats/detail paths.

These tests drive the historical-run aggregation endpoints
(`GET /api/runs`, `GET /api/runs/stats`) and the active-run detail path
(`GET /api/runs/{run_id}`) through their edge-case branches:

- a configured run root that does not exist (skip / continue)
- `OSError` while stat-ing a run dir (swallowed)
- malformed `task_results` (not a dict) in a manifest
- `plan_path` filtering for stats (success and load failure)
- duration / cost / token parse failures (swallowed, defaulted)
- non-dict task results in stats aggregation
- the `>= 20` / `>= 12` truncation breaks for cost-by-run / cost-by-model /
  tokens-by-model
- a finished active run whose manifest JSON is corrupt (decode error path)

Everything that touches the filesystem uses tmp_path; the run-root discovery
helper is monkeypatched so no real `.maestro-runs` directories are scanned.
No engine, network, or git calls occur.
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

from maestro_cli.web import create_app
from maestro_cli.web.state import (
    RunState,
    _active_runs,
    _lock,
    register_run,
    set_project_roots,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Create <run_dir>/run_manifest.json with the given content."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _patch_run_roots(
    monkeypatch: pytest.MonkeyPatch, roots: list[Path],
) -> None:
    """Force _discover_run_roots() to return exactly `roots`."""
    monkeypatch.setattr(
        "maestro_cli.web.routes_api._discover_run_roots",
        lambda: list(roots),
    )


class _FakeDirEntry:
    """A directory-like entry whose stat() raises OSError.

    Mimics just enough of the Path interface that routes_api iterates over:
    `.is_dir()`, `.name`, `.stat()`, and the `/` join operator.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def is_dir(self) -> bool:
        return True

    def stat(self, *args: Any, **kwargs: Any) -> Any:
        raise OSError("stat failed")

    def __truediv__(self, other: str) -> Path:  # pragma: no cover - unused path
        return Path(self.name) / other


def _patch_iterdir_with_failing_entry(
    monkeypatch: pytest.MonkeyPatch, run_root: Path,
) -> None:
    """Make run_root.iterdir() yield one entry whose stat() raises OSError."""
    real_iterdir = Path.iterdir

    def _iterdir(self: Path) -> Any:
        if self == run_root:
            return iter([_FakeDirEntry("20260101_a")])
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _iterdir)


def _make_run_state(
    run_id: str,
    run_path: Path,
    *,
    finished: bool = True,
    task_ids: list[str] | None = None,
) -> RunState:
    """Build a RunState with a controllable thread-alive status."""
    thread = MagicMock(spec=threading.Thread)
    thread.is_alive.return_value = not finished
    return RunState(
        run_id=run_id,
        plan_name="p",
        task_ids=task_ids or [],
        run_path=run_path,
        started_at=datetime.now(UTC),
        thread=thread,
        result=None,
        task_graph={},
    )


# ===========================================================================
# GET /api/runs — historical run discovery edge cases
# ===========================================================================

class TestListRunsEdges:
    def test_nonexistent_run_root_is_skipped(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A configured run root that doesn't exist is skipped (continue)."""
        missing = tmp_path / "does-not-exist"
        _patch_run_roots(monkeypatch, [missing])
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_stat_oserror_is_swallowed(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If d.stat() raises OSError while listing run dirs, it is ignored."""
        run_root = tmp_path / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        _patch_run_roots(monkeypatch, [run_root])
        _patch_iterdir_with_failing_entry(monkeypatch, run_root)

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        # The directory whose stat() failed is dropped, so no runs survive.
        assert resp.json() == []

    def test_manifest_task_results_not_a_dict(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A manifest where task_results is not a dict is coerced to {}."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {"plan_name": "a", "task_results": ["not", "a", "dict"]},
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        # task_results coerced to {} -> task_count 0
        assert runs[0]["task_count"] == 0
        assert runs[0]["plan_name"] == "a"

    def test_active_run_excluded_from_filesystem_listing(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A dir whose name matches an active run id is not listed twice."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "active-001"
        _write_manifest(run_dir, {"plan_name": "a", "task_results": {}})
        _patch_run_roots(monkeypatch, [run_root])
        register_run(_make_run_state("active-001", run_dir, finished=True))

        resp = client.get("/api/runs")
        assert resp.status_code == 200
        run_ids = [r["run_id"] for r in resp.json()]
        # exactly one entry for active-001 (the active-state one, not fs dup)
        assert run_ids.count("active-001") == 1


# ===========================================================================
# GET /api/runs/stats — aggregation edge cases
# ===========================================================================

class TestRunsStatsEdges:
    def test_plan_path_resolves_run_root(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """plan_path -> load_plan + resolve_path determines the single root."""
        run_root = tmp_path / "planruns"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {"plan_name": "a", "success": True, "task_results": {}},
        )

        plan = MagicMock()
        plan.source_dir = tmp_path
        plan.run_dir = "planruns"
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.load_plan", lambda _p: plan,
        )
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.resolve_path",
            lambda _src, _rd: run_root,
        )

        resp = client.get("/api/runs/stats", params={"plan_path": "plan.yaml"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 1
        assert data["success_count"] == 1

    def test_plan_path_load_failure_yields_empty(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If load_plan raises, the plan_path branch leaves run_roots empty."""
        def _boom(_p: Any) -> Any:
            raise ValueError("bad plan")

        monkeypatch.setattr("maestro_cli.web.routes_api.load_plan", _boom)

        resp = client.get("/api/runs/stats", params={"plan_path": "x.yaml"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 0

    def test_nonexistent_run_root_skipped(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run root that does not exist is skipped (continue)."""
        _patch_run_roots(monkeypatch, [tmp_path / "nope"])
        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        assert resp.json()["total_runs"] == 0

    def test_stat_oserror_swallowed(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError from d.stat() while collecting dirs for stats is ignored."""
        run_root = tmp_path / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        _patch_run_roots(monkeypatch, [run_root])
        _patch_iterdir_with_failing_entry(monkeypatch, run_root)

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        # The dir was skipped, so no manifest collected.
        assert resp.json()["total_runs"] == 0

    def test_duration_parse_failure_swallowed(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unparseable started_at/finished_at -> duration parse except path."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {
                "plan_name": "a",
                "success": True,
                "started_at": "not-a-timestamp",
                "finished_at": "also-not",
                "task_results": {},
            },
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        # No durations collected -> avg stays 0.0
        assert data["avg_duration_sec"] == 0.0
        # recent_runs duration falls back to None
        assert data["recent_runs"][0]["duration_sec"] is None

    def test_non_dict_task_result_skipped(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A task_results value that is not a dict is skipped in aggregation."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {
                "plan_name": "a",
                "success": True,
                "task_results": {
                    "t1": "this is not a dict",
                    "t2": {"status": "success", "command": "echo hi"},
                },
            },
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        dist = resp.json()["status_distribution"]
        # Only the dict task contributes a status.
        assert dist.get("success") == 1
        assert "unknown" not in dist or dist.get("unknown", 0) == 0

    def test_bad_cost_value_defaults_to_zero(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-numeric cost_usd hits the float() except path (-> 0, dropped)."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {
                "plan_name": "a",
                "success": True,
                "task_results": {
                    "t1": {
                        "status": "success",
                        "command": "claude --model sonnet",
                        "cost_usd": "not-a-number",
                    },
                },
            },
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        # cost_val defaulted to 0.0 -> no model cost bucket created.
        assert data["cost_by_model"] == []

    def test_bad_token_value_defaults_to_zero(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-numeric token total hits the int() except path (-> 0, dropped)."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {
                "plan_name": "a",
                "success": True,
                "task_results": {
                    "t1": {
                        "status": "success",
                        "command": "claude --model sonnet",
                        "token_usage": {"total_tokens": "lots"},
                    },
                },
            },
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        # tok_val defaulted to 0 -> no model token bucket created.
        assert data["tokens_by_model"] == []

    def test_recent_runs_duration_parse_failure(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """recent_runs duration parse failure leaves dur None (except path)."""
        run_root = tmp_path / "runs"
        run_dir = run_root / "20260101_a"
        _write_manifest(
            run_dir,
            {
                "plan_name": "a",
                "success": True,
                # well-formed start but garbage finish triggers ValueError
                "started_at": "2026-01-01T00:00:00",
                "finished_at": "garbage",
                "task_results": {},
            },
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        recent = resp.json()["recent_runs"]
        assert len(recent) == 1
        assert recent[0]["duration_sec"] is None

    def test_cost_by_run_truncates_at_20(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """More than 20 manifests with cost -> cost_by_run breaks at 20."""
        run_root = tmp_path / "runs"
        for i in range(25):
            run_dir = run_root / f"20260101_{i:03d}"
            _write_manifest(
                run_dir,
                {
                    "plan_name": f"p{i}",
                    "success": True,
                    "total_cost_usd": 1.0 + i,
                    "task_results": {},
                },
            )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        assert len(resp.json()["cost_by_run"]) == 20

    def test_cost_by_model_truncates_at_12(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """More than 12 distinct models with cost -> cost_by_model breaks at 12."""
        run_root = tmp_path / "runs"
        task_results: dict[str, Any] = {}
        for i in range(15):
            task_results[f"t{i}"] = {
                "status": "success",
                "command": f"claude --model model-{i:02d}",
                "cost_usd": 1.0 + i,
            }
        _write_manifest(
            run_root / "20260101_a",
            {"plan_name": "a", "success": True, "task_results": task_results},
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        assert len(resp.json()["cost_by_model"]) == 12

    def test_tokens_by_model_truncates_at_12(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """More than 12 distinct models with tokens -> tokens_by_model breaks at 12."""
        run_root = tmp_path / "runs"
        task_results: dict[str, Any] = {}
        for i in range(15):
            task_results[f"t{i}"] = {
                "status": "success",
                "command": f"claude --model tokmodel-{i:02d}",
                "token_usage": {"total_tokens": 100 + i},
            }
        _write_manifest(
            run_root / "20260101_a",
            {"plan_name": "a", "success": True, "task_results": task_results},
        )
        _patch_run_roots(monkeypatch, [run_root])

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        assert len(resp.json()["tokens_by_model"]) == 12


# ===========================================================================
# GET /api/runs/{run_id} — finished active run with corrupt manifest
# ===========================================================================

class TestRunDetailCorruptManifest:
    def test_finished_active_run_bad_manifest_falls_through(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A finished active run with corrupt manifest JSON -> decode except path.

        The endpoint catches JSONDecodeError and falls through to the
        active-summary branch, returning a 200 with a task_results payload
        rather than raising.
        """
        run_dir = tmp_path / "runs" / "active-bad"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Corrupt manifest so json.loads raises JSONDecodeError.
        (run_dir / "run_manifest.json").write_text("{ not json", encoding="utf-8")
        _patch_run_roots(monkeypatch, [tmp_path / "runs"])

        register_run(
            _make_run_state(
                "active-bad", run_dir, finished=True, task_ids=["t1"],
            ),
        )

        resp = client.get("/api/runs/active-bad")
        assert resp.status_code == 200
        data = resp.json()
        # Fell through to the summary branch (carries run_id + task_results).
        assert data["run_id"] == "active-bad"
        assert "task_results" in data
