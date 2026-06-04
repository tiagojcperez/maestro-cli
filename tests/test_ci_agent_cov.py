"""Coverage tests for maestro_cli.ci_agent.

Drives the failure-analysis, blame-attribution, knowledge-learning, and
formatting branches of ci_agent.py. All external boundaries (blame_run,
extract_knowledge, store_knowledge) are monkeypatched so nothing real is
invoked. The manifest is read from a real temp file the test writes.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from maestro_cli import ci_agent
from maestro_cli.ci_agent import (
    CiFailureAnalysis,
    CiRemediationAction,
    analyze_ci_failure,
    format_ci_analysis,
    learn_from_ci_run,
)
from maestro_cli.models import BlameChain, BlameNode, PlanRunResult


def _write_manifest(run_path: Path, task_results: dict[str, object]) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "run_manifest.json").write_text(
        json.dumps({"task_results": task_results}, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CiFailureAnalysis.to_dict
# ---------------------------------------------------------------------------


def test_failure_analysis_to_dict_serializes_all_fields(tmp_path: Path) -> None:
    action = CiRemediationAction(
        task_id="t1",
        action="escalate_model",
        reason="failed",
        params={"escalation": ["sonnet", "opus"]},
    )
    analysis = CiFailureAnalysis(
        run_path=tmp_path / "run",
        failed_tasks=["t1"],
        root_causes=["t1"],
        cascading_failures=["t2"],
        remediation_actions=[action],
        should_retry=True,
        retry_config={"resume": True, "only": ["t1"]},
    )

    d = analysis.to_dict()

    assert d["run_path"] == str(tmp_path / "run")
    assert d["failed_tasks"] == ["t1"]
    assert d["root_causes"] == ["t1"]
    assert d["cascading_failures"] == ["t2"]
    # nested remediation action dicts are serialized via their own to_dict
    assert d["remediation_actions"] == [action.to_dict()]
    assert d["should_retry"] is True
    assert d["retry_config"] == {"resume": True, "only": ["t1"]}


# ---------------------------------------------------------------------------
# analyze_ci_failure: malformed manifest JSON
# ---------------------------------------------------------------------------


def test_analyze_returns_empty_on_malformed_manifest(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    # Invalid JSON triggers the JSONDecodeError except branch.
    (run_path / "run_manifest.json").write_text("{not valid json", encoding="utf-8")

    analysis = analyze_ci_failure(run_path)

    assert analysis.run_path == run_path
    assert analysis.failed_tasks == []
    assert analysis.should_retry is False


# ---------------------------------------------------------------------------
# analyze_ci_failure: blame attribution, cascading branch
# ---------------------------------------------------------------------------


def test_analyze_records_cascading_failures_from_blame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = tmp_path / "run"
    _write_manifest(
        run_path,
        {
            "root-task": {"status": "failed", "failure_history": []},
            "cascade-task": {"status": "failed", "failure_history": []},
        },
    )

    chain = BlameChain(
        root_task_id="root-task",
        nodes=[
            BlameNode(
                task_id="root-task",
                category="root_cause",
                confidence=0.9,
                message="boom",
            ),
            BlameNode(
                task_id="cascade-task",
                category="dependency_cascade",
                confidence=0.95,
                message="downstream",
            ),
        ],
    )
    monkeypatch.setattr(ci_agent, "blame_run", lambda rp: chain)

    analysis = analyze_ci_failure(run_path)

    assert "root-task" in analysis.root_causes
    assert "cascade-task" in analysis.cascading_failures


# ---------------------------------------------------------------------------
# analyze_ci_failure: blame raises -> fallback
# ---------------------------------------------------------------------------


def test_analyze_falls_back_when_blame_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = tmp_path / "run"
    _write_manifest(
        run_path,
        {
            "a": {"status": "failed", "failure_history": []},
            "b": {"status": "success"},
        },
    )

    def _boom(rp: Path) -> BlameChain:
        raise RuntimeError("blame exploded")

    monkeypatch.setattr(ci_agent, "blame_run", _boom)

    analysis = analyze_ci_failure(run_path)

    # Fallback assigns root_causes from failed_tasks when blame fails.
    assert analysis.failed_tasks == ["a"]
    assert analysis.root_causes == ["a"]


# ---------------------------------------------------------------------------
# analyze_ci_failure: context_exceeded remediation
# ---------------------------------------------------------------------------


def test_analyze_context_exceeded_produces_reduce_context_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_path = tmp_path / "run"
    _write_manifest(
        run_path,
        {
            "ctx-task": {
                "status": "failed",
                "failure_history": [{"category": "context_exceeded"}],
            },
        },
    )
    # No blame nodes -> root_causes stays empty, loop falls back to failed_tasks.
    monkeypatch.setattr(ci_agent, "blame_run", lambda rp: BlameChain())

    analysis = analyze_ci_failure(run_path)

    assert analysis.should_retry is True
    actions = [a for a in analysis.remediation_actions if a.action == "reduce_context"]
    assert len(actions) == 1
    action = actions[0]
    assert action.task_id == "ctx-task"
    assert action.params == {"context_compaction": "progressive"}
    # retry_config gets populated when should_retry is True.
    assert analysis.retry_config.get("resume") is True


# ---------------------------------------------------------------------------
# learn_from_ci_run
# ---------------------------------------------------------------------------


def _make_run_result(plan_name: str = "demo") -> PlanRunResult:
    now = datetime(2026, 1, 1, 12, 0, 0)
    return PlanRunResult(
        plan_name=plan_name,
        run_id="r1",
        run_path=Path("/tmp/run"),
        started_at=now,
        finished_at=now,
        success=True,
    )


def test_learn_from_ci_run_returns_zero_when_no_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ci_agent, "extract_knowledge", lambda rr: [])
    store_calls: list[object] = []
    monkeypatch.setattr(
        ci_agent,
        "store_knowledge",
        lambda *a, **k: store_calls.append((a, k)),
    )

    count = learn_from_ci_run(_make_run_result(), tmp_path)

    assert count == 0
    # Early-return path: store must NOT be called when there are no records.
    assert store_calls == []


def test_learn_from_ci_run_stores_records_and_returns_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_records = ["rec1", "rec2", "rec3"]
    monkeypatch.setattr(ci_agent, "extract_knowledge", lambda rr: list(fake_records))

    captured: dict[str, object] = {}

    def _store(plan_name: str, source_dir: Path, records: list[object]) -> None:
        captured["plan_name"] = plan_name
        captured["source_dir"] = source_dir
        captured["records"] = records

    monkeypatch.setattr(ci_agent, "store_knowledge", _store)

    run_result = _make_run_result(plan_name="my-plan")
    count = learn_from_ci_run(run_result, tmp_path)

    assert count == 3
    assert captured["plan_name"] == "my-plan"
    assert captured["source_dir"] == tmp_path
    assert captured["records"] == fake_records


# ---------------------------------------------------------------------------
# format_ci_analysis: cascading line
# ---------------------------------------------------------------------------


def test_format_ci_analysis_includes_cascading_line(tmp_path: Path) -> None:
    analysis = CiFailureAnalysis(
        run_path=tmp_path,
        root_causes=["root-x"],
        cascading_failures=["casc-a", "casc-b"],
        remediation_actions=[
            CiRemediationAction(
                task_id="root-x",
                action="escalate_model",
                reason="failed with test_failure",
            )
        ],
        should_retry=True,
        retry_config={"only": ["root-x"]},
    )

    out = format_ci_analysis(analysis)

    assert "Root causes: root-x" in out
    # The cascading branch renders both cascading task ids.
    assert "Cascading: casc-a, casc-b" in out
    assert "escalate_model" in out
    assert "retry with --only root-x" in out
