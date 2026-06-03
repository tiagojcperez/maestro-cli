from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.blame import (
    _classify_failed_task,
    _confidence_for_category,
    _load_events_evidence,
    _suggested_fix,
    blame_run,
    format_blame,
    format_blame_json,
)
from maestro_cli.models import BlameChain, BlameNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(tmp_path: Path, tasks: dict[str, object]) -> None:
    """Write a run_manifest.json with the given task results dict.

    Each task result should have: task_id, status, exit_code, message,
    depends_on, duration_sec.
    """
    manifest = {"task_results": tasks}
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_events(tmp_path: Path, events: list[dict[str, object]]) -> None:
    """Write events.jsonl with the given list of event dicts."""
    lines = [json.dumps(e) for e in events]
    (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# TestClassifyFailedTask
# ---------------------------------------------------------------------------


class TestClassifyFailedTask:
    def test_dependency_cascade(self) -> None:
        cat = _classify_failed_task(1, "some error", ["dep-a"], {"dep-a"})
        assert cat == "dependency_cascade"

    def test_no_cascade_when_dep_not_failed(self) -> None:
        cat = _classify_failed_task(1, "some error", ["dep-a"], set())
        assert cat == "root_cause"

    def test_timeout(self) -> None:
        cat = _classify_failed_task(124, "timed out", [], set())
        assert cat == "timeout_propagation"

    def test_budget_exhaustion_budget(self) -> None:
        cat = _classify_failed_task(1, "budget exceeded", [], set())
        assert cat == "budget_exhaustion"

    def test_budget_exhaustion_cost(self) -> None:
        cat = _classify_failed_task(1, "over cost limit", [], set())
        assert cat == "budget_exhaustion"

    def test_budget_exhaustion_skipped(self) -> None:
        cat = _classify_failed_task(1, "task was skipped due to budget", [], set())
        assert cat == "budget_exhaustion"

    def test_budget_exhaustion_max_cost(self) -> None:
        cat = _classify_failed_task(1, "max_cost exceeded", [], set())
        assert cat == "budget_exhaustion"

    def test_context_corruption_context(self) -> None:
        cat = _classify_failed_task(1, "context window exceeded", [], set())
        assert cat == "context_corruption"

    def test_context_corruption_token(self) -> None:
        cat = _classify_failed_task(1, "token limit reached", [], set())
        assert cat == "context_corruption"

    def test_root_cause_default(self) -> None:
        cat = _classify_failed_task(1, "unknown error", [], set())
        assert cat == "root_cause"

    def test_cascade_takes_priority_over_timeout(self) -> None:
        cat = _classify_failed_task(124, "timed out", ["dep-a"], {"dep-a"})
        assert cat == "dependency_cascade"


# ---------------------------------------------------------------------------
# TestConfidenceForCategory
# ---------------------------------------------------------------------------


class TestConfidenceForCategory:
    @pytest.mark.parametrize(
        ("category", "expected"),
        [
            ("root_cause", 0.9),
            ("timeout_propagation", 0.85),
            ("budget_exhaustion", 0.9),
            ("dependency_cascade", 0.95),
            ("context_corruption", 0.85),
        ],
    )
    def test_known_categories(self, category: str, expected: float) -> None:
        assert _confidence_for_category(category) == expected  # type: ignore[arg-type]

    def test_unknown_category_returns_06(self) -> None:
        assert _confidence_for_category("unknown") == 0.6  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestSuggestedFix
# ---------------------------------------------------------------------------


class TestSuggestedFix:
    def test_timeout(self) -> None:
        fix = _suggested_fix("timeout_propagation")
        assert "timeout" in fix.lower()

    def test_root_cause(self) -> None:
        fix = _suggested_fix("root_cause")
        assert "logs" in fix.lower()

    def test_budget(self) -> None:
        fix = _suggested_fix("budget_exhaustion")
        assert "max_cost" in fix.lower()

    def test_context(self) -> None:
        fix = _suggested_fix("context_corruption")
        assert "context_budget" in fix.lower()

    def test_cascade_includes_root(self) -> None:
        fix = _suggested_fix("dependency_cascade", "task-root")
        assert "task-root" in fix


# ---------------------------------------------------------------------------
# TestLoadEventsEvidence
# ---------------------------------------------------------------------------


class TestLoadEventsEvidence:
    def test_loads_failed_task_events(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "t1", "status": "failed", "exit_code": 1, "message": "boom"},
            {"event": "task_complete", "task_id": "t2", "status": "success"},
        ])
        evidence = _load_events_evidence(tmp_path)
        assert "t1" in evidence
        assert "t2" not in evidence
        assert any("exit_code=1" in e for e in evidence["t1"])

    def test_no_events_file(self, tmp_path: Path) -> None:
        evidence = _load_events_evidence(tmp_path)
        assert evidence == {}

    def test_malformed_json_lines_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "events.jsonl").write_text(
            "not json\n" + json.dumps({"event": "task_complete", "task_id": "t1", "status": "failed"}),
            encoding="utf-8",
        )
        evidence = _load_events_evidence(tmp_path)
        assert "t1" in evidence

    def test_soft_failed_included(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "t1", "status": "soft_failed"},
        ])
        evidence = _load_events_evidence(tmp_path)
        assert "t1" in evidence


# ---------------------------------------------------------------------------
# TestBlameRun
# ---------------------------------------------------------------------------


class TestBlameRun:
    def test_single_root_cause(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "task-a"
        assert len(chain.nodes) == 1
        assert chain.nodes[0].category == "root_cause"

    def test_dependency_cascade(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "", "depends_on": [], "duration_sec": 5.0},
            "task-b": {"task_id": "task-b", "status": "failed", "exit_code": 1, "message": "", "depends_on": ["task-a"], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "task-a"
        node_by_id = {n.task_id: n for n in chain.nodes}
        assert node_by_id["task-a"].category == "root_cause"
        assert node_by_id["task-b"].category == "dependency_cascade"
        assert node_by_id["task-b"].caused_by == "task-a"

    def test_timeout(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 124, "message": "", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "task-a"
        assert chain.nodes[0].category == "timeout_propagation"

    def test_budget_exhaustion(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "budget exceeded", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "task-a"
        assert chain.nodes[0].category == "budget_exhaustion"

    def test_no_failures(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "success", "exit_code": 0, "message": "", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == ""
        assert chain.nodes == []

    def test_missing_manifest(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "nonexistent_run"
        chain = blame_run(run_dir)
        assert isinstance(chain, BlameChain)
        assert chain.root_task_id == ""
        assert chain.nodes == []

    def test_multiple_root_causes(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "error A", "depends_on": [], "duration_sec": 5.0},
            "task-c": {"task_id": "task-c", "status": "failed", "exit_code": 1, "message": "error C", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert len(chain.nodes) == 2
        task_ids = {n.task_id for n in chain.nodes}
        assert task_ids == {"task-a", "task-c"}
        # Both are independent — no cascades
        assert all(n.category != "dependency_cascade" for n in chain.nodes)
        # The reported root is one of them
        assert chain.root_task_id in {"task-a", "task-c"}

    def test_suggested_fixes_present(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        assert len(chain.suggested_fixes) > 0

    def test_suggested_fixes_deduped(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 1, "message": "error"},
            "t2": {"status": "failed", "exit_code": 1, "message": "error"},
        })
        chain = blame_run(tmp_path)
        assert len(chain.suggested_fixes) == len(set(chain.suggested_fixes))

    def test_evidence_from_events(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 1, "message": "error"},
        })
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "t1", "status": "failed", "exit_code": 1, "message": "runtime error"},
        ])
        chain = blame_run(tmp_path)
        assert chain.nodes[0].evidence
        assert any("runtime error" in e for e in chain.nodes[0].evidence)

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text("not json", encoding="utf-8")
        chain = blame_run(tmp_path)
        assert chain.nodes == []
        assert any("Failed to load" in f for f in chain.suggested_fixes)

    def test_manifest_not_dict(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text('"just a string"', encoding="utf-8")
        chain = blame_run(tmp_path)
        assert chain.nodes == []

    def test_mixed_success_and_failure(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "ok": {"status": "success", "exit_code": 0, "message": ""},
            "bad": {"status": "failed", "exit_code": 1, "message": "fail"},
        })
        chain = blame_run(tmp_path)
        assert len(chain.nodes) == 1
        assert chain.nodes[0].task_id == "bad"

    def test_without_depends_on_in_manifest(self, tmp_path: Path) -> None:
        # depends_on not present in task results — treat as independent roots
        _write_manifest(tmp_path, {
            "a": {"status": "failed", "exit_code": 1, "message": "err"},
            "b": {"status": "failed", "exit_code": 1, "message": "err"},
        })
        chain = blame_run(tmp_path)
        for node in chain.nodes:
            assert node.category != "dependency_cascade"


# ---------------------------------------------------------------------------
# TestFormatBlame
# ---------------------------------------------------------------------------


class TestFormatBlame:
    def test_human_readable(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        output = format_blame(chain)
        assert isinstance(output, str)
        assert "task-a" in output
        assert "root_cause" in output

    def test_json_valid(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "", "depends_on": [], "duration_sec": 5.0},
        })
        chain = blame_run(tmp_path)
        raw_json = format_blame_json(chain)
        parsed = json.loads(raw_json)
        assert isinstance(parsed, dict)
        assert "root_task_id" in parsed
        assert "nodes" in parsed

    def test_empty_chain(self) -> None:
        chain = BlameChain(root_task_id="", nodes=[], suggested_fixes=[])
        output = format_blame(chain)
        assert "No failures found" in output

    def test_cascade_section(self) -> None:
        chain = BlameChain(
            root_task_id="root",
            nodes=[
                BlameNode(task_id="root", category="root_cause", confidence=0.9, message="err"),
                BlameNode(task_id="child", category="dependency_cascade", confidence=0.95, message="cascade", caused_by="root"),
            ],
            suggested_fixes=["Fix root"],
        )
        output = format_blame(chain)
        assert "Cascade" in output
        assert "child" in output
        assert "<- root" in output

    def test_suggested_fixes_numbered(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err")],
            suggested_fixes=["Fix A", "Fix B"],
        )
        output = format_blame(chain)
        assert "1. Fix A" in output
        assert "2. Fix B" in output


# ---------------------------------------------------------------------------
# TestBlameIntegration
# ---------------------------------------------------------------------------


class TestBlameIntegration:
    def test_full_cascade_scenario(self, tmp_path: Path) -> None:
        """4 tasks (A→B→C, A→D); A fails → B, C, D are dependency cascades."""
        _write_manifest(tmp_path, {
            "task-a": {"task_id": "task-a", "status": "failed", "exit_code": 1, "message": "compile error", "depends_on": [], "duration_sec": 3.0},
            "task-b": {"task_id": "task-b", "status": "failed", "exit_code": 1, "message": "dep failed", "depends_on": ["task-a"], "duration_sec": 1.0},
            "task-c": {"task_id": "task-c", "status": "failed", "exit_code": 1, "message": "dep failed", "depends_on": ["task-b"], "duration_sec": 1.0},
            "task-d": {"task_id": "task-d", "status": "failed", "exit_code": 1, "message": "dep failed", "depends_on": ["task-a"], "duration_sec": 1.0},
        })
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "task-a", "status": "failed", "exit_code": 1},
            {"event": "task_complete", "task_id": "task-b", "status": "failed", "exit_code": 1},
            {"event": "task_complete", "task_id": "task-c", "status": "failed", "exit_code": 1},
            {"event": "task_complete", "task_id": "task-d", "status": "failed", "exit_code": 1},
        ])

        chain = blame_run(tmp_path)

        assert chain.root_task_id == "task-a"
        assert len(chain.nodes) == 4

        node_by_id = {n.task_id: n for n in chain.nodes}
        assert node_by_id["task-a"].category == "root_cause"
        assert node_by_id["task-b"].category == "dependency_cascade"
        assert node_by_id["task-c"].category == "dependency_cascade"
        assert node_by_id["task-d"].category == "dependency_cascade"

        assert node_by_id["task-b"].caused_by == "task-a"
        assert node_by_id["task-c"].caused_by == "task-b"
        assert node_by_id["task-d"].caused_by == "task-a"

        assert len(chain.suggested_fixes) > 0
        assert any("task-a" in fix for fix in chain.suggested_fixes)


# ---------------------------------------------------------------------------
# Additional tests appended below
# ---------------------------------------------------------------------------

from maestro_cli.blame import _safe_float, _safe_int, _safe_str


# ---------------------------------------------------------------------------
# TestSafeStr
# ---------------------------------------------------------------------------


class TestSafeStr:
    def test_string_passthrough(self) -> None:
        assert _safe_str("hello") == "hello"

    def test_empty_string(self) -> None:
        assert _safe_str("") == ""

    def test_none_returns_empty(self) -> None:
        assert _safe_str(None) == ""

    def test_int_returns_empty(self) -> None:
        assert _safe_str(42) == ""

    def test_list_returns_empty(self) -> None:
        assert _safe_str([1, 2]) == ""

    def test_bool_returns_empty(self) -> None:
        assert _safe_str(True) == ""


# ---------------------------------------------------------------------------
# TestSafeInt
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_int_passthrough(self) -> None:
        assert _safe_int(42) == 42

    def test_zero(self) -> None:
        assert _safe_int(0) == 0

    def test_negative(self) -> None:
        assert _safe_int(-5) == -5

    def test_float_truncated(self) -> None:
        assert _safe_int(3.7) == 3

    def test_bool_returns_none(self) -> None:
        # bool is a subclass of int — but _safe_int rejects booleans
        assert _safe_int(True) is None
        assert _safe_int(False) is None

    def test_string_returns_none(self) -> None:
        assert _safe_int("42") is None

    def test_none_returns_none(self) -> None:
        assert _safe_int(None) is None

    def test_list_returns_none(self) -> None:
        assert _safe_int([1]) is None


# ---------------------------------------------------------------------------
# TestSafeFloat
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_float_passthrough(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_int_promoted(self) -> None:
        assert _safe_float(5) == 5.0
        assert isinstance(_safe_float(5), float)

    def test_bool_returns_none(self) -> None:
        assert _safe_float(True) is None
        assert _safe_float(False) is None

    def test_string_returns_none(self) -> None:
        assert _safe_float("3.14") is None

    def test_none_returns_none(self) -> None:
        assert _safe_float(None) is None

    def test_zero(self) -> None:
        assert _safe_float(0) == 0.0


# ---------------------------------------------------------------------------
# TestClassifyFailedTaskAdditional
# ---------------------------------------------------------------------------


class TestClassifyFailedTaskAdditional:
    def test_none_exit_code_no_matching_message(self) -> None:
        cat = _classify_failed_task(None, "something went wrong", [], set())
        assert cat == "root_cause"

    def test_none_exit_code_with_budget_message(self) -> None:
        cat = _classify_failed_task(None, "budget limit hit", [], set())
        assert cat == "budget_exhaustion"

    def test_cascade_with_multiple_failed_deps(self) -> None:
        cat = _classify_failed_task(1, "error", ["dep-a", "dep-b"], {"dep-a", "dep-b"})
        assert cat == "dependency_cascade"

    def test_cascade_with_only_some_deps_failed(self) -> None:
        # dep-a failed, dep-b did not
        cat = _classify_failed_task(1, "error", ["dep-a", "dep-b"], {"dep-a"})
        assert cat == "dependency_cascade"

    def test_empty_message_and_zero_exit(self) -> None:
        cat = _classify_failed_task(0, "", [], set())
        assert cat == "root_cause"

    def test_context_window_keyword(self) -> None:
        cat = _classify_failed_task(1, "exceeded context window limit", [], set())
        assert cat == "context_corruption"

    def test_case_insensitive_budget(self) -> None:
        cat = _classify_failed_task(1, "BUDGET exceeded", [], set())
        assert cat == "budget_exhaustion"

    def test_case_insensitive_token(self) -> None:
        cat = _classify_failed_task(1, "TOKEN limit hit", [], set())
        assert cat == "context_corruption"


# ---------------------------------------------------------------------------
# TestConfidenceForCategoryAdditional
# ---------------------------------------------------------------------------


class TestConfidenceForCategoryAdditional:
    def test_unknown_string_returns_default(self) -> None:
        assert _confidence_for_category("totally_made_up") == 0.6  # type: ignore[arg-type]

    def test_all_known_categories_above_06(self) -> None:
        for cat in ("root_cause", "dependency_cascade", "context_corruption",
                     "timeout_propagation", "budget_exhaustion"):
            assert _confidence_for_category(cat) > 0.6  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestSuggestedFixAdditional
# ---------------------------------------------------------------------------


class TestSuggestedFixAdditional:
    def test_unknown_category(self) -> None:
        fix = _suggested_fix("unknown")  # type: ignore[arg-type]
        assert "Investigate" in fix

    def test_cascade_without_root_id(self) -> None:
        fix = _suggested_fix("dependency_cascade")
        assert "root cause task" in fix

    def test_cascade_with_root_id(self) -> None:
        fix = _suggested_fix("dependency_cascade", "task-root")
        assert "task-root" in fix
        assert fix.endswith("task-root")


# ---------------------------------------------------------------------------
# TestLoadEventsEvidenceAdditional
# ---------------------------------------------------------------------------


class TestLoadEventsEvidenceAdditional:
    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        content = "\n\n" + json.dumps({"event": "task_complete", "task_id": "t1", "status": "failed"}) + "\n\n"
        (tmp_path / "events.jsonl").write_text(content, encoding="utf-8")
        evidence = _load_events_evidence(tmp_path)
        assert "t1" in evidence

    def test_non_dict_lines_skipped(self, tmp_path: Path) -> None:
        lines = ['"just a string"', json.dumps({"event": "task_complete", "task_id": "t1", "status": "failed"})]
        (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
        evidence = _load_events_evidence(tmp_path)
        assert "t1" in evidence

    def test_non_task_complete_events_ignored(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_start", "task_id": "t1", "status": "failed"},
            {"event": "run_complete", "task_id": "t1", "status": "failed"},
        ])
        evidence = _load_events_evidence(tmp_path)
        assert evidence == {}

    def test_message_truncated_at_120(self, tmp_path: Path) -> None:
        long_msg = "x" * 200
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "t1", "status": "failed", "message": long_msg},
        ])
        evidence = _load_events_evidence(tmp_path)
        for ev in evidence["t1"]:
            if "message=" in ev:
                # message= prefix plus max 120 chars of the message
                msg_part = ev.split("message=", 1)[1]
                assert len(msg_part) <= 120

    def test_error_field_used_as_fallback(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "t1", "status": "failed", "error": "something broke"},
        ])
        evidence = _load_events_evidence(tmp_path)
        assert any("something broke" in e for e in evidence["t1"])

    def test_missing_task_id_skipped(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_complete", "status": "failed"},
        ])
        evidence = _load_events_evidence(tmp_path)
        assert evidence == {}

    def test_multiple_events_same_task(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_complete", "task_id": "t1", "status": "failed", "exit_code": 1},
            {"event": "task_complete", "task_id": "t1", "status": "failed", "exit_code": 2},
        ])
        evidence = _load_events_evidence(tmp_path)
        assert len(evidence["t1"]) >= 4  # 2x status + 2x exit_code


# ---------------------------------------------------------------------------
# TestBlameRunAdditional
# ---------------------------------------------------------------------------


class TestBlameRunAdditional:
    def test_context_corruption_root(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 1, "message": "context window overflow"},
        })
        chain = blame_run(tmp_path)
        assert chain.nodes[0].category == "context_corruption"
        assert chain.nodes[0].confidence == 0.85

    def test_timeout_confidence(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 124, "message": ""},
        })
        chain = blame_run(tmp_path)
        assert chain.nodes[0].confidence == 0.85

    def test_root_cause_confidence(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 1, "message": "generic error"},
        })
        chain = blame_run(tmp_path)
        assert chain.nodes[0].confidence == 0.9

    def test_cascade_confidence(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 1, "message": "", "depends_on": []},
            "t2": {"status": "failed", "exit_code": 1, "message": "", "depends_on": ["t1"]},
        })
        chain = blame_run(tmp_path)
        node_by_id = {n.task_id: n for n in chain.nodes}
        assert node_by_id["t2"].confidence == 0.95

    def test_manifest_array_at_root(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
        chain = blame_run(tmp_path)
        assert chain.nodes == []
        assert any("not a valid JSON object" in f for f in chain.suggested_fixes)

    def test_manifest_missing_task_results(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text('{"other_key": 1}', encoding="utf-8")
        chain = blame_run(tmp_path)
        assert chain.nodes == []
        assert any("no task_results" in f for f in chain.suggested_fixes)

    def test_task_result_not_dict_skipped(self, tmp_path: Path) -> None:
        manifest = {"task_results": {"t1": "not_a_dict", "t2": {"status": "failed", "exit_code": 1, "message": "err"}}}
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        chain = blame_run(tmp_path)
        assert len(chain.nodes) == 1
        assert chain.nodes[0].task_id == "t2"

    def test_all_tasks_successful(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "success", "exit_code": 0, "message": ""},
            "t2": {"status": "success", "exit_code": 0, "message": ""},
            "t3": {"status": "dry_run", "exit_code": 0, "message": ""},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == ""
        assert chain.nodes == []
        assert chain.suggested_fixes == []

    def test_soft_failed_not_blamed(self, tmp_path: Path) -> None:
        """soft_failed tasks are NOT included in blame analysis (only 'failed')."""
        _write_manifest(tmp_path, {
            "t1": {"status": "soft_failed", "exit_code": 1, "message": "soft fail"},
        })
        chain = blame_run(tmp_path)
        assert chain.nodes == []

    def test_evidence_falls_back_to_manifest(self, tmp_path: Path) -> None:
        """When events.jsonl is absent, evidence comes from the manifest."""
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 42, "message": "fatal error"},
        })
        # no events.jsonl
        chain = blame_run(tmp_path)
        assert chain.nodes[0].evidence
        assert any("exit_code=42" in e for e in chain.nodes[0].evidence)
        assert any("fatal error" in e for e in chain.nodes[0].evidence)

    def test_empty_message_fallback(self, tmp_path: Path) -> None:
        """When message is empty, the node message includes exit_code."""
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 7, "message": ""},
        })
        chain = blame_run(tmp_path)
        assert "exit_code=7" in chain.nodes[0].message

    def test_root_prefers_non_cascade(self, tmp_path: Path) -> None:
        """Root cause selection prefers non-cascade nodes."""
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 124, "message": "", "depends_on": []},
            "t2": {"status": "failed", "exit_code": 1, "message": "", "depends_on": ["t1"]},
        })
        chain = blame_run(tmp_path)
        # t1 is timeout (non-cascade), t2 is cascade — root should be t1
        assert chain.root_task_id == "t1"

    def test_all_cascade_selects_highest_confidence(self, tmp_path: Path) -> None:
        """When all nodes are cascades, root is the one with highest confidence."""
        # Both tasks depend on each other — circular deps (both will be cascade)
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 1, "message": "", "depends_on": ["t2"]},
            "t2": {"status": "failed", "exit_code": 1, "message": "", "depends_on": ["t1"]},
        })
        chain = blame_run(tmp_path)
        # Both are cascades — root should still be selected
        assert chain.root_task_id in {"t1", "t2"}


# ---------------------------------------------------------------------------
# TestFormatBlameAdditional
# ---------------------------------------------------------------------------


class TestFormatBlameAdditional:
    def test_root_message_displayed(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="compile error")],
            suggested_fixes=[],
        )
        output = format_blame(chain)
        assert "compile error" in output

    def test_evidence_displayed(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(
                task_id="t1", category="root_cause", confidence=0.9,
                message="err", evidence=["exit_code=1", "message=boom"],
            )],
            suggested_fixes=[],
        )
        output = format_blame(chain)
        assert "Evidence:" in output
        assert "exit_code=1" in output
        assert "message=boom" in output

    def test_confidence_percentage_format(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err")],
            suggested_fixes=[],
        )
        output = format_blame(chain)
        assert "90%" in output

    def test_additional_root_causes_section(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[
                BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err1"),
                BlameNode(task_id="t2", category="timeout_propagation", confidence=0.85, message="timeout"),
            ],
            suggested_fixes=[],
        )
        output = format_blame(chain)
        assert "Additional root causes" in output
        assert "t2" in output
        assert "timeout_propagation" in output

    def test_cascade_without_caused_by(self) -> None:
        chain = BlameChain(
            root_task_id="root",
            nodes=[
                BlameNode(task_id="root", category="root_cause", confidence=0.9, message="err"),
                BlameNode(task_id="child", category="dependency_cascade", confidence=0.95, message="cascade"),
            ],
            suggested_fixes=[],
        )
        output = format_blame(chain)
        assert "child" in output
        assert "<-" not in output  # no caused_by, so no arrow

    def test_no_suggested_fixes_section(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err")],
            suggested_fixes=[],
        )
        output = format_blame(chain)
        assert "Suggested fixes:" not in output


# ---------------------------------------------------------------------------
# TestFormatBlameJsonAdditional
# ---------------------------------------------------------------------------


class TestFormatBlameJsonAdditional:
    def test_empty_chain_json(self) -> None:
        chain = BlameChain(root_task_id="", nodes=[], suggested_fixes=[])
        raw = format_blame_json(chain)
        parsed = json.loads(raw)
        assert parsed["root_task_id"] == ""
        assert parsed["nodes"] == []
        assert parsed["suggested_fixes"] == []

    def test_node_fields_in_json(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(
                task_id="t1", category="timeout_propagation", confidence=0.85,
                message="timed out", caused_by=None, evidence=["exit_code=124"],
            )],
            suggested_fixes=["Increase timeout"],
        )
        raw = format_blame_json(chain)
        parsed = json.loads(raw)
        node = parsed["nodes"][0]
        assert node["task_id"] == "t1"
        assert node["category"] == "timeout_propagation"
        assert node["confidence"] == 0.85
        assert node["message"] == "timed out"
        assert node["caused_by"] is None
        assert node["evidence"] == ["exit_code=124"]

    def test_caused_by_in_json(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[
                BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err"),
                BlameNode(task_id="t2", category="dependency_cascade", confidence=0.95, message="cascade", caused_by="t1"),
            ],
            suggested_fixes=[],
        )
        raw = format_blame_json(chain)
        parsed = json.loads(raw)
        assert parsed["nodes"][1]["caused_by"] == "t1"

    def test_json_is_valid_and_indented(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err")],
            suggested_fixes=["Fix it"],
        )
        raw = format_blame_json(chain)
        # Verify it's indented (pretty-printed)
        assert "\n" in raw
        assert "  " in raw


# ---------------------------------------------------------------------------
# TestBlameNodeToDict
# ---------------------------------------------------------------------------


class TestBlameNodeToDict:
    def test_all_fields_present(self) -> None:
        node = BlameNode(
            task_id="t1", category="root_cause", confidence=0.9,
            message="err", caused_by="t0", evidence=["e1", "e2"],
        )
        d = node.to_dict()
        assert d["task_id"] == "t1"
        assert d["category"] == "root_cause"
        assert d["confidence"] == 0.9
        assert d["message"] == "err"
        assert d["caused_by"] == "t0"
        assert d["evidence"] == ["e1", "e2"]

    def test_defaults(self) -> None:
        node = BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err")
        d = node.to_dict()
        assert d["caused_by"] is None
        assert d["evidence"] == []


# ---------------------------------------------------------------------------
# TestBlameChainToDict
# ---------------------------------------------------------------------------


class TestBlameChainToDict:
    def test_empty_chain(self) -> None:
        chain = BlameChain()
        d = chain.to_dict()
        assert d["root_task_id"] == ""
        assert d["nodes"] == []
        assert d["suggested_fixes"] == []

    def test_populated_chain(self) -> None:
        chain = BlameChain(
            root_task_id="t1",
            nodes=[BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="err")],
            suggested_fixes=["Fix it"],
        )
        d = chain.to_dict()
        assert d["root_task_id"] == "t1"
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["task_id"] == "t1"
        assert d["suggested_fixes"] == ["Fix it"]


# ---------------------------------------------------------------------------
# TestBlameIntegrationAdditional
# ---------------------------------------------------------------------------


class TestBlameIntegrationAdditional:
    def test_mixed_categories(self, tmp_path: Path) -> None:
        """Multiple failure types in the same run."""
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 124, "message": "timed out", "depends_on": []},
            "t2": {"status": "failed", "exit_code": 1, "message": "budget exceeded", "depends_on": []},
            "t3": {"status": "failed", "exit_code": 1, "message": "", "depends_on": ["t1"]},
        })
        chain = blame_run(tmp_path)
        node_by_id = {n.task_id: n for n in chain.nodes}
        assert node_by_id["t1"].category == "timeout_propagation"
        assert node_by_id["t2"].category == "budget_exhaustion"
        assert node_by_id["t3"].category == "dependency_cascade"
        assert node_by_id["t3"].caused_by == "t1"

    def test_deep_cascade_chain(self, tmp_path: Path) -> None:
        """A -> B -> C -> D cascade chain."""
        _write_manifest(tmp_path, {
            "a": {"status": "failed", "exit_code": 1, "message": "root fail", "depends_on": []},
            "b": {"status": "failed", "exit_code": 1, "message": "dep", "depends_on": ["a"]},
            "c": {"status": "failed", "exit_code": 1, "message": "dep", "depends_on": ["b"]},
            "d": {"status": "failed", "exit_code": 1, "message": "dep", "depends_on": ["c"]},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "a"
        node_by_id = {n.task_id: n for n in chain.nodes}
        assert node_by_id["a"].category == "root_cause"
        assert node_by_id["b"].caused_by == "a"
        assert node_by_id["c"].caused_by == "b"
        assert node_by_id["d"].caused_by == "c"

    def test_single_task_no_message_no_exit(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, {
            "t1": {"status": "failed"},
        })
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "t1"
        assert chain.nodes[0].category == "root_cause"

    def test_suggested_fixes_root_first(self, tmp_path: Path) -> None:
        """Root cause fix should appear before cascade fix."""
        _write_manifest(tmp_path, {
            "t1": {"status": "failed", "exit_code": 124, "message": "", "depends_on": []},
            "t2": {"status": "failed", "exit_code": 1, "message": "", "depends_on": ["t1"]},
        })
        chain = blame_run(tmp_path)
        # First fix should be for timeout (root cause), not cascade
        assert "timeout" in chain.suggested_fixes[0].lower()

    def test_large_number_of_failures(self, tmp_path: Path) -> None:
        tasks = {}
        for i in range(20):
            tasks[f"t{i}"] = {
                "status": "failed",
                "exit_code": 1,
                "message": "error",
                "depends_on": [f"t{i-1}"] if i > 0 else [],
            }
        _write_manifest(tmp_path, tasks)
        chain = blame_run(tmp_path)
        assert chain.root_task_id == "t0"
        assert len(chain.nodes) == 20
        # Only t0 should be root_cause, rest are cascades
        node_by_id = {n.task_id: n for n in chain.nodes}
        assert node_by_id["t0"].category == "root_cause"
        for i in range(1, 20):
            assert node_by_id[f"t{i}"].category == "dependency_cascade"
