"""Tests for v1.26.0 features:
- Staged Progressive Compaction
- Privacy-Aware Context Pipeline
- Trajectory-Level Guardrails
- Phantom Output Interception
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from maestro_cli.audit import audit_plan
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    CONTEXT_COMPACTION_VALUES,
    TRAJECTORY_GUARD_ACTIONS,
    PlanDefaults,
    PlanSpec,
    TaskResult,
    TaskSpec,
    TrajectoryGuardSpec,
)
from maestro_cli.runners import (
    _apply_progressive_compaction,
    _compact_context,
    _extract_l0_summary,
    _extract_l1_sections,
    _filter_context_fields,
    _prune_low_signal_sections,
    _redact_output,
    _truncate_with_markers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_plan(*tasks: TaskSpec, **kwargs: object) -> PlanSpec:
    return PlanSpec(name="test", tasks=list(tasks), **kwargs)  # type: ignore[arg-type]


def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return p


# ===========================================================================
# Feature 1: Staged Progressive Compaction
# ===========================================================================


class TestContextCompactionValues:
    def test_values(self) -> None:
        assert CONTEXT_COMPACTION_VALUES == {"none", "standard", "progressive"}


class TestCompactContext:
    def test_empty(self) -> None:
        assert _compact_context("") == ""

    def test_diff_header_stripping(self) -> None:
        text = (
            "diff --git a/foo.py b/foo.py\n"
            "index 1234..5678 100644\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+new line\n"
            " existing\n"
        )
        result = _compact_context(text)
        assert "diff --git" not in result
        assert "index 1234" not in result
        assert "--- foo.py" in result or "+new line" in result

    def test_stack_trace_compression(self) -> None:
        frames = "".join(
            f'  File "mod{i}.py", line {i}\n    call{i}()\n'
            for i in range(10)
        )
        text = f"Traceback (most recent call last):\n{frames}"
        result = _compact_context(text)
        assert "frames omitted" in result

    def test_json_minification(self) -> None:
        text = '{\n  "key": "value",\n  "num": 42\n}'
        result = _compact_context(text)
        assert '"key":"value"' in result or '"key": "value"' in result


class TestPruneLowSignalSections:
    def test_keeps_headings(self) -> None:
        text = "# Important\nSome body text\n## Also Important\nMore text"
        result = _prune_low_signal_sections(text, target_chars=60)
        assert "# Important" in result
        assert "## Also Important" in result

    def test_keeps_error_lines(self) -> None:
        text = "normal line\nError: something broke\nnormal line 2"
        result = _prune_low_signal_sections(text, target_chars=40)
        assert "Error:" in result

    def test_adds_marker_when_pruning(self) -> None:
        text = "\n".join(f"line {i} of body text content" for i in range(50))
        result = _prune_low_signal_sections(text, target_chars=200)
        assert "compacted" in result


class TestTruncateWithMarkers:
    def test_short_text_unchanged(self) -> None:
        assert _truncate_with_markers("hello", 100) == "hello"

    def test_long_text_truncated(self) -> None:
        text = "x" * 1000
        result = _truncate_with_markers(text, 200)
        assert len(result) <= 250  # marker adds some chars
        assert "compacted" in result

    def test_preserves_head_and_tail(self) -> None:
        text = "HEAD_MARKER" + "x" * 1000 + "TAIL_MARKER"
        result = _truncate_with_markers(text, 300)
        assert "HEAD_MARKER" in result
        assert "TAIL_MARKER" in result


class TestApplyProgressiveCompaction:
    def test_empty(self) -> None:
        result, stage = _apply_progressive_compaction({}, 1000)
        assert result == {}
        assert stage == 0

    def test_within_budget_no_compaction(self) -> None:
        texts = {"a": "short text"}
        result, stage = _apply_progressive_compaction(texts, 1000)
        assert result == texts
        assert stage == 0

    def test_stage_1_structural(self) -> None:
        # Create text that structural compaction can shrink
        frames = "".join(
            f'  File "mod{i}.py", line {i}\n    call{i}()\n'
            for i in range(20)
        )
        big_trace = f"Traceback (most recent call last):\n{frames}"
        texts = {"a": big_trace}
        result, stage = _apply_progressive_compaction(texts, 200)
        assert stage >= 1
        assert len(result["a"]) < len(big_trace)

    def test_stage_5_l0_summary(self) -> None:
        # Create very large text with tiny budget
        texts = {"a": "x" * 5000, "b": "y" * 5000}
        result, stage = _apply_progressive_compaction(texts, 10)
        assert stage == 5
        for v in result.values():
            assert len(v) < 500

    def test_lowest_scored_compacted_first(self) -> None:
        texts = {"high": "H" * 200, "low": "L" * 200}
        scores = {"high": 1.0, "low": 0.1}
        result, stage = _apply_progressive_compaction(texts, 80, scores)
        assert stage >= 1
        # High-scored upstream should retain more content
        assert len(result.get("high", "")) >= len(result.get("low", ""))


class TestContextCompactionLoader:
    def test_parse_progressive(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                context_compaction: progressive
        """)
        plan = load_plan(p)
        assert plan.tasks[0].context_compaction == "progressive"

    def test_parse_standard(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                context_compaction: standard
        """)
        plan = load_plan(p)
        assert plan.tasks[0].context_compaction == "standard"

    def test_parse_none_value(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                context_compaction: none
        """)
        plan = load_plan(p)
        assert plan.tasks[0].context_compaction == "none"

    def test_invalid_value_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                context_compaction: aggressive
        """)
        with pytest.raises(Exception, match="E068"):
            load_plan(p)

    def test_context_compact_true_maps_to_standard(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                context_compact: true
        """)
        plan = load_plan(p)
        assert plan.tasks[0].context_compaction == "standard"

    def test_defaults_level(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            defaults:
              context_compaction: progressive
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
        """)
        plan = load_plan(p)
        assert plan.defaults.context_compaction == "progressive"

    def test_defaults_invalid_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            defaults:
              context_compaction: turbo
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
        """)
        with pytest.raises(Exception, match="E068"):
            load_plan(p)


# ===========================================================================
# Feature 2: Privacy-Aware Context Pipeline
# ===========================================================================


class TestRedactOutput:
    def test_empty(self) -> None:
        assert _redact_output("", [r"\d+"]) == ""
        assert _redact_output("hello", []) == "hello"

    def test_single_pattern(self) -> None:
        result = _redact_output("SSN: 123-45-6789", [r"\d{3}-\d{2}-\d{4}"])
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result

    def test_multiple_patterns(self) -> None:
        text = "email: user@test.com, phone: 555-1234"
        result = _redact_output(text, [
            r"[\w.]+@[\w.]+",
            r"\d{3}-\d{4}",
        ])
        assert "user@test.com" not in result
        assert "555-1234" not in result
        assert result.count("[REDACTED]") == 2

    def test_no_match_unchanged(self) -> None:
        text = "nothing sensitive here"
        assert _redact_output(text, [r"\d{16}"]) == text


class TestFilterContextFields:
    def test_empty_allowlist_returns_original(self) -> None:
        r = TaskResult(task_id="t1", status="success", stdout_tail="out")
        assert _filter_context_fields(r, []) is r

    def test_filter_stdout_tail(self) -> None:
        r = TaskResult(
            task_id="t1", status="success", stdout_tail="secret data",
            exit_code=0, duration_sec=1.5,
        )
        filtered = _filter_context_fields(r, ["exit_code", "status"])
        assert filtered.stdout_tail == ""
        assert filtered.exit_code == 0
        assert filtered.status == "success"

    def test_filter_duration(self) -> None:
        r = TaskResult(
            task_id="t1", status="success", stdout_tail="out",
            duration_sec=42.0,
        )
        filtered = _filter_context_fields(r, ["stdout_tail"])
        assert filtered.stdout_tail == "out"
        assert filtered.duration_sec == 0.0


class TestOutputRedactLoader:
    def test_parse_list(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                output_redact:
                  - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
                  - "secret_\\\\w+"
        """)
        plan = load_plan(p)
        assert len(plan.tasks[0].output_redact) == 2

    def test_parse_single_string(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                output_redact: "password"
        """)
        plan = load_plan(p)
        assert plan.tasks[0].output_redact == ["password"]

    def test_invalid_regex_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                output_redact:
                  - "[invalid"
        """)
        with pytest.raises(Exception, match="E008"):
            load_plan(p)


class TestContextAllowlistLoader:
    def test_parse(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "first"
              - id: t2
                engine: claude
                prompt: "second"
                depends_on: [t1]
                context_from: [t1]
                context_allowlist: [stdout_tail, status]
        """)
        plan = load_plan(p)
        assert plan.tasks[1].context_allowlist == ["stdout_tail", "status"]


class TestSEC020:
    def test_pii_without_redact(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="extract",
                engine="claude",
                prompt="Extract all email addresses and passwords",
            ),
            TaskSpec(
                id="process",
                engine="claude",
                prompt="Process the data",
                depends_on=["extract"],
                context_from=["extract"],
            ),
        )
        findings = audit_plan(plan)
        sec020 = [f for f in findings if f.rule == "SEC020"]
        assert len(sec020) == 1
        assert "output_redact" in sec020[0].message

    def test_pii_with_redact_ok(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="extract",
                engine="claude",
                prompt="Extract all email addresses",
                output_redact=[r"[\w.]+@[\w.]+"],
            ),
            TaskSpec(
                id="process",
                engine="claude",
                prompt="Process",
                depends_on=["extract"],
                context_from=["extract"],
            ),
        )
        findings = audit_plan(plan)
        sec020 = [f for f in findings if f.rule == "SEC020"]
        assert len(sec020) == 0

    def test_no_pii_no_finding(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="compute",
                engine="claude",
                prompt="Calculate fibonacci numbers",
            ),
            TaskSpec(
                id="report",
                engine="claude",
                prompt="Summarize",
                depends_on=["compute"],
                context_from=["compute"],
            ),
        )
        findings = audit_plan(plan)
        sec020 = [f for f in findings if f.rule == "SEC020"]
        assert len(sec020) == 0


# ===========================================================================
# Feature 3: Trajectory-Level Guardrails
# ===========================================================================


class TestTrajectoryGuardSpec:
    def test_defaults(self) -> None:
        spec = TrajectoryGuardSpec()
        assert spec.on_violation == "warn"
        assert spec.max_tool_calls is None
        assert spec.max_retries_without_progress is None
        assert spec.scope_pattern is None

    def test_to_dict(self) -> None:
        spec = TrajectoryGuardSpec(
            max_tool_calls=10,
            on_violation="abort",
            scope_pattern="/etc/.*",
        )
        d = spec.to_dict()
        assert d["max_tool_calls"] == 10
        assert d["on_violation"] == "abort"
        assert d["scope_pattern"] == "/etc/.*"

    def test_actions_constant(self) -> None:
        assert TRAJECTORY_GUARD_ACTIONS == {"warn", "abort", "escalate"}


class TestTrajectoryGuardLoader:
    def test_parse_basic(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                trajectory_guard:
                  max_tool_calls: 50
                  on_violation: abort
        """)
        plan = load_plan(p)
        tg = plan.tasks[0].trajectory_guard
        assert tg is not None
        assert tg.max_tool_calls == 50
        assert tg.on_violation == "abort"

    def test_parse_with_scope(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                trajectory_guard:
                  scope_pattern: "/etc/.*"
                  max_retries_without_progress: 3
        """)
        plan = load_plan(p)
        tg = plan.tasks[0].trajectory_guard
        assert tg is not None
        assert tg.scope_pattern == "/etc/.*"
        assert tg.max_retries_without_progress == 3
        assert tg.on_violation == "warn"  # default

    def test_invalid_action_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                trajectory_guard:
                  on_violation: explode
        """)
        with pytest.raises(Exception, match="E008"):
            load_plan(p)

    def test_invalid_max_tool_calls_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                trajectory_guard:
                  max_tool_calls: 0
        """)
        with pytest.raises(Exception, match="E012"):
            load_plan(p)

    def test_invalid_scope_regex_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                trajectory_guard:
                  scope_pattern: "[invalid"
        """)
        with pytest.raises(Exception, match="E008"):
            load_plan(p)

    def test_no_guard_is_none(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
        """)
        plan = load_plan(p)
        assert plan.tasks[0].trajectory_guard is None


class TestToolCallCount:
    def test_task_result_default(self) -> None:
        r = TaskResult(task_id="t1", status="success")
        assert r.tool_call_count == 0

    def test_task_result_serialization(self) -> None:
        r = TaskResult(task_id="t1", status="success", tool_call_count=42)
        d = r.to_dict()
        assert d["tool_call_count"] == 42


# ===========================================================================
# Feature 4: Phantom Output Interception
# ===========================================================================


class TestPhantomWorkspaceLoader:
    def test_parse_true(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "destructive task"
                phantom_workspace: true
        """)
        plan = load_plan(p)
        assert plan.tasks[0].phantom_workspace is True

    def test_parse_default_false(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "safe task"
        """)
        plan = load_plan(p)
        assert plan.tasks[0].phantom_workspace is False


class TestPhantomWorkspaceFunctions:
    def test_setup_creates_dir(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _setup_phantom_workspace
        phantom = _setup_phantom_workspace(tmp_path, "task-1")
        assert phantom.exists()
        assert phantom.is_dir()
        assert "task-1" in phantom.name

    def test_commit_copies_files(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _commit_phantom_workspace, _setup_phantom_workspace
        phantom = _setup_phantom_workspace(tmp_path, "task-1")
        target = tmp_path / "target"
        target.mkdir()
        # Create file in phantom
        (phantom / "output.txt").write_text("result", encoding="utf-8")
        committed = _commit_phantom_workspace(phantom, target)
        assert "output.txt" in committed
        assert (target / "output.txt").read_text(encoding="utf-8") == "result"

    def test_commit_empty_phantom(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _commit_phantom_workspace, _setup_phantom_workspace
        phantom = _setup_phantom_workspace(tmp_path, "task-2")
        target = tmp_path / "target"
        target.mkdir()
        committed = _commit_phantom_workspace(phantom, target)
        assert committed == []

    def test_cleanup_removes_dir(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _cleanup_phantom_workspace, _setup_phantom_workspace
        phantom = _setup_phantom_workspace(tmp_path, "task-3")
        (phantom / "temp.txt").write_text("tmp", encoding="utf-8")
        _cleanup_phantom_workspace(phantom)
        assert not phantom.exists()

    def test_commit_nested_files(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _commit_phantom_workspace, _setup_phantom_workspace
        phantom = _setup_phantom_workspace(tmp_path, "task-4")
        target = tmp_path / "target"
        target.mkdir()
        # Create nested structure
        (phantom / "sub" / "dir").mkdir(parents=True)
        (phantom / "sub" / "dir" / "deep.txt").write_text("deep", encoding="utf-8")
        committed = _commit_phantom_workspace(phantom, target)
        assert any("deep.txt" in c for c in committed)
        assert (target / "sub" / "dir" / "deep.txt").read_text(encoding="utf-8") == "deep"


class TestSEC021:
    def test_destructive_without_phantom_or_approval(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="cleanup",
                command="rm -rf /tmp/old_data",
            ),
        )
        findings = audit_plan(plan)
        sec021 = [f for f in findings if f.rule == "SEC021"]
        assert len(sec021) == 1

    def test_destructive_with_phantom_ok(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="cleanup",
                command="rm -rf /tmp/old_data",
                phantom_workspace=True,
            ),
        )
        findings = audit_plan(plan)
        sec021 = [f for f in findings if f.rule == "SEC021"]
        assert len(sec021) == 0

    def test_destructive_with_approval_ok(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="cleanup",
                command="rm -rf /tmp/old_data",
                requires_approval=True,
            ),
        )
        findings = audit_plan(plan)
        sec021 = [f for f in findings if f.rule == "SEC021"]
        assert len(sec021) == 0

    def test_non_destructive_no_finding(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="safe",
                command="echo hello",
            ),
        )
        findings = audit_plan(plan)
        sec021 = [f for f in findings if f.rule == "SEC021"]
        assert len(sec021) == 0

    def test_destructive_in_prompt(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="db-cleanup",
                engine="claude",
                prompt="DROP TABLE users; TRUNCATE logs;",
            ),
        )
        findings = audit_plan(plan)
        sec021 = [f for f in findings if f.rule == "SEC021"]
        assert len(sec021) == 1

    def test_git_reset_hard_detected(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="reset",
                command="git reset --hard HEAD~5",
            ),
        )
        findings = audit_plan(plan)
        sec021 = [f for f in findings if f.rule == "SEC021"]
        assert len(sec021) == 1


# ===========================================================================
# Cross-feature integration
# ===========================================================================


class TestModelSerialization:
    def test_context_compaction_in_to_dict(self) -> None:
        t = TaskSpec(id="t1", context_compaction="progressive")
        d = t.to_dict()
        assert d["context_compaction"] == "progressive"

    def test_output_redact_in_to_dict(self) -> None:
        t = TaskSpec(id="t1", output_redact=[r"\d+"])
        d = t.to_dict()
        assert d["output_redact"] == [r"\d+"]

    def test_context_allowlist_in_to_dict(self) -> None:
        t = TaskSpec(id="t1", context_allowlist=["stdout_tail"])
        d = t.to_dict()
        assert d["context_allowlist"] == ["stdout_tail"]

    def test_trajectory_guard_in_to_dict(self) -> None:
        t = TaskSpec(
            id="t1",
            trajectory_guard=TrajectoryGuardSpec(max_tool_calls=10),
        )
        d = t.to_dict()
        assert d["trajectory_guard"]["max_tool_calls"] == 10

    def test_phantom_workspace_in_to_dict(self) -> None:
        t = TaskSpec(id="t1", phantom_workspace=True)
        d = t.to_dict()
        assert d["phantom_workspace"] is True

    def test_plan_defaults_context_compaction(self) -> None:
        d = PlanDefaults(context_compaction="standard")
        assert d.context_compaction == "standard"
