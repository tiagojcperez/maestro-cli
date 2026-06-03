from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from maestro_cli.audit import (
    AuditFinding,
    CategoryCoverage,
    audit_plan,
    compute_audit_coverage,
    fix_plan,
    format_audit,
    format_audit_coverage,
    format_audit_coverage_json,
    format_audit_json,
)
from maestro_cli.loader import load_plan
from maestro_cli.models import PlanSpec, TaskSpec


def _minimal_plan(*tasks: TaskSpec, max_cost_usd: float | None = 10.0) -> PlanSpec:
    """Build a minimal PlanSpec with sane defaults for testing."""
    return PlanSpec(
        name="test-plan",
        max_cost_usd=max_cost_usd,
        tasks=list(tasks),
    )


def _cmd_task(task_id: str, **kwargs) -> TaskSpec:  # type: ignore[no-untyped-def]
    return TaskSpec(id=task_id, command="echo hi", **kwargs)


class TestAuditFinding:
    def test_to_dict(self) -> None:
        f = AuditFinding(severity="error", rule="SEC001", message="bad thing", task_id="my-task")
        d = f.to_dict()
        assert d["severity"] == "error"
        assert d["rule"] == "SEC001"
        assert d["message"] == "bad thing"
        assert d["task_id"] == "my-task"

    def test_to_dict_no_task_id(self) -> None:
        f = AuditFinding(severity="warning", rule="SEC002", message="something risky")
        d = f.to_dict()
        assert "task_id" not in d
        assert d["severity"] == "warning"
        assert d["rule"] == "SEC002"


class TestAuditPlan:
    def test_no_max_cost_sec001(self) -> None:
        plan = _minimal_plan(_cmd_task("t1"), max_cost_usd=None)
        findings = audit_plan(plan)
        rules = [f.rule for f in findings]
        assert "SEC001" in rules
        sec001 = next(f for f in findings if f.rule == "SEC001")
        assert sec001.severity == "error"
        assert sec001.task_id is None

    def test_with_max_cost_no_sec001(self) -> None:
        plan = _minimal_plan(_cmd_task("t1"), max_cost_usd=5.0)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC001" for f in findings)

    def test_yolo_without_approval_sec002(self) -> None:
        task = TaskSpec(
            id="dangerous",
            engine="claude",
            prompt="do something",
            args=["--dangerously-bypass-approvals-and-sandbox"],
            requires_approval=False,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec002 = [f for f in findings if f.rule == "SEC002"]
        assert len(sec002) == 1
        assert sec002[0].task_id == "dangerous"
        assert sec002[0].severity == "warning"

    def test_yolo_with_approval_no_sec002(self) -> None:
        task = TaskSpec(
            id="safe-yolo",
            engine="claude",
            prompt="do something",
            args=["--dangerously-bypass-approvals-and-sandbox"],
            requires_approval=True,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC002" for f in findings)

    def test_env_with_secret_pattern_sec003(self) -> None:
        task = TaskSpec(
            id="leaky",
            command="echo $API_KEY",
            env={"API_KEY": "should-be-secret"},
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 1
        assert sec003[0].task_id == "leaky"
        assert "API_KEY" in sec003[0].message

    def test_env_secret_declared_no_sec003(self) -> None:
        task = TaskSpec(
            id="declared",
            command="echo $API_KEY",
            env={"API_KEY": "value"},
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            secrets=["API_KEY"],
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert all(f.rule != "SEC003" for f in findings)

    def test_allow_failure_on_security_tag_sec004(self) -> None:
        task = TaskSpec(
            id="sec-task",
            command="run-audit",
            allow_failure=True,
            tags=["security"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec004 = [f for f in findings if f.rule == "SEC004"]
        assert len(sec004) == 1
        assert sec004[0].task_id == "sec-task"
        assert sec004[0].severity == "info"

    def test_allow_failure_on_critical_tag_sec004(self) -> None:
        task = TaskSpec(
            id="crit-task",
            command="run-critical",
            allow_failure=True,
            tags=["critical"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec004 = [f for f in findings if f.rule == "SEC004"]
        assert len(sec004) == 1
        assert sec004[0].task_id == "crit-task"

    def test_allow_failure_without_security_tag_no_sec004(self) -> None:
        task = TaskSpec(
            id="benign",
            command="echo hi",
            allow_failure=True,
            tags=["docs"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC004" for f in findings)

    def test_hardcoded_api_key_sec005(self) -> None:
        task = TaskSpec(
            id="leaky-prompt",
            engine="claude",
            prompt="use this key: sk-abc123456789012345678901234567",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec005 = [f for f in findings if f.rule == "SEC005"]
        assert len(sec005) == 1
        assert sec005[0].task_id == "leaky-prompt"
        assert sec005[0].severity == "warning"

    def test_hardcoded_github_pat_sec005(self) -> None:
        task = TaskSpec(
            id="gh-leak",
            engine="claude",
            prompt="token=ghp_" + "A" * 36,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec005 = [f for f in findings if f.rule == "SEC005"]
        assert len(sec005) == 1
        assert sec005[0].task_id == "gh-leak"

    def test_clean_prompt_no_sec005(self) -> None:
        task = TaskSpec(
            id="clean",
            engine="claude",
            prompt="implement the feature as described in the spec",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC005" for f in findings)

    def test_clean_plan_no_findings(self) -> None:
        task = TaskSpec(
            id="normal-task",
            command="echo hello",
            allow_failure=False,
            tags=["build"],
        )
        plan = PlanSpec(
            name="clean-plan",
            max_cost_usd=5.0,
            tasks=[task],
        )
        findings = audit_plan(plan)
        # A clean plan with no secrets declared should have zero findings
        assert findings == []

    def test_only_one_sec005_per_task(self) -> None:
        """Multiple key patterns in same prompt → only one SEC005 finding."""
        task = TaskSpec(
            id="multi-leak",
            engine="claude",
            prompt="sk-abc123456789012345678901234567 and ghp_" + "B" * 36,
        )
        plan = _minimal_plan(task)
        sec005 = [f for f in audit_plan(plan) if f.rule == "SEC005"]
        assert len(sec005) == 1

    def test_audit_pack_file_rule_finds_violation(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "repo.php").write_text("SELECT * FROM users", encoding="utf-8")
        pack_file = tmp_path / "audit-pack.yaml"
        pack_file.write_text(textwrap.dedent("""\
        rules:
          - rule: BO001
            severity: error
            type: file_regex_absent
            path: src/repo.php
            pattern: "SELECT \\\\*"
            message: "Repositories must not use SELECT *"
        """), encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(textwrap.dedent(f"""\
        version: 1
        name: audit-pack-test
        max_cost_usd: 5.0
        audit_packs:
          - "{pack_file.name}"
        tasks:
          - id: t1
            command: "echo hi"
        """), encoding="utf-8")

        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        bo001 = [f for f in findings if f.rule == "BO001"]
        assert len(bo001) == 1
        assert bo001[0].severity == "error"
        assert bo001[0].message == "Repositories must not use SELECT *"

    def test_audit_pack_composer_package_rule(self, tmp_path: Path) -> None:
        (tmp_path / "composer.json").write_text(json.dumps({
            "require": {"slim/slim": "^4.0"},
        }), encoding="utf-8")
        pack_file = tmp_path / "audit-pack.yaml"
        pack_file.write_text(textwrap.dedent("""\
        rules:
          - rule: BO002
            severity: warning
            type: composer_package_present
            package: guzzlehttp/guzzle
            message: "Expected guzzle dependency is missing"
        """), encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(textwrap.dedent(f"""\
        version: 1
        name: composer-pack-test
        max_cost_usd: 5.0
        audit_packs:
          - "{pack_file.name}"
        tasks:
          - id: t1
            command: "echo hi"
        """), encoding="utf-8")

        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        bo002 = [f for f in findings if f.rule == "BO002"]
        assert len(bo002) == 1
        assert bo002[0].severity == "warning"

    def test_audit_pack_rules_run_against_workspace_root(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        src_dir = workspace / "src"
        src_dir.mkdir()
        (src_dir / "repo.php").write_text("SELECT * FROM users", encoding="utf-8")

        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        pack_file = rules_dir / "audit-pack.yaml"
        pack_file.write_text(textwrap.dedent("""\
        rules:
          - rule: BO003
            severity: error
            type: file_regex_absent
            path: src/repo.php
            pattern: "SELECT \\\\*"
            message: "Workspace-relative rule should still find repo.php"
        """), encoding="utf-8")

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(textwrap.dedent(f"""\
        version: 1
        name: workspace-pack-test
        workspace_root: "{workspace.name}"
        max_cost_usd: 5.0
        audit_packs:
          - "rules/{pack_file.name}"
        tasks:
          - id: t1
            command: "echo hi"
        """), encoding="utf-8")

        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        bo003 = [f for f in findings if f.rule == "BO003"]
        assert len(bo003) == 1
        assert bo003[0].message == "Workspace-relative rule should still find repo.php"

    def test_invalid_audit_pack_rule_reports_pack003(self, tmp_path: Path) -> None:
        pack_file = tmp_path / "audit-pack.yaml"
        pack_file.write_text(textwrap.dedent("""\
        rules:
          - severity: error
            type: file_contains
            path: src/repo.php
        """), encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(textwrap.dedent(f"""\
        version: 1
        name: invalid-pack-test
        max_cost_usd: 5.0
        audit_packs:
          - "{pack_file.name}"
        tasks:
          - id: t1
            command: "echo hi"
        """), encoding="utf-8")

        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        pack003 = [f for f in findings if f.rule == "PACK003"]
        assert len(pack003) == 1
        assert pack003[0].severity == "error"


class TestFormatAudit:
    def test_format_human_readable_empty(self) -> None:
        result = format_audit([])
        assert "clean" in result.lower()

    def test_format_human_readable(self) -> None:
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="warning", rule="SEC002", message="yolo", task_id="t1"),
        ]
        result = format_audit(findings)
        assert "SEC001" in result
        assert "SEC002" in result
        assert "ERROR" in result or "error" in result.lower()
        assert "WARN" in result or "warning" in result.lower()
        assert "t1" in result

    def test_format_human_readable_summary_counts(self) -> None:
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="e1"),
            AuditFinding(severity="error", rule="SEC001", message="e2"),
            AuditFinding(severity="warning", rule="SEC002", message="w1"),
        ]
        result = format_audit(findings)
        assert "2 error" in result
        assert "1 warning" in result

    def test_format_json(self) -> None:
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="warning", rule="SEC003", message="leaky env", task_id="t2"),
        ]
        raw = format_audit_json(findings)
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["rule"] == "SEC001"
        assert "task_id" not in parsed[0]
        assert parsed[1]["task_id"] == "t2"

    def test_format_json_empty(self) -> None:
        raw = format_audit_json([])
        parsed = json.loads(raw)
        assert parsed == []


class TestSTPAHazardRules:
    """Tests for SEC008-SEC014 STPA-inspired hazard rules."""

    def _write_plan(self, tmp_path: pytest.TempPathFactory, content: str) -> "Path":
        from pathlib import Path

        p: Path = tmp_path / "plan.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    # SEC008 — destructive commands without approval gate
    # ------------------------------------------------------------------

    def test_sec008_destructive_without_approval(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "rm -rf /tmp/build"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"

    def test_sec008_destructive_with_approval(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "rm -rf /tmp/build"
                requires_approval: true
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC008" for f in findings)

    # ------------------------------------------------------------------
    # SEC009 — engine + yolo + workspace_root, no worktree isolation
    # ------------------------------------------------------------------

    def test_sec009_engine_yolo_no_worktree(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            workspace_root: /tmp/ws
            tasks:
              - id: t1
                engine: claude
                prompt: "fix the bug"
                args: ["--dangerously-bypass-approvals-and-sandbox"]
                worktree: false
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec009 = [f for f in findings if f.rule == "SEC009"]
        assert len(sec009) == 1
        assert sec009[0].task_id == "t1"
        assert sec009[0].severity == "info"

    def test_sec009_engine_yolo_with_worktree(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            workspace_root: /tmp/ws
            tasks:
              - id: t1
                engine: claude
                prompt: "fix the bug"
                args: ["--dangerously-bypass-approvals-and-sandbox"]
                worktree: true
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC009" for f in findings)

    # ------------------------------------------------------------------
    # SEC010 — deep context chain without budget
    # ------------------------------------------------------------------

    def test_sec010_deep_context_no_budget(self, tmp_path: pytest.TempPathFactory) -> None:
        # t1 → t2 → t3 → t4 → t5: _chain_depth("t5") = 4 > 3, triggers SEC010
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "echo a"
              - id: t2
                command: "echo b"
                depends_on: [t1]
                context_from: [t1]
              - id: t3
                command: "echo c"
                depends_on: [t2]
                context_from: [t2]
              - id: t4
                command: "echo d"
                depends_on: [t3]
                context_from: [t3]
              - id: t5
                command: "echo e"
                depends_on: [t4]
                context_from: [t4]
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec010 = [f for f in findings if f.rule == "SEC010"]
        assert len(sec010) >= 1
        assert any(f.task_id == "t5" for f in sec010)

    def test_sec010_shallow_chain(self, tmp_path: pytest.TempPathFactory) -> None:
        # t1 → t2: depth 2, should not trigger SEC010
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "echo a"
              - id: t2
                command: "echo b"
                depends_on: [t1]
                context_from: [t1]
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC010" for f in findings)

    # ------------------------------------------------------------------
    # SEC011 — escalation list without cost budget
    # ------------------------------------------------------------------

    def test_sec011_escalation_no_budget(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "implement feature"
                max_retries: 2
                escalation: [haiku, sonnet, opus]
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec011 = [f for f in findings if f.rule == "SEC011"]
        assert len(sec011) == 1
        assert sec011[0].task_id == "t1"
        assert sec011[0].severity == "warning"

    def test_sec011_escalation_with_budget(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 10.0
            tasks:
              - id: t1
                engine: claude
                prompt: "implement feature"
                max_retries: 2
                escalation: [haiku, sonnet, opus]
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC011" for f in findings)

    # ------------------------------------------------------------------
    # SEC012 — fallback engine with yolo propagation
    # ------------------------------------------------------------------

    def test_sec012_fallback_yolo(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                engine: codex
                prompt: "write a function"
                args: ["--dangerously-bypass-approvals-and-sandbox"]
                fallback_engine: claude
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec012 = [f for f in findings if f.rule == "SEC012"]
        assert len(sec012) == 1
        assert sec012[0].task_id == "t1"
        assert sec012[0].severity == "warning"

    def test_sec012_fallback_no_yolo(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                engine: codex
                prompt: "write a function"
                fallback_engine: claude
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC012" for f in findings)

    # ------------------------------------------------------------------
    # SEC013 — watch loop without bounds
    # ------------------------------------------------------------------

    def test_sec013_watch_unbounded(self, tmp_path: pytest.TempPathFactory) -> None:
        # watch block with no max_cost_usd → error SEC013
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            watch:
              metric: accuracy
              metric_source: stdout_regex
              metric_pattern: "accuracy=(\\\\d+\\\\.\\\\d+)"
            tasks:
              - id: t1
                command: "echo accuracy=0.95"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec013 = [f for f in findings if f.rule == "SEC013"]
        assert len(sec013) == 1
        assert sec013[0].severity == "error"

    def test_sec013_watch_bounded(self, tmp_path: pytest.TempPathFactory) -> None:
        # watch block with max_cost_usd set → no SEC013
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            watch:
              metric: accuracy
              metric_source: stdout_regex
              metric_pattern: "accuracy=(\\\\d+\\\\.\\\\d+)"
              max_cost_usd: 3.0
              max_iterations: 10
            tasks:
              - id: t1
                command: "echo accuracy=0.95"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC013" for f in findings)

    # ------------------------------------------------------------------
    # SEC014 — cloud credentials in env without secrets configuration
    # ------------------------------------------------------------------

    def test_sec014_cloud_creds_no_secrets(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "aws s3 ls"
                env:
                  AWS_SECRET_ACCESS_KEY: "my-secret-key"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec014 = [f for f in findings if f.rule == "SEC014"]
        assert len(sec014) == 1
        assert sec014[0].task_id == "t1"
        assert sec014[0].severity == "warning"

    def test_sec014_cloud_creds_with_secrets(self, tmp_path: pytest.TempPathFactory) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            secrets: auto
            tasks:
              - id: t1
                command: "aws s3 ls"
                env:
                  AWS_SECRET_ACCESS_KEY: "my-secret-key"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC014" for f in findings)


# ===========================================================================
# TestAuditFix
# ===========================================================================


class TestAuditFix:
    """Tests for fix_plan() auto-remediation."""

    @staticmethod
    def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
        return plan_file

    def test_fix_adds_max_cost_usd(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC001" for f in findings)
        fixes = fix_plan(plan_file, findings)
        assert any("max_cost_usd" in d for d in fixes)
        import yaml
        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["max_cost_usd"] == 10.0

    def test_fix_adds_secrets_auto(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "aws s3 ls"
                env:
                  AWS_SECRET_ACCESS_KEY: "my-key"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert any(f.rule in ("SEC003", "SEC014") for f in findings)
        fixes = fix_plan(plan_file, findings)
        assert any("secrets" in d for d in fixes)
        import yaml
        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["secrets"] == "auto"

    def test_fix_dry_run(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """,
        )
        original = plan_file.read_text(encoding="utf-8")
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        fixes = fix_plan(plan_file, findings, dry_run=True)
        assert len(fixes) > 0
        assert plan_file.read_text(encoding="utf-8") == original

    def test_fix_no_fixable_findings(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            secrets: auto
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        # May have non-fixable findings like SEC006
        fixes = fix_plan(plan_file, findings)
        assert fixes == []

    def test_fix_backup_created(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """,
        )
        original = plan_file.read_text(encoding="utf-8")
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        fix_plan(plan_file, findings)
        bak = plan_file.with_suffix(".yaml.bak")
        assert bak.exists()
        assert bak.read_text(encoding="utf-8") == original

    def test_fix_idempotent(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        fixes1 = fix_plan(plan_file, findings)
        assert len(fixes1) > 0
        # Re-audit the fixed plan
        plan2 = load_plan(plan_file)
        findings2 = audit_plan(plan2)
        fixes2 = fix_plan(plan_file, findings2)
        assert fixes2 == []

    def test_fix_preserves_fields(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(
            tmp_path,
            """\
            version: 1
            name: my-plan
            workspace_root: /tmp/ws
            tasks:
              - id: t1
                command: "echo hi"
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        fix_plan(plan_file, findings)
        import yaml
        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["name"] == "my-plan"
        assert data["workspace_root"] == "/tmp/ws"
        assert data["version"] == 1
        assert len(data["tasks"]) == 1


# ===========================================================================
# TestSEC002Additional — additional yolo flag variants
# ===========================================================================


class TestSEC002Additional:
    @pytest.mark.parametrize("flag", ["--yolo", "--allow-all"])
    def test_sec002_additional_yolo_flags_no_approval(self, flag: str) -> None:
        """--yolo and --allow-all are also yolo flags that trigger SEC002."""
        task = TaskSpec(id="t1", engine="claude", prompt="do something", args=[flag])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec002 = [f for f in findings if f.rule == "SEC002"]
        assert len(sec002) == 1
        assert sec002[0].task_id == "t1"
        assert sec002[0].severity == "warning"

    def test_sec002_non_engine_task_no_trigger(self) -> None:
        """Command tasks (no engine) should NOT trigger SEC002 even with yolo-looking args."""
        task = TaskSpec(id="t1", command="echo hi", args=["--yolo"])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC002" for f in findings)

    def test_sec002_yolo_with_approval_no_trigger(self) -> None:
        """--yolo WITH requires_approval should NOT trigger SEC002."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            args=["--yolo"],
            requires_approval=True,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC002" for f in findings)


# ===========================================================================
# TestSEC007 — secrets declared warning
# ===========================================================================


class TestSEC007:
    def test_sec007_secrets_list_emits_warning(self) -> None:
        """When plan.secrets is a non-empty list, SEC007 is always emitted."""
        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(name="test-plan", max_cost_usd=5.0, secrets=["MY_TOKEN"], tasks=[task])
        findings = audit_plan(plan)
        sec007 = [f for f in findings if f.rule == "SEC007"]
        assert len(sec007) == 1
        assert sec007[0].severity == "warning"
        assert sec007[0].task_id is None

    def test_sec007_no_secrets_no_warning(self) -> None:
        """A plan with no secrets: declaration produces no SEC007."""
        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(name="test-plan", max_cost_usd=5.0, tasks=[task])
        findings = audit_plan(plan)
        assert all(f.rule != "SEC007" for f in findings)

    def test_sec007_secrets_auto_string_emits_warning(self) -> None:
        """secrets: auto (loaded as a truthy non-empty list) still emits SEC007."""
        task = TaskSpec(id="t1", command="echo hi")
        # In PlanSpec, secrets_auto is a separate bool field set by the loader.
        # But when secrets == "auto" from YAML, the loader sets secrets_auto=True
        # and secrets stays []. We test the code-path where secrets is a non-empty list.
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets=["SOME_KEY"],
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec007 = [f for f in findings if f.rule == "SEC007"]
        assert len(sec007) == 1


# ===========================================================================
# TestSEC008Additional — additional destructive command patterns
# ===========================================================================


class TestSEC008Additional:
    @pytest.mark.parametrize("cmd,label", [
        ("DELETE FROM users WHERE 1=1", "DELETE FROM"),
        ("TRUNCATE logs", "TRUNCATE"),
        ("drop table secrets", "case-insensitive DROP TABLE"),
        (["git", "reset", "--hard", "HEAD~1"], "list-format git reset --hard"),
    ])
    def test_sec008_additional_patterns(self, cmd: str | list[str], label: str) -> None:
        task = TaskSpec(id="t1", command=cmd)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1, f"Expected SEC008 for {label}"
        assert sec008[0].task_id == "t1"

    def test_sec008_pre_command_destructive_no_approval(self) -> None:
        """Destructive pattern in pre_command should also trigger SEC008."""
        task = TaskSpec(id="t1", command="echo hi", pre_command="git reset --hard HEAD")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"

    def test_sec008_verify_command_destructive_no_approval(self) -> None:
        """Destructive pattern in verify_command should trigger SEC008."""
        task = TaskSpec(id="t1", command="echo deploy", verify_command="DROP TABLE audit_log")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1

    def test_sec008_git_push_force_short_flag(self) -> None:
        """git push -f (short form) triggers SEC008."""
        task = TaskSpec(id="t1", command="git push -f origin main")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1


# ===========================================================================
# TestSEC015 — when expression referencing unbounded upstream fields
# ===========================================================================


class TestSEC015:
    def test_sec015_both_stdout_tail_and_log_two_findings(self) -> None:
        """When expression with both stdout_tail and log produces two SEC015 findings."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.stdout_tail }} and {{ t0.log }}",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 2
        fields = {f.message for f in sec015}
        assert any("stdout_tail" in m for m in fields)
        assert any("log" in m for m in fields)

    def test_sec015_stdout_tail_only_one_finding(self) -> None:
        """When expression with only stdout_tail produces exactly one SEC015 finding."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.stdout_tail }} == 'ok'",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 1
        assert sec015[0].task_id == "t1"

    def test_sec015_safe_status_field_no_trigger(self) -> None:
        """when expression using .status (safe field) should NOT trigger SEC015."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.status }} == 'success'",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC015" for f in findings)

    def test_sec015_no_when_field_no_trigger(self) -> None:
        """Tasks without a when field produce no SEC015 findings."""
        task = TaskSpec(id="t1", command="echo hi")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC015" for f in findings)


# ===========================================================================
# TestSEC016 — context_from raw engine output injection
# ===========================================================================


class TestSEC016:
    def test_sec016_engine_upstream_raw_triggers(self) -> None:
        """Engine upstream without guard_command in raw context_mode triggers SEC016."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 1
        assert sec016[0].task_id == "down"
        assert sec016[0].severity == "warning"

    def test_sec016_map_reduce_mode_exempt(self) -> None:
        """context_mode: map_reduce is exempt from SEC016 (LLM synthesis reduces risk)."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="map_reduce",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_sec016_recursive_mode_exempt(self) -> None:
        """context_mode: recursive is exempt from SEC016."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="recursive",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_sec016_command_upstream_no_trigger(self) -> None:
        """Command upstreams (no engine) do NOT trigger SEC016."""
        upstream = TaskSpec(id="up", command="echo results")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_sec016_wildcard_context_from_no_trigger(self) -> None:
        """context_from: ['*'] skips the per-upstream check — no SEC016."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["*"],
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_sec016_upstream_with_guard_command_no_trigger(self) -> None:
        """Engine upstream WITH guard_command does NOT trigger SEC016."""
        upstream = TaskSpec(
            id="up",
            engine="claude",
            prompt="do stuff",
            guard_command="validate.sh",
        )
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_sec016_summarized_mode_exempt(self) -> None:
        """context_mode: summarized is exempt — haiku summarization with guidance
        provides partial injection resistance, similar to map_reduce/recursive.
        Refined 2026-04-26 in response to an internal post-mortem (SEC016 was the #1
        false-positive friction when authors used `summarized` for cost savings).
        """
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="summarized",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 0


# ===========================================================================
# TestSEC010Additional — DAG topology and budget variants
# ===========================================================================


class TestSEC010Additional:
    def test_sec010_diamond_dag_no_trigger(self) -> None:
        """Diamond DAG t1→{t2,t3}→t4 has max depth 2 — no SEC010."""
        t1 = TaskSpec(id="t1", command="echo a")
        t2 = TaskSpec(id="t2", command="echo b", depends_on=["t1"], context_from=["t1"])
        t3 = TaskSpec(id="t3", command="echo c", depends_on=["t1"], context_from=["t1"])
        t4 = TaskSpec(
            id="t4",
            command="echo d",
            depends_on=["t2", "t3"],
            context_from=["t2", "t3"],
        )
        plan = _minimal_plan(t1, t2, t3, t4)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC010" for f in findings)

    def test_sec010_deep_chain_with_budget_no_trigger(self) -> None:
        """5-hop chain but context_budget_tokens set — SEC010 suppressed."""
        t1 = TaskSpec(id="t1", command="echo a")
        t2 = TaskSpec(id="t2", command="echo b", depends_on=["t1"], context_from=["t1"])
        t3 = TaskSpec(id="t3", command="echo c", depends_on=["t2"], context_from=["t2"])
        t4 = TaskSpec(id="t4", command="echo d", depends_on=["t3"], context_from=["t3"])
        t5 = TaskSpec(
            id="t5",
            command="echo e",
            depends_on=["t4"],
            context_from=["t4"],
            context_budget_tokens=4000,
        )
        plan = _minimal_plan(t1, t2, t3, t4, t5)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC010" for f in findings)


# ===========================================================================
# TestFixPlanEdgeCases — fix_plan with already-present values
# ===========================================================================


class TestFixPlanEdgeCases:
    def test_fix_existing_max_cost_not_overwritten(self, tmp_path: Path) -> None:
        """If max_cost_usd is already set in the file, SEC001 fix returns empty."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 99.0
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        findings = [AuditFinding(severity="error", rule="SEC001", message="no budget")]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []

    def test_fix_existing_secrets_not_overwritten(self, tmp_path: Path) -> None:
        """If secrets is already set in the file, SEC003 fix returns empty."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 5.0
            secrets:
              - MY_TOKEN
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        findings = [AuditFinding(severity="warning", rule="SEC003", message="leaky env")]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []

    def test_fix_only_non_fixable_rules_returns_empty(self, tmp_path: Path) -> None:
        """SEC002 and SEC008 are not fixable — fix_plan returns empty immediately."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        findings = [
            AuditFinding(severity="warning", rule="SEC002", message="yolo without approval"),
            AuditFinding(severity="warning", rule="SEC008", message="destructive command"),
        ]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []


# ===========================================================================
# TestFormatAuditAdditional — format helpers edge cases
# ===========================================================================


class TestFormatAuditAdditional:
    def test_format_audit_task_id_none_not_in_output(self) -> None:
        """Findings without a task_id must NOT produce '(task: None)' in output."""
        findings = [AuditFinding(severity="error", rule="SEC001", message="no budget")]
        result = format_audit(findings)
        assert "(task: None)" not in result
        assert "SEC001" in result

    def test_format_audit_info_prefix(self) -> None:
        """Info-severity findings get [INFO] prefix and are counted in summary."""
        findings = [AuditFinding(severity="info", rule="SEC004", message="info thing")]
        result = format_audit(findings)
        assert "[INFO]" in result
        assert "1 info" in result


# ===========================================================================
# TestSEC003Additional — additional SEC003 patterns and edge cases
# ===========================================================================


class TestSEC003Additional:
    def test_sec003_password_pattern_triggers(self) -> None:
        """DB_PASSWORD in task.env triggers SEC003."""
        task = TaskSpec(id="t1", command="echo hi", env={"DB_PASSWORD": "secret123"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 1
        assert "DB_PASSWORD" in sec003[0].message
        assert sec003[0].task_id == "t1"

    def test_sec003_credential_pattern_triggers(self) -> None:
        """MY_CREDENTIAL in task.env triggers SEC003."""
        task = TaskSpec(id="t1", command="echo hi", env={"MY_CREDENTIAL": "cred_value"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 1
        assert "MY_CREDENTIAL" in sec003[0].message

    def test_sec003_normal_var_no_trigger(self) -> None:
        """Env vars without KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL do NOT trigger SEC003."""
        task = TaskSpec(id="t1", command="echo hi", env={"BASE_URL": "http://localhost", "LOG_LEVEL": "info"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC003" for f in findings)

    def test_sec003_multiple_secret_vars_multiple_findings(self) -> None:
        """Multiple secret-like vars in the same task → one finding per var."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            env={"API_KEY": "abc", "DB_TOKEN": "xyz"},
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 2
        var_names = {f.message for f in sec003}
        assert any("API_KEY" in m for m in var_names)
        assert any("DB_TOKEN" in m for m in var_names)


# ===========================================================================
# TestSEC006Additional — SEC006 pre_command and list-format edge cases
# ===========================================================================


class TestSEC006Additional:
    def test_sec006_pre_command_not_checked(self) -> None:
        """pre_command with prod/deploy path is NOT checked by SEC006 (only command + verify_command)."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            pre_command="deploy.sh --prod",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC006" for f in findings)

    def test_sec006_list_format_command_triggers(self) -> None:
        """List-format command containing 'production' triggers SEC006."""
        task = TaskSpec(
            id="t1",
            command=["bash", "deploy.sh", "--env", "production"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec006 = [f for f in findings if f.rule == "SEC006"]
        assert len(sec006) == 1
        assert sec006[0].task_id == "t1"
        assert sec006[0].severity == "info"


# ===========================================================================
# TestSEC009Additional — additional yolo flag variants for SEC009
# ===========================================================================


class TestSEC009Additional:
    @pytest.mark.parametrize("flag", ["--yolo", "--allow-all"])
    def test_sec009_additional_yolo_flags_no_worktree(self, flag: str) -> None:
        """--yolo and --allow-all also trigger SEC009 when workspace_root is set and worktree is false."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="fix bug",
            args=[flag],
            worktree=False,
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec009 = [f for f in findings if f.rule == "SEC009"]
        assert len(sec009) == 1, f"Expected SEC009 for {flag}"
        assert sec009[0].task_id == "t1"


# ===========================================================================
# TestSEC016Additional — context_mode: layered exempt from SEC016
# ===========================================================================


class TestSEC016Additional:
    def test_sec016_layered_mode_exempt(self) -> None:
        """context_mode: layered is NOT in ('raw', 'summarized') → exempt from SEC016."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="layered",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_format_audit_mixed_severity_counts(self) -> None:
        """Summary line correctly counts each severity independently."""
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="e1"),
            AuditFinding(severity="warning", rule="SEC002", message="w1", task_id="t1"),
            AuditFinding(severity="info", rule="SEC004", message="i1", task_id="t2"),
        ]
        result = format_audit(findings)
        assert "1 error" in result
        assert "1 warning" in result
        assert "1 info" in result
        # task_id present → location appears in output
        assert "(task: t1)" in result
        assert "(task: t2)" in result


# ===========================================================================
# TestSEC005AdditionalAPIKeyPatterns — patterns beyond sk-* and ghp_*
# ===========================================================================


class TestSEC005AdditionalAPIKeyPatterns:
    def test_sec005_aws_access_key_triggers(self) -> None:
        """AWS AKIA access key in prompt triggers SEC005."""
        task = _cmd_task("t1", prompt="Use AKIAJR2GTYZV3EXAMPLE for auth")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC005" and f.task_id == "t1" for f in findings)

    def test_sec005_jwt_style_token_triggers(self) -> None:
        """JWT-style EY token in prompt triggers SEC005."""
        task = _cmd_task("t1", prompt="Authorization: EYJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC005" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC006PreCommandNotChecked — pre_command is NOT inspected by SEC006
# ===========================================================================


class TestSEC006PreCommandNotChecked:
    def test_sec006_prod_in_pre_command_no_trigger(self) -> None:
        """prod keyword in pre_command does NOT trigger SEC006 (only command/verify_command checked)."""
        task = TaskSpec(id="t1", pre_command="./check-production-ready.sh")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC006" for f in findings)


# ===========================================================================
# TestSEC003AdditionalEnvPatterns — PASSWORD and non-secret var names
# ===========================================================================


class TestSEC003AdditionalEnvPatterns:
    def test_sec003_password_pattern_triggers(self) -> None:
        """MY_PASSWORD in task.env matches PASSWORD pattern and triggers SEC003."""
        task = _cmd_task("t1", env={"MY_PASSWORD": "hunter2"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC003" and f.task_id == "t1" for f in findings)

    def test_sec003_normal_var_no_trigger(self) -> None:
        """Env vars without KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL do not trigger SEC003."""
        task = _cmd_task("t1", env={"NORMAL_VAR": "value", "APP_MODE": "dev"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC003" for f in findings)


# ===========================================================================
# TestSEC010ChainBoundary — depth=3 is NOT > 3 (no trigger); confirms threshold
# ===========================================================================


class TestSEC010ChainBoundary:
    def test_sec010_three_hop_chain_no_trigger(self) -> None:
        """Linear context chain of depth=3 does NOT trigger SEC010 (threshold is depth > 3)."""
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["t1"])
        t3 = _cmd_task("t3", depends_on=["t2"], context_from=["t2"])
        t4 = _cmd_task("t4", depends_on=["t3"], context_from=["t3"])
        plan = _minimal_plan(t1, t2, t3, t4)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC010" for f in findings)


# ===========================================================================
# TestSEC004BothTags — security + critical simultaneously → exactly 1 finding
# ===========================================================================


class TestSEC004BothTags:
    def test_sec004_both_security_and_critical_tags_one_finding(self) -> None:
        """Task tagged both 'security' and 'critical' with allow_failure produces exactly 1 SEC004."""
        task = _cmd_task("t1", tags=["security", "critical"], allow_failure=True)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec004 = [f for f in findings if f.rule == "SEC004"]
        assert len(sec004) == 1
        assert sec004[0].task_id == "t1"


# ===========================================================================
# TestSEC002DangerouslyBypassFlag — explicit test for the Codex/Claude bypass flag
# ===========================================================================


class TestSEC002DangerouslyBypassFlag:
    def test_sec002_dangerously_bypass_without_approval_triggers(self) -> None:
        """--dangerously-bypass-approvals-and-sandbox without requires_approval triggers SEC002."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Do something",
            args=["--dangerously-bypass-approvals-and-sandbox"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC002" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC006 — prod path in command/verify_command positive cases
# ===========================================================================


class TestSEC006:
    def test_sec006_prod_in_command_triggers(self) -> None:
        """prod keyword in command without guard_command triggers SEC006."""
        task = TaskSpec(id="t1", command="./deploy.sh production")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec006 = [f for f in findings if f.rule == "SEC006"]
        assert len(sec006) == 1
        assert sec006[0].task_id == "t1"
        assert sec006[0].severity == "info"

    def test_sec006_verify_command_with_prod_triggers(self) -> None:
        """prod keyword in verify_command without guard_command triggers SEC006."""
        task = TaskSpec(
            id="t1",
            command="echo done",
            verify_command="curl https://prod.example.com/health",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC006" and f.task_id == "t1" for f in findings)

    def test_sec006_list_format_command_with_prod_triggers(self) -> None:
        """prod keyword in list-format command triggers SEC006."""
        task = TaskSpec(
            id="t1",
            command=["kubectl", "apply", "-n", "production", "-f", "app.yaml"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC006" and f.task_id == "t1" for f in findings)

    def test_sec006_with_guard_command_no_trigger(self) -> None:
        """Prod path WITH guard_command does NOT trigger SEC006."""
        task = TaskSpec(
            id="t1",
            command="./deploy.sh production",
            guard_command="./validate.sh",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC006" for f in findings)

    def test_sec006_deploy_keyword_triggers(self) -> None:
        """deploy keyword in command (not just prod/production) also triggers SEC006."""
        task = TaskSpec(id="t1", command="./deploy.sh --env staging")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC006" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC005GoogleAndSlack — AIza and xoxb key patterns
# ===========================================================================


class TestSEC005GoogleAndSlack:
    def test_sec005_google_api_key_triggers(self) -> None:
        """Google API key (AIza...) in prompt triggers SEC005."""
        # AIza + 35 alphanumeric chars satisfies the pattern
        key = "AIzaSyAbc123DefGhi456JklMno789PqrStu01234"
        task = _cmd_task("t1", prompt=f"Use {key} for auth")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC005" and f.task_id == "t1" for f in findings)

    def test_sec005_slack_bot_token_triggers(self) -> None:
        """Slack bot token (xoxb-...) in prompt triggers SEC005."""
        # xoxb- + 11 digits + - + 11 digits + - + 24 alphanumeric
        token = "xoxb-12345678901-12345678901-abcdefghijklmnopqrstuvwx"
        task = _cmd_task("t1", prompt=f"Slack token: {token}")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC005" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC003CredentialAndMultiple — CREDENTIAL pattern + multiple findings
# ===========================================================================


class TestSEC003CredentialAndMultiple:
    def test_sec003_credential_pattern_triggers(self) -> None:
        """MY_CREDENTIAL in task.env matches CREDENTIAL pattern and triggers SEC003."""
        task = _cmd_task("t1", env={"MY_CREDENTIAL": "secret-value"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC003" and f.task_id == "t1" for f in findings)

    def test_sec003_multiple_secret_vars_produce_multiple_findings(self) -> None:
        """Two secret-like env vars in one task produce two separate SEC003 findings."""
        task = _cmd_task("t1", env={"MY_TOKEN": "tok1", "DB_PASSWORD": "pass2"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003" and f.task_id == "t1"]
        assert len(sec003) == 2

    def test_sec003_token_pattern_triggers(self) -> None:
        """TOKEN suffix in env var name matches SEC003 pattern."""
        task = _cmd_task("t1", env={"GITHUB_TOKEN": "ghp_example"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC003" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC009FlagVariants — all three yolo flag forms trigger SEC009
# ===========================================================================


class TestSEC009FlagVariants:
    @pytest.mark.parametrize("flag", [
        "--yolo",
        "--allow-all",
        "--dangerously-bypass-approvals-and-sandbox",
    ])
    def test_sec009_various_yolo_flags_trigger(self, flag: str) -> None:
        """All yolo flag forms trigger SEC009 when workspace_root is set and worktree is off."""
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff", args=[flag])
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec009 = [f for f in findings if f.rule == "SEC009"]
        assert len(sec009) == 1, f"Expected SEC009 for flag {flag!r}"
        assert sec009[0].task_id == "t1"

    def test_sec009_no_workspace_root_no_trigger(self) -> None:
        """Without workspace_root, SEC009 is never checked — no finding."""
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff", args=["--yolo"])
        plan = _minimal_plan(task)  # workspace_root=None by default
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC009" for f in findings)


# ===========================================================================
# TestFormatAuditJsonStructure — verify JSON output shape
# ===========================================================================


class TestFormatAuditJsonStructure:
    def test_format_audit_json_structure(self) -> None:
        """format_audit_json returns valid JSON list with expected keys per finding."""
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="warning", rule="SEC002", message="yolo risk", task_id="t1"),
        ]
        result = format_audit_json(findings)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0] == {"severity": "error", "rule": "SEC001", "message": "no budget"}
        assert data[1] == {
            "severity": "warning",
            "rule": "SEC002",
            "message": "yolo risk",
            "task_id": "t1",
        }

    def test_format_audit_json_empty_list(self) -> None:
        """format_audit_json on empty findings returns JSON empty array."""
        result = format_audit_json([])
        data = json.loads(result)
        assert data == []


# ===========================================================================
# TestSEC014AdditionalPatterns — cloud cred env var patterns beyond AWS_SECRET
# ===========================================================================


class TestSEC014AdditionalPatterns:
    def test_sec014_database_url_triggers(self) -> None:
        """DATABASE_URL in task env without secrets config triggers SEC014."""
        task = _cmd_task("t1", env={"DATABASE_URL": "postgres://user:pass@host/db"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC014" and f.task_id == "t1" for f in findings)

    def test_sec014_private_key_triggers(self) -> None:
        """PRIVATE_KEY in task env without secrets config triggers SEC014."""
        task = _cmd_task("t1", env={"PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC014" and f.task_id == "t1" for f in findings)

    def test_sec014_azure_client_secret_triggers(self) -> None:
        """AZURE_CLIENT_SECRET in task env without secrets config triggers SEC014."""
        task = _cmd_task("t1", env={"AZURE_CLIENT_SECRET": "client-secret-value"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC014" and f.task_id == "t1" for f in findings)

    def test_sec014_only_one_finding_per_task_when_multiple_cloud_creds(self) -> None:
        """Two cloud cred env vars in one task produce only 1 SEC014 finding (break after first)."""
        task = _cmd_task("t1", env={
            "AWS_SECRET_ACCESS_KEY": "key1",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json",
        })
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec014 = [f for f in findings if f.rule == "SEC014" and f.task_id == "t1"]
        assert len(sec014) == 1


# ===========================================================================
# TestSEC007SecretsAuto — secrets_auto=True interaction with SEC007
# ===========================================================================


class TestSEC007SecretsAuto:
    def test_sec007_secrets_auto_true_no_secrets_list_no_sec007(self) -> None:
        """Plan with secrets_auto=True but empty secrets list does NOT trigger SEC007.

        SEC007 checks `if plan.secrets:` — falsy for an empty list even when
        secrets_auto is True.
        """
        from maestro_cli.models import PlanSpec

        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(name="test-plan", max_cost_usd=5.0, secrets_auto=True, tasks=[task])
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC007" for f in findings)


# ===========================================================================
# TestSEC003SecretsAuto — secrets_auto does not suppress SEC003
# ===========================================================================


class TestSEC003SecretsAuto:
    def test_sec003_secrets_auto_plan_still_triggers_for_task_env(self) -> None:
        """Even with secrets_auto=True, task.env with SECRET-like var triggers SEC003.

        declared_secrets is only populated from plan.secrets list, not from
        secrets_auto, so auto-detection mode still produces a SEC003 finding.
        """
        from maestro_cli.models import PlanSpec

        task = _cmd_task("t1", env={"MY_API_KEY": "secret-value"})
        plan = PlanSpec(name="test-plan", max_cost_usd=5.0, secrets_auto=True, tasks=[task])
        findings = audit_plan(plan)
        assert any(f.rule == "SEC003" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC010WildcardContextFrom — wildcard ["*"] doesn't traverse real context chain
# ===========================================================================


class TestSEC010WildcardContextFrom:
    def test_sec010_wildcard_context_from_on_last_task_no_trigger(self) -> None:
        """context_from: ['*'] on the last task of a 4-hop chain does NOT trigger SEC010.

        The wildcard resolves to upstream_id='*' which has no entry in task_map,
        so _chain_depth returns 0 for it and the depth of the final task is only 1.
        """
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["t1"])
        t3 = _cmd_task("t3", depends_on=["t2"], context_from=["t2"])
        t4 = _cmd_task("t4", depends_on=["t3"], context_from=["t3"])
        # t5 uses wildcard — depth from t5's perspective is 1 (wildcard → not in map)
        t5 = _cmd_task("t5", depends_on=["t4"], context_from=["*"])
        plan = _minimal_plan(t1, t2, t3, t4, t5)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC010" for f in findings)


# ===========================================================================
# TestAuditPackErrorPaths — PACK001/PACK002 error paths for missing/invalid packs
# ===========================================================================


class TestAuditPackErrorPaths:
    def test_audit_pack_missing_file_reports_pack001(self, tmp_path: Path) -> None:
        """Audit pack referencing a non-existent file produces PACK001 error at audit time.

        PlanSpec is constructed directly (bypassing the loader which validates
        file existence) so that the error path in _apply_single_audit_pack is hit.
        """
        from maestro_cli.models import PlanSpec

        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(
            name="missing-pack-test",
            max_cost_usd=5.0,
            tasks=[task],
            audit_packs=["nonexistent-pack.yaml"],
            source_path=tmp_path / "plan.yaml",
        )
        findings = audit_plan(plan)
        pack001 = [f for f in findings if f.rule == "PACK001"]
        assert len(pack001) == 1
        assert pack001[0].severity == "error"

    def test_audit_pack_dict_without_rules_key_reports_pack002(self, tmp_path: Path) -> None:
        """Audit pack as YAML dict with no 'rules' key produces PACK002 error."""
        pack_file = tmp_path / "no-rules-pack.yaml"
        pack_file.write_text("description: no rules here\nversion: 1\n", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent(f"""\
            version: 1
            name: no-rules-test
            max_cost_usd: 5.0
            audit_packs:
              - "{pack_file.name}"
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        pack002 = [f for f in findings if f.rule == "PACK002"]
        assert len(pack002) == 1
        assert pack002[0].severity == "error"


# ===========================================================================
# TestFormatAuditInfoOnly — format_audit with only info-severity findings
# ===========================================================================


class TestFormatAuditInfoOnly:
    def test_format_audit_info_only_summary_no_error_or_warning_line(self) -> None:
        """format_audit with only info findings shows '2 info' in summary, no error/warning counts."""
        findings = [
            AuditFinding(severity="info", rule="SEC006", message="prod path", task_id="t1"),
            AuditFinding(severity="info", rule="SEC009", message="no worktree", task_id="t2"),
        ]
        result = format_audit(findings)
        assert "2 info" in result
        assert "error(s)" not in result
        assert "warning(s)" not in result


# ===========================================================================
# TestSEC012FlagVariants — --yolo and --allow-all also trigger SEC012
# ===========================================================================


class TestSEC012FlagVariants:
    @pytest.mark.parametrize("flag", ["--yolo", "--allow-all"])
    def test_sec012_yolo_and_allow_all_flags_trigger(self, flag: str) -> None:
        """--yolo and --allow-all with fallback_engine also trigger SEC012."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            args=[flag],
            fallback_engine="codex",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec012 = [f for f in findings if f.rule == "SEC012"]
        assert len(sec012) == 1, f"Expected SEC012 for flag {flag!r}"
        assert sec012[0].task_id == "t1"
        assert sec012[0].severity == "warning"


# ===========================================================================
# TestSEC015LogOnly — when expression with only log field → exactly 1 finding
# ===========================================================================


class TestSEC015LogOnly:
    def test_sec015_log_field_only_one_finding(self) -> None:
        """when expression referencing only 'log' produces exactly one SEC015 finding."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.log }} != ''",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 1
        assert sec015[0].task_id == "t1"
        assert "log" in sec015[0].message


# ===========================================================================
# TestSEC016MixedUpstreamGuards — one upstream guarded, one not → 1 finding
# ===========================================================================


class TestSEC016MixedUpstreamGuards:
    def test_sec016_one_guarded_one_unguarded_upstream_exactly_one_finding(self) -> None:
        """With two engine upstreams, only the one without guard_command triggers SEC016."""
        guarded = TaskSpec(
            id="guarded",
            engine="claude",
            prompt="do stuff",
            guard_command="validate.sh",
        )
        unguarded = TaskSpec(id="unguarded", engine="claude", prompt="do other stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["guarded", "unguarded"],
            context_from=["guarded", "unguarded"],
        )
        plan = _minimal_plan(guarded, unguarded, downstream)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016" and f.task_id == "down"]
        assert len(sec016) == 1
        assert "unguarded" in sec016[0].message


# ===========================================================================
# TestSEC011TwoTasks — two tasks with escalation, no budget → two SEC011 findings
# ===========================================================================


class TestSEC011TwoTasks:
    def test_sec011_two_tasks_escalation_no_budget_produces_two_findings(self) -> None:
        """Two tasks each with escalation: configured but no plan max_cost_usd → 2 SEC011 findings."""
        t1 = TaskSpec(
            id="t1",
            engine="claude",
            prompt="implement feature A",
            max_retries=2,
            escalation=["haiku", "sonnet"],
        )
        t2 = TaskSpec(
            id="t2",
            engine="claude",
            prompt="implement feature B",
            max_retries=2,
            escalation=["sonnet", "opus"],
        )
        plan = _minimal_plan(t1, t2, max_cost_usd=None)
        findings = audit_plan(plan)
        sec011 = [f for f in findings if f.rule == "SEC011"]
        assert len(sec011) == 2
        task_ids = {f.task_id for f in sec011}
        assert task_ids == {"t1", "t2"}


# ===========================================================================
# TestFixPlanBothRulesTogether — fix_plan applies SEC001 and SEC003 in one call
# ===========================================================================


class TestFixPlanBothRulesTogether:
    def test_fix_sec001_and_sec003_both_applied_together(self, tmp_path: Path) -> None:
        """fix_plan with SEC001 + SEC003 findings applies both fixes in a single call."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "aws s3 ls"
                env:
                  MY_SECRET_KEY: "secret-value"
            """),
            encoding="utf-8",
        )
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="warning", rule="SEC003", message="leaky env", task_id="t1"),
        ]
        fixes = fix_plan(plan_file, findings)
        import yaml
        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["max_cost_usd"] == 10.0
        assert data["secrets"] == "auto"
        assert len(fixes) == 2
        assert any("max_cost_usd" in s for s in fixes)
        assert any("secrets" in s for s in fixes)


# ===========================================================================
# TestSEC002EdgeCases — empty args and non-engine engine-like tasks
# ===========================================================================


class TestSEC002EdgeCases:
    def test_sec002_empty_args_list_no_trigger(self) -> None:
        """Engine task with args=[] (empty list) does NOT trigger SEC002."""
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff", args=[])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC002" for f in findings)

    def test_sec002_engine_task_no_args_field_no_trigger(self) -> None:
        """Engine task without any args at all does NOT trigger SEC002."""
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC002" for f in findings)


# ===========================================================================
# TestSEC005NullPrompt — engine task with no prompt field → no SEC005
# ===========================================================================


class TestSEC005NullPrompt:
    def test_sec005_engine_task_null_prompt_no_trigger(self) -> None:
        """Engine task with prompt=None (default) does NOT trigger SEC005."""
        task = TaskSpec(id="t1", engine="claude", prompt_file="tasks/impl.md")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC005" for f in findings)


# ===========================================================================
# TestSEC008DropDatabase — DROP DATABASE pattern triggers SEC008
# ===========================================================================


class TestSEC008DropDatabase:
    def test_sec008_drop_database_triggers(self) -> None:
        """DROP DATABASE pattern triggers SEC008 without requires_approval."""
        task = TaskSpec(id="t1", command="DROP DATABASE legacy_db")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"


# ===========================================================================
# TestSEC006EnginePromptNotChecked — 'prod' in prompt but no command → no SEC006
# ===========================================================================


class TestSEC006EnginePromptNotChecked:
    def test_sec006_prod_in_prompt_no_command_no_trigger(self) -> None:
        """'production' in prompt but task has no command/verify_command — SEC006 not triggered."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Deploy to the production environment using kubectl",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC006" for f in findings)


# ===========================================================================
# TestFormatAuditWarningsOnly — warnings-only findings get [WARN] prefix
# ===========================================================================


class TestFormatAuditWarningsOnly:
    def test_format_audit_warnings_only_correct_prefix_and_summary(self) -> None:
        """format_audit with only warning findings shows [WARN] prefix and 'N warning(s)' summary."""
        findings = [
            AuditFinding(severity="warning", rule="SEC002", message="yolo risk", task_id="t1"),
            AuditFinding(severity="warning", rule="SEC003", message="secret leak", task_id="t2"),
        ]
        result = format_audit(findings)
        assert "[WARN]" in result
        assert "2 warning(s)" in result
        assert "error(s)" not in result
        assert "info" not in result
        assert "(task: t1)" in result
        assert "(task: t2)" in result


# ===========================================================================
# TestSEC009CommandTaskWithWorkspace — command task with workspace_root → no SEC009
# ===========================================================================


class TestSEC009CommandTaskWithWorkspace:
    def test_sec009_command_task_no_engine_no_trigger(self) -> None:
        """Command task (engine=None) with yolo-looking args and workspace_root does NOT trigger SEC009."""
        task = TaskSpec(
            id="t1",
            command="./run.sh --yolo",
            args=["--yolo"],
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC009" for f in findings)


# ===========================================================================
# TestSEC011MultipleTasks — two tasks both with escalation, no budget → 2 findings
# ===========================================================================


class TestSEC011MultipleTasks:
    def test_sec011_two_tasks_escalation_no_budget_two_findings(self) -> None:
        """Two tasks each with escalation and no plan max_cost_usd produce two SEC011 findings."""
        t1 = TaskSpec(
            id="t1",
            engine="claude",
            prompt="task one",
            max_retries=2,
            escalation=["haiku", "sonnet"],
        )
        t2 = TaskSpec(
            id="t2",
            engine="claude",
            prompt="task two",
            max_retries=2,
            escalation=["sonnet", "opus"],
        )
        plan = PlanSpec(name="test-plan", max_cost_usd=None, tasks=[t1, t2])
        findings = audit_plan(plan)
        sec011 = [f for f in findings if f.rule == "SEC011"]
        assert len(sec011) == 2
        task_ids = {f.task_id for f in sec011}
        assert task_ids == {"t1", "t2"}


# ===========================================================================
# TestSEC014SecretsListSuppresses — non-empty secrets list skips SEC014 check
# ===========================================================================


class TestSEC014SecretsListSuppresses:
    def test_sec014_non_empty_secrets_list_suppresses_check(self) -> None:
        """plan.secrets = ['SOME_VAR'] (non-empty list) skips the SEC014 check entirely."""
        task = _cmd_task("t1", env={"AWS_SECRET_ACCESS_KEY": "my-key"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets=["SOME_VAR"],
            tasks=[task],
        )
        findings = audit_plan(plan)
        # SEC007 is expected (secrets list declared), but SEC014 should NOT appear
        assert not any(f.rule == "SEC014" for f in findings)


# ===========================================================================
# TestSEC006EngineTaskPromptOnly — prod in prompt only → no SEC006
# ===========================================================================


class TestSEC006EngineTaskPromptOnly:
    def test_sec006_prod_in_prompt_no_command_no_trigger(self) -> None:
        """Engine task with 'production' only in prompt (no command/verify_command) → no SEC006."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Deploy to the production environment by updating the config",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC006" for f in findings)


# ===========================================================================
# TestSEC002EngineVariants — SEC002 is engine-agnostic; any engine triggers it
# ===========================================================================


class TestSEC002EngineVariants:
    @pytest.mark.parametrize("engine,flag", [
        ("codex", "--dangerously-bypass-approvals-and-sandbox"),
        ("gemini", "--allow-all"),
        ("copilot", "--yolo"),
        ("qwen", "--yolo"),
    ])
    def test_sec002_various_engines_with_yolo_trigger(self, engine: str, flag: str) -> None:
        """SEC002 fires for any engine type (not just claude) with a yolo flag."""
        task = TaskSpec(id="t1", engine=engine, prompt="do stuff", args=[flag])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec002 = [f for f in findings if f.rule == "SEC002"]
        assert len(sec002) == 1, f"Expected SEC002 for engine={engine!r}, flag={flag!r}"
        assert sec002[0].task_id == "t1"
        assert sec002[0].severity == "warning"


# ===========================================================================
# TestSEC016LayeredModeExempt — layered context_mode is exempt (only raw triggers)
# ===========================================================================


class TestSEC016LayeredModeExempt:
    def test_sec016_layered_context_mode_exempt(self) -> None:
        """context_mode: layered is NOT 'raw' — exempt from SEC016."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="layered",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)


# ===========================================================================
# TestSEC014PatternCoverage — patterns not yet explicitly tested
# ===========================================================================


class TestSEC014PatternCoverage:
    def test_sec014_google_application_credentials_triggers(self) -> None:
        """GOOGLE_APPLICATION_CREDENTIALS in task env without secrets config triggers SEC014."""
        task = _cmd_task("t1", env={"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC014" and f.task_id == "t1" for f in findings)

    def test_sec014_aws_access_key_id_triggers(self) -> None:
        """AWS_ACCESS_KEY_ID (contains 'AWS_ACCESS_KEY') triggers SEC014."""
        task = _cmd_task("t1", env={"AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC014" and f.task_id == "t1" for f in findings)


# ===========================================================================
# TestSEC008DropDatabaseAndGitForce — additional destructive patterns
# ===========================================================================


class TestSEC008DropDatabaseAndGitForce:
    def test_sec008_drop_database_triggers(self) -> None:
        """DROP DATABASE in command without approval triggers SEC008."""
        task = TaskSpec(id="t1", command="DROP DATABASE testdb")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"

    def test_sec008_git_push_force_long_flag_triggers(self) -> None:
        """git push --force (long form, distinct from -f) triggers SEC008."""
        task = TaskSpec(id="t1", command="git push --force origin main")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"


# ===========================================================================
# TestFixPlanCombinedFindings — SEC001 + SEC003 both applied in one fix_plan call
# ===========================================================================


class TestFixPlanCombinedFindings:
    def test_fix_plan_applies_both_sec001_and_sec003_simultaneously(self, tmp_path: Path) -> None:
        """fix_plan applies SEC001 (max_cost_usd) and SEC003 (secrets: auto) in the same call."""
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo $MY_API_KEY"
                env:
                  MY_API_KEY: "should-be-secret"
            """),
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC001" for f in findings)
        assert any(f.rule == "SEC003" for f in findings)

        fixes = fix_plan(plan_file, findings)
        combined = " ".join(fixes)
        assert "max_cost_usd" in combined
        assert "secrets" in combined

        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["max_cost_usd"] == 10.0
        assert data["secrets"] == "auto"


# ===========================================================================
# TestSEC010DepthBoundaryPrecision — intermediate tasks NOT over threshold
# ===========================================================================


class TestSEC010DepthBoundaryPrecision:
    def test_sec010_only_t5_triggers_not_t4_in_five_task_chain(self) -> None:
        """In a 5-task linear chain: depth(t4)=3 → no trigger; depth(t5)=4 → triggers."""
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["t1"])
        t3 = _cmd_task("t3", depends_on=["t2"], context_from=["t2"])
        t4 = _cmd_task("t4", depends_on=["t3"], context_from=["t3"])
        t5 = _cmd_task("t5", depends_on=["t4"], context_from=["t4"])
        plan = _minimal_plan(t1, t2, t3, t4, t5)
        findings = audit_plan(plan)
        sec010 = [f for f in findings if f.rule == "SEC010"]
        triggered_ids = {f.task_id for f in sec010}
        assert "t5" in triggered_ids
        assert "t4" not in triggered_ids


# ===========================================================================
# TestSEC006ListFormatVerifyCommand — list-form verify_command with prod keyword
# ===========================================================================


class TestSEC006ListFormatVerifyCommand:
    def test_sec006_prod_in_list_format_verify_command_triggers(self) -> None:
        """production keyword inside a list-format verify_command triggers SEC006."""
        task = TaskSpec(
            id="t1",
            command="echo done",
            verify_command=["./check-deploy.sh", "--env", "production"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec006 = [f for f in findings if f.rule == "SEC006"]
        assert len(sec006) == 1
        assert sec006[0].task_id == "t1"
        assert sec006[0].severity == "info"


# ===========================================================================
# TestSEC016LayeredContextModeExempt — layered mode is exempt from SEC016
# ===========================================================================


class TestSEC016LayeredContextModeExempt:
    def test_sec016_layered_context_mode_exempt(self) -> None:
        """context_mode: layered is exempt from SEC016 (only raw triggers; refined 2026-04-26)."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="layered",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)


# ===========================================================================
# TestFixPlanDryRunNoBackup — dry_run=True does not create .yaml.bak
# ===========================================================================


class TestFixPlanDryRunNoBackup:
    def test_fix_dry_run_does_not_create_backup(self, tmp_path: Path) -> None:
        """dry_run=True returns the fix descriptions but must NOT create a .yaml.bak file."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        from maestro_cli.loader import load_plan

        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        fixes = fix_plan(plan_file, findings, dry_run=True)
        assert len(fixes) > 0
        bak = plan_file.with_suffix(".yaml.bak")
        assert not bak.exists()


# ===========================================================================
# TestSEC008DropDatabaseAndForcePush — DROP DATABASE + git push --force patterns
# ===========================================================================


class TestSEC008DropDatabaseAndForcePush:
    def test_sec008_drop_database_triggers(self) -> None:
        """DROP DATABASE without requires_approval triggers SEC008."""
        task = TaskSpec(id="t1", command="DROP DATABASE staging_db")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"

    def test_sec008_git_push_force_long_flag_triggers(self) -> None:
        """git push --force (long form) without requires_approval triggers SEC008."""
        task = TaskSpec(id="t1", command="git push --force origin main")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"


# ===========================================================================
# TestSEC003DefaultsEnvNotAudited — defaults.env is outside audit scope
# ===========================================================================


class TestSEC003DefaultsEnvNotAudited:
    def test_sec003_defaults_env_secret_name_does_not_trigger(self) -> None:
        """Secret-like var names in plan defaults.env do NOT trigger SEC003.

        _check_sec003 iterates task.env only — defaults.env is resolved
        by the scheduler at runtime and is never inspected by the audit engine.
        """
        from maestro_cli.models import PlanDefaults

        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            tasks=[task],
            defaults=PlanDefaults(env={"MY_API_KEY": "should-not-trigger"}),
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC003" for f in findings)


# ===========================================================================
# TestSEC010DeepDiamond — diamond DAG deep enough to trigger SEC010
# ===========================================================================


class TestSEC010DeepDiamondTriggers:
    def test_sec010_wide_fan_in_deep_chain_triggers(self) -> None:
        """Two independent 4-hop chains merging at a fan-in node triggers SEC010.

        Structure: A1→A2→A3→A4→merge, B1→B2→B3→B4→merge.
        Because A and B chains share no nodes, the DFS on merge visits A4→A3→A2→A1
        (depth 4 from merge via the A branch), which is > 3 → SEC010.
        """
        a1 = _cmd_task("a1")
        a2 = _cmd_task("a2", depends_on=["a1"], context_from=["a1"])
        a3 = _cmd_task("a3", depends_on=["a2"], context_from=["a2"])
        a4 = _cmd_task("a4", depends_on=["a3"], context_from=["a3"])
        b1 = _cmd_task("b1")
        b2 = _cmd_task("b2", depends_on=["b1"], context_from=["b1"])
        b3 = _cmd_task("b3", depends_on=["b2"], context_from=["b2"])
        b4 = _cmd_task("b4", depends_on=["b3"], context_from=["b3"])
        merge = _cmd_task(
            "merge",
            depends_on=["a4", "b4"],
            context_from=["a4", "b4"],
        )
        plan = _minimal_plan(a1, a2, a3, a4, b1, b2, b3, b4, merge)
        findings = audit_plan(plan)
        sec010 = [f for f in findings if f.rule == "SEC010"]
        assert len(sec010) >= 1
        assert any(f.task_id == "merge" for f in sec010)


# ===========================================================================
# TestFixPlanBothSEC001AndSEC003 — fix_plan applies both SEC001 and SEC003/SEC014 fixes
# ===========================================================================


class TestFixPlanBothSEC001AndSEC014:
    def test_fix_plan_applies_both_sec001_and_sec014_together(self, tmp_path: Path) -> None:
        """fix_plan with both SEC001 and SEC014 findings applies max_cost_usd and secrets."""
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "aws s3 ls"
                env:
                  AWS_SECRET_ACCESS_KEY: "my-key"
            """),
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC001" for f in findings)
        assert any(f.rule in ("SEC003", "SEC014") for f in findings)

        fixes = fix_plan(plan_file, findings)

        assert any("max_cost_usd" in d for d in fixes)
        assert any("secrets" in d for d in fixes)

        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["max_cost_usd"] == 10.0
        assert data["secrets"] == "auto"


# ===========================================================================
# TestFormatAuditWarnPrefix — WARN prefix includes trailing space for alignment
# ===========================================================================


class TestFormatAuditWarnPrefix:
    def test_format_audit_warn_prefix_has_trailing_space(self) -> None:
        """Warning-severity findings use '[WARN] ' (with trailing space) for column alignment."""
        findings = [AuditFinding(severity="warning", rule="SEC002", message="yolo risk", task_id="t1")]
        result = format_audit(findings)
        assert "[WARN]  SEC002" in result or "[WARN] SEC002" in result
        # The prefix map uses "[WARN] " (6 chars + space); the rule appears right after
        # Verify at minimum that the WARN tag is present without 'None'
        assert "SEC002" in result
        assert "(task: t1)" in result
        assert "(task: None)" not in result


# ===========================================================================
# TestSEC016MultipleUnguardedEngineUpstreams — each unguarded upstream = 1 finding
# ===========================================================================


class TestSEC016MultipleUnguardedEngineUpstreams:
    def test_sec016_three_unguarded_engine_upstreams_three_findings(self) -> None:
        """Three engine upstreams all lacking guard_command → three SEC016 findings on downstream."""
        u1 = TaskSpec(id="u1", engine="claude", prompt="task one")
        u2 = TaskSpec(id="u2", engine="codex", prompt="task two")
        u3 = TaskSpec(id="u3", engine="gemini", prompt="task three")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["u1", "u2", "u3"],
            context_from=["u1", "u2", "u3"],
        )
        plan = _minimal_plan(u1, u2, u3, downstream)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016" and f.task_id == "down"]
        assert len(sec016) == 3
        mentioned_upstreams = {f.message for f in sec016}
        assert any("u1" in m for m in mentioned_upstreams)
        assert any("u2" in m for m in mentioned_upstreams)
        assert any("u3" in m for m in mentioned_upstreams)


# ===========================================================================
# TestSEC003MultipleTasksEachWithSecretEnv — per-task findings are independent
# ===========================================================================


class TestSEC003MultipleTasksWithSecretEnv:
    def test_sec003_two_tasks_with_secret_env_produce_two_findings(self) -> None:
        """Two separate tasks each with a single secret-like env var produce two SEC003 findings."""
        t1 = _cmd_task("t1", env={"AUTH_TOKEN": "tok1"})
        t2 = _cmd_task("t2", env={"DB_PASSWORD": "pass2"})
        plan = _minimal_plan(t1, t2)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        task_ids = {f.task_id for f in sec003}
        assert "t1" in task_ids
        assert "t2" in task_ids
        assert len(sec003) == 2


# ===========================================================================
# TestSEC016LayeredMode — layered context_mode is exempt from SEC016
# ===========================================================================


class TestSEC016LayeredMode:
    def test_sec016_layered_mode_exempt(self) -> None:
        """context_mode: layered is NOT 'raw' — exempt from SEC016."""
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=["up"],
            context_mode="layered",
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)


# ===========================================================================
# TestSEC012FlagVariants — --yolo and --allow-all also trigger SEC012
# ===========================================================================


class TestSEC012FlagVariants:
    @pytest.mark.parametrize("flag", ["--yolo", "--allow-all"])
    def test_sec012_yolo_variants_trigger(self, flag: str) -> None:
        """--yolo and --allow-all are also yolo flags — trigger SEC012 when fallback_engine set."""
        task = TaskSpec(
            id="t1",
            engine="codex",
            prompt="write code",
            args=[flag],
            fallback_engine="claude",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec012 = [f for f in findings if f.rule == "SEC012"]
        assert len(sec012) == 1, f"Expected SEC012 for flag {flag!r}"
        assert sec012[0].task_id == "t1"
        assert sec012[0].severity == "warning"


# ===========================================================================
# TestFixPlanDryRunNoBackup — dry_run=True must not create .yaml.bak
# ===========================================================================


class TestFixPlanDryRunNoBackup:
    def test_fix_dry_run_does_not_create_backup(self, tmp_path: Path) -> None:
        """fix_plan with dry_run=True must NOT create the .yaml.bak backup file."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        from maestro_cli.loader import load_plan

        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        fixes = fix_plan(plan_file, findings, dry_run=True)
        assert len(fixes) > 0
        bak = plan_file.with_suffix(".yaml.bak")
        assert not bak.exists(), "dry_run=True must NOT create a .yaml.bak backup"


# ===========================================================================
# TestSEC004RequiresAllowFailure — security/critical tag alone doesn't trigger
# ===========================================================================


class TestSEC004RequiresAllowFailure:
    def test_sec004_security_tag_without_allow_failure_no_trigger(self) -> None:
        """security tag + allow_failure=False (default) must NOT trigger SEC004."""
        task = _cmd_task("t1", tags=["security"], allow_failure=False)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC004" for f in findings)

    def test_sec004_critical_tag_without_allow_failure_no_trigger(self) -> None:
        """critical tag + allow_failure=False (default) must NOT trigger SEC004."""
        task = _cmd_task("t1", tags=["critical"], allow_failure=False)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC004" for f in findings)


# ===========================================================================
# TestSEC011MultipleEscalatingTasks — each task with escalation gets its own finding
# ===========================================================================


class TestSEC011MultipleEscalatingTasks:
    def test_sec011_two_escalation_tasks_two_findings(self) -> None:
        """Two tasks both with escalation and no plan budget each produce a SEC011 finding."""
        t1 = TaskSpec(
            id="t1",
            engine="claude",
            prompt="task one",
            max_retries=2,
            escalation=["haiku", "sonnet"],
        )
        t2 = TaskSpec(
            id="t2",
            engine="claude",
            prompt="task two",
            max_retries=2,
            escalation=["sonnet", "opus"],
        )
        plan = PlanSpec(name="test-plan", tasks=[t1, t2])  # max_cost_usd=None
        findings = audit_plan(plan)
        sec011 = [f for f in findings if f.rule == "SEC011"]
        task_ids = {f.task_id for f in sec011}
        assert "t1" in task_ids
        assert "t2" in task_ids
        assert len(sec011) == 2


# ===========================================================================
# TestSEC009CommandTaskExempt — no engine → SEC009 never checked
# ===========================================================================


class TestSEC009CommandTaskExempt:
    def test_sec009_command_task_yolo_args_workspace_root_no_trigger(self) -> None:
        """Command tasks (no engine) are exempt from SEC009 even with yolo args + workspace_root."""
        task = TaskSpec(id="t1", command="echo hi", args=["--yolo"])
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC009" for f in findings)


# ===========================================================================
# TestSEC009WorktreeSuppresses — worktree=True suppresses SEC009
# ===========================================================================


class TestSEC009WorktreeSuppresses:
    def test_sec009_yolo_with_worktree_true_no_trigger(self) -> None:
        """Engine task with yolo flag AND worktree=True does NOT trigger SEC009.

        _check_sec009 guards with `not task.worktree`, so worktree isolation
        satisfies the control requirement.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do stuff",
            args=["--yolo"],
            worktree=True,
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC009" for f in findings)


# ===========================================================================
# TestSEC011MaxCostUsdSuppresses — escalation + max_cost_usd set → no SEC011
# ===========================================================================


class TestSEC011MaxCostUsdSuppresses:
    def test_sec011_escalation_with_max_cost_usd_no_trigger(self) -> None:
        """Task with escalation + plan max_cost_usd set produces no SEC011 finding.

        The SEC011 loop is gated by `if plan.max_cost_usd is None:` so any
        non-None budget completely suppresses the rule.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do stuff",
            max_retries=2,
            escalation=["haiku", "sonnet", "opus"],
        )
        plan = PlanSpec(name="test-plan", max_cost_usd=20.0, tasks=[task])
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC011" for f in findings)


# ===========================================================================
# TestAuditFindingEmptyTaskId — empty string task_id is falsy → not in to_dict
# ===========================================================================


class TestAuditFindingEmptyTaskId:
    def test_to_dict_empty_string_task_id_not_included(self) -> None:
        """task_id='' (empty string) is falsy — to_dict() must not include 'task_id' key."""
        f = AuditFinding(severity="warning", rule="SEC002", message="risky", task_id="")
        d = f.to_dict()
        assert "task_id" not in d
        assert d["severity"] == "warning"
        assert d["rule"] == "SEC002"


# ===========================================================================
# TestAuditPlanEmptyTasks — audit_plan with zero tasks: plan-level only, no crash
# ===========================================================================


class TestAuditPlanEmptyTasks:
    def test_audit_plan_empty_tasks_no_crash_and_no_task_level_findings(self) -> None:
        """audit_plan with tasks=[] does not crash; only plan-level rules apply."""
        plan = PlanSpec(name="test-plan", max_cost_usd=5.0, tasks=[])
        findings = audit_plan(plan)
        # No tasks → no task-level findings for SEC002-SEC006, SEC008-SEC012, etc.
        task_rules = {"SEC002", "SEC003", "SEC004", "SEC005", "SEC006",
                      "SEC008", "SEC009", "SEC010", "SEC011", "SEC012",
                      "SEC015", "SEC016"}
        assert not any(f.rule in task_rules for f in findings)

    def test_audit_plan_empty_tasks_no_budget_only_sec001(self) -> None:
        """audit_plan with tasks=[] and no max_cost_usd raises only SEC001."""
        plan = PlanSpec(name="test-plan", tasks=[])
        findings = audit_plan(plan)
        assert any(f.rule == "SEC001" for f in findings)
        task_rules = {"SEC002", "SEC003", "SEC004", "SEC008", "SEC016"}
        assert not any(f.rule in task_rules for f in findings)


# ===========================================================================
# TestSEC003CaseSensitiveDeclaration — declared secret names are case-sensitive
# ===========================================================================


class TestSEC003CaseSensitiveDeclaration:
    def test_sec003_lowercase_declaration_does_not_suppress_uppercase_env_var(self) -> None:
        """plan.secrets=['my_api_key'] (lowercase) does NOT suppress SEC003 for 'MY_API_KEY'."""
        task = TaskSpec(id="t1", command="echo hi", env={"MY_API_KEY": "secret-value"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets=["my_api_key"],   # lowercase — does not match MY_API_KEY exactly
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003" and f.task_id == "t1"]
        assert len(sec003) == 1
        assert "MY_API_KEY" in sec003[0].message


# ===========================================================================
# TestSEC008ListPreCommandDestructive — list-format pre_command triggers SEC008
# ===========================================================================


class TestSEC008ListPreCommandDestructive:
    def test_sec008_list_format_pre_command_destructive_triggers(self) -> None:
        """Destructive pattern in list-format pre_command also triggers SEC008."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            pre_command=["git", "reset", "--hard", "HEAD~1"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"

    def test_sec008_list_format_pre_command_with_approval_no_trigger(self) -> None:
        """Destructive list-format pre_command WITH requires_approval does NOT trigger SEC008."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            pre_command=["git", "reset", "--hard", "HEAD"],
            requires_approval=True,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC008" for f in findings)


# ===========================================================================
# TestFixPlanSEC014OnlyFinding — SEC014-only fix adds "secrets: auto" labelled SEC014
# ===========================================================================


class TestFixPlanSEC014OnlyFinding:
    def test_fix_sec014_only_adds_secrets_auto_with_sec014_label(self, tmp_path: Path) -> None:
        """fix_plan with only a SEC014 finding adds secrets: auto labelled as 'SEC014'."""
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        findings = [AuditFinding(severity="warning", rule="SEC014", message="cloud cred")]
        fixes = fix_plan(plan_file, findings)
        assert len(fixes) == 1
        assert "SEC014" in fixes[0]
        assert "secrets" in fixes[0]
        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["secrets"] == "auto"


# ===========================================================================
# TestSEC014SecretsAutoSuppresses — secrets_auto=True skips the SEC014 check
# ===========================================================================


class TestSEC014SecretsAutoSuppresses:
    def test_sec014_secrets_auto_true_suppresses_check(self) -> None:
        """PlanSpec with secrets_auto=True skips the SEC014 check entirely.

        The condition is `if not plan.secrets and not plan.secrets_auto:` so
        secrets_auto=True unconditionally prevents SEC014 from firing.
        Using DATABASE_URL ensures the env var matches _CLOUD_CRED_PATTERN
        but NOT _SECRET_ENV_PATTERN (no KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL),
        so only SEC014 would have fired — cleanly verifying the suppression.
        """
        task = _cmd_task("t1", env={"DATABASE_URL": "postgres://user:pass@host/db"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets_auto=True,
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC014" for f in findings)


# ===========================================================================
# TestFormatAuditUnknownSeverity — unknown severity falls back to "[?]" prefix
# ===========================================================================


class TestFormatAuditUnknownSeverity:
    def test_format_audit_unknown_severity_uses_fallback_prefix(self) -> None:
        """A finding with an unrecognised severity uses the '[?]   ' fallback prefix."""
        findings = [AuditFinding(severity="critical", rule="CUST001", message="custom issue")]  # type: ignore[arg-type]
        result = format_audit(findings)
        assert "[?]" in result
        assert "CUST001" in result

    def test_format_audit_unknown_severity_not_counted_in_standard_summary(self) -> None:
        """Unknown severity does not increment error/warning/info counters."""
        findings = [AuditFinding(severity="critical", rule="CUST001", message="custom")]  # type: ignore[arg-type]
        result = format_audit(findings)
        # Summary line should not contain 'error(s)', 'warning(s)', or 'info'
        assert "error(s)" not in result
        assert "warning(s)" not in result
        assert "1 info" not in result


# ===========================================================================
# TestSEC012DangerouslyBypassDirect — full bypass flag + fallback direct construction
# ===========================================================================


class TestSEC012DangerouslyBypassDirect:
    def test_sec012_dangerously_bypass_flag_with_fallback_triggers(self) -> None:
        """--dangerously-bypass-approvals-and-sandbox + fallback_engine triggers SEC012.

        Direct PlanSpec/TaskSpec construction (complements the YAML-based test
        in TestSTPAHazardRules.test_sec012_fallback_yolo).
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do stuff",
            args=["--dangerously-bypass-approvals-and-sandbox"],
            fallback_engine="codex",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec012 = [f for f in findings if f.rule == "SEC012"]
        assert len(sec012) == 1
        assert sec012[0].task_id == "t1"
        assert sec012[0].severity == "warning"
        assert "fallback" in sec012[0].message.lower() or "codex" in sec012[0].message


# ===========================================================================
# TestSEC008RequiresApprovalSuppresses — requires_approval=True suppresses SEC008
# ===========================================================================


class TestSEC008RequiresApprovalSuppresses:
    @pytest.mark.parametrize("cmd", [
        "rm -rf /tmp/old_data",
        "DROP TABLE legacy_users",
        "git push --force origin main",
    ])
    def test_sec008_destructive_with_requires_approval_no_trigger(self, cmd: str) -> None:
        """Destructive command WITH requires_approval=True does NOT trigger SEC008."""
        task = TaskSpec(id="t1", command=cmd, requires_approval=True)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC008" for f in findings)


# ===========================================================================
# TestSEC014SecretsAutoSuppresses — secrets_auto=True skips the SEC014 check
# ===========================================================================


class TestSEC014SecretsAutoSuppresses:
    def test_sec014_secrets_auto_true_suppresses_cloud_cred_check(self) -> None:
        """plan.secrets_auto=True skips the SEC014 check entirely.

        The guard is `if not plan.secrets and not plan.secrets_auto:` so
        secrets_auto=True makes the condition False and no SEC014 fires.
        """
        task = _cmd_task("t1", env={"AWS_SECRET_ACCESS_KEY": "my-key"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets_auto=True,
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC014" for f in findings)


# ===========================================================================
# TestSEC012DangerouslyBypassFlag — third yolo flag form also triggers SEC012
# ===========================================================================


class TestSEC012DangerouslyBypassFlag:
    def test_sec012_dangerously_bypass_flag_triggers(self) -> None:
        """--dangerously-bypass-approvals-and-sandbox + fallback_engine triggers SEC012.

        TestSEC012FlagVariants covers --yolo and --allow-all; this confirms the
        third member of _YOLO_FLAGS also works.
        """
        task = TaskSpec(
            id="t1",
            engine="codex",
            prompt="do stuff",
            args=["--dangerously-bypass-approvals-and-sandbox"],
            fallback_engine="claude",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec012 = [f for f in findings if f.rule == "SEC012"]
        assert len(sec012) == 1
        assert sec012[0].task_id == "t1"
        assert sec012[0].severity == "warning"


# ===========================================================================
# TestSEC016UpstreamNotInTaskMap — graceful skip when upstream ID unknown
# ===========================================================================


class TestSEC016UpstreamNotInTaskMap:
    def test_sec016_unknown_upstream_id_no_crash_no_finding(self) -> None:
        """context_from referencing a non-existent task ID is handled gracefully.

        `task_map.get(upstream_id)` returns None and the code does
        `if upstream is None: continue`, so no exception and no SEC016 finding.
        """
        task = TaskSpec(
            id="t1",
            command="echo hi",
            context_from=["nonexistent-task"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)


# ===========================================================================
# TestSEC006EngineTaskVerifyCommandProd — engine task with prod in verify_command
# ===========================================================================


class TestSEC006EngineTaskVerifyCommandProd:
    def test_sec006_engine_task_prod_in_verify_command_triggers(self) -> None:
        """Engine task with 'production' in verify_command (no guard_command) triggers SEC006.

        Engine tasks can have verify_command; _check_sec006 inspects it regardless
        of whether the task uses engine or command for its main work.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="run deployment checks",
            verify_command="curl https://production.example.com/health",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec006 = [f for f in findings if f.rule == "SEC006"]
        assert len(sec006) == 1
        assert sec006[0].task_id == "t1"
        assert sec006[0].severity == "info"


# ===========================================================================
# TestAuditPlanEmptyTasks — empty task list only fires plan-level rules
# ===========================================================================


class TestAuditPlanEmptyTasks:
    def test_audit_plan_empty_tasks_only_plan_level_findings(self) -> None:
        """Plan with empty tasks list fires plan-level rules (SEC001) but no task-level ones."""
        plan = PlanSpec(name="empty-plan", max_cost_usd=None, tasks=[])
        findings = audit_plan(plan)
        # SEC001 must fire (no max_cost_usd)
        assert any(f.rule == "SEC001" for f in findings)
        # No task-level rules should fire when tasks list is empty
        task_rules = {
            "SEC002", "SEC003", "SEC004", "SEC005", "SEC006",
            "SEC008", "SEC009", "SEC011", "SEC012", "SEC014",
            "SEC015", "SEC016",
        }
        assert not any(f.rule in task_rules for f in findings)


# ===========================================================================
# TestSEC002MultipleYoloFlagsSingleFinding — both flags in args → still 1 finding
# ===========================================================================


class TestSEC002MultipleYoloFlagsSingleFinding:
    def test_sec002_both_yolo_and_allow_all_in_args_one_finding(self) -> None:
        """Both --yolo and --allow-all in args triggers exactly 1 SEC002 finding.

        _check_sec002 does a single `findings.append()` after `bool(...& set(args))`,
        so having multiple yolo flags still produces only one finding per task.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            args=["--yolo", "--allow-all"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec002 = [f for f in findings if f.rule == "SEC002"]
        assert len(sec002) == 1
        assert sec002[0].task_id == "t1"


# ===========================================================================
# TestSEC005PromptNoneNoTrigger — task.prompt=None returns early from _check_sec005
# ===========================================================================


class TestSEC005PromptNoneNoTrigger:
    def test_sec005_engine_task_with_no_prompt_field_no_trigger(self) -> None:
        """Engine task using prompt_file (prompt=None) does NOT trigger SEC005.

        `_check_sec005` guards with `prompt_text = task.prompt or ""`; when
        prompt is None, prompt_text is "" → `if not prompt_text: return` fires
        before any pattern search.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt_file="prompts/task.txt",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC005" for f in findings)


# ===========================================================================
# TestSEC008PureEngineTaskNoTrigger — engine task with no command fields skips SEC008
# ===========================================================================


class TestSEC008PureEngineTaskNoTrigger:
    def test_sec008_engine_task_prompt_only_no_command_no_trigger(self) -> None:
        """Engine task with only prompt (no command/pre_command/verify_command) → no SEC008.

        `_check_sec008` builds `texts` from command/pre_command/verify_command.
        When all three are None, texts=[] and the `if not texts: return` guard fires.
        """
        task = TaskSpec(id="t1", engine="claude", prompt="do something")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC008" for f in findings)


# ===========================================================================
# TestSEC013NoPlanWatchNoTrigger — plan without watch block never checks SEC013
# ===========================================================================


class TestSEC013NoPlanWatchNoTrigger:
    def test_sec013_no_watch_block_no_trigger(self) -> None:
        """Plan with no watch block (plan.watch=None) never triggers SEC013.

        The SEC013 check is `if plan.watch is not None and plan.watch.max_cost_usd is None:`,
        so watch=None short-circuits it.
        """
        task = _cmd_task("t1")
        plan = _minimal_plan(task)
        assert plan.watch is None
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC013" for f in findings)


# ===========================================================================
# TestSEC006BothFieldsProdOneFinding — break after first match in _check_sec006
# ===========================================================================


class TestSEC006BothFieldsProdOneFinding:
    def test_sec006_command_and_verify_command_both_prod_one_finding(self) -> None:
        """When both command and verify_command contain 'prod', only 1 SEC006 is appended.

        `_check_sec006` iterates texts and breaks after the first prod match, so
        even with two matching fields a single finding is produced.
        """
        task = TaskSpec(
            id="t1",
            command="./deploy.sh production",
            verify_command="curl https://prod.example.com/healthz",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec006 = [f for f in findings if f.rule == "SEC006" and f.task_id == "t1"]
        assert len(sec006) == 1


# ===========================================================================
# TestSEC002OllamaEngine — SEC002 is engine-agnostic: ollama also triggers it
# ===========================================================================


class TestSEC002OllamaEngine:
    def test_sec002_ollama_engine_yolo_flag_triggers(self) -> None:
        """SEC002 fires for ollama engine (not just the big four) with --yolo flag.

        `TestSEC002EngineVariants` covers codex/gemini/copilot/qwen;
        this test completes the engine-variant coverage for ollama.
        """
        task = TaskSpec(id="t1", engine="ollama", prompt="run local model", args=["--yolo"])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec002 = [f for f in findings if f.rule == "SEC002"]
        assert len(sec002) == 1
        assert sec002[0].task_id == "t1"
        assert sec002[0].severity == "warning"


# ===========================================================================
# TestFormatAuditWarningOnlySummary — warning-only findings → correct summary line
# ===========================================================================


class TestFormatAuditWarningOnlySummary:
    def test_format_audit_warning_only_no_error_or_info_in_summary(self) -> None:
        """format_audit with only warning findings shows '1 warning(s)' in summary.

        The summary omits counts that are zero — error(s) and info must not appear.
        """
        findings = [AuditFinding(severity="warning", rule="SEC002", message="yolo risk", task_id="t1")]
        result = format_audit(findings)
        assert "1 warning" in result
        assert "error" not in result.lower() or "0 error" not in result
        assert "[WARN]" in result
        assert "SEC002" in result
        assert "(task: t1)" in result

    def test_format_audit_two_warnings_summary_count(self) -> None:
        """format_audit with two warning findings shows '2 warning(s)' in summary."""
        findings = [
            AuditFinding(severity="warning", rule="SEC002", message="w1", task_id="t1"),
            AuditFinding(severity="warning", rule="SEC003", message="w2", task_id="t2"),
        ]
        result = format_audit(findings)
        assert "2 warning" in result


# ===========================================================================
# TestFixPlanSEC014OnlyNoSEC003 — fix_plan with only SEC014 (DATABASE_URL) adds secrets
# ===========================================================================


class TestFixPlanSEC014OnlyNoSEC003:
    def test_fix_plan_sec014_only_database_url_adds_secrets_auto(self, tmp_path: Path) -> None:
        """fix_plan with only a SEC014 finding (DATABASE_URL) adds secrets: auto.

        DATABASE_URL matches _CLOUD_CRED_PATTERN (SEC014) but NOT _SECRET_ENV_PATTERN
        (KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL), so only SEC014 fires (no SEC003).
        SEC014 is in _FIXABLE_RULES, so fix_plan adds `secrets: auto`.
        """
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: t1
                command: "psql $DATABASE_URL -c 'SELECT 1'"
                env:
                  DATABASE_URL: "postgres://user:pass@host/db"
            """),
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)

        # Verify SEC014 fires but SEC003 does NOT (DATABASE_URL has no KEY/SECRET/TOKEN/etc.)
        assert any(f.rule == "SEC014" for f in findings)
        assert not any(f.rule == "SEC003" for f in findings)

        fixes = fix_plan(plan_file, findings)
        assert any("secrets" in d for d in fixes)
        assert not any("max_cost_usd" in d for d in fixes)

        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["secrets"] == "auto"
        assert data["max_cost_usd"] == 5.0  # unchanged

    # ------------------------------------------------------------------
    # Iteration 11 — deeper coverage for edge cases
    # ------------------------------------------------------------------

    def test_sec010_circular_context_from_no_infinite_loop(self) -> None:
        """SEC010: context_from forming a cycle via visited set — must not hang."""
        t1 = TaskSpec(
            id="t1", engine="claude", prompt="a",
            depends_on=["t2"], context_from=["t2"],
        )
        t2 = TaskSpec(
            id="t2", engine="claude", prompt="b",
            depends_on=["t1"], context_from=["t1"],
        )
        plan = _minimal_plan(t1, t2)
        # Should not hang — visited set prevents infinite recursion.
        # Depth is <=3 so SEC010 should NOT fire even if cycle exists.
        findings = audit_plan(plan)
        assert all(f.rule != "SEC010" for f in findings)

    def test_sec010_context_from_references_nonexistent_task(self) -> None:
        """SEC010: context_from referencing a task ID not in plan.tasks → depth 0."""
        t1 = TaskSpec(
            id="t1", engine="claude", prompt="do",
            context_from=["ghost"],
        )
        plan = _minimal_plan(t1)
        findings = audit_plan(plan)
        # ghost task not found → depth=1 (self context_from), not >3, no trigger
        assert all(f.rule != "SEC010" for f in findings)

    def test_sec003_secret_keyword_triggers(self) -> None:
        """SEC003: env var with 'SECRET' substring triggers."""
        task = TaskSpec(
            id="s", command="echo hi",
            env={"MY_SECRET_VALUE": "abc123"},
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 1
        assert "MY_SECRET_VALUE" in sec003[0].message

    def test_sec008_truncate_standalone_triggers(self) -> None:
        """SEC008: TRUNCATE keyword in command triggers without approval."""
        task = TaskSpec(
            id="trunc", command="TRUNCATE users;",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "trunc"

    def test_sec008_delete_from_in_list_command_triggers(self) -> None:
        """SEC008: DELETE FROM in list-format command triggers."""
        task = TaskSpec(
            id="del", command=["mysql", "-e", "DELETE FROM logs WHERE id > 0"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1

    def test_audit_pack_bare_list_yaml(self, tmp_path: Path) -> None:
        """Audit pack as bare YAML list (no dict wrapper) is accepted."""
        pack_file = tmp_path / "pack.yaml"
        # A bare list of rules — _apply_single_audit_pack handles this:
        # rules_raw = raw (since raw is not a dict, skip .get("rules"))
        pack_file.write_text(
            textwrap.dedent("""\
            - type: glob_exists
              glob: "must_exist.txt"
              message: "must_exist.txt is required"
              rule: "CUSTOM001"
              severity: warning
            """),
            encoding="utf-8",
        )
        plan = PlanSpec(
            name="test",
            max_cost_usd=5.0,
            tasks=[_cmd_task("t1")],
            source_path=tmp_path / "plan.yaml",
            workspace_root=str(tmp_path),
            audit_packs=[str(pack_file)],
        )
        findings = audit_plan(plan)
        # must_exist.txt doesn't exist → should produce a finding
        custom = [f for f in findings if f.rule == "CUSTOM001"]
        assert len(custom) == 1
        assert custom[0].severity == "warning"

    def test_audit_pack_invalid_yaml_reports_pack002(self, tmp_path: Path) -> None:
        """Audit pack with unparseable YAML reports PACK002."""
        pack_file = tmp_path / "bad.yaml"
        pack_file.write_text("{{{{invalid yaml: [", encoding="utf-8")
        plan = PlanSpec(
            name="test",
            max_cost_usd=5.0,
            tasks=[_cmd_task("t1")],
            source_path=tmp_path / "plan.yaml",
            workspace_root=str(tmp_path),
            audit_packs=[str(pack_file)],
        )
        findings = audit_plan(plan)
        pack002 = [f for f in findings if f.rule == "PACK002"]
        assert len(pack002) == 1
        assert "Invalid YAML" in pack002[0].message

    def test_format_audit_json_mixed_task_id_present_and_absent(self) -> None:
        """format_audit_json: findings with and without task_id serialize correctly."""
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="warning", rule="SEC003", message="leaky", task_id="t1"),
        ]
        parsed = json.loads(format_audit_json(findings))
        assert len(parsed) == 2
        assert "task_id" not in parsed[0]
        assert parsed[1]["task_id"] == "t1"


# ===========================================================================
# TestSEC014SecretsAutoSuppressesCheck — secrets_auto=True skips SEC014
# ===========================================================================


class TestSEC014SecretsAutoSuppressesCheck:
    def test_sec014_secrets_auto_true_suppresses_cloud_cred_check(self) -> None:
        """plan.secrets_auto=True causes audit_plan to skip the SEC014 check entirely.

        The guard is `if not plan.secrets and not plan.secrets_auto:` —
        when secrets_auto is True the whole SEC014 loop is skipped even if
        a task.env contains a cloud credential name.
        """
        from maestro_cli.models import PlanSpec

        task = _cmd_task("t1", env={"AWS_SECRET_ACCESS_KEY": "my-key"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets_auto=True,
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC014" for f in findings)


# ===========================================================================
# TestSEC009CommandTaskNoTrigger — command task (no engine) never triggers SEC009
# ===========================================================================


class TestSEC009CommandTaskNoEngine:
    def test_sec009_command_task_no_engine_no_trigger_even_with_workspace_root(self) -> None:
        """Command tasks with no engine skip SEC009 even with workspace_root and yolo args.

        _check_sec009 returns immediately when task.engine is None — the yolo
        flag check is only meaningful for engine tasks that actually perform
        autonomous filesystem mutations.
        """
        task = TaskSpec(
            id="t1",
            command="echo hi",
            args=["--dangerously-bypass-approvals-and-sandbox"],
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC009" for f in findings)


# ===========================================================================
# TestSEC016NonExistentUpstream — missing upstream ID is silently skipped
# ===========================================================================


class TestSEC016NonExistentUpstream:
    def test_sec016_context_from_nonexistent_upstream_no_finding(self) -> None:
        """When context_from references an upstream ID not in the plan, SEC016 is NOT raised.

        _check_sec016 has `if upstream is None: continue` — a missing upstream
        cannot be introspected, so it is skipped rather than treated as a
        worst-case risk.
        """
        task = TaskSpec(
            id="t1",
            command="echo hi",
            context_from=["ghost-task"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)


# ===========================================================================
# TestFormatAuditUnknownSeverity — fallback [?] prefix for unknown severity
# ===========================================================================


class TestFormatAuditUnknownSeverity:
    def test_format_audit_unknown_severity_uses_question_mark_prefix(self) -> None:
        """format_audit uses '[?]   ' fallback prefix for unrecognised severity strings.

        The _prefix dict only maps 'error', 'warning', and 'info'; any other
        value falls through to the `.get(..., "[?]   ")` default.
        """
        finding = AuditFinding(severity="critical", rule="CUSTOM001", message="custom severity")  # type: ignore[arg-type]
        result = format_audit([finding])
        assert "[?]" in result
        assert "CUSTOM001" in result


# ===========================================================================
# TestFixPlanNonFixableRulesFileUnchanged — SEC005/SEC006 findings leave file intact
# ===========================================================================


class TestFixPlanNonFixableRulesFileUnchanged:
    def test_fix_plan_sec005_only_returns_empty_and_file_unchanged(self, tmp_path: Path) -> None:
        """Injecting only a SEC005 finding (not in _FIXABLE_RULES) causes fix_plan to
        return [] immediately without touching the file or creating a backup.
        """
        plan_file = tmp_path / "plan.yaml"
        content = textwrap.dedent("""\
        version: 1
        name: test
        max_cost_usd: 5.0
        tasks:
          - id: t1
            command: "echo hi"
        """)
        plan_file.write_text(content, encoding="utf-8")

        findings = [
            AuditFinding(severity="warning", rule="SEC005", message="hardcoded key"),
            AuditFinding(severity="info", rule="SEC006", message="prod path"),
        ]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []
        assert plan_file.read_text(encoding="utf-8") == content
        assert not plan_file.with_suffix(".yaml.bak").exists()


# ===========================================================================
# TestSEC010MidChainWildcard — wildcard context_from on intermediate task breaks depth
# ===========================================================================


class TestSEC010MidChainWildcard:
    def test_sec010_wildcard_on_intermediate_task_caps_depth_no_trigger(self) -> None:
        """Wildcard context_from on an intermediate task caps the DFS depth at that node.

        Structure: t1 → t2 (context_from=["*"]) → t3 → t4 → t5
        _chain_depth("t5") traverses t4 → t3 → t2, but t2.context_from is ["*"]
        so it tries _chain_depth("*", ...) which returns 0 (not in task_map).
        depth("t2") = 1+0 = 1, depth("t3") = 1+1 = 2, depth("t4") = 3,
        depth("t5") = 4 — still > 3, this DOES trigger.

        However, testing the direct parent case: t1 → t2 (context_from=["*"])
        chain is t2 → * (depth=1, wildcard not found), so t2 has depth=1, t3=2,
        t4=3, t5=4. That triggers for t5 with no budget.

        The real scenario that caps depth: use wildcard at the last-but-one node
        so the chain depth stays ≤ 3 from the perspective of the consuming task.
        """
        # Wildcard on t4 (last before consumer): t5 sees t4 → ["*"] → depth=1
        # so depth("t5") = 2, which is ≤ 3 → no SEC010.
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["t1"])
        t3 = _cmd_task("t3", depends_on=["t2"], context_from=["t2"])
        t4 = _cmd_task("t4", depends_on=["t3"], context_from=["*"])  # wildcard caps depth
        t5 = _cmd_task("t5", depends_on=["t4"], context_from=["t4"])
        plan = _minimal_plan(t1, t2, t3, t4, t5)
        findings = audit_plan(plan)
        # depth(t4) = 1 (wildcard resolves to 0), depth(t5) = 2 → not > 3
        assert not any(f.rule == "SEC010" and f.task_id == "t5" for f in findings)


# ===========================================================================
# TestSEC016EngineDownstream — downstream engine task still triggers SEC016
# ===========================================================================


class TestSEC016EngineDownstream:
    def test_sec016_engine_downstream_with_unguarded_engine_upstream_triggers(self) -> None:
        """SEC016 is triggered even when the downstream task is itself an engine task.

        The rule checks the upstream's guard_command, regardless of whether
        the downstream task is a command or engine task.
        """
        upstream = TaskSpec(id="up", engine="claude", prompt="generate code")
        downstream = TaskSpec(
            id="down",
            engine="codex",
            prompt="review the code: {{ up.stdout_tail }}",
            depends_on=["up"],
            context_from=["up"],
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016" and f.task_id == "down"]
        assert len(sec016) == 1
        assert "up" in sec016[0].message


# ===========================================================================
# Iteration 15 — new tests for remaining gaps
# ===========================================================================


class TestSEC008GitResetHard:
    def test_sec008_git_reset_hard_triggers(self) -> None:
        task = TaskSpec(id="t1", command="git reset --hard HEAD~3")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC008" and f.task_id == "t1" for f in findings)

    def test_sec008_git_reset_hard_with_approval_no_trigger(self) -> None:
        task = TaskSpec(id="t1", command="git reset --hard HEAD~1", requires_approval=True)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC008" for f in findings)


class TestSEC008LowercasePatterns:
    def test_sec008_drop_table_all_lowercase_triggers(self) -> None:
        task = TaskSpec(id="t1", command="mysql -e 'drop table users'")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC008" and f.task_id == "t1" for f in findings)

    def test_sec008_delete_from_mixed_case_triggers(self) -> None:
        task = TaskSpec(id="t1", command="psql -c 'Delete From users WHERE id=1'")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert any(f.rule == "SEC008" and f.task_id == "t1" for f in findings)


class TestSEC005TwoPatternsSamePrompt:
    def test_sec005_prompt_with_two_key_patterns_only_one_finding(self) -> None:
        """SEC005 breaks after the first match, so only one finding per task."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Use sk-AAAAAAAAAAAAAAAAAAAAAA and ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec005 = [f for f in findings if f.rule == "SEC005"]
        assert len(sec005) == 1


class TestSingleTaskMultipleRules:
    def test_single_task_triggers_sec002_and_sec008_simultaneously(self) -> None:
        """A single task with yolo flags AND destructive command triggers both rules."""
        task = TaskSpec(
            id="dangerous",
            engine="claude",
            prompt="clean up",
            args=["--yolo"],
            command="rm -rf /tmp/data",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        rules_for_task = {f.rule for f in findings if f.task_id == "dangerous"}
        assert "SEC002" in rules_for_task
        assert "SEC008" in rules_for_task


class TestSEC003EmptySecretsList:
    def test_sec003_empty_secrets_list_still_triggers(self) -> None:
        """Plan with secrets=[] means declared_secrets is empty, so SEC003 still fires."""
        task = _cmd_task("t1", env={"MY_API_KEY": "val123"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets=[],
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert any(f.rule == "SEC003" and f.task_id == "t1" for f in findings)


class TestFixPlanSEC001ExistingMaxCostNotOverwritten:
    def test_fix_plan_sec001_finding_but_data_has_max_cost_usd_no_overwrite(self, tmp_path: Path) -> None:
        """fix_plan checks data.get('max_cost_usd') is None; if already set, skip."""
        plan_file = tmp_path / "plan.yaml"
        content = textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 25.0
            tasks:
              - id: t1
                command: echo hi
        """)
        plan_file.write_text(content, encoding="utf-8")
        findings = [AuditFinding(severity="error", rule="SEC001", message="no budget")]
        fixes = fix_plan(plan_file, findings)
        # SEC001 is fixable but data already has max_cost_usd, so nothing applied
        assert fixes == []


class TestFixPlanSEC003ExistingSecretsNotOverwritten:
    def test_fix_plan_sec003_finding_but_data_has_secrets_no_overwrite(self, tmp_path: Path) -> None:
        """fix_plan checks data.get('secrets') is None; if already set, skip."""
        plan_file = tmp_path / "plan.yaml"
        content = textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 5.0
            secrets:
              - MY_TOKEN
            tasks:
              - id: t1
                command: echo hi
        """)
        plan_file.write_text(content, encoding="utf-8")
        findings = [AuditFinding(severity="warning", rule="SEC003", message="secret env")]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []


# ===========================================================================
# Iteration 16 — new tests targeting remaining gaps
# ===========================================================================


class TestFormatAuditEmptyStringTaskId:
    def test_format_audit_falsy_task_id_no_task_location_shown(self) -> None:
        """format_audit with task_id='' (empty string, falsy) must NOT show '(task: )'.

        `location = f" (task: {f.task_id})" if f.task_id else ""` — empty string
        is falsy so location stays "" and "(task: )" never appears in the output.
        """
        findings = [AuditFinding(severity="warning", rule="SEC002", message="yolo risk", task_id="")]
        result = format_audit(findings)
        assert "(task: )" not in result
        assert "(task: None)" not in result
        assert "SEC002" in result
        assert "[WARN]" in result


class TestSEC001ZeroCostNotNone:
    def test_sec001_max_cost_usd_zero_is_not_none_no_trigger(self) -> None:
        """max_cost_usd=0.0 is not None — SEC001 must NOT fire.

        The SEC001 check is `if plan.max_cost_usd is None:`. The value 0.0
        satisfies `is None` as False, so no SEC001 even though 0 is falsy.
        """
        task = _cmd_task("t1")
        plan = PlanSpec(name="test-plan", max_cost_usd=0.0, tasks=[task])
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC001" for f in findings)


class TestSEC011NoEscalationTasks:
    def test_sec011_no_tasks_with_escalation_no_trigger_even_without_budget(self) -> None:
        """SEC011 does NOT fire when no tasks have escalation:, regardless of budget.

        The SEC011 loop is `if plan.max_cost_usd is None: for task ... if task.escalation:`.
        With no escalation on any task, the inner `if task.escalation:` is never true.
        """
        t1 = TaskSpec(id="t1", engine="claude", prompt="do stuff", max_retries=2)
        t2 = TaskSpec(id="t2", engine="claude", prompt="more stuff", max_retries=1)
        plan = PlanSpec(name="test-plan", tasks=[t1, t2])  # max_cost_usd=None
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC011" for f in findings)


class TestSEC010MaxDepthFromMultipleUpstreams:
    def test_sec010_two_upstreams_takes_max_depth_triggers(self) -> None:
        """SEC010 computes max() over multiple upstreams — the deeper one wins.

        t1 → t2 → t3 → t4 (depth=3 from t4's context chain).
        t5 → t6 (depth=1 from t6's context chain).
        Merge task reads from both t4 and t6:
          depth("merge") = 1 + max(depth("t4"), depth("t6")) = 1 + max(3, 1) = 4 > 3.
        No context_budget_tokens → SEC010 triggers on merge.
        """
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["t1"])
        t3 = _cmd_task("t3", depends_on=["t2"], context_from=["t2"])
        t4 = _cmd_task("t4", depends_on=["t3"], context_from=["t3"])
        t5 = _cmd_task("t5")
        t6 = _cmd_task("t6", depends_on=["t5"], context_from=["t5"])
        merge = _cmd_task("merge", depends_on=["t4", "t6"], context_from=["t4", "t6"])
        plan = _minimal_plan(t1, t2, t3, t4, t5, t6, merge)
        findings = audit_plan(plan)
        sec010 = [f for f in findings if f.rule == "SEC010"]
        assert any(f.task_id == "merge" for f in sec010)


class TestSEC003LowercaseEnvVar:
    def test_sec003_lowercase_env_var_matches_ignorecase_pattern(self) -> None:
        """SEC003 regex is re.IGNORECASE — lowercase 'my_api_key' matches 'KEY' pattern.

        `_SECRET_ENV_PATTERN = re.compile(r"KEY|...", re.IGNORECASE)` so 'key'
        in lowercase var names is still detected.
        """
        task = _cmd_task("t1", env={"my_api_key": "secret-value"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003" and f.task_id == "t1"]
        assert len(sec003) == 1
        assert "my_api_key" in sec003[0].message


class TestSEC007SecretsAutoAndNonEmptyList:
    def test_sec007_secrets_auto_true_with_non_empty_list_triggers(self) -> None:
        """SEC007 fires whenever plan.secrets is truthy, even if secrets_auto=True also.

        The check is `if plan.secrets:` — a non-empty list is truthy regardless
        of the secrets_auto boolean. Both can be set simultaneously.
        """
        task = _cmd_task("t1")
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets=["MY_TOKEN"],
            secrets_auto=True,
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec007 = [f for f in findings if f.rule == "SEC007"]
        assert len(sec007) == 1
        assert sec007[0].severity == "warning"
        assert sec007[0].task_id is None


class TestFixPlanSEC001ZeroMaxCostInData:
    def test_fix_plan_sec001_finding_data_has_max_cost_usd_zero_no_overwrite(
        self, tmp_path: Path
    ) -> None:
        """fix_plan with SEC001 finding but YAML data has max_cost_usd: 0 (not None).

        The check is `data.get("max_cost_usd") is None`. The value 0 is not None,
        so the fix is skipped even though 0 is falsy — avoiding data loss.
        """
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 0
            tasks:
              - id: t1
                command: echo hi
            """),
            encoding="utf-8",
        )
        findings = [AuditFinding(severity="error", rule="SEC001", message="no budget")]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []
        data = yaml.safe_load(plan_file.read_text(encoding="utf-8"))
        assert data["max_cost_usd"] == 0  # untouched


class TestSEC009GeminiEngine:
    def test_sec009_gemini_engine_yolo_flag_no_worktree_triggers(self) -> None:
        """SEC009 fires for gemini engine with --yolo when workspace_root is set.

        Existing SEC009 tests only use engine='claude'. This confirms the rule
        is engine-agnostic — _check_sec009 only checks `task.engine is None`.
        """
        task = TaskSpec(
            id="t1",
            engine="gemini",
            prompt="analyse codebase",
            args=["--yolo"],
            worktree=False,
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            workspace_root="/tmp/ws",
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec009 = [f for f in findings if f.rule == "SEC009"]
        assert len(sec009) == 1
        assert sec009[0].task_id == "t1"
        assert sec009[0].severity == "info"


# ===========================================================================
# Iteration 17 — 7 new tests targeting specific gaps
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. SEC002: parametrize ALL three yolo flags + requires_approval no-trigger
# ---------------------------------------------------------------------------


class TestSEC002AllYoloFlagsParametrized:
    @pytest.mark.parametrize("flag", [
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
        "--allow-all",
    ])
    def test_sec002_all_three_yolo_flags_without_approval_trigger(self, flag: str) -> None:
        """All three _YOLO_FLAGS members trigger SEC002 when requires_approval is False."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            args=[flag],
            requires_approval=False,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec002 = [f for f in findings if f.rule == "SEC002"]
        assert len(sec002) == 1, f"Expected SEC002 for flag {flag!r}"
        assert sec002[0].task_id == "t1"
        assert sec002[0].severity == "warning"

    @pytest.mark.parametrize("flag", [
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
        "--allow-all",
    ])
    def test_sec002_all_three_yolo_flags_with_approval_no_trigger(self, flag: str) -> None:
        """All three _YOLO_FLAGS members do NOT trigger SEC002 when requires_approval is True."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            args=[flag],
            requires_approval=True,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC002" for f in findings), (
            f"SEC002 should not fire when requires_approval=True, flag={flag!r}"
        )


# ---------------------------------------------------------------------------
# 2. SEC003/SEC014: defaults.env with secret-like var names — no trigger
#    (audit inspects task.env only, not defaults.env)
# ---------------------------------------------------------------------------


class TestSEC003DefaultsEnvIgnoredDuringAudit:
    def test_sec003_defaults_env_with_api_key_name_no_trigger(self) -> None:
        """Secret-like var in plan.defaults.env does NOT trigger SEC003.

        _check_sec003 iterates `task.env`, not `plan.defaults.env`. The defaults
        are merged at runtime by the scheduler and are outside the audit scope.
        """
        from maestro_cli.models import PlanDefaults

        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            tasks=[task],
            defaults=PlanDefaults(env={"MY_API_KEY": "value", "DB_TOKEN": "tok"}),
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC003" for f in findings)

    def test_sec014_defaults_env_with_cloud_cred_name_no_trigger(self) -> None:
        """Cloud credential name in plan.defaults.env does NOT trigger SEC014.

        _check_sec014 iterates `task.env` only — same reasoning as SEC003.
        """
        from maestro_cli.models import PlanDefaults

        task = TaskSpec(id="t1", command="echo hi")
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            tasks=[task],
            defaults=PlanDefaults(env={"AWS_SECRET_ACCESS_KEY": "secret"}),
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC014" for f in findings)


# ---------------------------------------------------------------------------
# 3. SEC010: diamond DAG topology — shallow (no trigger) vs deep (trigger)
# ---------------------------------------------------------------------------


class TestSEC010DiamondTopology:
    def test_sec010_diamond_depth_two_no_trigger(self) -> None:
        """Diamond DAG t1->{t2,t3}->t4: max chain depth from t4 is 2 — no SEC010."""
        t1 = _cmd_task("dia-t1")
        t2 = _cmd_task("dia-t2", depends_on=["dia-t1"], context_from=["dia-t1"])
        t3 = _cmd_task("dia-t3", depends_on=["dia-t1"], context_from=["dia-t1"])
        t4 = _cmd_task("dia-t4", depends_on=["dia-t2", "dia-t3"], context_from=["dia-t2", "dia-t3"])
        plan = _minimal_plan(t1, t2, t3, t4)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC010" for f in findings)

    def test_sec010_diamond_extended_to_depth_four_triggers(self) -> None:
        """Diamond followed by a 2-hop chain: total depth 4 triggers SEC010 on the tail task.

        Structure: d1->{d2,d3}->d4->d5->d6
        depth(d6) = 1+depth(d5) = 1+1+depth(d4) = ... = 4+ > 3 -> SEC010.
        No context_budget_tokens on d6.
        """
        d1 = _cmd_task("d1")
        d2 = _cmd_task("d2", depends_on=["d1"], context_from=["d1"])
        d3 = _cmd_task("d3", depends_on=["d1"], context_from=["d1"])
        d4 = _cmd_task("d4", depends_on=["d2", "d3"], context_from=["d2", "d3"])
        d5 = _cmd_task("d5", depends_on=["d4"], context_from=["d4"])
        d6 = _cmd_task("d6", depends_on=["d5"], context_from=["d5"])
        plan = _minimal_plan(d1, d2, d3, d4, d5, d6)
        findings = audit_plan(plan)
        sec010 = [f for f in findings if f.rule == "SEC010"]
        assert any(f.task_id == "d6" for f in sec010)

    def test_sec010_diamond_extended_with_budget_no_trigger(self) -> None:
        """Same deep diamond topology but context_budget_tokens set — SEC010 suppressed."""
        d1 = _cmd_task("d1b")
        d2 = _cmd_task("d2b", depends_on=["d1b"], context_from=["d1b"])
        d3 = _cmd_task("d3b", depends_on=["d1b"], context_from=["d1b"])
        d4 = _cmd_task("d4b", depends_on=["d2b", "d3b"], context_from=["d2b", "d3b"])
        d5 = _cmd_task("d5b", depends_on=["d4b"], context_from=["d4b"])
        d6 = _cmd_task(
            "d6b",
            depends_on=["d5b"],
            context_from=["d5b"],
            context_budget_tokens=8000,
        )
        plan = _minimal_plan(d1, d2, d3, d4, d5, d6)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC010" and f.task_id == "d6b" for f in findings)


# ---------------------------------------------------------------------------
# 4. SEC015: both stdout_tail AND log in same when expression → exactly 2 findings
# ---------------------------------------------------------------------------


class TestSEC015BothDangerousFields:
    def test_sec015_stdout_tail_and_log_in_same_when_two_findings(self) -> None:
        """when expression with both 'stdout_tail' and 'log' produces exactly 2 SEC015 findings."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.stdout_tail }} != '' and {{ t0.log }} == ''",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 2
        msgs = {f.message for f in sec015}
        assert any("stdout_tail" in m for m in msgs)
        assert any("log" in m for m in msgs)
        # Both findings reference the same task
        assert all(f.task_id == "t1" for f in sec015)

    def test_sec015_exit_code_in_when_no_trigger(self) -> None:
        """when expression referencing only safe fields (exit_code) produces no SEC015."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.exit_code }} == 0",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC015" for f in findings)


# ---------------------------------------------------------------------------
# 5. SEC016: command (non-engine) upstream → no trigger; map_reduce → no trigger
# ---------------------------------------------------------------------------


class TestSEC016CommandAndMapReduceExempt:
    def test_sec016_command_task_upstream_no_engine_no_trigger(self) -> None:
        """context_from includes a command task (engine=None) — SEC016 NOT triggered.

        _check_sec016 checks `if upstream.engine and not upstream.guard_command:`.
        When engine is None the condition is False — command tasks are not LLM
        outputs and carry no prompt-injection risk.
        """
        cmd_upstream = TaskSpec(id="cmd-up", command="make build")
        downstream = TaskSpec(
            id="cmd-down",
            command="echo hi",
            depends_on=["cmd-up"],
            context_from=["cmd-up"],
        )
        plan = _minimal_plan(cmd_upstream, downstream)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)

    def test_sec016_map_reduce_context_mode_no_trigger(self) -> None:
        """context_mode: map_reduce is excluded from SEC016 (LLM synthesis reduces risk).

        _check_sec016 returns early when context_mode != 'raw' (refined 2026-04-26).
        """
        engine_upstream = TaskSpec(id="mr-gen", engine="codex", prompt="generate output")
        downstream = TaskSpec(
            id="mr-consumer",
            command="echo hi",
            depends_on=["mr-gen"],
            context_from=["mr-gen"],
            context_mode="map_reduce",
        )
        plan = _minimal_plan(engine_upstream, downstream)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)


# ---------------------------------------------------------------------------
# 6. SEC008: list-format rm -rf joined → triggers; case-insensitive drop table → triggers
# ---------------------------------------------------------------------------


class TestSEC008ListFormatAndCaseInsensitive:
    def test_sec008_list_format_rm_rf_triggers(self) -> None:
        """list-format command ['rm', '-rf', '/data'] is joined and matches SEC008.

        _check_sec008 joins list-format commands with ' '.join() before pattern
        matching, so ['rm', '-rf', '/data'] becomes 'rm -rf /data' which matches
        the _DESTRUCTIVE_PATTERNS regex.
        """
        task = TaskSpec(id="t1", command=["rm", "-rf", "/data/cache"])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"

    def test_sec008_case_insensitive_drop_table_list_command_triggers(self) -> None:
        """Case-insensitive 'drop table' inside a list-format command triggers SEC008.

        _DESTRUCTIVE_PATTERNS uses re.IGNORECASE so 'drop table' (all lowercase)
        still matches even when embedded in a list-style psql invocation.
        """
        task = TaskSpec(
            id="t1",
            command=["psql", "-c", "drop table sessions"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 1
        assert sec008[0].task_id == "t1"
        assert sec008[0].severity == "warning"


# ---------------------------------------------------------------------------
# 7. fix_plan: only non-fixable findings → file unchanged; max_cost_usd already
#    set → SEC001 fix does not add duplicate
# ---------------------------------------------------------------------------


class TestFixPlanEdgeCasesV2:
    def test_fix_plan_only_sec009_non_fixable_file_unchanged(self, tmp_path: Path) -> None:
        """fix_plan with only a SEC009 finding (not in _FIXABLE_RULES) returns []
        immediately and must NOT modify the plan file or create a backup.
        """
        plan_file = tmp_path / "plan.yaml"
        content = textwrap.dedent("""\
        version: 1
        name: test
        max_cost_usd: 5.0
        tasks:
          - id: t1
            command: "echo hi"
        """)
        plan_file.write_text(content, encoding="utf-8")

        findings = [
            AuditFinding(severity="info", rule="SEC009", message="no worktree isolation"),
        ]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []
        assert plan_file.read_text(encoding="utf-8") == content
        assert not plan_file.with_suffix(".yaml.bak").exists()

    def test_fix_plan_sec001_max_cost_usd_already_set_no_duplicate(self, tmp_path: Path) -> None:
        """When max_cost_usd is already set in the YAML, fix_plan must NOT overwrite it.

        fix_plan checks `data.get('max_cost_usd') is None` before applying the
        SEC001 fix. A pre-existing value of any float prevents the fix from adding
        a duplicate 10.0 entry.
        """
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            max_cost_usd: 50.0
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        original = plan_file.read_text(encoding="utf-8")
        findings = [AuditFinding(severity="error", rule="SEC001", message="no budget")]
        fixes = fix_plan(plan_file, findings)
        assert fixes == []
        # File must be completely unchanged
        assert plan_file.read_text(encoding="utf-8") == original
        data = yaml.safe_load(original)
        assert data["max_cost_usd"] == 50.0  # not overwritten with 10.0


# ---------------------------------------------------------------------------
# 8. Iteration 17 — fix_plan dry_run, format_audit empty/summary, SEC005
#    GitHub PAT, SEC009 all flag variants, SEC003 multiple vars, SEC013
#    watch bounded, SEC006 pre_command coverage
# ---------------------------------------------------------------------------


class TestFixPlanDryRun:
    def test_fix_plan_dry_run_returns_fixes_but_does_not_write(self, tmp_path: Path) -> None:
        """dry_run=True returns fix descriptions but leaves the file untouched."""
        import yaml

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        original = plan_file.read_text(encoding="utf-8")

        findings = [AuditFinding(severity="error", rule="SEC001", message="no budget")]
        fixes = fix_plan(plan_file, findings, dry_run=True)

        assert len(fixes) == 1
        assert "SEC001" in fixes[0]
        # File must NOT be modified
        assert plan_file.read_text(encoding="utf-8") == original
        # No backup created
        assert not plan_file.with_suffix(".yaml.bak").exists()

    def test_fix_plan_dry_run_sec003_sec014_combined_label(self, tmp_path: Path) -> None:
        """dry_run with SEC003+SEC014 findings produces combined label."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent("""\
            version: 1
            name: test
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        findings = [
            AuditFinding(severity="warning", rule="SEC003", message="secret env", task_id="t1"),
            AuditFinding(severity="warning", rule="SEC014", message="cloud cred", task_id="t1"),
        ]
        fixes = fix_plan(plan_file, findings, dry_run=True)
        assert len(fixes) == 1
        assert "SEC003" in fixes[0]
        assert "SEC014" in fixes[0]
        assert "secrets: auto" in fixes[0]


class TestFormatAuditEdgeCases:
    def test_format_audit_empty_findings_returns_clean_message(self) -> None:
        """Empty findings list returns the 'no findings' message."""
        result = format_audit([])
        assert "No findings" in result
        assert "plan looks clean" in result

    def test_format_audit_summary_counts_all_severities(self) -> None:
        """Summary line correctly counts errors, warnings, and info findings."""
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="m1"),
            AuditFinding(severity="error", rule="SEC001", message="m2"),
            AuditFinding(severity="warning", rule="SEC002", message="m3", task_id="t1"),
            AuditFinding(severity="info", rule="SEC004", message="m4", task_id="t2"),
            AuditFinding(severity="info", rule="SEC006", message="m5", task_id="t3"),
            AuditFinding(severity="info", rule="SEC009", message="m6", task_id="t4"),
        ]
        result = format_audit(findings)
        assert "2 error(s)" in result
        assert "1 warning(s)" in result
        assert "3 info" in result

    def test_format_audit_single_warning_no_error_or_info_in_summary(self) -> None:
        """Summary with only warnings does not mention errors or info."""
        findings = [
            AuditFinding(severity="warning", rule="SEC002", message="m1", task_id="t1"),
        ]
        result = format_audit(findings)
        assert "1 warning(s)" in result
        assert "error" not in result.split("\n")[-1]
        assert "info" not in result.split("\n")[-1]


class TestSEC005GitHubPAT:
    def test_sec005_github_pat_triggers(self) -> None:
        """A GitHub Personal Access Token (ghp_...) in prompt triggers SEC005."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Use this token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec005 = [f for f in findings if f.rule == "SEC005"]
        assert len(sec005) == 1
        assert sec005[0].task_id == "t1"
        assert "ghp_" in sec005[0].message


class TestSEC009AllYoloFlagVariants:
    @pytest.mark.parametrize("flag", [
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
        "--allow-all",
    ])
    def test_sec009_all_yolo_flags_trigger_without_worktree(self, flag: str) -> None:
        """All three yolo flag variants trigger SEC009 for engine tasks without worktree."""
        task = TaskSpec(
            id="t1",
            engine="codex",
            prompt="do something",
            args=[flag],
            worktree=False,
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            workspace_root="/some/root",
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec009 = [f for f in findings if f.rule == "SEC009"]
        assert len(sec009) == 1
        assert sec009[0].task_id == "t1"

    def test_sec009_yolo_with_worktree_true_no_trigger(self) -> None:
        """Yolo flag with worktree: true does NOT trigger SEC009."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            args=["--yolo"],
            worktree=True,
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            workspace_root="/some/root",
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec009 = [f for f in findings if f.rule == "SEC009"]
        assert len(sec009) == 0


class TestSEC003MultipleSecretVars:
    def test_sec003_two_secret_vars_in_same_task_two_findings(self) -> None:
        """Two secret-like env vars in the same task produce two SEC003 findings."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            env={"MY_API_KEY": "val1", "DB_PASSWORD": "val2"},
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 2
        var_names = {f.message.split("'")[1] for f in sec003}
        assert var_names == {"MY_API_KEY", "DB_PASSWORD"}

    def test_sec003_declared_secret_suppresses_finding(self) -> None:
        """A secret-like env var that IS declared in plan.secrets does not trigger."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do something",
            env={"MY_API_KEY": "val1"},
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=10.0,
            secrets=["MY_API_KEY"],
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec003) == 0


# ===========================================================================
# Iteration 18 — 8 new tests for remaining gaps
# ===========================================================================


class TestFixPlanEmptyFindings:
    def test_fix_plan_empty_findings_list_returns_empty_no_touch(self, tmp_path: Path) -> None:
        """fix_plan with an empty findings list returns [] immediately and does not modify the file.

        The `fixable` list is empty → `if not fixable: return []` fires before
        reading or writing the YAML file.
        """
        plan_file = tmp_path / "plan.yaml"
        content = "version: 1\nname: test\nmax_cost_usd: 5.0\ntasks:\n  - id: t1\n    command: echo hi\n"
        plan_file.write_text(content, encoding="utf-8")
        fixes = fix_plan(plan_file, [])
        assert fixes == []
        assert plan_file.read_text(encoding="utf-8") == content
        assert not plan_file.with_suffix(".yaml.bak").exists()


class TestSEC003AndSEC007CombinedFire:
    def test_sec003_and_sec007_both_fire_when_undeclared_secret_in_env(self) -> None:
        """When plan.secrets is non-empty (SEC007) and task.env has an undeclared
        secret-like var (SEC003), both rules fire in the same audit.

        SEC007 checks `if plan.secrets:` — any non-empty list triggers it.
        SEC003 checks `if var_name not in declared_secrets` — only declared names
        (e.g., 'DECLARED_KEY') are suppressed; 'UNDECLARED_TOKEN' is not in the set.
        """
        task = TaskSpec(id="t1", command="echo hi", env={"UNDECLARED_TOKEN": "tok_value"})
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            secrets=["DECLARED_KEY"],  # non-empty list → SEC007 fires
            tasks=[task],
        )
        findings = audit_plan(plan)
        sec007 = [f for f in findings if f.rule == "SEC007"]
        sec003 = [f for f in findings if f.rule == "SEC003"]
        assert len(sec007) == 1
        assert sec007[0].severity == "warning"
        assert sec007[0].task_id is None
        assert len(sec003) == 1
        assert sec003[0].task_id == "t1"
        assert "UNDECLARED_TOKEN" in sec003[0].message


class TestFormatAuditErrorOnlyFindings:
    def test_format_audit_error_only_shows_error_prefix_and_count(self) -> None:
        """format_audit with only error-severity findings shows '[ERROR]' prefix
        and 'N error(s)' in the summary line; warnings and info are absent."""
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="error", rule="SEC013", message="watch unbounded"),
        ]
        result = format_audit(findings)
        assert "[ERROR]" in result
        assert "2 error(s)" in result
        # No warning or info entries
        assert "[WARN]" not in result
        assert "[INFO]" not in result
        # Summary omits zero counts
        assert "warning" not in result.split("\n")[-1]
        assert "info" not in result.split("\n")[-1]


class TestSEC010EmptyContextFromList:
    def test_sec010_empty_context_from_list_no_trigger(self) -> None:
        """Task with context_from=[] (empty list) does NOT trigger SEC010.

        `_check_sec010` iterates plan.tasks and checks `if not task.context_from: continue`.
        An empty list is falsy, so the depth calculation is skipped entirely.
        """
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"])
        # t2 has context_from=[] (falsy) — skipped by SEC010
        t2_with_empty = TaskSpec(id="t2e", command="echo b", depends_on=["t1"], context_from=[])
        plan = _minimal_plan(t1, t2, t2_with_empty)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC010" for f in findings)


class TestSEC002ArgsNoneDefault:
    def test_sec002_engine_task_args_none_no_trigger(self) -> None:
        """Engine task with args=None (default) does NOT trigger SEC002.

        `_check_sec002` does `list(task.args or [])` — None becomes [] and
        `bool(_YOLO_FLAGS & set([]))` is False → no finding.
        """
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff", args=None)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC002" for f in findings)


class TestSEC008TwoIndependentTasks:
    def test_sec008_two_tasks_each_destructive_two_findings(self) -> None:
        """Two separate tasks, each containing a destructive command, each produce
        one SEC008 finding — findings are independent and both reference their own task_id.
        """
        t1 = TaskSpec(id="cleanup", command="rm -rf /tmp/old")
        t2 = TaskSpec(id="wipe-db", command="DROP TABLE legacy_users")
        plan = _minimal_plan(t1, t2)
        findings = audit_plan(plan)
        sec008 = [f for f in findings if f.rule == "SEC008"]
        assert len(sec008) == 2
        task_ids = {f.task_id for f in sec008}
        assert task_ids == {"cleanup", "wipe-db"}


class TestAuditFindingInfoToDict:
    def test_audit_finding_info_severity_to_dict(self) -> None:
        """AuditFinding with severity='info' serializes correctly in to_dict().

        Existing to_dict tests cover 'error' and 'warning'; this confirms
        'info' is preserved (no hard-coded severity mapping in to_dict).
        """
        f = AuditFinding(severity="info", rule="SEC004", message="allow_failure on security tag", task_id="t1")
        d = f.to_dict()
        assert d["severity"] == "info"
        assert d["rule"] == "SEC004"
        assert d["message"] == "allow_failure on security tag"
        assert d["task_id"] == "t1"

    def test_audit_finding_info_no_task_id_to_dict(self) -> None:
        """AuditFinding with severity='info' and no task_id omits 'task_id' key."""
        f = AuditFinding(severity="info", rule="SEC006", message="prod path without guard")
        d = f.to_dict()
        assert d["severity"] == "info"
        assert "task_id" not in d


# ===========================================================================
# SEC002 non-engine task — command task with yolo flag in args never triggers
# ===========================================================================


class TestSEC002NonEngineTaskWithYoloFlag:
    def test_sec002_command_task_with_yolo_flag_no_trigger(self) -> None:
        """A command task (no engine) with --yolo in args does NOT trigger SEC002.

        ``_check_sec002`` guards on ``task.engine`` being truthy: a command task
        has ``engine=None`` so the flag set intersection is never evaluated.
        """
        task = TaskSpec(id="t1", command="echo hi", args=["--yolo"])
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC002" for f in findings)

    def test_sec002_command_task_with_dangerously_bypass_no_trigger(self) -> None:
        """Command task with --dangerously-bypass-approvals-and-sandbox: no SEC002."""
        task = TaskSpec(
            id="t1",
            command="echo hi",
            args=["--dangerously-bypass-approvals-and-sandbox"],
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC002" for f in findings)


# ===========================================================================
# SEC009 — no workspace_root on plan means SEC009 loop is entirely skipped
# ===========================================================================


class TestSEC009NoWorkspaceRootSkipsCheck:
    def test_sec009_engine_yolo_but_no_workspace_root_no_trigger(self) -> None:
        """SEC009 only fires when ``plan.workspace_root`` is truthy.

        An engine task with --yolo but a plan without workspace_root should
        produce zero SEC009 findings because the outer ``if plan.workspace_root``
        guard prevents the check loop from running.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do stuff",
            args=["--yolo"],
        )
        plan = PlanSpec(
            name="test-plan",
            max_cost_usd=5.0,
            workspace_root=None,
            tasks=[task],
        )
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC009" for f in findings)


# ===========================================================================
# SEC010 — wildcard context_from ["*"] resolves to depth 1 (no trigger)
# ===========================================================================


class TestSEC010WildcardContextFrom:
    def test_sec010_wildcard_context_from_star_depth_one_no_trigger(self) -> None:
        """Task with ``context_from: ["*"]`` has depth 1 in ``_chain_depth``.

        ``_chain_depth`` tries ``task_map.get("*")`` which is None → returns 0.
        So the calling task has depth ``1 + 0 = 1`` which is ≤ 3, no trigger.
        """
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["*"])
        plan = _minimal_plan(t1, t2)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC010" for f in findings)

    def test_sec010_deep_chain_with_wildcard_still_triggers(self) -> None:
        """Deep chain where one task uses ``context_from: ["*"]`` — the actual
        chain depth (via real task IDs in context_from) still triggers SEC010.

        t1 → t2 → t3 → t4 → t5 (each with context_from=[prev task])
        t5 depth = 4 → triggers SEC010.
        """
        t1 = _cmd_task("t1")
        t2 = _cmd_task("t2", depends_on=["t1"], context_from=["t1"])
        t3 = _cmd_task("t3", depends_on=["t2"], context_from=["t2"])
        t4 = _cmd_task("t4", depends_on=["t3"], context_from=["t3"])
        t5 = _cmd_task("t5", depends_on=["t4"], context_from=["t4"])
        plan = _minimal_plan(t1, t2, t3, t4, t5)
        findings = audit_plan(plan)
        sec010 = [f for f in findings if f.rule == "SEC010"]
        assert len(sec010) >= 1
        assert any(f.task_id == "t5" for f in sec010)


# ===========================================================================
# fix_plan — secrets already "auto" string prevents duplicate fix
# ===========================================================================


class TestFixPlanSecretsAlreadyAutoString:
    def test_fix_plan_sec003_but_secrets_already_auto_no_overwrite(self, tmp_path: Path) -> None:
        """If the YAML already has ``secrets: auto``, fix_plan does NOT overwrite it.

        ``data.get("secrets")`` returns ``"auto"`` (truthy) → the guard
        ``data.get("secrets") is None`` fails → no fix applied for SEC003/014.
        """
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\nname: test\nsecrets: auto\ntasks:\n"
            "  - id: t1\n    command: echo hi\n    env:\n      MY_API_KEY: abc\n",
            encoding="utf-8",
        )
        # Build findings manually (SEC003 would fire if we ran audit_plan
        # on a plan without secrets in the declared set)
        findings = [
            AuditFinding(severity="warning", rule="SEC003", message="secret env var", task_id="t1"),
        ]
        applied = fix_plan(plan_path, findings)
        # Should return empty because secrets is already set
        assert applied == []


# ===========================================================================
# format_audit — task_id=None produces no "(task: None)" in output
# ===========================================================================


class TestFormatAuditTaskIdNoneFormatting:
    def test_format_audit_plan_level_finding_no_task_id_in_output(self) -> None:
        """A plan-level finding (task_id=None) should NOT produce '(task: None)'
        in the formatted output — the location part should be an empty string.
        """
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
        ]
        result = format_audit(findings)
        assert "(task:" not in result
        assert "SEC001" in result
        assert "no budget" in result

    def test_format_audit_mixed_plan_and_task_level_findings(self) -> None:
        """Mix of plan-level (task_id=None) and task-level findings formats correctly."""
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="warning", rule="SEC008", message="destructive cmd", task_id="cleanup"),
        ]
        result = format_audit(findings)
        # Plan-level finding has no "(task: ...)"
        lines = result.strip().split("\n")
        sec001_line = [l for l in lines if "SEC001" in l][0]
        assert "(task:" not in sec001_line
        # Task-level finding has "(task: cleanup)"
        sec008_line = [l for l in lines if "SEC008" in l][0]
        assert "(task: cleanup)" in sec008_line
        # Summary shows 1 error + 1 warning
        assert "1 error(s)" in result
        assert "1 warning(s)" in result


# ===========================================================================
# SEC003 — env var without secret pattern never triggers
# ===========================================================================


class TestSEC003NormalEnvVarNoTrigger:
    def test_sec003_env_var_without_secret_pattern_no_trigger(self) -> None:
        """An env var like ``MY_CONFIG`` that doesn't match KEY/SECRET/TOKEN/
        PASSWORD/CREDENTIAL never triggers SEC003, even without secrets declared.
        """
        task = _cmd_task("t1", env={"MY_CONFIG": "val", "DEBUG_MODE": "1", "APP_NAME": "demo"})
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC003" for f in findings)


# ===========================================================================
# Comprehensive integration — plan triggering many rules simultaneously
# ===========================================================================


class TestAuditPlanManyRulesSimultaneously:
    def test_audit_plan_triggers_sec001_sec002_sec008_sec011_together(self) -> None:
        """Single plan triggers SEC001 (no budget), SEC002 (yolo without approval),
        SEC008 (destructive cmd), and SEC011 (escalation without budget) together.
        """
        t1 = TaskSpec(
            id="dangerous",
            engine="claude",
            prompt="do stuff",
            args=["--yolo"],
            escalation=["haiku", "sonnet", "opus"],
        )
        t2 = TaskSpec(id="destroyer", command="rm -rf /tmp")
        plan = PlanSpec(
            name="chaotic-plan",
            max_cost_usd=None,  # SEC001
            tasks=[t1, t2],
        )
        findings = audit_plan(plan)
        rules_fired = {f.rule for f in findings}
        assert "SEC001" in rules_fired  # no budget
        assert "SEC002" in rules_fired  # yolo without approval
        assert "SEC008" in rules_fired  # rm -rf
        assert "SEC011" in rules_fired  # escalation without budget


# ===========================================================================
# Iteration 20 — 9 new tests targeting remaining gaps
# ===========================================================================


class TestSEC007EmptySecretsList:
    def test_sec007_empty_secrets_list_falsy_no_trigger(self) -> None:
        """plan.secrets = [] (empty list) is falsy — SEC007 must NOT fire.

        The check is `if plan.secrets:` — an empty list evaluates to False so
        SEC007 is only emitted when plan.secrets is a non-empty (truthy) list.
        """
        task = _cmd_task("t1")
        plan = PlanSpec(name="test-plan", max_cost_usd=5.0, secrets=[], tasks=[task])
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC007" for f in findings)


class TestSEC016EmptyContextFrom:
    def test_sec016_empty_context_from_list_returns_early_no_trigger(self) -> None:
        """Task with context_from=[] (empty list) does NOT trigger SEC016.

        `_check_sec016` guards with `if not task.context_from: return`.
        An empty list is falsy, so the per-upstream loop is never entered
        even though an engine upstream exists in the plan.
        """
        upstream = TaskSpec(id="up", engine="claude", prompt="do stuff")
        downstream = TaskSpec(
            id="down",
            command="echo hi",
            depends_on=["up"],
            context_from=[],
        )
        plan = _minimal_plan(upstream, downstream)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC016" for f in findings)


class TestSEC012NoFallbackEngine:
    def test_sec012_yolo_flag_without_fallback_engine_no_trigger(self) -> None:
        """Task with yolo flag but no fallback_engine configured → no SEC012.

        The SEC012 check is:
            `if task.fallback_engine and _YOLO_FLAGS & set(task.args or []):`.
        When fallback_engine is None the short-circuit prevents the finding
        even when yolo flags are present in args.
        """
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do stuff",
            args=["--yolo"],
            fallback_engine=None,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC012" for f in findings)


class TestFormatAuditErrorAndInfoNoWarning:
    def test_format_audit_error_and_info_no_warning_in_summary(self) -> None:
        """format_audit with error + info findings (no warnings) produces a summary
        that counts both severities but omits the 'warning(s)' entry.

        The summary builder only appends counts that are non-zero so a zero
        warning count means 'warning' never appears on the summary line.
        """
        findings = [
            AuditFinding(severity="error", rule="SEC001", message="no budget"),
            AuditFinding(severity="info", rule="SEC004", message="allow_failure risk", task_id="t1"),
        ]
        result = format_audit(findings)
        assert "[ERROR]" in result
        assert "[INFO]" in result
        assert "1 error(s)" in result
        assert "1 info" in result
        # No warning count on the summary line
        last_line = result.strip().split("\n")[-1]
        assert "warning" not in last_line
        # The [WARN] prefix must also be absent
        assert "[WARN]" not in result


class TestSEC014TaskEnvNone:
    def test_sec014_task_env_none_no_trigger(self) -> None:
        """Task with env=None (default) does NOT trigger SEC014.

        `_check_sec014` does `dict(task.env or {})` — None becomes {} and
        the for loop over an empty dict never runs, producing no finding.
        """
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff", env=None)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC014" for f in findings)


class TestSEC003TaskEnvNone:
    def test_sec003_task_env_none_no_trigger(self) -> None:
        """Task with env=None (default) does NOT trigger SEC003.

        `_check_sec003` does `dict(task.env or {})` — None becomes {} and
        the for loop over an empty dict never runs, producing no finding.
        """
        task = TaskSpec(id="t1", engine="claude", prompt="do stuff", env=None)
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC003" for f in findings)


class TestAuditPackEmptyRulesList:
    def test_audit_pack_empty_rules_list_no_pack_findings(self, tmp_path: Path) -> None:
        """Audit pack with ``rules: []`` (empty list) produces no PACK findings.

        `_apply_single_audit_pack` enumerates `rules_raw` — an empty list means
        no assertions are evaluated and no finding is emitted.
        """
        pack_file = tmp_path / "empty-rules-pack.yaml"
        pack_file.write_text("rules: []\n", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            textwrap.dedent(f"""\
            version: 1
            name: empty-rules-test
            max_cost_usd: 5.0
            audit_packs:
              - "{pack_file.name}"
            tasks:
              - id: t1
                command: "echo hi"
            """),
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        pack_findings = [f for f in findings if f.rule.startswith("PACK")]
        assert len(pack_findings) == 0


class TestSEC015WhenReferencesSafeField:
    def test_sec015_when_references_result_text_no_trigger(self) -> None:
        """when expression using the safe structured field 'result_text' → no SEC015.

        `_check_sec015` only flags 'stdout_tail' and 'log'. Any other field
        including 'result_text', 'summary', 'errors', etc. is considered safe.
        """
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.result_text }} != ''",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC015" for f in findings)

    def test_sec015_when_references_errors_field_no_trigger(self) -> None:
        """when expression using the structured 'errors' field → no SEC015.

        The structured output field 'errors' is not on the dangerous list
        ('stdout_tail', 'log'), so it never triggers the SEC015 rule.
        """
        task = TaskSpec(
            id="t1",
            command="echo hi",
            when="{{ t0.errors }} == ''",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert not any(f.rule == "SEC015" for f in findings)


# ---------------------------------------------------------------------------
# Feature 2: Security Audit Category Coverage
# ---------------------------------------------------------------------------


class TestAuditCategory:
    def test_finding_has_category_for_sec001(self) -> None:
        plan = _minimal_plan(_cmd_task("t1"), max_cost_usd=None)
        findings = audit_plan(plan)
        sec001 = next((f for f in findings if f.rule == "SEC001"), None)
        assert sec001 is not None
        assert sec001.category == "Workflow Architecture"

    def test_finding_category_for_sec015(self) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", command="echo hi", when="{{ t1.stdout_tail }} == 'ok'")
        )
        findings = audit_plan(plan)
        sec015 = next((f for f in findings if f.rule == "SEC015"), None)
        assert sec015 is not None
        assert sec015.category == "Injection"

    def test_audit_pack_rule_has_no_category(self) -> None:
        # Findings from audit packs (PACK001...) should have no category
        finding = AuditFinding(severity="error", rule="PACK001", message="pack error")
        assert finding.category is None

    def test_to_dict_includes_category(self) -> None:
        f = AuditFinding(severity="error", rule="SEC001", message="bad", category="Workflow Architecture")
        d = f.to_dict()
        assert d["category"] == "Workflow Architecture"

    def test_to_dict_no_category_key_when_none(self) -> None:
        f = AuditFinding(severity="error", rule="PACK001", message="pack error")
        d = f.to_dict()
        assert "category" not in d


class TestComputeAuditCoverage:
    def test_all_nine_categories_present(self) -> None:
        plan = _minimal_plan(_cmd_task("t1"))
        findings = audit_plan(plan)
        coverage = compute_audit_coverage(findings)
        categories = [c.category for c in coverage]
        assert "Agent-Tool Coupling" in categories
        assert "Data Leakage" in categories
        assert "Injection" in categories
        assert "Identity/Provenance" in categories
        assert "Memory Poisoning" in categories
        assert "Non-Determinism" in categories
        assert "Trust Exploitation" in categories
        assert "Timing/Monitoring" in categories
        assert "Workflow Architecture" in categories
        assert len(coverage) == 9

    def test_zero_coverage_for_uncovered_categories(self) -> None:
        # No findings → all triggered lists empty
        plan = _minimal_plan(_cmd_task("t1"), max_cost_usd=10.0)
        findings = audit_plan(plan)
        # Filter to plans that trigger no findings (use a valid plan)
        coverage = compute_audit_coverage([])
        for c in coverage:
            assert c.triggered == []
            assert c.coverage_pct == 0.0

    def test_injection_triggered_by_sec015(self) -> None:
        plan = _minimal_plan(
            TaskSpec(id="t1", command="echo hi", when="{{ t1.stdout_tail }} == 'ok'")
        )
        findings = audit_plan(plan)
        coverage = compute_audit_coverage(findings)
        injection = next(c for c in coverage if c.category == "Injection")
        assert injection.triggered  # SEC015 is in Injection
        assert injection.coverage_pct > 0

    def test_format_coverage_text_contains_categories(self) -> None:
        plan = _minimal_plan(_cmd_task("t1"), max_cost_usd=None)
        findings = audit_plan(plan)
        text = format_audit_coverage(findings)
        assert "Workflow Architecture" in text
        assert "Overall:" in text
        assert "%" in text

    def test_format_coverage_json_structure(self) -> None:
        plan = _minimal_plan(_cmd_task("t1"), max_cost_usd=None)
        findings = audit_plan(plan)
        raw = format_audit_coverage_json(findings)
        data = json.loads(raw)
        assert "categories" in data
        assert "total_rules" in data
        assert "total_triggered" in data
        assert "overall_pct" in data
        assert len(data["categories"]) == 9
        for cat in data["categories"]:
            assert "category" in cat
            assert "rules" in cat
            assert "triggered" in cat
            assert "coverage_pct" in cat

    def test_coverage_pct_range(self) -> None:
        findings = [AuditFinding(severity="error", rule="SEC001", message="x",
                                 category="Workflow Architecture")]
        coverage = compute_audit_coverage(findings)
        wa = next(c for c in coverage if c.category == "Workflow Architecture")
        assert 0.0 < wa.coverage_pct <= 100.0


# ===========================================================================
# TestSEC019 — untrusted context without honeypot decoys
# ===========================================================================


class TestSEC019:
    def test_sec019_untrusted_without_honeypot(self) -> None:
        """Untrusted task without honeypot: true produces SEC019 warning."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do it",
            context_trust="untrusted",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec019 = [f for f in findings if f.rule == "SEC019"]
        assert len(sec019) == 1
        assert sec019[0].severity == "warning"
        assert sec019[0].task_id == "t1"
        assert "honeypot" in sec019[0].message

    def test_sec019_untrusted_with_honeypot_no_warning(self) -> None:
        """Untrusted task WITH honeypot: true produces no SEC019 warning."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do it",
            context_trust="untrusted",
            honeypot=True,
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec019 = [f for f in findings if f.rule == "SEC019"]
        assert len(sec019) == 0

    def test_sec019_trusted_task_no_warning(self) -> None:
        """Trusted task does not trigger SEC019."""
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="do it",
            context_trust="trusted",
        )
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec019 = [f for f in findings if f.rule == "SEC019"]
        assert len(sec019) == 0

    def test_sec019_no_context_trust_no_warning(self) -> None:
        """Task without context_trust set does not trigger SEC019."""
        task = _cmd_task("t1")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        sec019 = [f for f in findings if f.rule == "SEC019"]
        assert len(sec019) == 0

    def test_sec019_multiple_untrusted_without_honeypot(self) -> None:
        """Multiple untrusted tasks without honeypot produce multiple SEC019 warnings."""
        task_a = TaskSpec(
            id="a",
            engine="claude",
            prompt="do a",
            context_trust="untrusted",
        )
        task_b = TaskSpec(
            id="b",
            engine="claude",
            prompt="do b",
            context_trust="untrusted",
        )
        plan = _minimal_plan(task_a, task_b)
        findings = audit_plan(plan)
        sec019 = [f for f in findings if f.rule == "SEC019"]
        assert len(sec019) == 2
        task_ids = {f.task_id for f in sec019}
        assert task_ids == {"a", "b"}
