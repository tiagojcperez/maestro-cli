from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from maestro_cli.cli import _build_parser, _find_latest_run, _parse_set_vars, _split_csv, main

# ---------------------------------------------------------------------------
# Module-level YAML constants
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

_TWO_TASK_PLAN_YAML = """\
version: 1
name: two-task-plan
tasks:
  - id: first
    command: "echo first"
  - id: second
    depends_on: [first]
    command: "echo second"
"""

_PLAN_WITH_WEBHOOK_YAML = """\
version: 1
name: webhook-plan
webhook_url: "https://example.com/from-plan"
tasks:
  - id: t1
    command: "echo hello"
"""

_INVALID_YAML = """\
version: 1
name: test
tasks: "not a list"
"""

_MALFORMED_YAML = """\
version: 1
name: test
tasks:
  - id: t1
    command: "echo hi"
  bad indentation here
"""

_BRIEF_YAML = """\
name: test-feature
goal: "Add a new feature"
max_parallel: 3
tasks:
  - id: add-endpoint
    description: "Add REST endpoint"
    task_type: implementation
    prompt_hint: "Create GET /api/thing"
"""

_BRIEF_WITH_SECURITY_YAML = """\
name: security-project
goal: "Multiple security audits"
max_parallel: 2
include_quality_gates: false
include_build_verify: false
tasks:
  - id: audit-1
    description: "Audit auth flow"
    task_type: security-audit
  - id: audit-2
    description: "Audit data flow"
    task_type: security-audit
  - id: audit-3
    description: "Audit API keys"
    task_type: security-audit
"""

_BRIEF_INVALID_YAML = """\
not_name: oops
tasks:
  - id: x
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, content: str, name: str = "plan.yaml") -> Path:
    """Write *content* to a YAML file in *tmp_path* and return its path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# Tests for _split_csv
# ===========================================================================

class TestSplitCsv:
    def test_none_returns_empty_set(self) -> None:
        assert _split_csv(None) == set()

    def test_empty_list_returns_empty_set(self) -> None:
        assert _split_csv([]) == set()

    def test_single_csv_value(self) -> None:
        assert _split_csv(["a,b,c"]) == {"a", "b", "c"}

    def test_multiple_values(self) -> None:
        assert _split_csv(["a", "b,c"]) == {"a", "b", "c"}

    def test_whitespace_trimming(self) -> None:
        assert _split_csv(["  a , b  "]) == {"a", "b"}

    def test_empty_strings_filtered(self) -> None:
        assert _split_csv([",a,,b,"]) == {"a", "b"}

    def test_all_empty_returns_empty_set(self) -> None:
        assert _split_csv([",,,"]) == set()

    def test_single_item_no_comma(self) -> None:
        assert _split_csv(["task-1"]) == {"task-1"}

    def test_repeated_values_deduped(self) -> None:
        assert _split_csv(["a,b", "b,c"]) == {"a", "b", "c"}


class TestParseSetVars:
    def test_empty_list(self) -> None:
        assert _parse_set_vars([]) == {}

    def test_single_pair(self) -> None:
        assert _parse_set_vars(["env=prod"]) == {"env": "prod"}

    def test_multiple_pairs(self) -> None:
        result = _parse_set_vars(["env=prod", "region=eu-west-1"])
        assert result == {"env": "prod", "region": "eu-west-1"}

    def test_value_with_equals(self) -> None:
        assert _parse_set_vars(["query=a=b=c"]) == {"query": "a=b=c"}

    def test_empty_value(self) -> None:
        assert _parse_set_vars(["key="]) == {"key": ""}

    def test_duplicate_key_last_wins(self) -> None:
        assert _parse_set_vars(["k=a", "k=b"]) == {"k": "b"}

    def test_malformed_no_equals(self) -> None:
        with pytest.raises(SystemExit):
            _parse_set_vars(["no-equals-here"])

    def test_empty_key(self) -> None:
        with pytest.raises(SystemExit):
            _parse_set_vars(["=value"])

    def test_set_flag_on_run_parser(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--set", "a=1", "--set", "b=2"])
        assert args.set_vars == ["a=1", "b=2"]

    def test_set_flag_on_replan_parser(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["replan", "plan.yaml", "--set", "x=y"])
        assert args.set_vars == ["x=y"]

    def test_set_flag_on_watch_parser(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["watch", "plan.yaml", "--set", "x=y"])
        assert args.set_vars == ["x=y"]

    def test_set_flag_default_empty(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.set_vars == []


# ===========================================================================
# Tests for _build_parser
# ===========================================================================

class TestBuildParser:
    def test_subcommands_exist(self) -> None:
        parser = _build_parser()
        # Parsing each positional-arg subcommand with a dummy arg should work
        for cmd in ("validate", "run", "cleanup", "scaffold", "report"):
            args = parser.parse_args([cmd, "dummy.yaml"])
            assert args.command == cmd
        args = parser.parse_args(["backfill-costs"])
        assert args.command == "backfill-costs"
        args = parser.parse_args(["ui"])
        assert args.command == "ui"

    def test_report_has_output_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["report", "run-dir", "-o", "out.html"])
        assert args.command == "report"
        assert args.run_path == "run-dir"
        assert args.output == "out.html"

    def test_report_output_default_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["report", "run-dir"])
        assert args.output is None

    def test_run_has_dry_run_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--dry-run"])
        assert args.dry_run is True

    def test_run_dry_run_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.dry_run is False

    def test_run_has_max_parallel(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--max-parallel", "4"])
        assert args.max_parallel == 4

    def test_run_max_parallel_default_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.max_parallel is None

    def test_run_has_only(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--only", "t1,t2"])
        assert args.only == ["t1,t2"]

    def test_run_only_repeated(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--only", "t1", "--only", "t2"])
        assert args.only == ["t1", "t2"]

    def test_run_has_skip(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--skip", "t1"])
        assert args.skip == ["t1"]

    def test_run_has_execution_profile(self) -> None:
        for profile in ("plan", "safe", "yolo"):
            parser = _build_parser()
            args = parser.parse_args(["run", "plan.yaml", "--execution-profile", profile])
            assert args.execution_profile == profile

    def test_run_execution_profile_default_plan(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.execution_profile == "plan"

    def test_run_profile_mode_alias(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--profile-mode", "safe"])
        assert args.execution_profile == "safe"

    def test_run_mode_alias(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--mode", "yolo"])
        assert args.execution_profile == "yolo"

    def test_run_has_resume(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--resume", "/some/path"])
        assert args.resume == "/some/path"

    def test_run_resume_default_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.resume is None

    def test_run_has_run_dir(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--run-dir", "/custom/dir"])
        assert args.run_dir == "/custom/dir"

    def test_run_has_webhook(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--webhook", "https://example.com/hook"])
        assert args.webhook == "https://example.com/hook"

    def test_scaffold_has_brief_positional(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml"])
        assert args.brief == "brief.yaml"

    def test_scaffold_has_output(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml", "-o", "plan.yaml"])
        assert args.output == "plan.yaml"

    def test_scaffold_output_long_form(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml", "--output", "plan.yaml"])
        assert args.output == "plan.yaml"

    def test_scaffold_has_validate_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml", "--validate"])
        assert args.validate is True

    def test_scaffold_validate_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml"])
        assert args.validate is False

    def test_scaffold_has_cost_check_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml", "--cost-check"])
        assert args.cost_check is True

    def test_scaffold_cost_check_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["scaffold", "brief.yaml"])
        assert args.cost_check is False

    def test_cleanup_has_keep(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cleanup", "plan.yaml", "--keep", "5"])
        assert args.keep == 5

    def test_cleanup_keep_default_10(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cleanup", "plan.yaml"])
        assert args.keep == 10

    def test_cleanup_has_older_than(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cleanup", "plan.yaml", "--older-than", "30"])
        assert args.older_than == 30

    def test_cleanup_has_dry_run(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["cleanup", "plan.yaml", "--dry-run"])
        assert args.dry_run is True

    def test_backfill_costs_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["backfill-costs"])
        assert args.root == "."
        assert args.run_root is None
        assert args.dry_run is False
        assert args.codex_pricing_file is None

    def test_backfill_costs_with_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "backfill-costs",
                "--root",
                "/tmp/project",
                "--run-root",
                "/tmp/project/.maestro-runs",
                "--run-root",
                "/tmp/project/plans/.maestro-runs",
                "--dry-run",
                "--codex-pricing-file",
                "/tmp/pricing.json",
            ]
        )
        assert args.root == "/tmp/project"
        assert args.run_root == [
            "/tmp/project/.maestro-runs",
            "/tmp/project/plans/.maestro-runs",
        ]
        assert args.dry_run is True
        assert args.codex_pricing_file == "/tmp/pricing.json"

    def test_invalid_execution_profile_rejected(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "plan.yaml", "--execution-profile", "invalid"])

    def test_ui_project_root_repeatable(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "ui",
            "--project-root", "/tmp/a",
            "--project-root", "/tmp/b",
        ])
        assert args.project_root == ["/tmp/a", "/tmp/b"]

    def test_watch_parser_accepts_watch_specific_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([
            "watch",
            "plan.yaml",
            "--dry-run",
            "--execution-profile",
            "safe",
            "--max-parallel",
            "3",
            "--auto-approve",
            "--output",
            "jsonl",
            "--mask-secrets",
            "--resume-last",
        ])
        assert args.command == "watch"
        assert args.plan == "plan.yaml"
        assert args.dry_run is True
        assert args.execution_profile == "safe"
        assert args.max_parallel == 3
        assert args.auto_approve is True
        assert args.output == "jsonl"
        assert args.mask_secrets is True
        assert args.resume_last is True

    def test_run_resume_flags_are_mutually_exclusive(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "run",
                "plan.yaml",
                "--resume",
                "prior-run",
                "--resume-last",
            ])

    def test_replan_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["replan", "plan.yaml"])
        assert args.command == "replan"
        assert args.plan == "plan.yaml"
        assert args.max_attempts == 3
        assert args.model == "opus"
        assert args.variants == 1
        assert args.debug_prob == pytest.approx(0.5)
        assert args.selection_policy == "debug_prob"
        assert args.exploration_constant == pytest.approx(1.41421356237)
        assert args.population_strategy == "best"
        assert args.tournament_size == 2
        assert args.elite_count == 1
        assert args.diversity_floor == pytest.approx(0.25)
        assert args.auto_approve is False
        assert args.execution_profile == "plan"
        assert args.dry_run is False
        assert args.verbose is False
        assert args.quiet is False
        assert args.output == "text"

    def test_eval_parser_verbose_flag(self) -> None:
        parser = _build_parser()
        args_default = parser.parse_args(["eval", "suite.yaml", "run-dir"])
        assert args_default.verbose is False
        assert args_default.json is False
        args_with_flag = parser.parse_args(["eval", "suite.yaml", "run-dir", "-v", "--json"])
        assert args_with_flag.verbose is True
        assert args_with_flag.json is True

    def test_suggest_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["suggest", "plan.yaml"])
        assert args.command == "suggest"
        assert args.plan == "plan.yaml"
        assert args.min_runs == 3
        assert args.run_dir is None
        assert args.json is False

    def test_shell_parser_plan_default_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["shell"])
        assert args.command == "shell"
        assert args.plan is None
        args_with_plan = parser.parse_args(["shell", "--plan", "my.yaml"])
        assert args_with_plan.plan == "my.yaml"

    def test_run_verbose_and_quiet_are_mutually_exclusive(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "plan.yaml", "--verbose", "--quiet"])

    def test_ci_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["ci", "plan.yaml"])
        assert args.command == "ci"
        assert args.provider == "github_actions"
        assert args.workflow_name == "Maestro CI"
        assert args.python_version == "3.11"
        assert args.test_command == "python -m pytest -q"
        assert args.output is None

    def test_ui_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["ui"])
        assert args.command == "ui"
        assert args.host == "127.0.0.1"
        assert args.port == 8000
        assert args.no_browser is False
        assert args.project_root is None

    def test_status_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["status", "plan.yaml"])
        assert args.command == "status"
        assert args.plan == "plan.yaml"
        assert args.cache_dir is None
        assert args.run_dir is None
        assert args.json is False


# ===========================================================================
# Tests for _find_latest_run
# ===========================================================================

class TestFindLatestRun:
    def test_returns_latest_matching_run_with_manifest(self, tmp_path: Path) -> None:
        plan = SimpleNamespace(
            source_dir=tmp_path,
            run_dir=".maestro-runs",
            name="plan name",
        )
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        older = run_root / "20260308-120000_plan_name"
        older.mkdir()
        (older / "run_manifest.json").write_text("{}", encoding="utf-8")

        latest = run_root / "20260308-130000_plan_name"
        latest.mkdir()
        (latest / "run_manifest.json").write_text("{}", encoding="utf-8")

        ignored_missing_manifest = run_root / "20260308-140000_plan_name"
        ignored_missing_manifest.mkdir()

        ignored_other_plan = run_root / "20260308-150000_other_plan"
        ignored_other_plan.mkdir()
        (ignored_other_plan / "run_manifest.json").write_text("{}", encoding="utf-8")

        assert _find_latest_run(plan) == latest

    def test_uses_run_dir_override_instead_of_plan_default(self, tmp_path: Path) -> None:
        plan = SimpleNamespace(
            source_dir=tmp_path,
            run_dir=".maestro-runs",
            name="test-plan",
        )
        default_root = tmp_path / ".maestro-runs"
        default_root.mkdir()
        default_candidate = default_root / "20260308-150000_test-plan"
        default_candidate.mkdir()
        (default_candidate / "run_manifest.json").write_text("{}", encoding="utf-8")

        override_root = tmp_path / "custom-runs"
        override_root.mkdir()
        override_candidate = override_root / "20260308-120000_test-plan"
        override_candidate.mkdir()
        (override_candidate / "run_manifest.json").write_text("{}", encoding="utf-8")

        assert _find_latest_run(plan, run_dir="custom-runs") == override_candidate


# ===========================================================================
# Tests for main() — validate subcommand
# ===========================================================================

class TestMainValidate:
    def test_valid_plan_returns_zero(self, tmp_path: Path) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        rc = main(["validate", str(plan_file)])
        assert rc == 0

    def test_valid_plan_prints_metadata(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        main(["validate", str(plan_file)])
        out = capsys.readouterr().out
        assert "Plan is valid" in out
        assert "test-plan" in out
        assert "tasks: 2" in out
        assert "max_parallel: 2" in out
        assert "fail_fast: True" in out

    def test_valid_plan_prints_resolved_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        main(["validate", str(plan_file)])
        out = capsys.readouterr().out
        assert str(plan_file.resolve()) in out

    def test_missing_file_returns_one(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        rc = main(["validate", str(missing)])
        assert rc == 1

    def test_missing_file_prints_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        missing = tmp_path / "nonexistent.yaml"
        main(["validate", str(missing)])
        out = capsys.readouterr().out
        assert "[maestro] error:" in out

    def test_invalid_yaml_returns_one(self, tmp_path: Path) -> None:
        plan_file = _write_yaml(tmp_path, _INVALID_YAML)
        rc = main(["validate", str(plan_file)])
        assert rc == 1

    def test_invalid_yaml_prints_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plan_file = _write_yaml(tmp_path, _INVALID_YAML)
        main(["validate", str(plan_file)])
        out = capsys.readouterr().out
        assert "[maestro] error:" in out

    def test_malformed_yaml_returns_one(self, tmp_path: Path) -> None:
        plan_file = _write_yaml(tmp_path, _MALFORMED_YAML)
        rc = main(["validate", str(plan_file)])
        assert rc == 1

    def test_two_task_plan_shows_correct_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _TWO_TASK_PLAN_YAML)
        main(["validate", str(plan_file)])
        out = capsys.readouterr().out
        assert "tasks: 2" in out
        assert "two-task-plan" in out


# ===========================================================================
# Tests for main() — scaffold subcommand
# ===========================================================================

class TestMainScaffold:
    def test_valid_brief_returns_zero(self, tmp_path: Path) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        rc = main(["scaffold", str(brief_file)])
        assert rc == 0

    def test_valid_brief_prints_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        main(["scaffold", str(brief_file)])
        out = capsys.readouterr().out
        assert "version: 1" in out
        assert "name: test-feature" in out
        assert "add-endpoint" in out

    def test_output_flag_writes_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        output_file = tmp_path / "generated.yaml"
        rc = main(["scaffold", str(brief_file), "-o", str(output_file)])
        assert rc == 0
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "version: 1" in content
        assert "test-feature" in content
        out = capsys.readouterr().out
        assert f"plan written to {output_file}" in out

    def test_validate_flag_prints_valid(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        rc = main(["scaffold", str(brief_file), "--validate"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "generated plan is valid" in out

    def test_cost_check_flag_with_security_audits(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_WITH_SECURITY_YAML, "brief.yaml")
        rc = main(["scaffold", str(brief_file), "--cost-check"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "cost-warning:" in out

    def test_cost_check_no_quality_gates_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The security brief above has include_quality_gates=false
        # so the cost checker should warn about missing review/qa
        brief_file = _write_yaml(tmp_path, _BRIEF_WITH_SECURITY_YAML, "brief.yaml")
        main(["scaffold", str(brief_file), "--cost-check"])
        out = capsys.readouterr().out
        # Security-only brief with no quality gates should trigger warnings
        assert "cost-warning:" in out

    def test_invalid_brief_returns_one(self, tmp_path: Path) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_INVALID_YAML, "brief.yaml")
        rc = main(["scaffold", str(brief_file)])
        assert rc == 1

    def test_invalid_brief_prints_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_INVALID_YAML, "brief.yaml")
        main(["scaffold", str(brief_file)])
        out = capsys.readouterr().out
        assert "[maestro] error:" in out

    def test_missing_brief_returns_one(self, tmp_path: Path) -> None:
        missing = tmp_path / "noexist.yaml"
        rc = main(["scaffold", str(missing)])
        assert rc == 1

    def test_validate_and_output_combined(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        output_file = tmp_path / "combined.yaml"
        rc = main(["scaffold", str(brief_file), "--validate", "-o", str(output_file)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "generated plan is valid" in out
        assert f"plan written to {output_file}" in out
        assert output_file.exists()

    def test_all_flags_combined(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        output_file = tmp_path / "all_flags.yaml"
        rc = main([
            "scaffold", str(brief_file),
            "--validate", "--cost-check", "-o", str(output_file),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "generated plan is valid" in out
        assert f"plan written to {output_file}" in out


# ===========================================================================
# Tests for main() — ci and diff subcommands
# ===========================================================================

class TestMainCi:
    def test_ci_prints_yaml_and_uses_default_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_generate_ci_yaml = MagicMock(return_value="name: Maestro CI\n")

        monkeypatch.setattr("maestro_cli.cli.generate_ci_yaml", mock_generate_ci_yaml)

        rc = main(["ci", str(plan_file)])

        assert rc == 0
        mock_generate_ci_yaml.assert_called_once_with(
            str(plan_file),
            provider="github_actions",
            workflow_name="Maestro CI",
            python_version="3.11",
            test_command="python -m pytest -q",
        )
        out = capsys.readouterr().out
        assert out == "name: Maestro CI\n"

    def test_ci_output_flag_writes_config_and_passes_custom_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        output_file = tmp_path / ".gitlab-ci.yml"
        mock_generate_ci_yaml = MagicMock(return_value="stages:\n  - test\n")
        mock_write_ci_yaml = MagicMock(return_value=output_file)

        monkeypatch.setattr("maestro_cli.cli.generate_ci_yaml", mock_generate_ci_yaml)
        monkeypatch.setattr("maestro_cli.cli.write_ci_yaml", mock_write_ci_yaml)

        rc = main([
            "ci",
            str(plan_file),
            "--provider",
            "gitlab_ci",
            "--workflow-name",
            "Nightly Maestro",
            "--python-version",
            "3.12",
            "--test-command",
            "pytest tests/test_cli.py -q",
            "-o",
            str(output_file),
        ])

        assert rc == 0
        mock_generate_ci_yaml.assert_called_once_with(
            str(plan_file),
            provider="gitlab_ci",
            workflow_name="Nightly Maestro",
            python_version="3.12",
            test_command="pytest tests/test_cli.py -q",
        )
        mock_write_ci_yaml.assert_called_once_with("stages:\n  - test\n", str(output_file))
        out = capsys.readouterr().out
        assert out == f"[maestro] CI config written to {output_file}\n"


class TestMainDiff:
    def test_diff_json_formats_regressions_and_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        diff_result = SimpleNamespace(regressions=["task-1"])
        mock_diff_runs = MagicMock(return_value=diff_result)
        mock_format_diff_json = MagicMock(return_value='{"regressions":["task-1"]}')

        monkeypatch.setattr("maestro_cli.diff.diff_runs", mock_diff_runs)
        monkeypatch.setattr("maestro_cli.diff.format_diff_json", mock_format_diff_json)

        rc = main(["diff", str(run_a), str(run_b), "--json"])

        assert rc == 1
        mock_diff_runs.assert_called_once_with(run_a.resolve(), run_b.resolve())
        mock_format_diff_json.assert_called_once_with(diff_result)
        out = capsys.readouterr().out
        assert out == '{"regressions":["task-1"]}\n'

    def test_diff_missing_run_dir_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_a.mkdir()
        missing_run = tmp_path / "missing-run"
        mock_diff_runs = MagicMock()

        monkeypatch.setattr("maestro_cli.diff.diff_runs", mock_diff_runs)

        rc = main(["diff", str(run_a), str(missing_run)])

        assert rc == 1
        mock_diff_runs.assert_not_called()
        out = capsys.readouterr().out
        assert f"[maestro] error: run path is not a directory: {missing_run.resolve()}" in out


# ===========================================================================
# Tests for main() — run subcommand (dry-run, mocked)
# ===========================================================================

class TestMainRun:
    def _make_mock_result(self, success: bool = True) -> MagicMock:
        """Create a mock PlanRunResult with the given success value."""
        mock_result = MagicMock()
        mock_result.success = success
        return mock_result

    def test_dry_run_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--dry-run"])
        assert rc == 0

    def test_dry_run_calls_run_plan_with_dry_run_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--dry-run"])
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["dry_run"] is True

    def test_run_passes_execution_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--execution-profile", "yolo"])
        _, kwargs = mock_run.call_args
        assert kwargs["execution_profile"] == "yolo"

    def test_run_passes_max_parallel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--max-parallel", "8"])
        _, kwargs = mock_run.call_args
        assert kwargs["max_parallel_override"] == 8

    def test_run_passes_only_as_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--only", "t1,t2"])
        _, kwargs = mock_run.call_args
        assert kwargs["only"] == {"t1", "t2"}

    def test_run_passes_skip_as_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--skip", "t2"])
        _, kwargs = mock_run.call_args
        assert kwargs["skip"] == {"t2"}

    def test_run_empty_only_passed_as_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file)])
        _, kwargs = mock_run.call_args
        assert kwargs["only"] is None
        assert kwargs["skip"] is None

    def test_run_passes_run_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--run-dir", "/custom/dir"])
        _, kwargs = mock_run.call_args
        assert kwargs["run_dir_override"] == "/custom/dir"

    def test_run_passes_webhook_cli_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--webhook", "https://example.com/notify"])
        _, kwargs = mock_run.call_args
        assert kwargs["webhook_url"] == "https://example.com/notify"

    def test_run_uses_plan_webhook_when_no_cli_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _PLAN_WITH_WEBHOOK_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file)])
        _, kwargs = mock_run.call_args
        assert kwargs["webhook_url"] == "https://example.com/from-plan"

    def test_run_failed_result_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=False))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file)])
        assert rc == 1

    def test_run_resume_path_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        # Create the resume directory so the path existence check passes
        resume_dir = tmp_path / "prior-run"
        resume_dir.mkdir()
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--resume", str(resume_dir)])
        _, kwargs = mock_run.call_args
        assert kwargs["resume_path"] == resume_dir.resolve()

    def test_run_resume_nonexistent_path_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--resume", "/nonexistent/path"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "resume path does not exist" in out
        mock_run.assert_not_called()

    def test_run_passes_tags_skip_tags_mask_secrets_and_no_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main([
            "run",
            str(plan_file),
            "--tags",
            " smoke,fast ",
            "--skip-tags",
            "slow , flaky",
            "--mask-secrets",
            "--auto-approve",
            "--no-cache",
        ])

        assert rc == 0
        plan = mock_run.call_args.args[0]
        _, kwargs = mock_run.call_args
        assert plan.secrets_auto is True
        assert kwargs["tags"] == {"smoke", "fast"}
        assert kwargs["skip_tags"] == {"slow", "flaky"}
        assert kwargs["auto_approve"] is True
        assert kwargs["cache_dir"] is None

    def test_run_resume_last_passes_latest_run_and_prints_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        latest_run = tmp_path / ".maestro-runs" / "latest"
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        mock_find_latest = MagicMock(return_value=latest_run)
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", mock_find_latest)

        rc = main(["run", str(plan_file), "--resume-last"])

        assert rc == 0
        mock_find_latest.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["resume_path"] == latest_run
        out = capsys.readouterr().out
        assert f"[maestro] resuming from: {latest_run}" in out

    def test_run_resume_last_without_prior_run_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        mock_find_latest = MagicMock(return_value=None)
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", mock_find_latest)

        rc = main(["run", str(plan_file), "--resume-last"])

        assert rc == 1
        mock_find_latest.assert_called_once()
        mock_run.assert_not_called()
        out = capsys.readouterr().out
        assert "[maestro] error: no prior runs found for plan 'test-plan'" in out

    def test_run_missing_plan_returns_one(self, tmp_path: Path) -> None:
        missing = tmp_path / "noplan.yaml"
        rc = main(["run", str(missing)])
        assert rc == 1

    def test_run_multiple_plans_dispatches_to_run_multi_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_a = _write_yaml(tmp_path, _VALID_PLAN_YAML, "plan-a.yaml")
        plan_b = _write_yaml(tmp_path, _VALID_PLAN_YAML, "plan-b.yaml")
        mock_multi = MagicMock(return_value=self._make_mock_result(success=True))
        mock_load_plan = MagicMock()
        monkeypatch.setattr("maestro_cli.cli.run_multi_plan", mock_multi)
        monkeypatch.setattr("maestro_cli.cli.load_plan", mock_load_plan)

        rc = main([
            "run",
            str(plan_a),
            str(plan_b),
            "--parallel",
            "--mode",
            "safe",
            "--dry-run",
            "--auto-approve",
            "-q",
            "--output",
            "jsonl",
        ])

        assert rc == 0
        mock_multi.assert_called_once_with(
            [str(plan_a), str(plan_b)],
            parallel=True,
            execution_profile="safe",
            dry_run=True,
            verbosity="quiet",
            output_mode="jsonl",
            auto_approve=True,
        )
        mock_load_plan.assert_not_called()

    def test_run_multiple_plans_tui_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_a = _write_yaml(tmp_path, _VALID_PLAN_YAML, "plan-a.yaml")
        plan_b = _write_yaml(tmp_path, _VALID_PLAN_YAML, "plan-b.yaml")
        mock_multi = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_multi_plan", mock_multi)

        rc = main(["run", str(plan_a), str(plan_b), "--output", "tui"])

        assert rc == 1
        mock_multi.assert_not_called()
        out = capsys.readouterr().out
        assert "TUI mode does not support multi-plan execution yet" in out

    def test_run_live_missing_dependencies_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)
        monkeypatch.setitem(sys.modules, "maestro_cli.live", ModuleType("maestro_cli.live"))

        rc = main(["run", str(plan_file), "--output", "live"])

        assert rc == 1
        mock_run.assert_not_called()
        out = capsys.readouterr().out
        assert "live output dependencies not installed" in out

    def test_run_live_passes_quiet_text_and_event_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        event_callback = MagicMock()

        class _LiveContext:
            def __enter__(self) -> "_LiveContext":
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

        mock_create_live_callback = MagicMock(return_value=(_LiveContext(), event_callback))
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.live.create_live_callback", mock_create_live_callback)
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main([
            "run",
            str(plan_file),
            "--output",
            "live",
            "--max-parallel",
            "2",
            "--dry-run",
            "--auto-approve",
            "-v",
        ])

        assert rc == 0
        plan = mock_create_live_callback.call_args.args[0]
        assert plan.name == "test-plan"
        passed_plan = mock_run.call_args.args[0]
        assert passed_plan == plan
        kwargs = mock_run.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["execution_profile"] == "plan"
        assert kwargs["max_parallel_override"] == 2
        assert kwargs["verbosity"] == "quiet"
        assert kwargs["output_mode"] == "text"
        assert kwargs["auto_approve"] is True
        assert kwargs["event_callback"] is event_callback

    def test_run_tui_missing_dependencies_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)
        monkeypatch.setitem(sys.modules, "maestro_cli.tui", ModuleType("maestro_cli.tui"))

        rc = main(["run", str(plan_file), "--output", "tui"])

        assert rc == 1
        mock_run.assert_not_called()
        out = capsys.readouterr().out
        assert "TUI dependencies not installed" in out

    def test_run_tui_passes_normalized_args_to_app(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        resume_dir = tmp_path / "prior-run"
        cache_dir = tmp_path / "cache-dir"
        resume_dir.mkdir()
        captured: dict[str, object] = {}

        class _FakeApp:
            def __init__(self, plan: object, **kwargs: object) -> None:
                captured["plan"] = plan
                captured["kwargs"] = kwargs
                captured["app"] = self
                self.ran = False
                self._result = SimpleNamespace(success=True)

            def run(self) -> None:
                self.ran = True

        fake_tui_module = ModuleType("maestro_cli.tui")
        fake_tui_module.MaestroApp = _FakeApp
        monkeypatch.setitem(sys.modules, "maestro_cli.tui", fake_tui_module)

        rc = main([
            "run",
            str(plan_file),
            "--output",
            "tui",
            "--mode",
            "safe",
            "--max-parallel",
            "4",
            "--dry-run",
            "--auto-approve",
            "--run-dir",
            "custom-runs",
            "--resume",
            str(resume_dir),
            "--cache-dir",
            str(cache_dir),
            "--only",
            "t1,t2",
            "--skip",
            "t3",
            "--tags",
            " smoke,fast ",
            "--skip-tags",
            "slow , flaky",
            "--webhook",
            "https://example.com/notify",
        ])

        assert rc == 0
        plan = captured["plan"]
        assert getattr(plan, "name") == "test-plan"
        app = captured["app"]
        assert getattr(app, "ran") is True
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["dry_run"] is True
        assert kwargs["execution_profile"] == "safe"
        assert kwargs["max_parallel_override"] == 4
        assert kwargs["run_dir_override"] == "custom-runs"
        assert kwargs["auto_approve"] is True
        assert kwargs["resume_path"] == resume_dir.resolve()
        assert kwargs["cache_dir"] == cache_dir.resolve() / ".cache"
        assert kwargs["only"] == {"t1", "t2"}
        assert kwargs["skip"] == {"t3"}
        assert kwargs["tags"] == {"smoke", "fast"}
        assert kwargs["skip_tags"] == {"slow", "flaky"}
        assert kwargs["webhook_url"] == "https://example.com/notify"


# ===========================================================================
# Tests for main() — watch, replan, and doctor subcommands
# ===========================================================================

class TestMainWatch:
    def test_watch_live_missing_dependencies_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock()
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)
        monkeypatch.setitem(sys.modules, "maestro_cli.live", ModuleType("maestro_cli.live"))

        rc = main(["watch", str(plan_file), "--output", "live"])

        assert rc == 1
        mock_watch.assert_not_called()
        out = capsys.readouterr().out
        assert "live output dependencies not installed" in out

    def test_watch_tui_not_supported_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock()
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)

        rc = main(["watch", str(plan_file), "--output", "tui"])

        assert rc == 1
        mock_watch.assert_not_called()
        out = capsys.readouterr().out
        assert "not yet supported for watch" in out

    def test_watch_jsonl_passes_normalized_args(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured["plan_path"] = plan_path
            captured["kwargs"] = kwargs
            event_callback = kwargs["event_callback"]
            assert callable(event_callback)
            event_callback("iteration.started", {"iteration": 1, "score": 0.5})
            return SimpleNamespace(status="improved")

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main([
            "watch",
            str(plan_file),
            "--dry-run",
            "--execution-profile",
            "safe",
            "--max-parallel",
            "3",
            "--auto-approve",
            "--output",
            "jsonl",
            "-v",
            "-q",
        ])

        assert rc == 0
        assert captured["plan_path"] == str(plan_file)

        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["max_parallel_override"] == 3
        assert kwargs["dry_run"] is True
        assert kwargs["execution_profile"] == "safe"
        assert kwargs["verbosity"] == "verbose"
        assert kwargs["output_mode"] == "jsonl"
        assert kwargs["auto_approve"] is True
        out = capsys.readouterr().out
        assert json.loads(out) == {"event": "iteration.started", "iteration": 1, "score": 0.5}
        assert "watch complete" not in out

    def test_watch_text_quiet_prints_summary_and_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured["plan_path"] = plan_path
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                status="plateau",
                total_iterations=4,
                best_metric=100.0,
                best_iteration=2,
                total_cost_usd=0.0,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "-q"])

        assert rc == 0
        assert captured["plan_path"] == str(plan_file)
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["verbosity"] == "quiet"
        assert kwargs["output_mode"] == "text"
        assert kwargs["event_callback"] is None
        out = capsys.readouterr().out
        assert "[maestro] watch complete: plateau" in out
        assert "[maestro]   iterations: 4" in out
        assert "[maestro]   best metric: 100.0 (iteration 2)" in out

    def test_watch_failed_status_returns_one_and_prints_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured["plan_path"] = plan_path
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                status="failed",
                total_iterations=2,
                best_metric=None,
                best_iteration=None,
                total_cost_usd=1.25,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file)])

        assert rc == 1
        assert captured["plan_path"] == str(plan_file)
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["verbosity"] == "normal"
        assert kwargs["output_mode"] == "text"
        out = capsys.readouterr().out
        assert "[maestro] watch complete: failed" in out
        assert "[maestro]   iterations: 2" in out
        assert "[maestro]   total cost: $1.25" in out

    def test_watch_none_state_returns_one_without_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock(return_value=None)
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)

        rc = main(["watch", str(plan_file)])

        assert rc == 1
        mock_watch.assert_called_once()
        out = capsys.readouterr().out
        assert "watch complete" not in out

    def test_watch_live_passes_quiet_text_and_event_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}
        event_callback = MagicMock()

        class _LiveContext:
            def __enter__(self) -> "_LiveContext":
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

        mock_create_live_callback = MagicMock(return_value=(_LiveContext(), event_callback))

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured["plan_path"] = plan_path
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                status="max_iterations",
                total_iterations=3,
                best_metric=None,
                best_iteration=None,
                total_cost_usd=0.0,
            )

        monkeypatch.setattr("maestro_cli.live.create_live_callback", mock_create_live_callback)
        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main([
            "watch",
            str(plan_file),
            "--output",
            "live",
            "--max-parallel",
            "2",
            "--dry-run",
            "--auto-approve",
            "-v",
        ])

        assert rc == 0
        assert captured["plan_path"] == str(plan_file)
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["max_parallel_override"] == 2
        assert kwargs["dry_run"] is True
        assert kwargs["execution_profile"] == "plan"
        assert kwargs["verbosity"] == "quiet"
        assert kwargs["output_mode"] == "text"
        assert kwargs["auto_approve"] is True
        assert kwargs["event_callback"] is event_callback
        plan = mock_create_live_callback.call_args.args[0]
        assert plan.name == "test-plan"
        out = capsys.readouterr().out
        assert "[maestro] watch complete: max_iterations" in out
        assert "[maestro]   iterations: 3" in out

    def test_watch_tui_not_supported_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """TUI watch is not yet implemented — verify clear error message."""
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock()
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)

        rc = main([
            "watch",
            str(plan_file),
            "--output",
            "tui",
            "--max-parallel",
            "4",
            "--dry-run",
            "--auto-approve",
        ])

        assert rc == 1
        mock_watch.assert_not_called()
        out = capsys.readouterr().out
        assert "not yet supported for watch" in out
        assert "--output text" in out

    def test_watch_tui_suggests_alternatives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """TUI watch error message suggests valid alternatives."""
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock()
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)

        rc = main(["watch", str(plan_file), "--output", "tui"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "--output jsonl" in out
        assert "--output live" in out


class TestMainReplan:
    def test_replan_defaults_to_normal_text_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file)])

        assert rc == 0
        mock_replan.assert_called_once_with(
            str(plan_file),
            max_attempts=3,
            analysis_model="opus",
            dry_run=False,
            execution_profile="plan",
            verbosity="normal",
            output_mode="text",
            auto_approve=False,
            extra_template_vars=None,
            variants=1,
            debug_prob=0.5,
            selection_policy="debug_prob",
            exploration_constant=pytest.approx(1.41421356237),
            population_strategy="best",
            tournament_size=2,
            elite_count=1,
            diversity_floor=pytest.approx(0.25),
        )

    def test_replan_passes_flags_and_failure_rc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=False))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main([
            "replan",
            str(plan_file),
            "--max-attempts",
            "5",
            "--model",
            "sonnet",
            "--mode",
            "yolo",
            "--dry-run",
            "--auto-approve",
            "-q",
            "--output",
            "jsonl",
        ])

        assert rc == 1
        mock_replan.assert_called_once_with(
            str(plan_file),
            max_attempts=5,
            analysis_model="sonnet",
            dry_run=True,
            execution_profile="yolo",
            verbosity="quiet",
            output_mode="jsonl",
            auto_approve=True,
            extra_template_vars=None,
            variants=1,
            debug_prob=0.5,
            selection_policy="debug_prob",
            exploration_constant=pytest.approx(1.41421356237),
            population_strategy="best",
            tournament_size=2,
            elite_count=1,
            diversity_floor=pytest.approx(0.25),
        )

    def test_replan_tui_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--output", "tui"])

        assert rc == 1
        mock_replan.assert_not_called()
        out = capsys.readouterr().out
        assert "TUI mode is not supported for replan yet" in out

    def test_replan_live_missing_dependencies_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)
        monkeypatch.setitem(sys.modules, "maestro_cli.live", ModuleType("maestro_cli.live"))

        rc = main(["replan", str(plan_file), "--output", "live"])

        assert rc == 1
        mock_replan.assert_not_called()
        out = capsys.readouterr().out
        assert "live output dependencies not installed" in out

    def test_replan_live_passes_quiet_text_and_event_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        event_callback = MagicMock()

        class _LiveContext:
            def __enter__(self) -> "_LiveContext":
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

        mock_create_live_callback = MagicMock(return_value=(_LiveContext(), event_callback))
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.live.create_live_callback", mock_create_live_callback)
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main([
            "replan",
            str(plan_file),
            "--max-attempts",
            "4",
            "--model",
            "sonnet",
            "--execution-profile",
            "safe",
            "--dry-run",
            "--auto-approve",
            "--output",
            "live",
            "-v",
        ])

        assert rc == 0
        plan = mock_create_live_callback.call_args.args[0]
        assert plan.name == "test-plan"
        mock_replan.assert_called_once_with(
            str(plan_file),
            max_attempts=4,
            analysis_model="sonnet",
            dry_run=True,
            execution_profile="safe",
            verbosity="quiet",
            output_mode="text",
            auto_approve=True,
            event_callback=event_callback,
            extra_template_vars=None,
            variants=1,
            debug_prob=0.5,
            selection_policy="debug_prob",
            exploration_constant=pytest.approx(1.41421356237),
            population_strategy="best",
            tournament_size=2,
            elite_count=1,
            diversity_floor=pytest.approx(0.25),
        )


class TestMainDoctor:
    def test_doctor_passes_flags_and_returns_one_on_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_doctor = MagicMock(return_value=[
            ("python_version", "Python 3.11", "ok"),
            ("codex", "missing", "fail"),
        ])
        monkeypatch.setattr("maestro_cli.doctor.run_doctor", mock_run_doctor)

        rc = main(["doctor", "--json", "--run-dir", str(tmp_path)])

        assert rc == 1
        mock_run_doctor.assert_called_once_with(run_dir=str(tmp_path), json_output=True, full=False)

    def test_doctor_defaults_return_zero_when_all_checks_pass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_doctor = MagicMock(return_value=[
            ("python_version", "Python 3.11", "ok"),
            ("codex", "installed", "ok"),
        ])
        monkeypatch.setattr("maestro_cli.doctor.run_doctor", mock_run_doctor)

        rc = main(["doctor"])

        assert rc == 0
        mock_run_doctor.assert_called_once_with(run_dir=".maestro-runs", json_output=False, full=False)


class TestMainStatus:
    def test_status_text_uses_default_run_and_cache_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        status_result = SimpleNamespace(summary="ok")
        mock_find_latest_run = MagicMock(return_value=None)
        mock_plan_status = MagicMock(return_value=status_result)
        mock_format_status = MagicMock(return_value="status ok")

        monkeypatch.setattr("maestro_cli.cli._find_latest_run", mock_find_latest_run)
        monkeypatch.setattr("maestro_cli.status.plan_status", mock_plan_status)
        monkeypatch.setattr("maestro_cli.status.format_status", mock_format_status)

        rc = main(["status", str(plan_file)])

        assert rc == 0
        mock_find_latest_run.assert_called_once()
        plan = mock_find_latest_run.call_args.args[0]
        assert plan.name == "test-plan"
        assert mock_find_latest_run.call_args.kwargs == {"run_dir": None}
        passed_plan, passed_latest_run, passed_cache_dir = mock_plan_status.call_args.args
        assert passed_plan == plan
        assert passed_latest_run is None
        assert passed_cache_dir == (plan.source_dir / plan.run_dir).resolve() / ".cache"
        mock_format_status.assert_called_once_with(status_result)
        out = capsys.readouterr().out
        assert out == "status ok\n"

    def test_status_json_passes_latest_run_and_resolved_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        latest_run = tmp_path / ".maestro-runs" / "latest"
        cache_dir = tmp_path / "cache-dir"
        status_result = SimpleNamespace(summary="ok")
        mock_plan_status = MagicMock(return_value=status_result)
        mock_format_status_json = MagicMock(return_value='{"status":"ok"}')

        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=latest_run))
        monkeypatch.setattr("maestro_cli.status.plan_status", mock_plan_status)
        monkeypatch.setattr("maestro_cli.status.format_status_json", mock_format_status_json)

        rc = main([
            "status",
            str(plan_file),
            "--cache-dir",
            str(cache_dir),
            "--run-dir",
            "custom-runs",
            "--json",
        ])

        assert rc == 0
        plan, passed_latest_run, passed_cache_dir = mock_plan_status.call_args.args
        assert plan.name == "test-plan"
        assert passed_latest_run == latest_run
        assert passed_cache_dir == cache_dir.resolve()
        out = capsys.readouterr().out
        assert out == '{"status":"ok"}\n'


class TestMainExplain:
    def test_explain_text_uses_default_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        explanation = SimpleNamespace(summary="ok")
        mock_explain_plan = MagicMock(return_value=explanation)
        mock_format_explain = MagicMock(return_value="explain ok")

        monkeypatch.setattr("maestro_cli.explain.explain_plan", mock_explain_plan)
        monkeypatch.setattr("maestro_cli.explain.format_explain", mock_format_explain)

        rc = main(["explain", str(plan_file)])

        assert rc == 0
        plan, passed_cache_dir = mock_explain_plan.call_args.args
        assert plan.name == "test-plan"
        assert passed_cache_dir == (plan.source_dir / plan.run_dir).resolve() / ".cache"
        mock_format_explain.assert_called_once_with(explanation)
        out = capsys.readouterr().out
        assert out == "explain ok\n"

    def test_explain_json_passes_resolved_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        cache_dir = tmp_path / "cache-dir"
        explanation = SimpleNamespace(summary="ok")
        mock_explain_plan = MagicMock(return_value=explanation)
        mock_format_explain_json = MagicMock(return_value='{"explain":"ok"}')

        monkeypatch.setattr("maestro_cli.explain.explain_plan", mock_explain_plan)
        monkeypatch.setattr("maestro_cli.explain.format_explain_json", mock_format_explain_json)

        rc = main(["explain", str(plan_file), "--cache-dir", str(cache_dir), "--json"])

        assert rc == 0
        plan, passed_cache_dir = mock_explain_plan.call_args.args
        assert plan.name == "test-plan"
        assert passed_cache_dir == cache_dir.resolve()
        mock_format_explain_json.assert_called_once_with(explanation)
        out = capsys.readouterr().out
        assert out == '{"explain":"ok"}\n'


class TestMainEval:
    def test_eval_text_success_returns_zero_and_formats_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        run_dir = tmp_path / "completed-run"
        run_dir.mkdir()
        suite = SimpleNamespace(overall_pass=True)
        mock_run_eval = MagicMock(return_value=suite)
        mock_format_eval = MagicMock(return_value="eval ok")

        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)
        monkeypatch.setattr("maestro_cli.eval.format_eval", mock_format_eval)

        rc = main(["eval", str(eval_yaml), str(run_dir)])

        assert rc == 0
        mock_run_eval.assert_called_once_with(eval_yaml, run_dir.resolve())
        mock_format_eval.assert_called_once_with(suite)
        out = capsys.readouterr().out
        assert out == "eval ok\n"

    def test_eval_json_failure_returns_one_and_formats_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        run_dir = tmp_path / "completed-run"
        run_dir.mkdir()
        suite = SimpleNamespace(overall_pass=False)
        mock_run_eval = MagicMock(return_value=suite)
        mock_format_eval_json = MagicMock(return_value='{"overall_pass":false}')

        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)
        monkeypatch.setattr("maestro_cli.eval.format_eval_json", mock_format_eval_json)

        rc = main(["eval", str(eval_yaml), str(run_dir), "--json"])

        assert rc == 1
        mock_run_eval.assert_called_once_with(eval_yaml, run_dir.resolve())
        mock_format_eval_json.assert_called_once_with(suite)
        out = capsys.readouterr().out
        assert out == '{"overall_pass":false}\n'

    def test_eval_missing_run_dir_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        missing_run_dir = tmp_path / "missing-run"
        mock_run_eval = MagicMock()

        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)

        rc = main(["eval", str(eval_yaml), str(missing_run_dir)])

        assert rc == 1
        mock_run_eval.assert_not_called()
        out = capsys.readouterr().out
        assert f"[maestro] error: run path is not a directory: {missing_run_dir.resolve()}" in out


class TestMainShell:
    def test_shell_passes_plan_path_and_returns_handler_rc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run_shell = MagicMock(return_value=7)
        monkeypatch.setattr("maestro_cli.shell.run_shell", mock_run_shell)

        rc = main(["shell", "--plan", str(plan_file)])

        assert rc == 7
        mock_run_shell.assert_called_once_with(plan_file)

    def test_shell_without_plan_passes_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_shell = MagicMock(return_value=0)
        monkeypatch.setattr("maestro_cli.shell.run_shell", mock_run_shell)

        rc = main(["shell"])

        assert rc == 0
        mock_run_shell.assert_called_once_with(None)


class TestMainSuggest:
    def test_suggest_json_passes_run_dir_and_min_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        run_dir = tmp_path / "custom-runs"
        suggestions = SimpleNamespace(items=["cache"])
        mock_suggest_plan = MagicMock(return_value=suggestions)
        mock_format_suggestions_json = MagicMock(return_value='{"suggestions":["cache"]}')

        monkeypatch.setattr("maestro_cli.suggest.suggest_plan", mock_suggest_plan)
        monkeypatch.setattr("maestro_cli.suggest.format_suggestions_json", mock_format_suggestions_json)

        rc = main([
            "suggest",
            str(plan_file),
            "--run-dir",
            str(run_dir),
            "--min-runs",
            "5",
            "--json",
        ])

        assert rc == 0
        plan, passed_run_dir = mock_suggest_plan.call_args.args[:2]
        assert plan.name == "test-plan"
        assert passed_run_dir == run_dir
        assert mock_suggest_plan.call_args.kwargs["min_runs"] == 5
        mock_format_suggestions_json.assert_called_once_with(suggestions)
        out = capsys.readouterr().out
        assert out == '{"suggestions":["cache"]}\n'

    def test_suggest_text_uses_plan_run_dir_and_default_min_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        suggestions = SimpleNamespace(items=["cache"])
        mock_suggest_plan = MagicMock(return_value=suggestions)
        mock_format_suggestions = MagicMock(return_value="cache suggestion")

        monkeypatch.setattr("maestro_cli.suggest.suggest_plan", mock_suggest_plan)
        monkeypatch.setattr("maestro_cli.suggest.format_suggestions", mock_format_suggestions)

        rc = main(["suggest", str(plan_file)])

        assert rc == 0
        plan, passed_run_dir = mock_suggest_plan.call_args.args[:2]
        assert plan.name == "test-plan"
        assert passed_run_dir == Path(plan.run_dir)
        assert mock_suggest_plan.call_args.kwargs["min_runs"] == 3
        mock_format_suggestions.assert_called_once_with(suggestions)
        out = capsys.readouterr().out
        assert out == "cache suggestion\n"


# ===========================================================================
# Tests for main() — cleanup subcommand
# ===========================================================================

class TestMainCleanup:
    def test_cleanup_passes_flags_and_prints_dry_run_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        old_run = run_root / "20260308-120000_test-plan"
        new_run = run_root / "20260308-130000_test-plan"
        old_run.mkdir()
        new_run.mkdir()
        mock_cleanup_runs = MagicMock(return_value=[old_run, new_run])

        monkeypatch.setattr("maestro_cli.cli.cleanup_runs", mock_cleanup_runs)

        rc = main([
            "cleanup",
            str(plan_file),
            "--keep",
            "2",
            "--older-than",
            "30",
            "--dry-run",
        ])

        assert rc == 0
        mock_cleanup_runs.assert_called_once_with(
            run_root.resolve(),
            keep=2,
            older_than_days=30,
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert f"[maestro] would delete: {old_run.name}" in out
        assert f"[maestro] would delete: {new_run.name}" in out
        assert "[maestro] would delete 2 run(s), kept 2" in out

    def test_cleanup_no_run_dir_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cleanup on a plan whose run dir does not exist should exit cleanly."""
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        rc = main(["cleanup", str(plan_file)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no run directory found" in out

    def test_cleanup_missing_plan_returns_one(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.yaml"
        rc = main(["cleanup", str(missing)])
        assert rc == 1


# ===========================================================================
# Tests for main() — backfill-costs subcommand
# ===========================================================================

class TestMainBackfillCosts:
    def test_backfill_costs_dry_run_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["backfill-costs", "--root", str(tmp_path), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "backfill-costs mode=dry-run" in out

    def test_backfill_costs_missing_pricing_file_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "pricing.json"
        rc = main(["backfill-costs", "--codex-pricing-file", str(missing)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "pricing file does not exist" in out


# ===========================================================================
# Tests for main() — ui subcommand
# ===========================================================================

class TestMainUi:
    def test_ui_passes_deduped_project_roots_to_create_app(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        extra_a = tmp_path / "repo-a"
        extra_b = tmp_path / "repo-b"
        extra_a.mkdir()
        extra_b.mkdir()

        captured: dict[str, object] = {}

        def _fake_create_app(*, project_root: object = None, project_roots: object = None) -> object:
            captured["project_root"] = project_root
            captured["project_roots"] = project_roots
            return object()

        def _fake_uvicorn_run(app: object, **kwargs: object) -> None:
            captured["app"] = app
            captured["uvicorn_kwargs"] = kwargs

        monkeypatch.setattr("maestro_cli.web.create_app", _fake_create_app)
        monkeypatch.setattr("uvicorn.run", _fake_uvicorn_run)

        rc = main([
            "ui",
            "--no-browser",
            "--project-root", str(extra_a),
            "--project-root", str(extra_b),
            "--project-root", str(extra_a),
        ])
        assert rc == 0
        assert captured["project_root"] is None

        roots = captured["project_roots"]
        assert isinstance(roots, list)
        assert len(roots) == 3  # cwd + 2 unique extras
        assert roots[1] == extra_a.resolve()
        assert roots[2] == extra_b.resolve()


# ===========================================================================
# Tests for main() — report subcommand
# ===========================================================================

class TestMainReport:
    def _write_manifest(self, run_dir: Path) -> None:
        manifest = {
            "plan_name": "report-plan",
            "run_id": "rpt-001",
            "started_at": "2026-03-01T18:00:00+00:00",
            "finished_at": "2026-03-01T18:02:00+00:00",
            "success": True,
            "task_results": {
                "t1": {
                    "status": "success",
                    "duration_sec": 45.0,
                    "cost_usd": 0.11,
                    "token_usage": {"total_tokens": 1234},
                    "command": "codex -m gpt-5.3-codex",
                    "stdout_tail": "done",
                    "started_at": "2026-03-01T18:00:00+00:00",
                    "finished_at": "2026-03-01T18:00:45+00:00",
                },
                "t2": {
                    "status": "failed",
                    "duration_sec": 20.0,
                    "cost_usd": 0.05,
                    "token_usage": {"total_tokens": 456},
                    "command": "echo fail",
                    "stdout_tail": "error",
                    "started_at": "2026-03-01T18:01:00+00:00",
                    "finished_at": "2026-03-01T18:01:20+00:00",
                },
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def test_report_default_output_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run_dir = tmp_path / ".maestro-runs" / "rpt-001_report-plan"
        run_dir.mkdir(parents=True)
        self._write_manifest(run_dir)

        rc = main(["report", str(run_dir)])
        assert rc == 0

        report_path = run_dir / "report.html"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Task Table" in content
        assert "report-plan" in content

        out = capsys.readouterr().out
        assert "report written to" in out

    def test_report_custom_output_path(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        run_dir.mkdir()
        self._write_manifest(run_dir)

        out_path = tmp_path / "exports" / "my-report.html"
        rc = main(["report", str(run_dir), "-o", str(out_path)])
        assert rc == 0
        assert out_path.exists()

    def test_report_non_directory_path_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        not_dir = tmp_path / "nope"
        rc = main(["report", str(not_dir)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "run path is not a directory" in out

    def test_report_missing_manifest_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run-without-manifest"
        run_dir.mkdir()

        rc = main(["report", str(run_dir)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "run_manifest.json not found" in out


# ===========================================================================
# Tests for main() — error handling
# ===========================================================================

class TestMainErrorHandling:
    def test_exception_from_load_plan_caught(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _INVALID_YAML)
        rc = main(["validate", str(plan_file)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error:" in out

    def test_no_subcommand_shows_banner_and_exits_zero(self) -> None:
        """No args should show the banner + command list and return 0."""
        assert main([]) == 0

    def test_top_level_help_prints_banner_before_argparse_exit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_banner = MagicMock()
        monkeypatch.setattr("maestro_cli.cli._print_banner", mock_banner)

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        mock_banner.assert_called_once_with()
        out = capsys.readouterr().out
        assert "usage: maestro" in out

    def test_subcommand_help_does_not_print_banner(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_banner = MagicMock()
        monkeypatch.setattr("maestro_cli.cli._print_banner", mock_banner)

        with pytest.raises(SystemExit) as exc_info:
            main(["watch", "--help"])

        assert exc_info.value.code == 0
        mock_banner.assert_not_called()
        out = capsys.readouterr().out
        assert "usage: maestro watch" in out

    def test_unknown_subcommand_exits_with_error(self) -> None:
        """Argparse should exit with code 2 for an unrecognised subcommand."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent"])
        assert exc_info.value.code == 2

    def test_generic_exception_caught_and_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Any unhandled exception inside a command should be caught by main()."""
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("kaboom")

        monkeypatch.setattr("maestro_cli.cli.load_plan", _boom)
        rc = main(["validate", str(plan_file)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error: kaboom" in out

    def test_keyboard_interrupt_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """KeyboardInterrupt is not an Exception subclass and should propagate."""
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        def _interrupt(*_args: object, **_kwargs: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("maestro_cli.cli.load_plan", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            main(["validate", str(plan_file)])


# ===========================================================================
# Tests for main() — integration round-trip
# ===========================================================================

class TestMainIntegration:
    def test_scaffold_then_validate(self, tmp_path: Path) -> None:
        """Scaffold a plan, then validate the output file."""
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        output_file = tmp_path / "roundtrip.yaml"

        rc1 = main(["scaffold", str(brief_file), "-o", str(output_file)])
        assert rc1 == 0
        assert output_file.exists()

        rc2 = main(["validate", str(output_file)])
        assert rc2 == 0

    def test_scaffold_validate_flag_catches_invalid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The --validate flag on scaffold should successfully validate a well-formed brief."""
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        rc = main(["scaffold", str(brief_file), "--validate"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "generated plan is valid" in out


# ===========================================================================
# Additional targeted tests — parser gaps and flag propagation
# ===========================================================================

class TestEvalVerboseFlag:
    """eval --verbose is accepted by the parser but not forwarded to run_eval."""

    def test_eval_verbose_flag_accepted_and_dispatches_normally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        run_dir = tmp_path / "completed-run"
        run_dir.mkdir()
        suite = SimpleNamespace(overall_pass=True)
        mock_run_eval = MagicMock(return_value=suite)
        mock_format_eval = MagicMock(return_value="eval verbose ok")

        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)
        monkeypatch.setattr("maestro_cli.eval.format_eval", mock_format_eval)

        rc = main(["eval", str(eval_yaml), str(run_dir), "--verbose"])

        assert rc == 0
        # run_eval is still called once; --verbose has no effect on the call itself
        mock_run_eval.assert_called_once_with(eval_yaml, run_dir.resolve())
        out = capsys.readouterr().out
        assert out == "eval verbose ok\n"


class TestWatchParserFlags:
    """watch parser flags that aren't forwarded to watch() are still accepted."""

    def test_watch_mask_secrets_flag_accepted_by_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock(return_value=SimpleNamespace(
            status="improved",
            total_iterations=1,
            best_metric=None,
            total_cost_usd=None,
        ))
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)

        # --mask-secrets is defined in the parser; confirm it does not crash main()
        rc = main(["watch", str(plan_file), "--mask-secrets"])

        assert rc == 0
        mock_watch.assert_called_once()

    def test_watch_resume_last_flag_accepted_by_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_watch = MagicMock(return_value=SimpleNamespace(
            status="max_iterations",
            total_iterations=5,
            best_metric=None,
            total_cost_usd=None,
        ))
        monkeypatch.setattr("maestro_cli.watch.watch", mock_watch)

        # --resume-last is defined in the parser; confirm it does not crash main()
        rc = main(["watch", str(plan_file), "--resume-last"])

        assert rc == 0
        mock_watch.assert_called_once()


class TestReplanProfileModeAlias:
    """replan exposes --profile-mode and --mode as aliases for --execution-profile."""

    def test_replan_profile_mode_alias_forwards_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--profile-mode", "safe"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["execution_profile"] == "safe"


class TestMainDiffTextFormat:
    """diff subcommand — text (non-JSON) output path."""

    def test_diff_text_no_regressions_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        diff_result = SimpleNamespace(regressions=[])
        mock_diff_runs = MagicMock(return_value=diff_result)
        mock_format_diff = MagicMock(return_value="no regressions found")

        monkeypatch.setattr("maestro_cli.diff.diff_runs", mock_diff_runs)
        monkeypatch.setattr("maestro_cli.diff.format_diff", mock_format_diff)

        rc = main(["diff", str(run_a), str(run_b)])

        assert rc == 0
        mock_diff_runs.assert_called_once_with(run_a.resolve(), run_b.resolve())
        mock_format_diff.assert_called_once_with(diff_result)
        out = capsys.readouterr().out
        assert "no regressions found" in out

    def test_diff_runs_raises_value_error_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        monkeypatch.setattr(
            "maestro_cli.diff.diff_runs",
            MagicMock(side_effect=ValueError("manifest missing")),
        )

        rc = main(["diff", str(run_a), str(run_b)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error: manifest missing" in out


class TestWatchFlagPropagation:
    """watch subcommand — flag forwarding to watch()."""

    def test_watch_auto_approve_forwarded_to_watch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="max_iterations",
                total_iterations=1,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--auto-approve"])

        assert rc == 0
        assert captured.get("auto_approve") is True

    def test_watch_execution_profile_yolo_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="improved",
                total_iterations=1,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--execution-profile", "yolo"])

        assert rc == 0
        assert captured.get("execution_profile") == "yolo"

    def test_watch_dry_run_forwarded_to_watch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="max_iterations",
                total_iterations=2,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--dry-run"])

        assert rc == 0
        assert captured.get("dry_run") is True

    def test_watch_max_parallel_forwarded_to_watch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="improved",
                total_iterations=1,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--max-parallel", "4"])

        assert rc == 0
        assert captured.get("max_parallel_override") == 4

    def test_watch_plateau_status_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="plateau",
                total_iterations=5,
                best_metric=None,
                total_cost_usd=None,
            )),
        )

        rc = main(["watch", str(plan_file)])

        assert rc == 0


class TestReplanMaxAttempts:
    """replan --max-attempts forwarded to replan()."""

    def test_replan_max_attempts_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--max-attempts", "5"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["max_attempts"] == 5


class TestCiProviderAlias:
    """ci --provider short aliases ('github', 'gitlab') are forwarded as-is."""

    def test_ci_provider_github_alias_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_generate = MagicMock(return_value="# github ci\n")
        monkeypatch.setattr("maestro_cli.cli.generate_ci_yaml", mock_generate)

        rc = main(["ci", str(plan_file), "--provider", "github"])

        assert rc == 0
        mock_generate.assert_called_once_with(
            str(plan_file),
            provider="github",
            workflow_name="Maestro CI",
            python_version="3.11",
            test_command="python -m pytest -q",
        )
        assert "# github ci" in capsys.readouterr().out


class TestReplanVerboseFlag:
    """replan --verbose forwards verbosity='verbose' to replan()."""

    def test_replan_verbose_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--verbose"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["verbosity"] == "verbose"


class TestWatchVerboseTextMode:
    """watch --verbose in text mode forwards verbosity='verbose' to watch()."""

    def test_watch_verbose_text_mode_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="improved",
                total_iterations=1,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--verbose"])

        assert rc == 0
        assert captured.get("verbosity") == "verbose"
        assert captured.get("output_mode") == "text"


class TestWatchSummaryBestMetricAbsent:
    """watch summary omits 'best metric' line when best_metric is None."""

    def test_best_metric_none_not_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="max_iterations",
                total_iterations=3,
                best_metric=None,
                total_cost_usd=None,
            )),
        )

        rc = main(["watch", str(plan_file)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "[maestro] watch complete: max_iterations" in out
        assert "best metric" not in out
        assert "total cost" not in out


class TestDiffTextWithRegressions:
    """diff text format returns 1 when regressions are present."""

    def test_diff_text_regressions_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        diff_result = SimpleNamespace(regressions=["task-1", "task-2"])
        mock_diff_runs = MagicMock(return_value=diff_result)
        mock_format_diff = MagicMock(return_value="2 regressions found")

        monkeypatch.setattr("maestro_cli.diff.diff_runs", mock_diff_runs)
        monkeypatch.setattr("maestro_cli.diff.format_diff", mock_format_diff)

        rc = main(["diff", str(run_a), str(run_b)])

        assert rc == 1
        mock_diff_runs.assert_called_once_with(run_a.resolve(), run_b.resolve())
        mock_format_diff.assert_called_once_with(diff_result)
        out = capsys.readouterr().out
        assert "2 regressions found" in out


class TestDiffJsonNoRegressions:
    """diff --json returns 0 when there are no regressions."""

    def test_diff_json_no_regressions_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        diff_result = SimpleNamespace(regressions=[])
        mock_diff_runs = MagicMock(return_value=diff_result)
        mock_format_diff_json = MagicMock(return_value='{"regressions":[]}')

        monkeypatch.setattr("maestro_cli.diff.diff_runs", mock_diff_runs)
        monkeypatch.setattr("maestro_cli.diff.format_diff_json", mock_format_diff_json)

        rc = main(["diff", str(run_a), str(run_b), "--json"])

        assert rc == 0
        mock_diff_runs.assert_called_once_with(run_a.resolve(), run_b.resolve())
        mock_format_diff_json.assert_called_once_with(diff_result)
        out = capsys.readouterr().out
        assert out == '{"regressions":[]}\n'


class TestWatchJsonlFailedStatus:
    """watch --output jsonl with failed status suppresses summary and returns 1."""

    def test_watch_jsonl_failed_suppresses_summary_and_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="failed",
                total_iterations=2,
                best_metric=None,
                total_cost_usd=0.5,
            )),
        )

        rc = main(["watch", str(plan_file), "--output", "jsonl"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "watch complete" not in out
        assert "total cost" not in out


class TestExplainTextCustomCacheDir:
    """explain text mode with --cache-dir resolves the override path."""

    def test_explain_text_custom_cache_dir_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        cache_dir = tmp_path / "my-cache"
        explanation = SimpleNamespace(summary="ok")
        mock_explain_plan = MagicMock(return_value=explanation)
        mock_format_explain = MagicMock(return_value="explain custom")

        monkeypatch.setattr("maestro_cli.explain.explain_plan", mock_explain_plan)
        monkeypatch.setattr("maestro_cli.explain.format_explain", mock_format_explain)

        rc = main(["explain", str(plan_file), "--cache-dir", str(cache_dir)])

        assert rc == 0
        _plan, passed_cache_dir = mock_explain_plan.call_args.args
        assert passed_cache_dir == cache_dir.resolve()
        mock_format_explain.assert_called_once_with(explanation)
        out = capsys.readouterr().out
        assert out == "explain custom\n"


class TestEvalTextModeFailure:
    """eval text mode with a failing suite formats text output and returns 1."""

    def test_eval_text_failure_formats_text_and_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        suite = SimpleNamespace(overall_pass=False)
        mock_run_eval = MagicMock(return_value=suite)
        mock_format_eval = MagicMock(return_value="FAILED: criteria not met")
        mock_format_eval_json = MagicMock(return_value='{"pass": false}')

        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)
        monkeypatch.setattr("maestro_cli.eval.format_eval", mock_format_eval)
        monkeypatch.setattr("maestro_cli.eval.format_eval_json", mock_format_eval_json)

        rc = main(["eval", str(eval_yaml), str(run_dir)])

        assert rc == 1
        mock_format_eval.assert_called_once_with(suite)
        mock_format_eval_json.assert_not_called()
        out = capsys.readouterr().out
        assert out == "FAILED: criteria not met\n"


class TestDiffMissingFirstRunPath:
    """diff exits 1 with an error message when the first run path (run_a) is missing."""

    def test_diff_missing_run_a_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing_run_a = tmp_path / "missing-a"
        run_b = tmp_path / "run-b"
        run_b.mkdir()
        mock_diff_runs = MagicMock()

        monkeypatch.setattr("maestro_cli.diff.diff_runs", mock_diff_runs)

        rc = main(["diff", str(missing_run_a), str(run_b)])

        assert rc == 1
        mock_diff_runs.assert_not_called()
        out = capsys.readouterr().out
        assert f"[maestro] error: run path is not a directory: {missing_run_a.resolve()}" in out


class TestWatchZeroCostSuppressed:
    """watch summary omits cost line when total_cost_usd is 0.0 (falsy)."""

    def test_watch_zero_cost_not_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="improved",
                total_iterations=1,
                best_metric=None,
                total_cost_usd=0.0,
            )),
        )

        rc = main(["watch", str(plan_file)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "[maestro] watch complete: improved" in out
        assert "total cost" not in out


class TestDoctorJsonAllPass:
    """doctor --json with all checks passing returns 0."""

    def test_doctor_json_all_pass_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_doctor = MagicMock(return_value=[
            ("python_version", "Python 3.11", "ok"),
            ("pyyaml", "installed", "ok"),
            ("codex", "installed", "ok"),
        ])
        monkeypatch.setattr("maestro_cli.doctor.run_doctor", mock_run_doctor)

        rc = main(["doctor", "--json"])

        assert rc == 0
        mock_run_doctor.assert_called_once_with(run_dir=".maestro-runs", json_output=True, full=False)


_PLAN_WITH_APPROVAL_YAML = """\
version: 1
name: approval-plan
tasks:
  - id: t1
    command: "echo hi"
    requires_approval: true
"""


class TestRunCacheDirOverride:
    """run --cache-dir forwarded correctly to run_plan in normal (non-TUI, non-live) mode."""

    def test_run_cache_dir_override_forwarded_to_run_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        custom_cache = tmp_path / "my-cache"
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--cache-dir", str(custom_cache)])

        assert rc == 0
        _, kwargs = mock_run.call_args
        assert kwargs["cache_dir"] == custom_cache.resolve() / ".cache"


class TestRunNullResultReturnsOne:
    """run returns 1 when run_plan returns None (e.g. plan loading is aborted)."""

    def test_run_plan_none_result_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.cli.run_plan", MagicMock(return_value=None))

        rc = main(["run", str(plan_file)])

        assert rc == 1


class TestRunDryRunApprovalNotice:
    """run --dry-run prints the approval-gate notice when a task has requires_approval."""

    def test_dry_run_approval_task_prints_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _PLAN_WITH_APPROVAL_YAML)
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--dry-run"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Approval gates will be enforced interactively" in out


class TestCiGitlabAlias:
    """ci --provider gitlab alias is forwarded as-is to generate_ci_yaml."""

    def test_ci_provider_gitlab_alias_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_generate = MagicMock(return_value="# gitlab ci\n")
        monkeypatch.setattr("maestro_cli.cli.generate_ci_yaml", mock_generate)

        rc = main(["ci", str(plan_file), "--provider", "gitlab"])

        assert rc == 0
        mock_generate.assert_called_once_with(
            str(plan_file),
            provider="gitlab",
            workflow_name="Maestro CI",
            python_version="3.11",
            test_command="python -m pytest -q",
        )
        assert "# gitlab ci" in capsys.readouterr().out


class TestEvalExistingFileNotDirectory:
    """eval exits 1 when run_path exists on disk but is a file, not a directory."""

    def test_eval_existing_file_path_returns_one_without_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        file_path = tmp_path / "not-a-dir.json"
        file_path.write_text("{}", encoding="utf-8")
        mock_run_eval = MagicMock()
        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)

        rc = main(["eval", str(eval_yaml), str(file_path)])

        assert rc == 1
        mock_run_eval.assert_not_called()
        out = capsys.readouterr().out
        assert f"[maestro] error: run path is not a directory: {file_path.resolve()}" in out


class TestUiMissingWebDependencies:
    """ui exits 1 with a helpful message when web extras are not installed."""

    def test_ui_missing_web_dependencies_returns_one(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Replace maestro_cli.web with an empty module so 'from .web import create_app' raises ImportError
        monkeypatch.setitem(sys.modules, "maestro_cli.web", ModuleType("maestro_cli.web"))

        rc = main(["ui", "--no-browser"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "web dependencies not installed" in out
        assert "pip install maestro-ai-cli[web]" in out


class TestBackfillCostsValidPricingFile:
    """backfill-costs reads a pricing file and sets the env var before proceeding."""

    def test_backfill_costs_valid_pricing_file_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pricing_file = tmp_path / "pricing.json"
        pricing_file.write_text('{"default":{"input_per_million":1.0}}', encoding="utf-8")
        # Ensure the env var is cleaned up after the test regardless
        monkeypatch.delenv("MAESTRO_CODEX_PRICING_JSON", raising=False)

        rc = main([
            "backfill-costs",
            "--codex-pricing-file", str(pricing_file),
            "--root", str(tmp_path),
            "--dry-run",
        ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "backfill-costs mode=dry-run" in out


class TestWatchTextModeFailedStatus:
    """watch in text mode with a non-passing status prints summary and returns 1."""

    def test_watch_text_mode_failed_status_prints_summary_and_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="failed",
                total_iterations=3,
                best_metric=None,
                total_cost_usd=None,
            )),
        )

        rc = main(["watch", str(plan_file)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] watch complete: failed" in out
        assert "[maestro]   iterations: 3" in out


class TestRunVerbosityForwarding:
    """run --verbose / --quiet verbosity is forwarded to run_plan."""

    def test_run_verbose_forwards_verbosity_verbose(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--verbose"])

        assert rc == 0
        _, kwargs = mock_run.call_args
        assert kwargs["verbosity"] == "verbose"

    def test_run_quiet_forwards_verbosity_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "-q"])

        assert rc == 0
        _, kwargs = mock_run.call_args
        assert kwargs["verbosity"] == "quiet"


class TestRunJsonlOutputMode:
    """run --output jsonl forwards output_mode='jsonl' to run_plan."""

    def test_run_jsonl_output_mode_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--output", "jsonl"])

        assert rc == 0
        _, kwargs = mock_run.call_args
        assert kwargs["output_mode"] == "jsonl"


class TestWatchJsonlEventCallback:
    """watch --output jsonl passes a callable event_callback to watch()."""

    def test_watch_jsonl_passes_callable_event_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured_kwargs: dict = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                status="improved",
                total_iterations=1,
                best_metric=42.0,
                best_iteration=1,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--output", "jsonl"])

        assert rc == 0
        cb = captured_kwargs.get("event_callback")
        assert cb is not None
        assert callable(cb)


class TestReplanQuietVerbosity:
    """replan --quiet forwards verbosity='quiet' to replan()."""

    def test_replan_quiet_forwards_verbosity_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--quiet"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["verbosity"] == "quiet"


class TestSuggestMissingPlan:
    """suggest exits 1 when the plan file does not exist."""

    def test_suggest_missing_plan_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "no-such-plan.yaml"

        rc = main(["suggest", str(missing)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error" in out


class TestStatusMissingPlan:
    """status exits 1 when the plan file does not exist."""

    def test_status_missing_plan_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "no-such-plan.yaml"

        rc = main(["status", str(missing)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error" in out


class TestDoctorRunDirWithoutJson:
    """doctor --run-dir without --json passes run_dir and json_output=False."""

    def test_doctor_run_dir_no_json_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_doctor = MagicMock(return_value=[
            ("python_version", "Python 3.11", "ok"),
        ])
        monkeypatch.setattr("maestro_cli.doctor.run_doctor", mock_run_doctor)

        rc = main(["doctor", "--run-dir", str(tmp_path)])

        assert rc == 0
        mock_run_doctor.assert_called_once_with(run_dir=str(tmp_path), json_output=False, full=False)


# ===========================================================================
# Tests for main() — cleanup non-dry-run dispatch
# ===========================================================================

class TestMainCleanupNonDryRun:
    """cleanup without --dry-run prints 'deleted' (not 'would delete')."""

    def test_cleanup_non_dry_run_prints_deleted_action(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        old_run = run_root / "20260101-120000_test-plan"
        old_run.mkdir()
        mock_cleanup_runs = MagicMock(return_value=[old_run])

        monkeypatch.setattr("maestro_cli.cli.cleanup_runs", mock_cleanup_runs)

        rc = main(["cleanup", str(plan_file), "--keep", "5"])

        assert rc == 0
        mock_cleanup_runs.assert_called_once_with(
            run_root.resolve(),
            keep=5,
            older_than_days=None,
            dry_run=False,
        )
        out = capsys.readouterr().out
        assert f"[maestro] deleted: {old_run.name}" in out
        assert "[maestro] deleted 1 run(s), kept 5" in out
        assert "would delete" not in out


# ===========================================================================
# Tests for main() — backfill-costs --run-root explicit paths
# ===========================================================================

class TestBackfillCostsRunRoot:
    """backfill-costs --run-root bypasses discover_run_roots and uses explicit paths."""

    def test_backfill_costs_run_root_bypasses_discover(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir_a = tmp_path / "run-a"
        run_dir_b = tmp_path / "run-b"
        run_dir_a.mkdir()
        run_dir_b.mkdir()

        mock_discover = MagicMock()
        monkeypatch.setattr("maestro_cli.cli.discover_run_roots", mock_discover)

        rc = main([
            "backfill-costs",
            "--run-root", str(run_dir_a),
            "--run-root", str(run_dir_b),
            "--dry-run",
        ])

        assert rc == 0
        mock_discover.assert_not_called()
        out = capsys.readouterr().out
        assert "backfill-costs mode=dry-run" in out

    def test_backfill_costs_manifests_failed_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """backfill-costs returns 1 when manifests_failed > 0."""
        from maestro_cli.cost_backfill import BackfillSummary

        failed_summary = BackfillSummary(
            run_roots=1,
            runs_scanned=2,
            runs_updated=0,
            tasks_updated=0,
            manifests_failed=1,
        )
        monkeypatch.setattr("maestro_cli.cli.backfill_run_costs", MagicMock(return_value=failed_summary))

        rc = main(["backfill-costs", "--root", str(tmp_path)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "manifests_failed=1" in out

    def test_backfill_costs_write_mode_prints_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """backfill-costs without --dry-run prints mode=write."""
        from maestro_cli.cost_backfill import BackfillSummary

        summary = BackfillSummary(run_roots=1, runs_scanned=3, runs_updated=2, tasks_updated=5)
        monkeypatch.setattr("maestro_cli.cli.backfill_run_costs", MagicMock(return_value=summary))

        rc = main(["backfill-costs", "--root", str(tmp_path)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "mode=write" in out
        assert "runs_updated=2" in out


class TestVersionFlag:
    """--version exits 0 and includes the version string."""

    def test_version_flag_exits_zero_with_version_string(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        from maestro_cli import __version__

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert __version__ in out


class TestBackfillCostsMissingPricingFile:
    """backfill-costs --codex-pricing-file returns 1 when the file does not exist."""

    def test_missing_pricing_file_returns_one_without_backfill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing = tmp_path / "no-such-pricing.json"
        mock_backfill = MagicMock()
        monkeypatch.setattr("maestro_cli.cli.backfill_run_costs", mock_backfill)

        rc = main([
            "backfill-costs",
            "--codex-pricing-file", str(missing),
            "--root", str(tmp_path),
        ])

        assert rc == 1
        mock_backfill.assert_not_called()
        out = capsys.readouterr().out
        assert "pricing file does not exist" in out


class TestExplainParserDefaults:
    """explain subcommand parser defaults are sane."""

    def test_explain_parser_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["explain", "plan.yaml"])
        assert args.command == "explain"
        assert args.plan == "plan.yaml"
        assert args.cache_dir is None
        assert args.json is False


class TestRunResumeLast:
    """run --resume-last --quiet suppresses the 'resuming from' print."""

    def test_resume_last_quiet_suppresses_resuming_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        latest_run = tmp_path / ".maestro-runs" / "latest"
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        mock_find_latest = MagicMock(return_value=latest_run)
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", mock_find_latest)

        rc = main(["run", str(plan_file), "--resume-last", "--quiet"])

        assert rc == 0
        _, kwargs = mock_run.call_args
        assert kwargs["resume_path"] == latest_run
        out = capsys.readouterr().out
        assert "resuming from" not in out


class TestDiffFileNotFoundError:
    """diff returns 1 when diff_runs raises FileNotFoundError (e.g. missing manifest)."""

    def test_diff_runs_raises_file_not_found_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        monkeypatch.setattr(
            "maestro_cli.diff.diff_runs",
            MagicMock(side_effect=FileNotFoundError("run_manifest.json not found")),
        )

        rc = main(["diff", str(run_a), str(run_b)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error: run_manifest.json not found" in out


class TestValidateWithImports:
    """validate prints the imports count when the plan has imports."""

    def test_validate_prints_imports_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        sub_plan_yaml = """\
version: 1
name: sub-plan
tasks:
  - id: sub-t1
    command: "echo sub"
"""
        sub_plan_file = _write_yaml(tmp_path, sub_plan_yaml, "sub.yaml")
        plan_yaml = f"""\
version: 1
name: import-plan
imports:
  - path: {sub_plan_file.as_posix()}
    prefix: sub
tasks:
  - id: main-t1
    command: "echo main"
"""
        plan_file = _write_yaml(tmp_path, plan_yaml, "plan.yaml")

        rc = main(["validate", str(plan_file)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "- imports: 1 file(s)" in out


class TestScaffoldMockedDispatch:
    """scaffold without --output prints the generated plan YAML to stdout."""

    def test_scaffold_no_output_prints_plan_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        generated_yaml = "version: 1\nname: generated\ntasks: []\n"
        mock_load_brief = MagicMock(return_value=SimpleNamespace(name="generated"))
        mock_scaffold_plan = MagicMock(return_value=generated_yaml)
        monkeypatch.setattr("maestro_cli.scaffold.load_brief", mock_load_brief)
        monkeypatch.setattr("maestro_cli.scaffold.scaffold_plan", mock_scaffold_plan)

        rc = main(["scaffold", str(brief_file)])

        assert rc == 0
        mock_load_brief.assert_called_once()
        mock_scaffold_plan.assert_called_once()
        out = capsys.readouterr().out
        assert generated_yaml in out


class TestWatchParserDefaults:
    """watch parser defaults: verbose=False, quiet=False."""

    def test_watch_parser_verbose_quiet_default_false(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["watch", "plan.yaml"])
        assert args.command == "watch"
        assert args.verbose is False
        assert args.quiet is False
        assert args.dry_run is False
        assert args.auto_approve is False

    def test_watch_parser_output_profile_parallel_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["watch", "plan.yaml"])
        assert args.output == "text"
        assert args.execution_profile == "plan"
        assert args.max_parallel is None


class TestWatchTextModeKwargsForwarding:
    """watch text mode: max_parallel_override and execution_profile forwarded to watch()."""

    def test_watch_text_mode_forwards_parallel_and_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured_kwargs: dict = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                status="improved",
                total_iterations=2,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main([
            "watch",
            str(plan_file),
            "--max-parallel", "4",
            "--execution-profile", "yolo",
        ])

        assert rc == 0
        assert captured_kwargs["max_parallel_override"] == 4
        assert captured_kwargs["execution_profile"] == "yolo"
        assert captured_kwargs["output_mode"] == "text"
        assert captured_kwargs["event_callback"] is None


class TestScaffoldOutputFile:
    """scaffold --output FILE writes generated plan YAML to disk."""

    def test_scaffold_output_flag_writes_to_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        output_file = tmp_path / "generated.yaml"
        generated_yaml = "version: 1\nname: generated\ntasks: []\n"
        monkeypatch.setattr("maestro_cli.scaffold.load_brief", MagicMock(return_value=SimpleNamespace(name="generated")))
        monkeypatch.setattr("maestro_cli.scaffold.scaffold_plan", MagicMock(return_value=generated_yaml))

        rc = main(["scaffold", str(brief_file), "--output", str(output_file)])

        assert rc == 0
        assert output_file.read_text(encoding="utf-8") == generated_yaml
        out = capsys.readouterr().out
        assert "plan written to" in out

    def test_scaffold_cost_check_dispatches_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        brief_file = _write_yaml(tmp_path, _BRIEF_YAML, "brief.yaml")
        generated_yaml = "version: 1\nname: generated\ntasks: []\n"
        monkeypatch.setattr("maestro_cli.scaffold.load_brief", MagicMock(return_value=SimpleNamespace(name="generated")))
        monkeypatch.setattr("maestro_cli.scaffold.scaffold_plan", MagicMock(return_value=generated_yaml))
        monkeypatch.setattr(
            "maestro_cli.scaffold.validate_plan_cost_safety",
            MagicMock(return_value=["No max_cost_usd set", "No verify_command"]),
        )

        rc = main(["scaffold", str(brief_file), "--cost-check"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "[maestro] cost-warning: No max_cost_usd set" in out
        assert "[maestro] cost-warning: No verify_command" in out


class TestReplanModelFlag:
    """replan --model forwarded to replan() as analysis_model."""

    def test_replan_model_flag_forwarded_as_analysis_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--model", "sonnet"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["analysis_model"] == "sonnet"

    def test_replan_auto_approve_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=False))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--auto-approve"])

        assert rc == 1
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["auto_approve"] is True


class TestReplanSearchFlags:
    """replan search-specific CLI flags are parsed and forwarded."""

    def test_replan_variants_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--variants", "3"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["variants"] == 3

    def test_replan_debug_prob_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--debug-prob", "0.25"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["debug_prob"] == pytest.approx(0.25)

    def test_replan_selection_policy_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--selection-policy", "ucb1"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["selection_policy"] == "ucb1"

    def test_replan_exploration_constant_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--exploration-constant", "2.75"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["exploration_constant"] == pytest.approx(2.75)

    def test_replan_population_strategy_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--population-strategy", "tournament"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["population_strategy"] == "tournament"

    def test_replan_tournament_size_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--tournament-size", "3"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["tournament_size"] == 3

    def test_replan_elite_count_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--elite-count", "2"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["elite_count"] == 2

    def test_replan_diversity_floor_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--diversity-floor", "0.6"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["diversity_floor"] == pytest.approx(0.6)


class TestWatchAutoApprove:
    """watch --auto-approve forwarded to watch() as auto_approve=True."""

    def test_watch_auto_approve_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="improved",
                total_iterations=1,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--auto-approve"])

        assert rc == 0
        assert captured.get("auto_approve") is True


class TestStatusTextOutput:
    """status without --json calls format_status (not format_status_json)."""

    def test_status_text_output_calls_format_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        status_result = SimpleNamespace(summary="all good")
        mock_plan_status = MagicMock(return_value=status_result)
        mock_format_status = MagicMock(return_value="status text output")
        mock_format_status_json = MagicMock(return_value='{"should":"not appear"}')

        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=None))
        monkeypatch.setattr("maestro_cli.status.plan_status", mock_plan_status)
        monkeypatch.setattr("maestro_cli.status.format_status", mock_format_status)
        monkeypatch.setattr("maestro_cli.status.format_status_json", mock_format_status_json)

        rc = main(["status", str(plan_file)])

        assert rc == 0
        mock_format_status.assert_called_once_with(status_result)
        mock_format_status_json.assert_not_called()
        out = capsys.readouterr().out
        assert "status text output" in out


class TestEvalJsonPass:
    """eval --json with overall_pass=True returns 0 and prints JSON via format_eval_json."""

    def test_eval_json_pass_returns_zero_and_formats_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        run_dir = tmp_path / "completed-run"
        run_dir.mkdir()
        suite = SimpleNamespace(overall_pass=True)
        mock_run_eval = MagicMock(return_value=suite)
        mock_format_eval_json = MagicMock(return_value='{"overall_pass":true}')
        mock_format_eval = MagicMock(return_value="should not appear")

        monkeypatch.setattr("maestro_cli.eval.run_eval", mock_run_eval)
        monkeypatch.setattr("maestro_cli.eval.format_eval_json", mock_format_eval_json)
        monkeypatch.setattr("maestro_cli.eval.format_eval", mock_format_eval)

        rc = main(["eval", str(eval_yaml), str(run_dir), "--json"])

        assert rc == 0
        mock_format_eval_json.assert_called_once_with(suite)
        mock_format_eval.assert_not_called()
        out = capsys.readouterr().out
        assert out == '{"overall_pass":true}\n'


class TestWatchNoneStateReturnsOne:
    """watch returns 1 when watch() returns None (plan loading aborted or fatal error)."""

    def test_watch_none_state_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.watch.watch", MagicMock(return_value=None))

        rc = main(["watch", str(plan_file)])

        assert rc == 1


class TestRunResumeLastNoPriorRuns:
    """run --resume-last returns 1 when no prior runs exist for the plan."""

    def test_resume_last_no_prior_runs_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=None))
        mock_run = MagicMock()
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--resume-last"])

        assert rc == 1
        mock_run.assert_not_called()
        out = capsys.readouterr().out
        assert "no prior runs found" in out


class TestReplanTuiOutputReturnsOne:
    """replan --output tui returns 1 with an informative error message."""

    def test_replan_tui_output_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock()
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--output", "tui"])

        assert rc == 1
        mock_replan.assert_not_called()
        out = capsys.readouterr().out
        assert "TUI mode is not supported for replan" in out


class TestDoctorFailedCheckReturnsOne:
    """doctor returns 1 when at least one check has status 'fail'."""

    def test_doctor_with_failed_check_returns_one(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_doctor = MagicMock(return_value=[
            ("python_version", "Python 3.11", "ok"),
            ("codex", "not found", "fail"),
        ])
        monkeypatch.setattr("maestro_cli.doctor.run_doctor", mock_run_doctor)

        rc = main(["doctor"])

        assert rc == 1
        mock_run_doctor.assert_called_once_with(run_dir=".maestro-runs", json_output=False, full=False)


class TestReplanDryRunFlag:
    """replan --dry-run is forwarded to replan() as dry_run=True."""

    def test_replan_dry_run_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--dry-run"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["dry_run"] is True

    def test_replan_execution_profile_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=False))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--execution-profile", "safe"])

        assert rc == 1
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["execution_profile"] == "safe"


# ===========================================================================
# Tests for main() — diff ValueError path
# ===========================================================================

class TestDiffValueError:
    """diff returns 1 and prints the error when diff_runs raises ValueError."""

    def test_diff_runs_raises_value_error_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_a = tmp_path / "run-a"
        run_b = tmp_path / "run-b"
        run_a.mkdir()
        run_b.mkdir()
        monkeypatch.setattr(
            "maestro_cli.diff.diff_runs",
            MagicMock(side_effect=ValueError("incompatible plan names")),
        )

        rc = main(["diff", str(run_a), str(run_b)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error: incompatible plan names" in out


# ===========================================================================
# Tests for main() — run --output text explicit flag
# ===========================================================================

class TestRunOutputTextExplicit:
    """run --output text explicitly forwards output_mode='text' to run_plan."""

    def test_run_explicit_text_output_mode_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_run = MagicMock(return_value=SimpleNamespace(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        rc = main(["run", str(plan_file), "--output", "text"])

        assert rc == 0
        _, kwargs = mock_run.call_args
        assert kwargs["output_mode"] == "text"


# ===========================================================================
# Tests for main() — eval top-level exception handler
# ===========================================================================

class TestEvalTopLevelException:
    """eval returns 1 and prints error when run_eval raises an unexpected exception."""

    def test_eval_run_eval_raises_exception_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        eval_yaml = _write_yaml(tmp_path, _VALID_PLAN_YAML, "eval.yaml")
        run_dir = tmp_path / "completed-run"
        run_dir.mkdir()
        monkeypatch.setattr(
            "maestro_cli.eval.run_eval",
            MagicMock(side_effect=ValueError("malformed eval criteria")),
        )

        rc = main(["eval", str(eval_yaml), str(run_dir)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error: malformed eval criteria" in out


# ===========================================================================
# Tests for main() — status with no prior run (latest_run=None)
# ===========================================================================

class TestStatusNoLatestRun:
    """status passes None as latest_run to plan_status when no prior run exists."""

    def test_status_no_prior_run_passes_none_as_latest_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        status_result = SimpleNamespace(summary="no runs yet")
        mock_plan_status = MagicMock(return_value=status_result)
        mock_format_status = MagicMock(return_value="no runs yet")

        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=None))
        monkeypatch.setattr("maestro_cli.status.plan_status", mock_plan_status)
        monkeypatch.setattr("maestro_cli.status.format_status", mock_format_status)

        rc = main(["status", str(plan_file)])

        assert rc == 0
        _, passed_latest_run, _ = mock_plan_status.call_args.args
        assert passed_latest_run is None
        out = capsys.readouterr().out
        assert "no runs yet" in out


# ===========================================================================
# Tests for main() — backfill-costs OSError when reading pricing file
# ===========================================================================

class TestBackfillCostsPricingFileReadError:
    """backfill-costs returns 1 and prints error when read_text raises OSError."""

    def test_pricing_file_read_oserror_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pricing_file = tmp_path / "pricing.json"
        pricing_file.write_text("{}", encoding="utf-8")

        original_read_text = Path.read_text

        def _raise_on_pricing(self: Path, **kwargs: object) -> str:
            if self == pricing_file:
                raise OSError("permission denied")
            return original_read_text(self, **kwargs)

        monkeypatch.setattr(Path, "read_text", _raise_on_pricing)
        mock_backfill = MagicMock()
        monkeypatch.setattr("maestro_cli.cli.backfill_run_costs", mock_backfill)

        rc = main([
            "backfill-costs",
            "--codex-pricing-file", str(pricing_file),
            "--root", str(tmp_path),
        ])

        assert rc == 1
        mock_backfill.assert_not_called()
        out = capsys.readouterr().out
        assert "failed to read pricing file" in out


# ===========================================================================
# Tests for main() — replan --output jsonl forwards output_mode
# ===========================================================================

class TestReplanJsonlOutputMode:
    """replan --output jsonl calls replan() with output_mode='jsonl'."""

    def test_replan_jsonl_output_mode_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--output", "jsonl"])

        assert rc == 0
        call_kwargs = mock_replan.call_args.kwargs
        assert call_kwargs["output_mode"] == "jsonl"


# ===========================================================================
# Tests for main() — watch summary prints both best_metric and total_cost_usd
# ===========================================================================

class TestWatchSummaryBothMetricAndCost:
    """watch summary prints both 'best metric' and 'total cost' when both are set."""

    def test_watch_summary_both_metric_and_cost_lines_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="improved",
                total_iterations=3,
                best_metric=42.5,
                best_iteration=2,
                total_cost_usd=0.75,
            )),
        )

        rc = main(["watch", str(plan_file)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "[maestro] watch complete: improved" in out
        assert "[maestro]   best metric: 42.5 (iteration 2)" in out
        assert "[maestro]   total cost: $0.75" in out


# ===========================================================================
# Tests for main() — status --run-dir forwarded to _find_latest_run
# ===========================================================================

class TestStatusRunDirForwarded:
    """status --run-dir passes the override value to _find_latest_run."""

    def test_status_run_dir_kwarg_forwarded_to_find_latest_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        status_result = SimpleNamespace(summary="ok")
        mock_find_latest_run = MagicMock(return_value=None)

        monkeypatch.setattr("maestro_cli.cli._find_latest_run", mock_find_latest_run)
        monkeypatch.setattr("maestro_cli.status.plan_status", MagicMock(return_value=status_result))
        monkeypatch.setattr("maestro_cli.status.format_status", MagicMock(return_value="ok"))

        rc = main(["status", str(plan_file), "--run-dir", "custom-runs"])

        assert rc == 0
        mock_find_latest_run.assert_called_once()
        assert mock_find_latest_run.call_args.kwargs == {"run_dir": "custom-runs"}


# ===========================================================================
# Tests for main() — suggest --run-dir text mode
# ===========================================================================

class TestSuggestRunDirTextMode:
    """suggest --run-dir override is used even in text (non-JSON) output mode."""

    def test_suggest_text_mode_custom_run_dir_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        custom_run_dir = tmp_path / "alt-runs"
        suggestions = SimpleNamespace(items=[])
        mock_suggest_plan = MagicMock(return_value=suggestions)
        mock_format_suggestions = MagicMock(return_value="no suggestions")

        monkeypatch.setattr("maestro_cli.suggest.suggest_plan", mock_suggest_plan)
        monkeypatch.setattr("maestro_cli.suggest.format_suggestions", mock_format_suggestions)

        rc = main(["suggest", str(plan_file), "--run-dir", str(custom_run_dir)])

        assert rc == 0
        _plan, passed_run_dir = mock_suggest_plan.call_args.args[:2]
        assert passed_run_dir == custom_run_dir
        mock_format_suggestions.assert_called_once_with(suggestions)
        assert "no suggestions" in capsys.readouterr().out


# ===========================================================================
# Tests for main() — watch --output jsonl suppresses summary even on success
# ===========================================================================

class TestWatchJsonlSuppressesSummaryOnSuccess:
    """watch --output jsonl suppresses the summary block even when status=improved."""

    def test_watch_jsonl_improved_suppresses_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="improved",
                total_iterations=4,
                best_metric=99.0,
                best_iteration=4,
                total_cost_usd=2.50,
            )),
        )

        rc = main(["watch", str(plan_file), "--output", "jsonl"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "watch complete" not in out
        assert "best metric" not in out
        assert "total cost" not in out


class TestWatchQuietVerbosityForwarding:
    """watch --quiet forwards verbosity='quiet' to watch()."""

    def test_watch_quiet_text_mode_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        captured: dict[str, object] = {}

        def _fake_watch(plan_path: str, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                status="max_iterations",
                total_iterations=2,
                best_metric=None,
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.watch", _fake_watch)

        rc = main(["watch", str(plan_file), "--quiet"])

        assert rc == 0
        assert captured.get("verbosity") == "quiet"
        assert captured.get("output_mode") == "text"


class TestWatchRegressionStatusReturnsOne:
    """watch returns exit code 1 when state.status is 'regression'."""

    def test_regression_status_returns_one_with_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="regression",
                total_iterations=3,
                best_metric=42.5,
                best_iteration=2,
                total_cost_usd=0.75,
            )),
        )

        rc = main(["watch", str(plan_file)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] watch complete: regression" in out
        assert "best metric: 42.5" in out


class TestReplanModeAlias:
    """replan --mode is a valid alias for --execution-profile."""

    def test_replan_mode_alias_forwards_execution_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        mock_replan = MagicMock(return_value=SimpleNamespace(final_success=True))
        monkeypatch.setattr("maestro_cli.cli.replan", mock_replan)

        rc = main(["replan", str(plan_file), "--mode", "yolo"])

        assert rc == 0
        assert mock_replan.call_args.kwargs["execution_profile"] == "yolo"


class TestSuggestPlanRaisesExceptionReturnsOne:
    """suggest returns 1 and prints error when suggest_plan raises."""

    def test_suggest_plan_exception_propagates_to_main(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.suggest.suggest_plan",
            MagicMock(side_effect=RuntimeError("suggest failed unexpectedly")),
        )

        rc = main(["suggest", str(plan_file)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error" in out
        assert "suggest failed unexpectedly" in out


class TestCiGenerateYamlRaisesExceptionReturnsOne:
    """ci propagates generate_ci_yaml exceptions through main() → returns 1."""

    def test_ci_generate_raises_returns_one_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.cli.generate_ci_yaml",
            MagicMock(side_effect=ValueError("unsupported provider")),
        )

        rc = main(["ci", str(plan_file)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error" in out
        assert "unsupported provider" in out


class TestShellExceptionPropagatesToMain:
    """shell propagates run_shell exceptions through main() → returns 1."""

    def test_shell_run_shell_raises_returns_one_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.shell.run_shell",
            MagicMock(side_effect=RuntimeError("REPL initialisation failed")),
        )

        rc = main(["shell"])

        assert rc == 1
        out = capsys.readouterr().out
        assert "[maestro] error" in out
        assert "REPL initialisation failed" in out


class TestWatchLiveBestMetricInSummary:
    """watch --output live prints best-metric line when state.best_metric is set."""

    def test_watch_live_with_best_metric_prints_metric_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)

        class _LiveContext:
            def __enter__(self) -> "_LiveContext":
                return self

            def __exit__(self, *_args: object) -> bool:
                return False

        mock_create_live_callback = MagicMock(
            return_value=(_LiveContext(), MagicMock())
        )
        monkeypatch.setattr("maestro_cli.live.create_live_callback", mock_create_live_callback)
        monkeypatch.setattr(
            "maestro_cli.watch.watch",
            MagicMock(return_value=SimpleNamespace(
                status="improved",
                total_iterations=4,
                best_metric=99.5,
                best_iteration=3,
                total_cost_usd=0.0,
            )),
        )

        rc = main(["watch", str(plan_file), "--output", "live"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "best metric: 99.5 (iteration 3)" in out


class TestSuggestMinRunsCustomValue:
    """suggest --min-runs with a value below the default (3) is forwarded correctly."""

    def test_suggest_min_runs_one_forwarded_to_suggest_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        suggestions = SimpleNamespace(suggestions=[])
        mock_suggest_plan = MagicMock(return_value=suggestions)
        monkeypatch.setattr("maestro_cli.suggest.suggest_plan", mock_suggest_plan)
        monkeypatch.setattr(
            "maestro_cli.suggest.format_suggestions",
            MagicMock(return_value="no suggestions"),
        )

        rc = main(["suggest", str(plan_file), "--min-runs", "1"])

        assert rc == 0
        assert mock_suggest_plan.call_args.kwargs["min_runs"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# ──── Coverage expansion tests ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrintBanner:
    """Cover _print_banner() with and without colour (lines 53-72)."""

    def test_banner_no_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO
        from maestro_cli.cli import _print_banner

        buf = StringIO()
        buf.isatty = lambda: False  # type: ignore[attr-defined]
        monkeypatch.setattr("maestro_cli.cli.sys.stderr", buf)

        _print_banner()

        output = buf.getvalue()
        assert "Maestro CLI" in output
        assert "\033[" not in output  # no ANSI codes

    def test_banner_with_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO
        from maestro_cli.cli import _print_banner

        buf = StringIO()
        buf.isatty = lambda: True  # type: ignore[attr-defined]
        monkeypatch.setattr("maestro_cli.cli.sys.stderr", buf)

        _print_banner()

        output = buf.getvalue()
        assert "Maestro CLI" in output
        assert "\033[" in output  # ANSI codes present


class TestPrintCommands:
    """Cover _print_commands() colour and plain branches (lines 75-131)."""

    def test_commands_no_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO
        from maestro_cli.cli import _print_commands

        buf = StringIO()
        buf.isatty = lambda: False  # type: ignore[attr-defined]
        monkeypatch.setattr("maestro_cli.cli.sys.stderr", buf)

        _print_commands()

        output = buf.getvalue()
        assert "Commands:" in output
        assert "maestro run" in output
        assert "\033[" not in output

    def test_commands_with_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO
        from maestro_cli.cli import _print_commands

        buf = StringIO()
        buf.isatty = lambda: True  # type: ignore[attr-defined]
        monkeypatch.setattr("maestro_cli.cli.sys.stderr", buf)

        _print_commands()

        output = buf.getvalue()
        assert "Commands:" in output
        assert "\033[" in output


class TestMainNoArgs:
    """Cover the no-args path: banner + command list (lines 1478-1481)."""

    def test_no_args_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO

        buf = StringIO()
        buf.isatty = lambda: False  # type: ignore[attr-defined]
        monkeypatch.setattr("maestro_cli.cli.sys.stderr", buf)

        rc = main([])

        assert rc == 0
        output = buf.getvalue()
        assert "Maestro CLI" in output
        assert "Commands:" in output


class TestMainHelpBanner:
    """Cover --help path showing banner (line 1475)."""

    def test_help_shows_banner_and_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from io import StringIO

        buf = StringIO()
        buf.isatty = lambda: False  # type: ignore[attr-defined]
        monkeypatch.setattr("maestro_cli.cli.sys.stderr", buf)

        # argparse --help calls sys.exit(0) after printing help text,
        # so we need to catch SystemExit.  The banner is printed before
        # parse_args is called.
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        output = buf.getvalue()
        assert "Maestro CLI" in output


class TestPrintWarnings:
    """Cover _print_warnings (line 688)."""

    def test_print_warnings_empty_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.cli import _print_warnings

        _print_warnings([])
        out = capsys.readouterr().out
        assert out == ""

    def test_print_warnings_with_items(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.cli import _print_warnings

        _print_warnings(["warn1", "warn2"])
        out = capsys.readouterr().out
        assert "2 warning(s)" in out
        assert "warn1" in out
        assert "warn2" in out


class TestFindLatestRunEdgeCases:
    """Cover _find_latest_run edge (lines 648, 663)."""

    def test_run_root_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plan = SimpleNamespace(
            source_dir=Path("/nonexistent"),
            run_dir=".maestro-runs",
            name="test",
        )
        monkeypatch.setattr("maestro_cli.cli.resolve_path", lambda *a: None)

        result = _find_latest_run(plan)
        assert result is None

    def test_no_matching_candidates(self, tmp_path: Path) -> None:
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        # Create a dir that does NOT match the plan name suffix
        (run_root / "20260101_other-plan").mkdir()
        (run_root / "20260101_other-plan" / "run_manifest.json").write_text("{}", encoding="utf-8")

        plan = SimpleNamespace(
            source_dir=tmp_path,
            run_dir=".maestro-runs",
            name="test-plan",
        )

        result = _find_latest_run(plan)
        assert result is None


class TestCmdBlame:
    """Cover _cmd_blame (lines 1332-1345)."""

    def test_blame_missing_directory_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["blame", str(tmp_path / "nonexistent")])
        assert rc == 1
        out = capsys.readouterr().out
        assert "not a directory" in out

    def test_blame_text_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        chain = SimpleNamespace(nodes=[])
        monkeypatch.setattr("maestro_cli.blame.blame_run", MagicMock(return_value=chain))
        monkeypatch.setattr("maestro_cli.blame.format_blame", MagicMock(return_value="no blame"))
        monkeypatch.setattr("maestro_cli.blame.format_blame_json", MagicMock(return_value="{}"))

        rc = main(["blame", str(run_dir)])
        assert rc == 0  # nodes is empty => 0
        out = capsys.readouterr().out
        assert "no blame" in out

    def test_blame_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        chain = SimpleNamespace(nodes=["has-nodes"])
        monkeypatch.setattr("maestro_cli.blame.blame_run", MagicMock(return_value=chain))
        monkeypatch.setattr("maestro_cli.blame.format_blame", MagicMock(return_value="text"))
        monkeypatch.setattr("maestro_cli.blame.format_blame_json", MagicMock(return_value='{"blame":"data"}'))

        rc = main(["blame", str(run_dir), "--json"])
        assert rc == 1  # nodes is non-empty => 1
        out = capsys.readouterr().out
        assert '{"blame":"data"}' in out


class TestCmdBudget:
    """Cover _cmd_budget (lines 1349-1354)."""

    def test_budget_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.budget.format_budget",
            MagicMock(return_value="Budget: $0.00"),
        )
        monkeypatch.setattr(
            "maestro_cli.budget._DEFAULT_LEDGER_PATH",
            ".maestro-cache/budget.jsonl",
        )

        rc = main(["budget"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Budget:" in out

    def test_budget_with_root_arg(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.budget.format_budget",
            MagicMock(return_value="Budget: $5.00"),
        )
        monkeypatch.setattr(
            "maestro_cli.budget._DEFAULT_LEDGER_PATH",
            ".maestro-cache/budget.jsonl",
        )

        rc = main(["budget", "--root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Budget:" in out


class TestCmdExportOtel:
    """Cover _cmd_export_otel (lines 1358-1381)."""

    def test_export_otel_missing_dir_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["export-otel", str(tmp_path / "nonexistent")])
        assert rc == 1
        out = capsys.readouterr().out
        assert "not a directory" in out

    def test_export_otel_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        monkeypatch.setattr(
            "maestro_cli.otel.export_run",
            MagicMock(return_value={"success": True, "json": '{"spans":[]}', "span_count": 5}),
        )

        rc = main(["export-otel", str(run_dir), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '{"spans":[]}' in out

    def test_export_otel_success_no_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        monkeypatch.setattr(
            "maestro_cli.otel.export_run",
            MagicMock(return_value={
                "success": True,
                "span_count": 10,
                "format": "otlp",
                "endpoint": "http://localhost:4317",
            }),
        )

        rc = main(["export-otel", str(run_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "exported 10 spans" in out

    def test_export_otel_forwards_content_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        captured: dict[str, object] = {}

        def _mock_export_run(
            run_path: Path,
            endpoint: str | None = None,
            json_output: bool = False,
            include_content: bool = False,
            mask_content: bool = False,
        ) -> dict[str, object]:
            captured["run_path"] = run_path
            captured["endpoint"] = endpoint
            captured["json_output"] = json_output
            captured["include_content"] = include_content
            captured["mask_content"] = mask_content
            return {"success": True, "span_count": 1, "format": "json", "json": "{}"}

        monkeypatch.setattr("maestro_cli.otel.export_run", _mock_export_run)

        rc = main([
            "export-otel",
            str(run_dir),
            "--json",
            "--include-content",
            "--otel-mask-prompts",
        ])
        assert rc == 0
        assert captured["json_output"] is True
        assert captured["include_content"] is True
        assert captured["mask_content"] is True

    def test_export_otel_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        monkeypatch.setattr(
            "maestro_cli.otel.export_run",
            MagicMock(return_value={"success": False, "error": "connection refused"}),
        )

        rc = main(["export-otel", str(run_dir)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "connection refused" in out


class TestCmdVerify:
    """Cover _cmd_verify (lines 1385-1419)."""

    def test_verify_missing_dir_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["verify", str(tmp_path / "nonexistent")])
        assert rc == 1
        out = capsys.readouterr().out
        assert "not a directory" in out

    def test_verify_missing_events_jsonl_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        rc = main(["verify", str(run_dir)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "events.jsonl not found" in out

    def test_verify_valid_chain_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr("maestro_cli.cli.replay_events", MagicMock(return_value=[]))
        monkeypatch.setattr("maestro_cli.cli.verify_chain", MagicMock(return_value="valid"))
        monkeypatch.setattr("maestro_cli.cli.verify_artefact_hashes", MagicMock(return_value=[]))

        rc = main(["verify", str(run_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "chain=valid" in out
        assert "artefacts=valid" in out

    def test_verify_tampered_chain_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr("maestro_cli.cli.replay_events", MagicMock(return_value=["e1"]))
        monkeypatch.setattr("maestro_cli.cli.verify_chain", MagicMock(return_value="tampered"))
        monkeypatch.setattr("maestro_cli.cli.verify_artefact_hashes", MagicMock(return_value=[]))

        rc = main(["verify", str(run_dir), "--json"])
        assert rc == 1
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["status"] == "tampered"

    def test_verify_artefact_mismatch_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr("maestro_cli.cli.replay_events", MagicMock(return_value=[]))
        monkeypatch.setattr("maestro_cli.cli.verify_chain", MagicMock(return_value="valid"))
        monkeypatch.setattr("maestro_cli.cli.verify_artefact_hashes", MagicMock(return_value=["task-a.log"]))

        rc = main(["verify", str(run_dir)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "tampered: task-a.log" in out

    def test_verify_artefact_mismatch_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text("", encoding="utf-8")

        monkeypatch.setattr("maestro_cli.cli.replay_events", MagicMock(return_value=[]))
        monkeypatch.setattr("maestro_cli.cli.verify_chain", MagicMock(return_value="valid"))
        monkeypatch.setattr("maestro_cli.cli.verify_artefact_hashes", MagicMock(return_value=["bad.log"]))

        rc = main(["verify", str(run_dir), "--json"])
        assert rc == 1
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["artefact_status"] == "tampered"
        assert "bad.log" in data["artefact_mismatches"]


class TestCmdAudit:
    """Cover _cmd_audit (lines 1422-1456)."""

    def test_audit_text_no_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[]))
        monkeypatch.setattr("maestro_cli.cli.format_audit", MagicMock(return_value="no findings"))

        rc = main(["audit", str(plan_file)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no findings" in out
        assert "0 findings" in out

    def test_audit_json_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        finding = SimpleNamespace(severity="error")
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[finding]))
        monkeypatch.setattr("maestro_cli.cli.format_audit_json", MagicMock(return_value='[{"sev":"error"}]'))

        rc = main(["audit", str(plan_file), "--json"])
        assert rc == 1
        out = capsys.readouterr().out
        assert '[{"sev":"error"}]' in out
        assert "1 errors" in out

    def test_audit_fix_applies_fixes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        finding = SimpleNamespace(severity="warning")
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[finding]))
        monkeypatch.setattr("maestro_cli.cli.format_audit", MagicMock(return_value="1 warning"))
        monkeypatch.setattr(
            "maestro_cli.cli._audit_fix_plan",
            MagicMock(return_value=["Added max_cost_usd: 10.0"]),
        )

        rc = main(["audit", str(plan_file), "--fix"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "fix: Added max_cost_usd: 10.0" in out

    def test_audit_fix_no_fixable_findings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        finding = SimpleNamespace(severity="info")
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[finding]))
        monkeypatch.setattr("maestro_cli.cli.format_audit", MagicMock(return_value="1 info"))
        monkeypatch.setattr("maestro_cli.cli._audit_fix_plan", MagicMock(return_value=[]))

        rc = main(["audit", str(plan_file), "--fix"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No auto-fixable findings" in out

    def test_audit_coverage_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[]))
        monkeypatch.setattr(
            "maestro_cli.cli.format_audit_coverage",
            MagicMock(return_value="coverage report"),
        )

        rc = main(["audit", str(plan_file), "--coverage"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "coverage report" in out

    def test_audit_coverage_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[]))
        monkeypatch.setattr(
            "maestro_cli.cli.format_audit_coverage_json",
            MagicMock(return_value='{"coverage":"data"}'),
        )

        rc = main(["audit", str(plan_file), "--json", "--coverage"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '{"coverage":"data"}' in out

    def test_audit_fix_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        finding = SimpleNamespace(severity="warning")
        monkeypatch.setattr("maestro_cli.cli.audit_plan", MagicMock(return_value=[finding]))
        monkeypatch.setattr("maestro_cli.cli.format_audit_json", MagicMock(return_value="[]"))
        monkeypatch.setattr(
            "maestro_cli.cli._audit_fix_plan",
            MagicMock(return_value=["fix-applied"]),
        )

        rc = main(["audit", str(plan_file), "--json", "--fix"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '["fix-applied"]' in out


class TestCmdExplainContext:
    """Cover _cmd_explain --context trajectory (lines 1013-1028)."""

    def test_explain_context_no_prior_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=None))

        rc = main(["explain", str(plan_file), "--context"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "no previous run found" in out

    def test_explain_context_text_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=run_dir))
        monkeypatch.setattr(
            "maestro_cli.explain.explain_context_trajectory",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "maestro_cli.explain.format_context_trajectory",
            MagicMock(return_value="trajectory text"),
        )

        rc = main(["explain", str(plan_file), "--context"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "trajectory text" in out

    def test_explain_context_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        monkeypatch.setattr("maestro_cli.cli._find_latest_run", MagicMock(return_value=run_dir))
        monkeypatch.setattr(
            "maestro_cli.explain.explain_context_trajectory",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "maestro_cli.explain.format_context_trajectory_json",
            MagicMock(return_value='{"trajectory":[]}'),
        )

        rc = main(["explain", str(plan_file), "--json", "--context"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '{"trajectory":[]}' in out


class TestCmdCiAnalyze:
    """Cover _cmd_ci_analyze (lines 1090-1102)."""

    def test_ci_analyze_missing_dir_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main(["ci-analyze", str(tmp_path / "nonexistent")])
        assert rc == 1
        out = capsys.readouterr().out
        assert "not found" in out

    def test_ci_analyze_text_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        analysis = SimpleNamespace()
        monkeypatch.setattr(
            "maestro_cli.ci_agent.analyze_ci_failure",
            MagicMock(return_value=analysis),
        )
        monkeypatch.setattr(
            "maestro_cli.ci_agent.format_ci_analysis",
            MagicMock(return_value="ci analysis text"),
        )

        rc = main(["ci-analyze", str(run_dir)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ci analysis text" in out

    def test_ci_analyze_json_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        analysis = SimpleNamespace(to_dict=lambda: {"root_cause": "timeout"})
        monkeypatch.setattr(
            "maestro_cli.ci_agent.analyze_ci_failure",
            MagicMock(return_value=analysis),
        )

        rc = main(["ci-analyze", str(run_dir), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["root_cause"] == "timeout"


class TestCmdSkill:
    """Cover _cmd_skill (lines 1106-1125)."""

    def test_skill_list_text(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skills",
            MagicMock(return_value="no skills"),
        )

        rc = main(["skill"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no skills" in out

    def test_skill_list_json(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skills_json",
            MagicMock(return_value="[]"),
        )

        rc = main(["skill", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[]" in out

    def test_skill_search_no_query_returns_one(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=[]),
        )

        rc = main(["skill", "search"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "--query is required" in out

    def test_skill_search_with_query(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=["skill1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.search_skills",
            MagicMock(return_value=["skill1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skills",
            MagicMock(return_value="skill1 found"),
        )

        rc = main(["skill", "search", "-q", "test"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "skill1 found" in out

    def test_skill_recommend_no_query_returns_one(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=[]),
        )

        rc = main(["skill", "recommend"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "--query is required" in out

    def test_skill_recommend_with_query(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=["skill1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.recommend_skills",
            MagicMock(return_value=["rec1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skill_recommendations",
            MagicMock(return_value="rec1 found"),
        )

        rc = main(["skill", "recommend", "-q", "failed run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "rec1 found" in out

    def test_skill_recommend_json(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=["skill1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.recommend_skills",
            MagicMock(return_value=["rec1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skill_recommendations_json",
            MagicMock(return_value='{"skill":"rec1"}'),
        )

        rc = main(["skill", "recommend", "-q", "pytest", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '{"skill":"rec1"}' in out


class TestCmdChat:
    """Cover _cmd_chat dispatch (lines 1136-1138, 1523)."""

    def test_chat_dispatches_to_run_chat(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_chat = MagicMock(return_value=0)
        monkeypatch.setattr("maestro_cli.chat.run_chat", mock_run_chat)

        rc = main(["chat", "--engine", "claude", "--model", "haiku"])
        assert rc == 0
        mock_run_chat.assert_called_once_with(
            engine="claude",
            model="haiku",
            execution_profile="plan",
            auto_context=True,
        )

    def test_chat_no_auto_context_dispatches_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_run_chat = MagicMock(return_value=0)
        monkeypatch.setattr("maestro_cli.chat.run_chat", mock_run_chat)

        rc = main(["chat", "--no-auto-context"])
        assert rc == 0
        mock_run_chat.assert_called_once_with(
            engine="claude",
            model=None,
            execution_profile="plan",
            auto_context=False,
        )


class TestCmdScaffoldListLibraries:
    """Cover _cmd_scaffold --list-libraries (lines 1269-1277) and missing brief (1280-1281)."""

    def test_list_libraries_shows_available(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.scaffold.list_workflow_libraries",
            MagicMock(return_value=[
                {"name": "rest-api", "description": "REST API workflow"},
            ]),
        )

        rc = main(["scaffold", "--list-libraries"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "rest-api" in out
        assert "REST API workflow" in out

    def test_list_libraries_empty(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.scaffold.list_workflow_libraries",
            MagicMock(return_value=[]),
        )

        rc = main(["scaffold", "--list-libraries"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no built-in workflow libraries" in out

    def test_scaffold_missing_brief_returns_one(
        self, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # scaffold without --list-libraries and no brief arg
        monkeypatch.setattr(
            "maestro_cli.scaffold.list_workflow_libraries",
            MagicMock(return_value=[]),
        )

        rc = main(["scaffold"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "brief file is required" in out


class TestCmdMcpServer:
    """Cover _cmd_mcp_server dispatch (lines 901-910, 1501)."""

    def test_mcp_server_import_error_returns_one(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import builtins
        _original = builtins.__import__

        def _fail_mcp(name: str, *a: object, **kw: object) -> object:
            if "mcp_server" in name:
                raise ImportError("no mcp")
            return _original(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fail_mcp)

        rc = main(["mcp-server"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "MCP dependencies not installed" in out

    def test_mcp_server_success(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_main = MagicMock()
        monkeypatch.setattr("maestro_cli.mcp_server.main", mock_main)

        rc = main(["mcp-server"])
        assert rc == 0
        mock_main.assert_called_once()


class TestMainExceptionHandler:
    """Cover the top-level exception handler in main() (lines 1541-1543)."""

    def test_exception_in_subcommand_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_file = _write_yaml(tmp_path, _VALID_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.cli.load_plan",
            MagicMock(side_effect=RuntimeError("boom")),
        )

        rc = main(["validate", str(plan_file)])
        assert rc == 1
        out = capsys.readouterr().out
        assert "boom" in out


class TestResolveCacheDir:
    """Cover _resolve_cache_dir (line 1257)."""

    def test_resolve_cache_dir_with_override(self) -> None:
        from maestro_cli.cli import _resolve_cache_dir

        plan = SimpleNamespace(source_dir=Path("/src"), run_dir=".maestro-runs")
        result = _resolve_cache_dir(plan, "/tmp/my-cache")
        assert result == Path("/tmp/my-cache").resolve()

    def test_resolve_cache_dir_without_override_no_run_root(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.cli import _resolve_cache_dir

        plan = SimpleNamespace(source_dir=Path("/nonexistent"), run_dir=".maestro-runs")
        monkeypatch.setattr("maestro_cli.cli.resolve_path", lambda *a: None)

        result = _resolve_cache_dir(plan, None)
        assert result is None


class TestCheckCommand:
    """`maestro check` runs validate + audit (+ optional suggest) in one pass."""

    _CLEAN_PLAN_YAML = """\
version: 1
name: clean-plan
max_cost_usd: 10.0
budget_warning_pct: 0.8
defaults:
  timeout_sec: 1500
  retry_delay_sec: [60, 120]
  claude:
    model: sonnet
tasks:
  - id: t1
    engine: claude
    prompt: "do the thing"
    verify_command: ["test", "-f", "out.txt"]
    max_retries: 1
"""

    _AUDIT_ERROR_PLAN_YAML = """\
version: 1
name: dirty-plan
tasks:
  - id: t1
    command: "echo ok"
"""

    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "plan.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_check_clean_plan_returns_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = self._write(tmp_path, self._CLEAN_PLAN_YAML)
        rc = main(["check", str(plan_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "== Validation ==" in out
        assert "== Audit ==" in out

    def test_check_audit_error_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # SEC001 (no max_cost_usd) is an error severity finding.
        plan_path = self._write(tmp_path, self._AUDIT_ERROR_PLAN_YAML)
        rc = main(["check", str(plan_path)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "SEC001" in out

    def test_check_validation_failure_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Missing 'tasks' key → loader raises PlanValidationError.
        bad_yaml = "version: 1\nname: broken\n"
        plan_path = self._write(tmp_path, bad_yaml)
        rc = main(["check", str(plan_path)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "validation failed" in out

    def test_check_json_output_is_parseable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = self._write(tmp_path, self._CLEAN_PLAN_YAML)
        rc = main(["check", str(plan_path), "--json"])
        out = capsys.readouterr().out
        report = json.loads(out)
        assert rc == 0
        assert report["ok"] is True
        assert report["name"] == "clean-plan"
        assert "validation_warnings" in report
        assert "audit_findings" in report
        assert "summary" in report

    def test_check_json_validation_failure_includes_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = self._write(tmp_path, "version: 1\nname: broken\n")
        rc = main(["check", str(plan_path), "--json"])
        report = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert report["ok"] is False
        assert report["stage"] == "validate"
        assert "error" in report

    def test_check_strict_warns_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The dirty plan has W20 + SEC001 — without --strict, only SEC001
        # error counts. With --strict, even validation warnings escalate.
        plan_path = self._write(
            tmp_path,
            """\
version: 1
name: strict-test
max_cost_usd: 10.0
defaults:
  timeout_sec: 600
tasks:
  - id: t1
    engine: claude
    prompt: "do thing"
    max_retries: 2
""",
        )
        rc_default = main(["check", str(plan_path)])
        capsys.readouterr()  # drain
        rc_strict = main(["check", str(plan_path), "--strict"])
        out = capsys.readouterr().out
        # No audit errors → default exits 0; --strict escalates W20 to failure.
        assert rc_default == 0
        assert rc_strict == 1
        assert "W20" in out
