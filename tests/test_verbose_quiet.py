from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.cli import _build_parser, main


# Module-level plan YAML constant used across tests
_SIMPLE_PLAN_YAML = """\
version: 1
name: test-plan
defaults:
  timeout_sec: 60
tasks:
  - id: t1
    command: "echo hello"
"""


def _write_plan(tmp_path: Path, content: str = _SIMPLE_PLAN_YAML) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


# ===========================================================================
# Argument parsing
# ===========================================================================


class TestVerbosityArgParsing:
    def test_verbose_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--verbose"])
        assert args.verbose is True
        assert args.quiet is False

    def test_verbose_short_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "-v"])
        assert args.verbose is True

    def test_quiet_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--quiet"])
        assert args.quiet is True
        assert args.verbose is False

    def test_quiet_short_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "-q"])
        assert args.quiet is True

    def test_verbose_quiet_mutually_exclusive(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "plan.yaml", "--verbose", "--quiet"])

    def test_verbose_quiet_short_mutually_exclusive(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "plan.yaml", "-v", "-q"])

    def test_default_verbosity_is_normal(self) -> None:
        """No flags: args.verbose and args.quiet should both be False."""
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.verbose is False
        assert args.quiet is False

    def test_verbose_resolves_to_verbose_string(self, tmp_path: Path) -> None:
        """_cmd_run derives verbosity='verbose' when args.verbose=True."""
        # Check the logic inline (mirrors what _cmd_run does)
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--verbose"])
        verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"
        assert verbosity == "verbose"

    def test_quiet_resolves_to_quiet_string(self) -> None:
        """_cmd_run derives verbosity='quiet' when args.quiet=True."""
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--quiet"])
        verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"
        assert verbosity == "quiet"

    def test_default_resolves_to_normal_string(self) -> None:
        """_cmd_run derives verbosity='normal' when no flags given."""
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"
        assert verbosity == "normal"


# ===========================================================================
# Verbosity effect on output
# ===========================================================================


class TestVerbosityOutput:
    def test_quiet_suppresses_starting_lines(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """In quiet mode, 'starting' task lines should not appear."""
        plan_file = _write_plan(tmp_path)
        exit_code = main([
            "run", str(plan_file), "--dry-run", "--quiet",
            "--run-dir", str(tmp_path / "runs"),
        ])
        captured = capsys.readouterr()
        assert "starting" not in captured.out
        assert exit_code == 0

    def test_quiet_suppresses_meta_lines(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """In quiet mode, run_id= header lines should not appear."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file), "--dry-run", "--quiet",
            "--run-dir", str(tmp_path / "runs"),
        ])
        captured = capsys.readouterr()
        assert "run_id=" not in captured.out

    def test_quiet_still_shows_summary(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """In quiet mode, the final SUCCESS/FAILED summary should still appear."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file), "--dry-run", "--quiet",
            "--run-dir", str(tmp_path / "runs"),
        ])
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out or "FAILED" in captured.out

    def test_normal_shows_starting_lines(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """In normal mode, 'starting' task lines should appear."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file), "--dry-run",
            "--run-dir", str(tmp_path / "runs"),
        ])
        captured = capsys.readouterr()
        assert "starting" in captured.out

    def test_verbose_shows_starting_lines(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """In verbose mode, 'starting' task lines should appear."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file), "--dry-run", "--verbose",
            "--run-dir", str(tmp_path / "runs"),
        ])
        captured = capsys.readouterr()
        assert "starting" in captured.out

    def test_normal_shows_run_id_header(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """In normal mode, run_id= header should appear."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file), "--dry-run",
            "--run-dir", str(tmp_path / "runs"),
        ])
        captured = capsys.readouterr()
        assert "run_id=" in captured.out
