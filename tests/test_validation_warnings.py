from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.loader import _collect_warnings, load_plan
from maestro_cli.models import (
    EngineDefaults,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
)


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(tmp_path: Path, **overrides: Any) -> PlanSpec:
    defaults = {
        "version": 1,
        "name": "test",
        "max_parallel": 1,
        "fail_fast": True,
        "run_dir": (tmp_path / "runs").as_posix(),
        "defaults": PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
        ),
        "tasks": [],
    }
    defaults.update(overrides)
    return PlanSpec(**defaults)


# ===========================================================================
# Warning infrastructure
# ===========================================================================


class TestWarningInfrastructure:
    def test_warnings_field_default_empty(self) -> None:
        plan = PlanSpec(version=1, name="test")
        assert plan.validation_warnings == []

    def test_valid_plan_no_warnings(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    command: "echo hello"
    timeout_sec: 60
""")
        plan = load_plan(plan_file)
        # On non-Windows, no Windows-specific warnings; no backslashes; has timeout
        no_timeout_ws = [w for w in plan.validation_warnings if "timeout" in w.lower()]
        assert len(no_timeout_ws) == 0

    def test_warnings_populated_after_load(self, tmp_path: Path) -> None:
        """Load a plan with a known warning trigger and verify it's collected."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: nonexistent-model
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("nonexistent-model" in w for w in plan.validation_warnings)


# ===========================================================================
# Pitfall 3: Unicode in prompt_md_heading
# ===========================================================================


class TestUnicodeWarning:
    def test_non_ascii_heading_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                prompt_md_heading="W2: Overlays \u2014 Drawer",  # em-dash
                prompt_md_file="prompts.md",
            ),
        ])
        _collect_warnings(plan)
        assert any("non-ASCII" in w for w in plan.validation_warnings)

    def test_ascii_heading_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                prompt_md_heading="W2: Overlays -- Drawer",
                prompt_md_file="prompts.md",
            ),
        ])
        _collect_warnings(plan)
        assert not any("non-ASCII" in w for w in plan.validation_warnings)

    def test_em_dash_in_heading_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                prompt_md_heading="Title \u2013 Subtitle",  # en-dash
                prompt_md_file="prompts.md",
            ),
        ])
        _collect_warnings(plan)
        assert any("non-ASCII" in w for w in plan.validation_warnings)
        assert any("em-dash" in w for w in plan.validation_warnings)


# ===========================================================================
# Pitfall 4: Backslashes in path fields
# ===========================================================================


class TestPathBackslashWarning:
    def test_workspace_root_backslash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, workspace_root="C:\\path\\to\\dir")
        _collect_warnings(plan)
        assert any("workspace_root" in w and "backslash" in w for w in plan.validation_warnings)

    def test_workdir_backslash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo", workdir="C:\\path\\to\\dir"),
        ])
        _collect_warnings(plan)
        assert any("workdir" in w and "backslash" in w for w in plan.validation_warnings)

    def test_forward_slashes_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, workspace_root="C:/path/to/dir", tasks=[
            TaskSpec(id="t1", command="echo", workdir="C:/path/to/dir"),
        ])
        _collect_warnings(plan)
        assert not any("backslash" in w for w in plan.validation_warnings)

    def test_prompt_file_backslash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt_file="C:\\prompts\\task.txt",
            ),
        ])
        _collect_warnings(plan)
        assert any("prompt_file" in w and "backslash" in w for w in plan.validation_warnings)

    def test_prompt_md_file_backslash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                prompt_md_file="C:\\prompts\\tasks.md",
                prompt_md_heading="Heading",
            ),
        ])
        _collect_warnings(plan)
        assert any("prompt_md_file" in w and "backslash" in w for w in plan.validation_warnings)


# ===========================================================================
# Pitfall 8: Timeout defaults
# ===========================================================================


class TestTimeoutWarning:
    def test_no_timeout_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello"),
        ])
        _collect_warnings(plan)
        assert any("timeout" in w.lower() and "t1" in w for w in plan.validation_warnings)

    def test_explicit_timeout_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello", timeout_sec=60),
        ])
        _collect_warnings(plan)
        assert not any("timeout" in w.lower() and "t1" in w for w in plan.validation_warnings)

    def test_defaults_timeout_no_warning(self, tmp_path: Path) -> None:
        defaults = PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
            timeout_sec=300,
        )
        plan = _make_plan(tmp_path, defaults=defaults, tasks=[
            TaskSpec(id="t1", command="echo hello"),
        ])
        _collect_warnings(plan)
        assert not any("timeout" in w.lower() and "t1" in w for w in plan.validation_warnings)

    def test_max_warnings_capped(self, tmp_path: Path) -> None:
        """Many tasks without timeout should cap at _MAX_TIMEOUT_WARNINGS + summary."""
        tasks = [TaskSpec(id=f"t{i}", command="echo") for i in range(10)]
        plan = _make_plan(tmp_path, tasks=tasks)
        _collect_warnings(plan)
        timeout_ws = [w for w in plan.validation_warnings if "timeout" in w.lower()]
        # Should be at most 3 individual + 1 "... and N more"
        assert len(timeout_ws) <= 4
        assert any("more task(s)" in w for w in timeout_ws)


# ===========================================================================
# Pitfall 1: Windows shell execution (monkeypatched)
# ===========================================================================


class TestWindowsShellWarning:
    def test_string_command_warns_on_windows(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello"),
        ])
        _collect_warnings(plan)
        assert any("shell=True" in w and "t1" in w for w in plan.validation_warnings)

    def test_list_command_no_shell_warning(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["C:\\Program Files\\Git\\bin\\bash.exe", "-c", "echo"]),
        ])
        _collect_warnings(plan)
        assert not any("shell=True" in w for w in plan.validation_warnings)

    def test_usr_bin_bash_warns(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=[
                "C:\\Program Files\\Git\\usr\\bin\\bash.exe", "-c", "echo",
            ]),
        ])
        _collect_warnings(plan)
        assert any("usr\\bin\\bash" in w.lower() or "usr/bin/bash" in w.lower()
                    for w in plan.validation_warnings)

    def test_bin_bash_no_warning(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=[
                "C:\\Program Files\\Git\\bin\\bash.exe", "-c", "echo",
            ]),
        ])
        _collect_warnings(plan)
        assert not any("usr" in w.lower() and "bash" in w.lower()
                       for w in plan.validation_warnings)

    def test_pre_command_usr_bin_bash_warns(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo", pre_command=[
                "C:\\Program Files\\Git\\usr\\bin\\bash.exe", "-c", "echo",
            ]),
        ])
        _collect_warnings(plan)
        assert any("pre_command" in w for w in plan.validation_warnings)

    def test_no_warning_on_non_windows(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "posix")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello"),
        ])
        _collect_warnings(plan)
        assert not any("shell=True" in w for w in plan.validation_warnings)

    def test_windows_shell_warning_suggests_git_bash(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Shell=True warning on Windows should suggest using Git Bash."""
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello"),
        ])
        _collect_warnings(plan)
        shell_warnings = [w for w in plan.validation_warnings if "shell=True" in w]
        assert len(shell_warnings) >= 1
        # The warning should suggest using Git Bash as an alternative
        assert any("Git Bash" in w for w in shell_warnings)

    def test_wrong_bash_binary_warning_improved(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Wrong bash binary warning should suggest the correct Git\\bin\\bash.exe path."""
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=[
                "C:\\Program Files\\Git\\usr\\bin\\bash.exe", "-c", "echo",
            ]),
        ])
        _collect_warnings(plan)
        bash_warnings = [
            w for w in plan.validation_warnings
            if "usr" in w.lower() and "bash" in w.lower()
        ]
        assert len(bash_warnings) >= 1
        # Should suggest the correct alternative binary
        assert any("bin\\bash.exe" in w or "bin/bash.exe" in w for w in bash_warnings)


# ===========================================================================
# Pitfall 7: Dry run checklist (CLI output)
# ===========================================================================


class TestDryRunChecklist:
    def test_dry_run_shows_checklist(self, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
        """Dry-run should print a NOT-validated checklist."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 60
tasks:
  - id: t1
    command: "echo hello"
""")
        monkeypatch.setattr("sys.argv", [
            "maestro", "run", str(plan_file), "--dry-run",
        ])
        from maestro_cli.cli import main
        exit_code = main()
        captured = capsys.readouterr()
        assert "dry-run checklist" in captured.out
        assert "Engine CLIs on PATH" in captured.out
        assert exit_code == 0

    def test_non_dry_run_no_checklist(self, tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
        """Normal validate should NOT print dry-run checklist."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 60
tasks:
  - id: t1
    command: "echo hello"
""")
        monkeypatch.setattr("sys.argv", [
            "maestro", "validate", str(plan_file),
        ])
        from maestro_cli.cli import main
        main()
        captured = capsys.readouterr()
        assert "dry-run checklist" not in captured.out


# ===========================================================================
# Migrated warnings (from warnings.warn → validation_warnings)
# ===========================================================================


class TestMigratedWarnings:
    def test_unknown_codex_model_in_warnings(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    model: bad-model
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("bad-model" in w and "may not be valid" in w
                    for w in plan.validation_warnings)

    def test_unknown_claude_model_in_warnings(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: gpt-4
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("gpt-4" in w and "may not be valid" in w
                    for w in plan.validation_warnings)

    def test_edit_policy_on_shell_in_warnings(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    edit_policy: efficient
""")
        plan = load_plan(plan_file)
        assert any("no effect on shell" in w for w in plan.validation_warnings)


# ===========================================================================
# W1: guard_command in shell/bash warning loop
# ===========================================================================


class TestGuardCommandShellWarning:
    def test_string_guard_command_warns_on_windows(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude", prompt="test",
                guard_command="grep -q OK",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "guard_command" in w and "shell=True" in w
            for w in plan.validation_warnings
        )

    def test_guard_command_wrong_bash_warns(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude", prompt="test",
                guard_command=[
                    "C:\\Program Files\\Git\\usr\\bin\\bash.exe",
                    "-c", "echo ok",
                ],
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "guard_command" in w and "usr" in w.lower()
            for w in plan.validation_warnings
        )

    def test_guard_command_correct_bash_no_warning(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude", prompt="test",
                guard_command=[
                    "C:\\Program Files\\Git\\bin\\bash.exe",
                    "-c", "echo ok",
                ],
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "guard_command" in w and "usr" in w.lower()
            for w in plan.validation_warnings
        )


# ===========================================================================
# W2: prompt_md_heading starts with '#'
# ===========================================================================


class TestHeadingHashPrefixWarning:
    def test_heading_with_hash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude", prompt="test",
                prompt_md_heading="## My Heading",
                prompt_md_file="prompts.md",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "starts with '#'" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_heading_with_single_hash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude", prompt="test",
                prompt_md_heading="# My Heading",
                prompt_md_file="prompts.md",
            ),
        ])
        _collect_warnings(plan)
        assert any("starts with '#'" in w for w in plan.validation_warnings)

    def test_heading_without_hash_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude", prompt="test",
                prompt_md_heading="My Heading",
                prompt_md_file="prompts.md",
            ),
        ])
        _collect_warnings(plan)
        assert not any("starts with '#'" in w for w in plan.validation_warnings)


# ===========================================================================
# W3: Unrecognised template variables
# ===========================================================================


class TestTemplateVarWarning:
    def test_known_global_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ workspace_root }} do something",
            ),
        ])
        _collect_warnings(plan)
        assert not any("template variable" in w for w in plan.validation_warnings)

    def test_known_context_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="dep1", command="echo ok"),
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ dep1.status }} and {{ dep1.stdout_tail }}",
                depends_on=["dep1"],
                context_from=["dep1"],
            ),
        ])
        _collect_warnings(plan)
        assert not any("template variable" in w for w in plan.validation_warnings)

    def test_matrix_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ matrix.ENV }} deploy",
            ),
        ])
        _collect_warnings(plan)
        assert not any("template variable" in w for w in plan.validation_warnings)

    def test_misspelled_var_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ workspace_rooot }} do something",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "workspace_rooot" in w and "template variable" in w
            for w in plan.validation_warnings
        )

    def test_misspelled_context_suffix_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="dep1", command="echo ok"),
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ dep1.statuss }}",
                depends_on=["dep1"],
                context_from=["dep1"],
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "dep1.statuss" in w and "template variable" in w
            for w in plan.validation_warnings
        )

    def test_unknown_task_prefix_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ nonexistent.status }}",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "nonexistent.status" in w and "template variable" in w
            for w in plan.validation_warnings
        )

    def test_upstream_synthesis_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", engine="claude",
                prompt="{{ upstream_synthesis }} summarize",
            ),
        ])
        _collect_warnings(plan)
        assert not any("template variable" in w for w in plan.validation_warnings)

    def test_contract_template_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="schema", command="echo schema", contract_type="sql-schema"),
            TaskSpec(
                id="repo",
                engine="claude",
                prompt="{{ contract.schema.summary }}",
                consumes_contracts=["schema"],
            ),
        ])
        _collect_warnings(plan)
        assert not any("template variable" in w for w in plan.validation_warnings)

    def test_consistency_template_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="controller", command="echo ok", consistency_group=["di"]),
            TaskSpec(
                id="reconcile",
                engine="claude",
                prompt="{{ consistency.di.statuses }}",
                reconcile_after=["di"],
            ),
        ])
        _collect_warnings(plan)
        assert not any("template variable" in w for w in plan.validation_warnings)


# ===========================================================================
# W4: run_dir backslash check
# ===========================================================================


class TestRunDirBackslashWarning:
    def test_run_dir_backslash_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, run_dir="C:\\output\\runs")
        _collect_warnings(plan)
        assert any(
            "run_dir" in w and "backslash" in w
            for w in plan.validation_warnings
        )

    def test_run_dir_forward_slash_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, run_dir="C:/output/runs")
        _collect_warnings(plan)
        assert not any("run_dir" in w for w in plan.validation_warnings)


# ===========================================================================
# W5: Bash-only syntax in string commands (Windows)
# ===========================================================================


class TestBashSyntaxWarning:
    def test_heredoc_in_command_warns(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command="py << 'PYEOF'\nprint('hello')\nPYEOF",
            ),
        ])
        _collect_warnings(plan)
        assert any("heredoc" in w.lower() for w in plan.validation_warnings)

    def test_process_substitution_warns(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command="diff <(cmd1) <(cmd2)",
            ),
        ])
        _collect_warnings(plan)
        assert any("process substitution" in w for w in plan.validation_warnings)

    def test_no_bash_syntax_no_extra_warning(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello"),
        ])
        _collect_warnings(plan)
        assert not any("heredoc" in w.lower() for w in plan.validation_warnings)
        assert not any("process substitution" in w for w in plan.validation_warnings)

    def test_heredoc_in_verify_command_warns(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", command=["echo", "ok"],
                verify_command="py << 'EOF'\npass\nEOF",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "verify_command" in w and "heredoc" in w.lower()
            for w in plan.validation_warnings
        )

    def test_not_on_linux(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "posix")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command="py << 'PYEOF'\nprint('hello')\nPYEOF",
            ),
        ])
        _collect_warnings(plan)
        assert not any("heredoc" in w.lower() for w in plan.validation_warnings)


# ===========================================================================
# W6: retry_delay_sec list shorter than max_retries
# ===========================================================================


class TestRetryDelayLengthWarning:
    def test_short_list_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", command="echo",
                max_retries=3, retry_delay_sec=[1.0],
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "retry_delay_sec" in w and "1 value" in w and "max_retries is 3" in w
            for w in plan.validation_warnings
        )

    def test_matching_list_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", command="echo",
                max_retries=2, retry_delay_sec=[1.0, 2.0],
            ),
        ])
        _collect_warnings(plan)
        assert not any("retry_delay_sec" in w for w in plan.validation_warnings)

    def test_float_delay_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1", command="echo",
                max_retries=3, retry_delay_sec=5.0,
            ),
        ])
        _collect_warnings(plan)
        assert not any("retry_delay_sec" in w for w in plan.validation_warnings)


# ===========================================================================
# W7: Environment variable references vs available env
# ===========================================================================


class TestEnvRefWarning:
    def test_unknown_var_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command="echo $MY_CUSTOM_SECRET",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "MY_CUSTOM_SECRET" in w and "not in the env allowlist" in w
            for w in plan.validation_warnings
        )

    def test_allowlisted_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo $PATH"),
        ])
        _collect_warnings(plan)
        assert not any(
            "PATH" in w and "not in the env" in w
            for w in plan.validation_warnings
        )

    def test_plan_env_var_no_warning(self, tmp_path: Path) -> None:
        defaults = PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
            env={"MY_API_KEY": "xxx"},
        )
        plan = _make_plan(tmp_path, defaults=defaults, tasks=[
            TaskSpec(id="t1", command="echo $MY_API_KEY"),
        ])
        _collect_warnings(plan)
        assert not any(
            "MY_API_KEY" in w and "not in the env" in w
            for w in plan.validation_warnings
        )

    def test_task_env_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command="echo $DEPLOY_TARGET",
                env={"DEPLOY_TARGET": "prod"},
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "DEPLOY_TARGET" in w and "not in the env" in w
            for w in plan.validation_warnings
        )

    def test_bash_special_var_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo $? $! $@"),
        ])
        _collect_warnings(plan)
        assert not any("not in the env" in w for w in plan.validation_warnings)

    def test_braced_var_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo ${UNKNOWN_VAR}"),
        ])
        _collect_warnings(plan)
        assert any(
            "UNKNOWN_VAR" in w and "not in the env" in w
            for w in plan.validation_warnings
        )

    def test_list_command_not_checked(self, tmp_path: Path) -> None:
        """List commands are not checked (they don't use shell expansion)."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "$UNKNOWN_VAR"]),
        ])
        _collect_warnings(plan)
        assert not any(
            "UNKNOWN_VAR" in w and "not in the env" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W-pipes: pipe character in string commands (Windows only)
# ===========================================================================


class TestPipeWarning:
    def test_pipe_in_string_command_warns_on_windows(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello | grep hello"),
        ])
        _collect_warnings(plan)
        assert any(
            "pipe '|'" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_pipe_in_verify_command_warns_on_windows(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "hello"],
                verify_command="cat output.txt | grep ok",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "pipe '|'" in w and "verify_command" in w
            for w in plan.validation_warnings
        )

    def test_pipe_prefix_warns_on_windows(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="| sort -u"),
        ])
        _collect_warnings(plan)
        assert any("pipe '|'" in w for w in plan.validation_warnings)

    def test_double_pipe_or_no_warning_on_windows(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """|| (logical OR) should NOT trigger the pipe warning."""
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="cmd1 || cmd2"),
        ])
        _collect_warnings(plan)
        assert not any("pipe '|'" in w for w in plan.validation_warnings)

    def test_pipe_no_warning_on_linux(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.loader.os.name", "posix")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command="echo hello | grep hello"),
        ])
        _collect_warnings(plan)
        assert not any("pipe '|'" in w for w in plan.validation_warnings)

    def test_list_command_with_pipe_arg_no_warning(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """List commands are passed literally; no shell pipe processing."""
        monkeypatch.setattr("maestro_cli.loader.os.name", "nt")
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["grep", "|", "file.txt"]),
        ])
        _collect_warnings(plan)
        assert not any("pipe '|'" in w for w in plan.validation_warnings)


# ===========================================================================
# W-multiline-string-verify: multiline py -c in string verify_command
# ===========================================================================


class TestMultilinePyCWarning:
    def test_multiline_py_c_in_verify_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command="py -c \"\nimport sys\nsys.exit(0)\n\"",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "verify_command" in w and "multiline" in w and "py -c" in w
            for w in plan.validation_warnings
        )

    def test_multiline_python_c_in_verify_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command="python -c \"\nimport os\nprint(os.getcwd())\n\"",
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "verify_command" in w and "python -c" in w
            for w in plan.validation_warnings
        )

    def test_single_line_py_c_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command="py -c \"import sys; sys.exit(0)\"",
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "multiline" in w and "py -c" in w
            for w in plan.validation_warnings
        )

    def test_multiline_without_py_c_no_warning(self, tmp_path: Path) -> None:
        """A multiline verify_command without py -c should not warn."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command="grep -q ok\necho done",
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "multiline" in w and "py -c" in w
            for w in plan.validation_warnings
        )

    def test_list_verify_command_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command=["py", "-c", "import sys\nsys.exit(0)"],
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "multiline" in w and "py -c" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W-no-retry-with-verify: verify_command set but max_retries=0
# ===========================================================================


class TestNoRetryWithVerifyWarning:
    def test_verify_no_retry_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command=["test", "-f", "output.txt"],
                max_retries=0,
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "verify_command" in w and "max_retries=0" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_verify_with_retries_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                verify_command=["test", "-f", "output.txt"],
                max_retries=1,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "verify_command" in w and "max_retries=0" in w
            for w in plan.validation_warnings
        )

    def test_no_verify_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "ok"], max_retries=0),
        ])
        _collect_warnings(plan)
        assert not any(
            "max_retries=0" in w and "t1" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W-assert-no-retry: assert set but max_retries=0
# ===========================================================================


class TestNoRetryWithAssertWarning:
    def test_assert_no_retry_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Implement feature",
                assertions=[
                    {
                        "type": "file_contains",
                        "path": "src/app.py",
                        "pattern": "main",
                    }
                ],
                max_retries=0,
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "assert rules" in w and "max_retries=0" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_assert_with_retries_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Implement feature",
                assertions=[
                    {
                        "type": "file_contains",
                        "path": "src/app.py",
                        "pattern": "main",
                    }
                ],
                max_retries=1,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "assert rules" in w and "max_retries=0" in w
            for w in plan.validation_warnings
        )

    def test_shell_task_assert_no_retry_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["echo", "ok"],
                assertions=[
                    {
                        "type": "file_contains",
                        "path": "src/app.py",
                        "pattern": "main",
                    }
                ],
                max_retries=0,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "assert rules" in w and "t1" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W-judge-retry-no-iterations: judge on_fail=retry without max_iterations
# ===========================================================================


class TestJudgeRetryNoIterationsWarning:
    def test_judge_retry_no_iterations_warns(self, tmp_path: Path) -> None:
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Do something",
                judge=JudgeSpec(criteria=["Output is correct"], on_fail="retry"),
                max_iterations=None,
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "judge" in w and "on_fail='retry'" in w and "max_iterations" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_judge_retry_with_iterations_no_warning(self, tmp_path: Path) -> None:
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Do something",
                judge=JudgeSpec(criteria=["Output is correct"], on_fail="retry"),
                max_iterations=5,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "on_fail='retry'" in w and "max_iterations" in w
            for w in plan.validation_warnings
        )

    def test_judge_fail_no_iterations_no_warning(self, tmp_path: Path) -> None:
        """on_fail='fail' with no max_iterations is fine."""
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Do something",
                judge=JudgeSpec(criteria=["Output is correct"], on_fail="fail"),
                max_iterations=None,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "on_fail='retry'" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W-timeout-retry-futility: explicit timeout_sec with max_retries > 0
# ===========================================================================


class TestRetryDesignWarning:
    """Unified W20 (consolidated 2026-04-26) — retries without an escape valve.

    Replaces the legacy W20 + W21 + W-timeout-retry-futility chain that
    contradicted itself when authors had verify_command set (internal post-mortem).
    """

    def test_engine_retries_no_escape_valve_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                timeout_sec=300,
                max_retries=2,
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "W20" in w and "t1" in w for w in plan.validation_warnings
        )

    def test_command_task_warns_without_escape_valve(self, tmp_path: Path) -> None:
        # Shell tasks are equally vulnerable to retry-without-escape futility:
        # `sleep 999 timeout 30 retries 2` is exactly the timeout-loop case.
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command=["sleep", "999"],
                timeout_sec=30,
                max_retries=2,
            ),
        ])
        _collect_warnings(plan)
        msg = next((w for w in plan.validation_warnings if "W20" in w), "")
        assert msg
        assert "t1" in msg
        # Engine-only valves not suggested for shell tasks.
        assert "escalation" not in msg
        assert "fallback_engine" not in msg

    def test_no_retries_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                timeout_sec=60,
                max_retries=0,
            ),
        ])
        _collect_warnings(plan)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_verify_command_silences_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                timeout_sec=300,
                max_retries=2,
                verify_command=["test", "-f", "out.txt"],
            ),
        ])
        _collect_warnings(plan)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_progressive_delay_silences_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                timeout_sec=300,
                max_retries=2,
                retry_delay_sec=[60, 120],
            ),
        ])
        _collect_warnings(plan)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_escalation_silences_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                timeout_sec=300,
                max_retries=2,
                escalation=["haiku", "sonnet", "opus"],
            ),
        ])
        _collect_warnings(plan)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_fallback_engine_silences_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="test",
                timeout_sec=300,
                max_retries=1,
                fallback_engine="codex",
            ),
        ])
        _collect_warnings(plan)
        assert not any("W20" in w for w in plan.validation_warnings)


# ===========================================================================
# W-context-no-budget: context_from without context_budget_tokens
# ===========================================================================


class TestContextNoBudgetWarning:
    def test_context_from_no_budget_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="dep", command=["echo", "ok"]),
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Use upstream",
                depends_on=["dep"],
                context_from=["dep"],
                context_budget_tokens=None,
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "context_from" in w and "context_budget_tokens" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_context_from_with_task_budget_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="dep", command=["echo", "ok"]),
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Use upstream",
                depends_on=["dep"],
                context_from=["dep"],
                context_budget_tokens=4000,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "context_from" in w and "context_budget_tokens" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_context_from_with_plan_budget_no_warning(self, tmp_path: Path) -> None:
        defaults = PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
            context_budget_tokens=8000,
        )
        plan = _make_plan(tmp_path, defaults=defaults, tasks=[
            TaskSpec(id="dep", command=["echo", "ok"]),
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Use upstream",
                depends_on=["dep"],
                context_from=["dep"],
                context_budget_tokens=None,
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "context_from" in w and "context_budget_tokens" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_no_context_from_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", engine="claude", prompt="Standalone task"),
        ])
        _collect_warnings(plan)
        assert not any(
            "context_from" in w and "context_budget_tokens" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W-judge-contains-codex: judge 'contains' assertion on codex engine
# ===========================================================================


class TestJudgeContainsEngineWarning:
    def test_contains_assertion_on_codex_warns(self, tmp_path: Path) -> None:
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="codex",
                prompt="Write code",
                judge=JudgeSpec(
                    criteria=[{"type": "contains", "value": "def main"}],
                ),
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "judge 'contains' assertion on engine task" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_contains_on_claude_also_warns(self, tmp_path: Path) -> None:
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="claude",
                prompt="Write code",
                judge=JudgeSpec(
                    criteria=[{"type": "contains", "value": "def main"}],
                ),
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "judge 'contains' assertion on engine task" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_regex_on_engine_warns(self, tmp_path: Path) -> None:
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="gemini",
                prompt="Write code",
                judge=JudgeSpec(
                    criteria=[{"type": "regex", "pattern": "passed"}],
                ),
            ),
        ])
        _collect_warnings(plan)
        assert any(
            "judge 'regex' assertion on engine task" in w and "t1" in w
            for w in plan.validation_warnings
        )

    def test_contains_on_shell_task_no_warning(self, tmp_path: Path) -> None:
        """Shell tasks (no engine) should NOT trigger the warning."""
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                command="echo hello",
                judge=JudgeSpec(
                    criteria=[{"type": "contains", "value": "hello"}],
                ),
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "assertion on engine task" in w
            for w in plan.validation_warnings
        )

    def test_llm_rubric_on_engine_no_warning(self, tmp_path: Path) -> None:
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="codex",
                prompt="Write code",
                judge=JudgeSpec(
                    criteria=[{"type": "llm-rubric", "value": "Code is well-structured"}],
                ),
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "assertion on engine task" in w
            for w in plan.validation_warnings
        )

    def test_multiple_contains_only_one_warning(self, tmp_path: Path) -> None:
        """break after first contains — should emit exactly one warning per task."""
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="codex",
                prompt="Write code",
                judge=JudgeSpec(
                    criteria=[
                        {"type": "contains", "value": "def main"},
                        {"type": "contains", "value": "return"},
                    ],
                ),
            ),
        ])
        _collect_warnings(plan)
        matching = [w for w in plan.validation_warnings if "assertion on engine task" in w]
        assert len(matching) == 1

    def test_string_criteria_on_engine_no_warning(self, tmp_path: Path) -> None:
        """Plain string criteria (LLM-evaluated) on engine should not warn."""
        from maestro_cli.models import JudgeSpec
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(
                id="t1",
                engine="codex",
                prompt="Write code",
                judge=JudgeSpec(
                    criteria=["Output is correct", "Code compiles"],
                ),
            ),
        ])
        _collect_warnings(plan)
        assert not any(
            "assertion on engine task" in w
            for w in plan.validation_warnings
        )


# ===========================================================================
# W29: Codex runtime entitlement warning
# ===========================================================================


class TestCodexRuntimeEntitlementWarning:
    def test_fail_fast_codex_without_fallback_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, fail_fast=True, tasks=[
            TaskSpec(id="t1", engine="codex", prompt="Write code"),
            TaskSpec(id="t2", command=["echo", "ok"]),
        ])
        _collect_warnings(plan)
        assert any(
            w.startswith("W29:") and "t1" in w and "fallback_engine" in w
            for w in plan.validation_warnings
        )

    def test_codex_with_inherited_fallback_no_warning(self, tmp_path: Path) -> None:
        defaults = PlanDefaults(
            codex=EngineDefaults(fallback_engine="claude", fallback_model="sonnet"),
            claude=EngineDefaults(),
        )
        plan = _make_plan(tmp_path, defaults=defaults, fail_fast=True, tasks=[
            TaskSpec(id="t1", engine="codex", prompt="Write code"),
        ])
        _collect_warnings(plan)
        assert not any(w.startswith("W29:") for w in plan.validation_warnings)

    def test_non_fail_fast_codex_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, fail_fast=False, tasks=[
            TaskSpec(id="t1", engine="codex", prompt="Write code"),
        ])
        _collect_warnings(plan)
        assert not any(w.startswith("W29:") for w in plan.validation_warnings)


# ===========================================================================
# W30: repo-wide TypeScript compile gate warning
# ===========================================================================


class TestRepoWideTypescriptGateWarning:
    def test_tsc_no_emit_command_warns_when_it_blocks_downstream(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, fail_fast=True, tasks=[
            TaskSpec(id="build", command=["npx", "tsc", "--noEmit"]),
            TaskSpec(id="review", command=["echo", "ok"], depends_on=["build"]),
        ])
        _collect_warnings(plan)
        assert any(
            w.startswith("W30:") and "build" in w and "baseline compile" in w
            for w in plan.validation_warnings
        )

    def test_tsc_no_emit_verify_command_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, fail_fast=False, tasks=[
            TaskSpec(
                id="build",
                command=["echo", "generated"],
                verify_command=["pnpm", "exec", "tsc", "--noEmit"],
            ),
            TaskSpec(id="review", command=["echo", "ok"], depends_on=["build"]),
        ])
        _collect_warnings(plan)
        assert any(
            w.startswith("W30:") and "verify_command" in w
            for w in plan.validation_warnings
        )

    def test_tsc_no_emit_allow_failure_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, fail_fast=True, tasks=[
            TaskSpec(
                id="build",
                command=["npx", "tsc", "--noEmit"],
                allow_failure=True,
            ),
        ])
        _collect_warnings(plan)
        assert not any(w.startswith("W30:") for w in plan.validation_warnings)


# ===========================================================================
# W26: overlapping output_scope warning
# ===========================================================================


class TestOutputScopeOverlapWarning:
    def test_identical_literal_output_scope_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="register", command=["echo", "ok"], output_scope=["src/Config/dependencies.php"]),
            TaskSpec(id="update", command=["echo", "ok"], output_scope=["src/Config/dependencies.php"]),
        ])
        _collect_warnings(plan)
        w26 = [w for w in plan.validation_warnings if w.startswith("W26:")]
        assert len(w26) == 1
        assert "register" in w26[0]
        assert "update" in w26[0]
        assert "src/Config/dependencies.php" in w26[0]

    def test_nested_glob_and_literal_output_scope_warns(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="broad", command=["echo", "ok"], output_scope=["src/**/*.py"]),
            TaskSpec(id="narrow", command=["echo", "ok"], output_scope=["src/auth/service.py"]),
        ])
        _collect_warnings(plan)
        assert any(w.startswith("W26:") and "src/**/*.py" in w for w in plan.validation_warnings)

    def test_disjoint_output_scope_no_warning(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="backend", command=["echo", "ok"], output_scope=["src/**/*.py"]),
            TaskSpec(id="docs", command=["echo", "ok"], output_scope=["docs/**/*.md"]),
        ])
        _collect_warnings(plan)
        assert not any(w.startswith("W26:") for w in plan.validation_warnings)
