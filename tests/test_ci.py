from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maestro_cli.ci import (
    CiWorkflowSpec,
    build_ci_workflow_spec,
    default_ci_output_path,
    generate_ci_yaml,
    render_github_actions,
    render_gitlab_ci,
)
from maestro_cli.cli import _build_parser, main


_VALID_PLAN_YAML = """\
version: 1
name: ci-demo
tasks:
  - id: t1
    command: "echo hello"
"""


def _write_plan(tmp_path: Path, name: str = "plan.yaml") -> Path:
    plan_file = tmp_path / name
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")
    return plan_file


def _sample_spec() -> CiWorkflowSpec:
    return CiWorkflowSpec(
        plan_name="ci-demo",
        plan_path="plans/demo.yaml",
        workflow_name="Demo CI",
        python_version="3.12",
        bootstrap_command="python -m pip install --upgrade pip",
        install_command="python -m pip install -e . pytest",
        validate_command="maestro validate plans/demo.yaml",
        test_command="python -m pytest -q",
        maestro_command="maestro run plans/demo.yaml --auto-approve",
    )


def test_build_parser_supports_ci_command() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "ci",
            "plan.yaml",
            "--provider",
            "gitlab",
            "--workflow-name",
            "Release",
            "--python-version",
            "3.12",
            "--test-command",
            "python -m pytest tests/unit -q",
            "--output",
            ".gitlab-ci.yml",
        ]
    )
    assert args.command == "ci"
    assert args.plan == "plan.yaml"
    assert args.provider == "gitlab"
    assert args.workflow_name == "Release"
    assert args.python_version == "3.12"
    assert args.test_command == "python -m pytest tests/unit -q"
    assert args.output == ".gitlab-ci.yml"


def test_build_ci_workflow_spec_validates_plan_and_normalizes_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_file = _write_plan(tmp_path, "plans/demo.yaml")
    monkeypatch.chdir(tmp_path)

    spec = build_ci_workflow_spec(
        "plans/demo.yaml",
        workflow_name="Repo CI",
        python_version="3.12",
        test_command="python -m pytest tests/unit -q",
    )

    assert spec.plan_name == "ci-demo"
    assert spec.plan_path == "plans/demo.yaml"
    assert spec.workflow_name == "Repo CI"
    assert spec.python_version == "3.12"
    assert spec.validate_command == "maestro validate plans/demo.yaml"
    assert spec.test_command == "python -m pytest tests/unit -q"
    assert spec.maestro_command == "maestro run plans/demo.yaml --auto-approve"
    assert plan_file.exists()


def test_build_ci_workflow_spec_quotes_plan_paths_portably_for_all_platforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_plan(tmp_path, "plans/demo plan.yaml")
    monkeypatch.chdir(tmp_path)

    spec = build_ci_workflow_spec("plans/demo plan.yaml")

    assert spec.plan_path == "plans/demo plan.yaml"
    assert spec.validate_command == 'maestro validate "plans/demo plan.yaml"'
    assert spec.maestro_command == 'maestro run "plans/demo plan.yaml" --auto-approve'


def test_build_ci_workflow_spec_rejects_shell_sensitive_plan_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_plan(tmp_path, "plans/demo$plan.yaml")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="not portable across Windows, macOS, and Linux CI shells"):
        build_ci_workflow_spec("plans/demo$plan.yaml")


def test_generate_ci_yaml_rejects_plan_outside_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plan_dir = tmp_path / "external"
    plan_dir.mkdir()
    plan_file = _write_plan(plan_dir)
    monkeypatch.chdir(workspace)

    with pytest.raises(ValueError, match="current working directory"):
        generate_ci_yaml(plan_file)


def test_render_github_actions_includes_validation_and_test_lanes() -> None:
    rendered = render_github_actions(_sample_spec())
    parsed = yaml.load(rendered, Loader=yaml.BaseLoader)
    jobs = parsed["jobs"]
    assert "name: 'Demo CI'" in rendered
    assert "#   platform | validate job     | test job      | runner          | coverage" in rendered
    assert "#   linux    | validate_linux   | test_linux    | ubuntu-latest   | required" in rendered
    assert "#   windows  | validate_windows | test_windows  | windows-latest  | required" in rendered
    assert "#   macos    | validate_macos   | test_macos    | macos-latest    | required" in rendered
    assert '["C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "python -m pytest -q"]' in rendered
    assert 'bash -lc "python -m pytest -q"' in rendered
    assert "Inside Git Bash snippets, keep paths in forward-slash form such as `/c/project`." in rendered
    assert "validate_linux:" in rendered
    assert "validate_windows:" in rendered
    assert "validate_macos:" in rendered
    assert "test_linux:" in rendered
    assert "test_windows:" in rendered
    assert "test_macos:" in rendered
    assert jobs["validate_windows"]["runs-on"] == "windows-latest"
    assert jobs["validate_macos"]["runs-on"] == "macos-latest"
    assert jobs["test_windows"]["needs"] == ["validate_windows"]
    assert jobs["test_macos"]["needs"] == ["validate_macos"]
    assert {job_name.removeprefix("validate_") for job_name in jobs if job_name.startswith("validate_")} == {
        "linux",
        "windows",
        "macos",
    }
    assert {job_name.removeprefix("test_") for job_name in jobs if job_name.startswith("test_")} == {
        "linux",
        "windows",
        "macos",
    }
    assert rendered.count("| required") == 3
    assert "workflow_dispatch:" in rendered
    assert "run-real-engine:" in rendered
    assert "maestro validate plans/demo.yaml" in rendered
    assert "python -m pytest -q" in rendered
    assert "maestro run plans/demo.yaml --auto-approve" in rendered


def test_render_gitlab_ci_includes_validation_and_test_lanes() -> None:
    rendered = render_gitlab_ci(_sample_spec())
    parsed = yaml.load(rendered, Loader=yaml.BaseLoader)
    assert "#   platform | validate job     | test job      | runner                  | coverage" in rendered
    assert "#   linux    | validate_linux   | test_linux    | shared/container runner | default-on" in rendered
    assert "#   windows  | validate_windows | test_windows  | tagged windows runner   | opt-in" in rendered
    assert "#   macos    | validate_macos   | test_macos    | tagged macos runner     | opt-in" in rendered
    assert '["C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "python -m pytest -q"]' in rendered
    assert 'bash -lc "python -m pytest -q"' in rendered
    assert "Inside Git Bash snippets, keep paths in forward-slash form such as `/c/project`." in rendered
    assert "stages:" in rendered
    assert "validate_linux:" in rendered
    assert "validate_windows:" in rendered
    assert "validate_macos:" in rendered
    assert "test_linux:" in rendered
    assert "test_windows:" in rendered
    assert "test_macos:" in rendered
    assert parsed["validate_windows"]["extends"] == ".windows_job"
    assert parsed["validate_macos"]["extends"] == ".macos_job"
    assert parsed["test_windows"]["needs"] == [{"job": "validate_windows"}]
    assert parsed["test_macos"]["needs"] == [{"job": "validate_macos"}]
    assert {job_name.removeprefix("validate_") for job_name in parsed if job_name.startswith("validate_")} == {
        "linux",
        "windows",
        "macos",
    }
    assert {job_name.removeprefix("test_") for job_name in parsed if job_name.startswith("test_")} == {
        "linux",
        "windows",
        "macos",
    }
    assert rendered.count("| opt-in") == 2
    assert "run_maestro_real_engine:" in rendered
    assert "maestro validate plans/demo.yaml" in rendered
    assert "python -m pytest -q" in rendered
    assert "maestro run plans/demo.yaml --auto-approve" in rendered


def test_default_ci_output_paths_are_provider_specific() -> None:
    assert default_ci_output_path("github") == Path(".github/workflows/maestro.yml")
    assert default_ci_output_path("gitlab") == Path(".gitlab-ci.yml")


def test_main_ci_prints_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_plan(tmp_path)
    monkeypatch.chdir(tmp_path)

    rc = main(["ci", "plan.yaml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "name: 'Maestro CI'" in out
    assert "maestro validate plan.yaml" in out
    assert "python -m pytest -q" in out


def test_main_ci_prints_portable_quoted_plan_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_plan(tmp_path, "plans/demo plan.yaml")
    monkeypatch.chdir(tmp_path)

    rc = main(["ci", "plans/demo plan.yaml"])

    assert rc == 0
    out = capsys.readouterr().out
    assert 'maestro validate "plans/demo plan.yaml"' in out
    assert 'maestro run "plans/demo plan.yaml" --auto-approve' in out


def test_main_ci_writes_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / ".github" / "workflows" / "maestro.yml"

    rc = main(["ci", "plan.yaml", "--provider", "github", "--output", str(output_file)])

    assert rc == 0
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "maestro validate plan.yaml" in content
    assert "python -m pytest -q" in content

    out = capsys.readouterr().out
    assert f"CI config written to {output_file.resolve()}" in out
