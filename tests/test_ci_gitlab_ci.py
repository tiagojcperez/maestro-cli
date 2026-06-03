from __future__ import annotations

from pathlib import Path

import yaml

from maestro_cli.ci import CiWorkflowSpec, render_gitlab_ci


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


def _load_yaml(rendered: str) -> dict[str, object]:
    return yaml.load(rendered, Loader=yaml.BaseLoader)


def test_render_gitlab_ci_has_explicit_platform_coverage_and_opt_in_real_engine_lane() -> None:
    rendered = render_gitlab_ci(_sample_spec())
    parsed = _load_yaml(rendered)

    assert "# Platform matrix (rendered as explicit jobs for deterministic CI output):" in rendered
    assert "#   platform | validate job     | test job      | runner                  | coverage" in rendered
    assert "#   linux    | validate_linux   | test_linux    | shared/container runner | default-on" in rendered
    assert "#   windows  | validate_windows | test_windows  | tagged windows runner   | opt-in" in rendered
    assert "#   macos    | validate_macos   | test_macos    | tagged macos runner     | opt-in" in rendered
    assert "prefer YAML list commands" in rendered
    assert '["C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "python -m pytest -q"]' in rendered
    assert 'bash -lc "python -m pytest -q"' in rendered
    assert "Inside Git Bash snippets, keep paths in forward-slash form such as `/c/project`." in rendered
    assert "workflow:" in rendered
    assert "name: 'Demo CI'" in rendered
    assert "stages:" in rendered
    assert "linux runs by default. windows and macos lanes require matching tagged runners." in rendered
    assert parsed["variables"]["MAESTRO_ENABLE_WINDOWS"] == "0"
    assert parsed["variables"]["MAESTRO_ENABLE_MACOS"] == "0"
    assert parsed["variables"]["MAESTRO_RUN_REAL_ENGINE"] == "0"
    assert parsed[".linux_job"]["image"] == "python:3.12"
    assert parsed[".windows_job"]["tags"] == ["windows"]
    assert parsed[".macos_job"]["tags"] == ["macos"]
    assert parsed["validate_linux"]["extends"] == ".linux_job"
    assert parsed["validate_windows"]["extends"] == ".windows_job"
    assert parsed["validate_macos"]["extends"] == ".macos_job"
    assert {
        job_name: parsed[job_name]["extends"]
        for job_name in (
            "validate_linux",
            "validate_windows",
            "validate_macos",
            "test_linux",
            "test_windows",
            "test_macos",
        )
    } == {
        "validate_linux": ".linux_job",
        "validate_windows": ".windows_job",
        "validate_macos": ".macos_job",
        "test_linux": ".linux_job",
        "test_windows": ".windows_job",
        "test_macos": ".macos_job",
    }
    assert rendered.count("| opt-in") == 2
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
    assert 'if: $MAESTRO_ENABLE_WINDOWS == "1"' in rendered
    assert 'if: $MAESTRO_ENABLE_MACOS == "1"' in rendered
    assert parsed["test_linux"]["extends"] == ".linux_job"
    assert parsed["test_windows"]["extends"] == ".windows_job"
    assert parsed["test_macos"]["extends"] == ".macos_job"
    assert parsed["test_linux"]["needs"] == [{"job": "validate_linux"}]
    assert parsed["test_windows"]["needs"] == [{"job": "validate_windows"}]
    assert parsed["test_macos"]["needs"] == [{"job": "validate_macos"}]
    assert parsed["run_maestro_real_engine"]["extends"] == ".linux_job"
    assert 'if: $MAESTRO_RUN_REAL_ENGINE == "1"' in rendered
    assert "when: manual" in rendered
    assert parsed["validate_windows"]["script"] == ["maestro validate plans/demo.yaml"]
    assert parsed["test_windows"]["script"] == ["python -m pytest -q"]
    assert parsed["run_maestro_real_engine"]["script"] == [
        "maestro run plans/demo.yaml --auto-approve"
    ]
    assert parsed["run_maestro_real_engine"]["needs"] == [
        {"job": "test_linux"},
        {"job": "test_windows", "optional": "true"},
        {"job": "test_macos", "optional": "true"},
    ]
    assert rendered.count("extends: .linux_job") == 3
    assert rendered.count("extends: .windows_job") == 2
    assert rendered.count("extends: .macos_job") == 2
    assert "optional: true" in rendered
    assert "maestro validate plans/demo.yaml" in rendered
    assert "python -m pytest -q" in rendered
    assert "maestro run plans/demo.yaml --auto-approve" in rendered


def test_gitlab_ci_example_matches_renderer_output() -> None:
    example_path = Path("examples/ci/gitlab-ci.yml")

    assert example_path.read_text(encoding="utf-8") == render_gitlab_ci(_sample_spec())


def test_gitlab_ci_alias_example_matches_renderer_output() -> None:
    example_path = Path("examples/ci/gitlab_ci.yml")

    assert example_path.read_text(encoding="utf-8") == render_gitlab_ci(_sample_spec())
