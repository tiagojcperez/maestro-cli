from __future__ import annotations

from pathlib import Path

import yaml

from maestro_cli.ci import (
    CiWorkflowSpec,
    default_ci_output_path,
    get_ci_provider,
    render_github_actions,
)


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


def test_render_github_actions_has_cross_platform_validation_and_tests() -> None:
    rendered = render_github_actions(_sample_spec())
    parsed = _load_yaml(rendered)
    jobs = parsed["jobs"]

    assert "name: 'Demo CI'" in rendered
    assert "# Platform matrix (rendered as explicit jobs for deterministic CI output):" in rendered
    assert "#   platform | validate job     | test job      | runner          | coverage" in rendered
    assert "#   linux    | validate_linux   | test_linux    | ubuntu-latest   | required" in rendered
    assert "#   windows  | validate_windows | test_windows  | windows-latest  | required" in rendered
    assert "#   macos    | validate_macos   | test_macos    | macos-latest    | required" in rendered
    assert "prefer YAML list commands" in rendered
    assert '["C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "python -m pytest -q"]' in rendered
    assert 'bash -lc "python -m pytest -q"' in rendered
    assert "Inside Git Bash snippets, keep paths in forward-slash form such as `/c/project`." in rendered
    assert "workflow_dispatch:" in rendered
    assert "run-real-engine:" in rendered
    assert "github.event_name == 'workflow_dispatch' && inputs.run-real-engine" in rendered
    assert list(jobs) == [
        "validate_linux",
        "validate_windows",
        "validate_macos",
        "test_linux",
        "test_windows",
        "test_macos",
        "maestro_real_engine",
    ]
    assert set(jobs) == {
        "validate_linux",
        "validate_windows",
        "validate_macos",
        "test_linux",
        "test_windows",
        "test_macos",
        "maestro_real_engine",
    }
    assert jobs["validate_linux"]["name"] == "Validate (Linux)"
    assert jobs["validate_windows"]["name"] == "Validate (Windows)"
    assert jobs["validate_macos"]["name"] == "Validate (macOS)"
    assert jobs["test_linux"]["name"] == "Test (Linux)"
    assert jobs["test_windows"]["name"] == "Test (Windows)"
    assert jobs["test_macos"]["name"] == "Test (macOS)"
    assert jobs["validate_linux"]["runs-on"] == "ubuntu-latest"
    assert jobs["validate_windows"]["runs-on"] == "windows-latest"
    assert jobs["validate_macos"]["runs-on"] == "macos-latest"
    assert jobs["test_linux"]["runs-on"] == "ubuntu-latest"
    assert jobs["test_windows"]["runs-on"] == "windows-latest"
    assert jobs["test_macos"]["runs-on"] == "macos-latest"
    assert {
        job_name: jobs[job_name]["runs-on"]
        for job_name in (
            "validate_linux",
            "validate_windows",
            "validate_macos",
            "test_linux",
            "test_windows",
            "test_macos",
        )
    } == {
        "validate_linux": "ubuntu-latest",
        "validate_windows": "windows-latest",
        "validate_macos": "macos-latest",
        "test_linux": "ubuntu-latest",
        "test_windows": "windows-latest",
        "test_macos": "macos-latest",
    }
    assert rendered.count("| required") == 3
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
    assert jobs["test_linux"]["needs"] == ["validate_linux"]
    assert jobs["test_windows"]["needs"] == ["validate_windows"]
    assert jobs["test_macos"]["needs"] == ["validate_macos"]
    assert jobs["maestro_real_engine"]["needs"] == ["test_linux", "test_windows", "test_macos"]
    assert jobs["maestro_real_engine"]["runs-on"] == "ubuntu-latest"
    assert jobs["maestro_real_engine"]["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.run-real-engine"
    )
    windows_validate_steps = jobs["validate_windows"]["steps"]
    windows_test_steps = jobs["test_windows"]["steps"]
    assert windows_validate_steps[-1]["run"] == "maestro validate plans/demo.yaml"
    assert windows_test_steps[-1]["run"] == "python -m pytest -q"
    assert windows_validate_steps[2]["run"] == "python -m pip install --upgrade pip"
    assert windows_validate_steps[3]["run"] == "python -m pip install -e . pytest"
    assert "maestro validate plans/demo.yaml" in rendered
    assert "python -m pytest -q" in rendered
    assert "maestro run plans/demo.yaml --auto-approve" in rendered


def test_github_actions_example_matches_renderer_output() -> None:
    example_path = Path("examples/ci/github-actions.yml")

    assert example_path.read_text(encoding="utf-8") == render_github_actions(_sample_spec())


def test_all_github_actions_examples_match_renderer_output() -> None:
    expected = render_github_actions(
        CiWorkflowSpec(
            plan_name="demo_plan",
            plan_path="examples/demo_plan.yaml",
            workflow_name="Maestro CI",
            python_version="3.11",
            bootstrap_command="python -m pip install --upgrade pip",
            install_command="python -m pip install -e . pytest",
            validate_command="maestro validate examples/demo_plan.yaml",
            test_command="python -m pytest -q",
            maestro_command="maestro run examples/demo_plan.yaml --auto-approve",
        )
    )

    for example_name in ("github_actions.yml", "github_actions_maestro.yml"):
        example_path = Path("examples/ci") / example_name
        assert example_path.read_text(encoding="utf-8") == expected


def test_github_actions_provider_slot_resolves_to_github_actions_renderer() -> None:
    provider = get_ci_provider("github_actions")

    assert provider.name == "github_actions"
    assert provider.render is render_github_actions
    assert default_ci_output_path("github_actions") == Path(".github/workflows/maestro.yml")
