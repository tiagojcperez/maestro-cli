from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ci import CiWorkflowSpec


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _job_steps(spec: "CiWorkflowSpec", *, step_name: str, command: str) -> list[str]:
    return [
        "    steps:",
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "        with:",
        f"          python-version: {_yaml_quote(spec.python_version)}",
        "      - name: Upgrade pip",
        f"        run: {_yaml_quote(spec.bootstrap_command)}",
        "      - name: Install dependencies",
        f"        run: {_yaml_quote(spec.install_command)}",
        f"      - name: {_yaml_quote(step_name)}",
        f"        run: {_yaml_quote(command)}",
    ]


def _platform_job(
    job_id: str,
    *,
    label: str,
    runner: str,
    spec: "CiWorkflowSpec",
    step_name: str,
    command: str,
    needs: str | None = None,
) -> list[str]:
    lines = [
        f"  {job_id}:",
        f"    name: {_yaml_quote(label)}",
        f"    runs-on: {_yaml_quote(runner)}",
    ]
    if needs is not None:
        lines.extend(
            [
                "    needs:",
                f"      - {needs}",
            ]
        )
    lines.extend(_job_steps(spec, step_name=step_name, command=command))
    return lines


def render_github_actions(spec: "CiWorkflowSpec") -> str:
    lines = [
        f"name: {_yaml_quote(spec.workflow_name)}",
        "",
        "# Platform matrix (rendered as explicit jobs for deterministic CI output):",
        "#   platform | validate job     | test job      | runner          | coverage",
        "#   linux    | validate_linux   | test_linux    | ubuntu-latest   | required",
        "#   windows  | validate_windows | test_windows  | windows-latest  | required",
        "#   macos    | validate_macos   | test_macos    | macos-latest    | required",
        "# Windows guidance: keep CI run steps portable (`python -m ...`, `maestro ...`).",
        "# For Maestro plan tasks on Windows, prefer YAML list commands such as",
        '# `["C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "python -m pytest -q"]`',
        "# and avoid bash-only string commands such as `bash -lc \"python -m pytest -q\"`.",
        "# Inside Git Bash snippets, keep paths in forward-slash form such as `/c/project`.",
        "",
        "on:",
        "  push:",
        "    branches:",
        "      - main",
        "  pull_request:",
        "  workflow_dispatch:",
        "    inputs:",
        "      run-real-engine:",
        "        description: 'Run the opt-in real-engine Maestro lane'",
        "        required: false",
        "        default: false",
        "        type: boolean",
        "",
        "jobs:",
        "  # Validate and test run as explicit linux/windows/macos lanes.",
        "  # Avoid matrix indirection here so generated examples stay easy to audit.",
    ]
    lines.extend(
        _platform_job(
            "validate_linux",
            label="Validate (Linux)",
            runner="ubuntu-latest",
            spec=spec,
            step_name="Validate plan",
            command=spec.validate_command,
        )
    )
    lines.extend(
        _platform_job(
            "validate_windows",
            label="Validate (Windows)",
            runner="windows-latest",
            spec=spec,
            step_name="Validate plan",
            command=spec.validate_command,
        )
    )
    lines.extend(
        _platform_job(
            "validate_macos",
            label="Validate (macOS)",
            runner="macos-latest",
            spec=spec,
            step_name="Validate plan",
            command=spec.validate_command,
        )
    )
    lines.extend(
        _platform_job(
            "test_linux",
            label="Test (Linux)",
            runner="ubuntu-latest",
            spec=spec,
            step_name="Run tests",
            command=spec.test_command,
            needs="validate_linux",
        )
    )
    lines.extend(
        _platform_job(
            "test_windows",
            label="Test (Windows)",
            runner="windows-latest",
            spec=spec,
            step_name="Run tests",
            command=spec.test_command,
            needs="validate_windows",
        )
    )
    lines.extend(
        _platform_job(
            "test_macos",
            label="Test (macOS)",
            runner="macos-latest",
            spec=spec,
            step_name="Run tests",
            command=spec.test_command,
            needs="validate_macos",
        )
    )
    lines.extend(
        [
            "  maestro_real_engine:",
            "    name: Run Maestro plan (real engine)",
            "    if: github.event_name == 'workflow_dispatch' && inputs.run-real-engine",
            "    needs:",
            "      - test_linux",
            "      - test_windows",
            "      - test_macos",
            "    runs-on: ubuntu-latest",
        ]
    )
    lines.extend(
        _job_steps(
            spec,
            step_name="Run Maestro plan",
            command=spec.maestro_command,
        )
    )
    return "\n".join(lines) + "\n"
