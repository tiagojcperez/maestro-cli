from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ci import CiWorkflowSpec


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _job_definition(
    job_name: str,
    *,
    extends: str,
    stage: str,
    command: str,
    needs: list[str] | None = None,
    optional_needs: list[str] | None = None,
    rules: list[str] | None = None,
) -> list[str]:
    lines = [
        f"{job_name}:",
        f"  extends: {extends}",
        f"  stage: {stage}",
    ]
    if rules:
        lines.append("  rules:")
        lines.extend(f"    {rule}" for rule in rules)
    if needs or optional_needs:
        lines.append("  needs:")
        for need in needs or []:
            lines.append(f"    - job: {need}")
        for need in optional_needs or []:
            lines.extend(
                [
                    f"    - job: {need}",
                    "      optional: true",
                ]
            )
    lines.extend(
        [
            "  script:",
            f"    - {_yaml_quote(command)}",
        ]
    )
    return lines


def render_gitlab_ci(spec: "CiWorkflowSpec") -> str:
    lines = [
        "# Platform matrix (rendered as explicit jobs for deterministic CI output):",
        "#   platform | validate job     | test job      | runner                  | coverage",
        "#   linux    | validate_linux   | test_linux    | shared/container runner | default-on",
        "#   windows  | validate_windows | test_windows  | tagged windows runner   | opt-in",
        "#   macos    | validate_macos   | test_macos    | tagged macos runner     | opt-in",
        "# Windows guidance: keep CI scripts portable (`python -m ...`, `maestro ...`).",
        "# For Maestro plan tasks on Windows, prefer YAML list commands such as",
        '# `["C:\\Program Files\\Git\\bin\\bash.exe", "-lc", "python -m pytest -q"]`',
        "# and avoid bash-only string commands such as `bash -lc \"python -m pytest -q\"`.",
        "# Inside Git Bash snippets, keep paths in forward-slash form such as `/c/project`.",
        "",
        "workflow:",
        f"  name: {_yaml_quote(spec.workflow_name)}",
        "",
        "stages:",
        "  - validate",
        "  - test",
        "  - maestro",
        "",
        "variables:",
        "  # linux runs by default. windows and macos lanes require matching tagged runners.",
        "  MAESTRO_ENABLE_WINDOWS: '0'",
        "  MAESTRO_ENABLE_MACOS: '0'",
        "  # Set to 1 when manually starting a pipeline to expose the real-engine lane.",
        "  MAESTRO_RUN_REAL_ENGINE: '0'",
        "",
        ".linux_job:",
        f"  image: {_yaml_quote(f'python:{spec.python_version}')}",
        "  before_script:",
        f"    - {_yaml_quote(spec.bootstrap_command)}",
        f"    - {_yaml_quote(spec.install_command)}",
        "",
        ".windows_job:",
        "  tags:",
        "    - windows",
        "  before_script:",
        f"    - {_yaml_quote(spec.bootstrap_command)}",
        f"    - {_yaml_quote(spec.install_command)}",
        "",
        ".macos_job:",
        "  tags:",
        "    - macos",
        "  before_script:",
        f"    - {_yaml_quote(spec.bootstrap_command)}",
        f"    - {_yaml_quote(spec.install_command)}",
        "",
        "# GitLab cannot provide hosted Windows or macOS runners here, so those lanes are explicit",
        "# and only run when your project has matching tagged runners and the variables are enabled.",
    ]
    lines.extend(
        _job_definition(
            "validate_linux",
            extends=".linux_job",
            stage="validate",
            command=spec.validate_command,
        )
    )
    lines.append("")
    lines.extend(
        _job_definition(
            "validate_windows",
            extends=".windows_job",
            stage="validate",
            command=spec.validate_command,
            rules=['- if: $MAESTRO_ENABLE_WINDOWS == "1"', "- when: never"],
        )
    )
    lines.append("")
    lines.extend(
        _job_definition(
            "validate_macos",
            extends=".macos_job",
            stage="validate",
            command=spec.validate_command,
            rules=['- if: $MAESTRO_ENABLE_MACOS == "1"', "- when: never"],
        )
    )
    lines.append("")
    lines.extend(
        _job_definition(
            "test_linux",
            extends=".linux_job",
            stage="test",
            command=spec.test_command,
            needs=["validate_linux"],
        )
    )
    lines.append("")
    lines.extend(
        _job_definition(
            "test_windows",
            extends=".windows_job",
            stage="test",
            command=spec.test_command,
            needs=["validate_windows"],
            rules=['- if: $MAESTRO_ENABLE_WINDOWS == "1"', "- when: never"],
        )
    )
    lines.append("")
    lines.extend(
        _job_definition(
            "test_macos",
            extends=".macos_job",
            stage="test",
            command=spec.test_command,
            needs=["validate_macos"],
            rules=['- if: $MAESTRO_ENABLE_MACOS == "1"', "- when: never"],
        )
    )
    lines.append("")
    lines.extend(
        _job_definition(
            "run_maestro_real_engine",
            extends=".linux_job",
            stage="maestro",
            command=spec.maestro_command,
            needs=["test_linux"],
            optional_needs=["test_windows", "test_macos"],
            rules=[
                '- if: $MAESTRO_RUN_REAL_ENGINE == "1"',
                "  when: manual",
                "- when: never",
            ],
        )
    )
    return "\n".join(lines) + "\n"
