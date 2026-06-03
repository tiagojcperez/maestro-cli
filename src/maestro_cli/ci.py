from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

from .ci_github_actions import render_github_actions
from .ci_gitlab_ci import render_gitlab_ci
from .loader import load_plan

CiProviderName = Literal["github_actions", "gitlab_ci"]
CiProviderAlias = Literal["github_actions", "gitlab_ci", "github", "gitlab"]

DEFAULT_CI_WORKFLOW_NAME = "Maestro CI"
DEFAULT_CI_PYTHON_VERSION = "3.11"
DEFAULT_CI_INSTALL_COMMAND = "python -m pip install -e . pytest"
DEFAULT_CI_TEST_COMMAND = "python -m pytest -q"
DEFAULT_CI_BOOTSTRAP_COMMAND = "python -m pip install --upgrade pip"


@dataclass(frozen=True)
class CiWorkflowSpec:
    plan_name: str
    plan_path: str
    workflow_name: str
    python_version: str
    bootstrap_command: str
    install_command: str
    validate_command: str
    test_command: str
    maestro_command: str


@dataclass(frozen=True)
class CiProvider:
    name: CiProviderName
    default_output_path: Path
    render: Callable[[CiWorkflowSpec], str]


def _require_non_empty(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty string")
    return stripped


def _normalize_plan_path(plan_path: str | Path) -> str:
    candidate = Path(plan_path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(
                "plan_path must be inside the current working directory "
                "to generate portable CI config"
            ) from exc
    return candidate.as_posix()


def _format_portable_cli_arg(value: str, *, field_name: str) -> str:
    if any(char in value for char in ('"', "$", "`")):
        raise ValueError(
            f"{field_name} contains characters that are not portable across "
            'Windows, macOS, and Linux CI shells: ", $, `'
        )
    if all(char.isalnum() or char in "/._-" for char in value):
        return value
    return f'"{value}"'


def build_ci_workflow_spec(
    plan_path: str | Path,
    *,
    workflow_name: str = DEFAULT_CI_WORKFLOW_NAME,
    python_version: str = DEFAULT_CI_PYTHON_VERSION,
    bootstrap_command: str = DEFAULT_CI_BOOTSTRAP_COMMAND,
    install_command: str = DEFAULT_CI_INSTALL_COMMAND,
    test_command: str = DEFAULT_CI_TEST_COMMAND,
    maestro_command: str | None = None,
) -> CiWorkflowSpec:
    plan = load_plan(plan_path)
    portable_plan_path = _normalize_plan_path(plan_path)
    plan_arg = _format_portable_cli_arg(portable_plan_path, field_name="plan_path")

    workflow_name = _require_non_empty(workflow_name, "workflow_name")
    python_version = _require_non_empty(python_version, "python_version")
    bootstrap_command = _require_non_empty(bootstrap_command, "bootstrap_command")
    install_command = _require_non_empty(install_command, "install_command")
    test_command = _require_non_empty(test_command, "test_command")
    validate_command = f"maestro validate {plan_arg}"
    run_command = maestro_command or f"maestro run {plan_arg} --auto-approve"
    run_command = _require_non_empty(run_command, "maestro_command")

    return CiWorkflowSpec(
        plan_name=plan.name,
        plan_path=portable_plan_path,
        workflow_name=workflow_name,
        python_version=python_version,
        bootstrap_command=bootstrap_command,
        install_command=install_command,
        validate_command=validate_command,
        test_command=test_command,
        maestro_command=run_command,
    )


_CI_PROVIDERS: dict[CiProviderName, CiProvider] = {
    "github_actions": CiProvider(
        name="github_actions",
        default_output_path=Path(".github/workflows/maestro.yml"),
        render=render_github_actions,
    ),
    "gitlab_ci": CiProvider(
        name="gitlab_ci",
        default_output_path=Path(".gitlab-ci.yml"),
        render=render_gitlab_ci,
    ),
}

_CI_PROVIDER_ALIASES: dict[CiProviderAlias, CiProviderName] = {
    "github_actions": "github_actions",
    "gitlab_ci": "gitlab_ci",
    "github": "github_actions",
    "gitlab": "gitlab_ci",
}

SUPPORTED_CI_PROVIDERS: tuple[CiProviderAlias, ...] = (
    "github_actions",
    "gitlab_ci",
    "github",
    "gitlab",
)


def get_ci_provider(name: str) -> CiProvider:
    canonical_name = _CI_PROVIDER_ALIASES.get(cast(CiProviderAlias, name))
    provider = _CI_PROVIDERS.get(canonical_name) if canonical_name is not None else None
    if provider is not None:
        return provider
    supported = ", ".join(SUPPORTED_CI_PROVIDERS)
    raise ValueError(f"Unsupported CI provider '{name}'. Supported: {supported}")


def default_ci_output_path(provider: str) -> Path:
    return get_ci_provider(provider).default_output_path


def generate_ci_yaml(
    plan_path: str | Path,
    *,
    provider: str = "github_actions",
    workflow_name: str = DEFAULT_CI_WORKFLOW_NAME,
    python_version: str = DEFAULT_CI_PYTHON_VERSION,
    bootstrap_command: str = DEFAULT_CI_BOOTSTRAP_COMMAND,
    install_command: str = DEFAULT_CI_INSTALL_COMMAND,
    test_command: str = DEFAULT_CI_TEST_COMMAND,
    maestro_command: str | None = None,
) -> str:
    renderer = get_ci_provider(provider).render
    spec = build_ci_workflow_spec(
        plan_path,
        workflow_name=workflow_name,
        python_version=python_version,
        bootstrap_command=bootstrap_command,
        install_command=install_command,
        test_command=test_command,
        maestro_command=maestro_command,
    )
    return renderer(spec)


def write_ci_yaml(content: str, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination.resolve()
