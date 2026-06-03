from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .plugins import (
    BUILTIN_ENGINE_ORDER,
    BUILTIN_ENGINE_NAMES,
    ENTRY_POINT_GROUP,
    PluginResolutionError,
    discover_engine_plugins,
    get_engine_plugin,
    plugin_discovery_errors,
    supported_engine_names,
)

# Check result: (check_name, detail_message, status)
# status: "ok", "warn", "fail", "info"
class CheckResult(tuple[str, str, str]):
    def __new__(cls, check: str, detail: str, status: str) -> "CheckResult":
        return super().__new__(cls, (check, detail, status))

    def __getitem__(self, item: int | slice | str) -> Any:  # type: ignore[override]
        if isinstance(item, str):
            mapping = {"check": 0, "detail": 1, "status": 2}
            return super().__getitem__(mapping[item])
        return super().__getitem__(item)


def _result(check: str, detail: str, status: str) -> CheckResult:
    return CheckResult(check, detail, status)

_STATUS_PAD = {
    "ok": "ok  ",
    "warn": "warn",
    "fail": "FAIL",
    "info": "info",
}


def _check_python_version() -> CheckResult:
    version = sys.version.split()[0]
    if sys.version_info >= (3, 11):
        return _result("python_version", f"Python {version}", "ok")
    return _result("python_version", f"Python {version} — need >=3.11", "fail")


def _check_pyyaml() -> CheckResult:
    try:
        import yaml

        version = getattr(yaml, "__version__", "unknown")
        return _result("pyyaml", f"PyYAML {version}", "ok")
    except ImportError:
        return _result("pyyaml", "PyYAML not installed — run: pip install pyyaml", "fail")


def _check_engine(
    name: str,
    *,
    executable: str | None = None,
    check_name: str | None = None,
    install_hint: str | None = None,
) -> CheckResult:
    executable = executable or name
    path = shutil.which(executable)
    if path:
        return _result(check_name or f"engine_{name}", f"found: {path}", "ok")
    detail = f"{executable!r} not found on PATH"
    if install_hint:
        detail += f". {install_hint}"
    return _result(check_name or f"engine_{name}", detail, "warn")


def _check_git() -> CheckResult:
    path = shutil.which("git")
    if path:
        return _result("git", f"found: {path}", "ok")
    return _result("git", "git not found on PATH (needed for requires_clean_worktree)", "warn")


def _check_run_dir_writable(run_dir: str) -> CheckResult:
    run_path = Path(run_dir)
    try:
        run_path.mkdir(parents=True, exist_ok=True)
        probe = run_path / ".maestro_doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return _result("run_dir_writable", f"{run_path.resolve()} is writable", "ok")
    except OSError as exc:
        return _result("run_dir_writable", f"cannot write to {run_dir!r}: {exc}", "fail")


def _check_web_deps() -> CheckResult:
    missing: list[str] = []
    for pkg in ("fastapi", "uvicorn"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return _result("web_deps", "fastapi + uvicorn installed", "ok")
    return _result(
        "web_deps",
        f"optional web deps not installed: {', '.join(missing)} — run: pip install maestro-cli[web]",
        "warn",
    )


def _plugin_warning_names(errors: dict[str, str]) -> list[str]:
    return sorted(name for name in errors if name != "__entry_points__")


def _check_plugin_discovery() -> CheckResult:
    discovered = discover_engine_plugins()
    errors = plugin_discovery_errors()
    custom_names = sorted(discovered)

    if "__entry_points__" in errors:
        return _result("engine_plugins", errors["__entry_points__"], "warn")

    if custom_names:
        detail = (
            f"discovered {len(custom_names)} custom engine plugin(s) in "
            f"'{ENTRY_POINT_GROUP}': {', '.join(custom_names)}"
        )
    else:
        detail = f"no custom engine plugins discovered in '{ENTRY_POINT_GROUP}'"

    if errors:
        warning_names = ", ".join(_plugin_warning_names(errors))
        return _result(
            "engine_plugins",
            f"{detail}; plugin warning(s): {warning_names}",
            "warn",
        )
    return _result("engine_plugins", detail, "ok")


def _engine_check_results() -> list[CheckResult]:
    errors = plugin_discovery_errors()
    results: list[CheckResult] = [_check_plugin_discovery()]

    ordered_names = list(BUILTIN_ENGINE_ORDER)
    ordered_names.extend(
        name for name in supported_engine_names()
        if name not in BUILTIN_ENGINE_NAMES
    )

    for name in ordered_names:
        try:
            plugin = get_engine_plugin(name)
        except PluginResolutionError as exc:
            results.append(_result(f"engine_{name}", str(exc), "warn"))
            continue

        probe = plugin.doctor_probe
        if probe is None:
            continue
        results.append(_check_engine(
            name,
            executable=probe.executable,
            check_name=probe.resolved_check_name(name),
            install_hint=probe.install_hint,
        ))

    for name, detail in errors.items():
        if name == "__entry_points__":
            continue
        check_name = f"engine_{name}"
        results.append(_result(check_name, detail, "warn"))

    # -- Optional dependency checks --
    try:
        import textual

        version = getattr(textual, "__version__", "unknown")
        results.append(_result("tui_dependency", f"textual {version} available", "ok"))
    except ImportError:
        results.append(_result(
            "tui_dependency",
            "textual not installed (pip install maestro-cli[tui])",
            "info",
        ))

    try:
        import rich

        version = getattr(rich, "__version__", "unknown")
        results.append(_result("live_dependency", f"rich {version} available", "ok"))
    except ImportError:
        results.append(_result(
            "live_dependency",
            "rich not installed (pip install maestro-cli[live])",
            "info",
        ))

    try:
        import ag_ui

        results.append(_result("agui_protocol", "ag-ui-protocol available", "ok"))
    except ImportError:
        results.append(_result(
            "agui_protocol",
            "ag-ui-protocol not installed (pip install maestro-cli[agui])",
            "info",
        ))

    try:
        import mcp

        version = getattr(mcp, "__version__", "unknown")
        results.append(_result("mcp_protocol", f"mcp {version} available", "ok"))
    except ImportError:
        results.append(_result(
            "mcp_protocol",
            "mcp not installed (pip install maestro-cli[mcp])",
            "info",
        ))

    # OpenTelemetry
    try:
        import opentelemetry
        version = getattr(opentelemetry, "__version__", "unknown")
        results.append(_result("otel_protocol", f"opentelemetry {version} available", "ok"))
    except ImportError:
        results.append(_result(
            "otel_protocol",
            "opentelemetry not installed (pip install maestro-cli[otel])",
            "info",
        ))

    return results


def _check_cache_dir() -> CheckResult:
    cache_dir = Path.cwd() / ".maestro-cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return _result("cache_dir", f"{cache_dir} is writable", "ok")
    except OSError as exc:
        return _result("cache_dir", f"cannot write to .maestro-cache: {exc}", "warn")


def _check_knowledge_store() -> CheckResult:
    knowledge_dir = Path.cwd() / ".maestro-cache" / "knowledge"
    if knowledge_dir.is_dir():
        files = list(knowledge_dir.glob("*.jsonl"))
        return _result(
            "knowledge_store",
            f"{len(files)} knowledge file(s) in {knowledge_dir}",
            "ok",
        )
    return _result("knowledge_store", "no knowledge store yet (created after first run)", "info")


def _check_skill_registry() -> CheckResult:
    skills_dir = Path.cwd() / ".claude" / "skills"
    if not skills_dir.is_dir():
        return _result("skill_registry", "no .claude/skills/ directory", "info")
    skill_count = sum(
        1 for d in skills_dir.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    )
    return _result("skill_registry", f"{skill_count} skill(s) discovered", "ok")


def _check_plans_in_cwd() -> CheckResult:
    yaml_files = list(Path.cwd().glob("*.yaml")) + list(Path.cwd().glob("*.yml"))
    plan_count = 0
    errors: list[str] = []
    for yf in yaml_files:
        try:
            import yaml as _yaml
            raw = _yaml.safe_load(yf.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("version") == 1 and "tasks" in raw:
                plan_count += 1
        except Exception:
            pass
    if plan_count == 0:
        return _result("plans_in_cwd", "no Maestro plans found in cwd", "info")
    return _result("plans_in_cwd", f"{plan_count} plan(s) found in cwd", "ok")


def _check_prior_runs(run_dir: str) -> CheckResult:
    run_path = Path(run_dir)
    if not run_path.is_dir():
        return _result("prior_runs", "no runs directory yet", "info")
    run_dirs = [d for d in run_path.iterdir() if d.is_dir() and (d / "run_manifest.json").exists()]
    if not run_dirs:
        return _result("prior_runs", "runs directory exists but no completed runs", "info")
    return _result("prior_runs", f"{len(run_dirs)} completed run(s)", "ok")


def run_doctor(
    *,
    run_dir: str = ".maestro-runs",
    json_output: bool = False,
    full: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = [
        _check_python_version(),
        _check_pyyaml(),
        *_engine_check_results(),
        _check_git(),
        _check_run_dir_writable(run_dir),
        _check_web_deps(),
    ]

    if full:
        results.extend([
            _check_cache_dir(),
            _check_knowledge_store(),
            _check_skill_registry(),
            _check_plans_in_cwd(),
            _check_prior_runs(run_dir),
        ])

    if json_output:
        print(json.dumps(
            [{"check": name, "detail": detail, "status": status} for name, detail, status in results],
            indent=2,
        ))
    else:
        for name, detail, status in results:
            icon = _STATUS_PAD[status]
            print(f"  [{icon}]  {name}: {detail}")

        fail_count = sum(1 for _, _, s in results if s == "fail")
        warn_count = sum(1 for _, _, s in results if s == "warn")
        print(
            f"\n[maestro] doctor: {len(results)} checks — "
            f"{fail_count} failed, {warn_count} warning(s)"
        )

    return results
