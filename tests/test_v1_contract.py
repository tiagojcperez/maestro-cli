from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro_cli.cli import _build_parser
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import _CURRENT_SCHEMA_VERSION, _SUPPORTED_SCHEMA_VERSIONS, load_plan
from maestro_cli.models import (
    PlanDefaults,
    PlanSpec,
    StructuredContext,
    TaskResult,
    TaskSpec,
    TokenUsage,
    WorkspaceBrief,
)
from maestro_cli.scheduler import run_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
V1_CONTRACT_DOC = REPO_ROOT / "docs" / "V1_API_FREEZE.md"

V1_STABLE_COMMANDS = (
    "validate",
    "run",
    "cleanup",
    "backfill-costs",
    "scaffold",
    "doctor",
    "ui",
    "report",
    "diff",
    "explain",
    "status",
    "eval",
    "suggest",
    "shell",
    "replan",
)

V1_STABLE_ENTRYPOINTS = (
    "maestro --help",
    "maestro --version",
    *(f"maestro {command}" for command in V1_STABLE_COMMANDS),
)

V1_STABLE_CLI_SURFACE = {
    "validate": {
        "positionals": ("plan",),
        "options": frozenset(),
        "doc_fragments": ("positional plan",),
    },
    "run": {
        "positionals": ("plan",),
        "options": frozenset({
            "--dry-run",
            "--max-parallel",
            "--only",
            "--skip",
            "--run-dir",
            "--webhook",
            "--execution-profile",
            "--profile-mode",
            "--mode",
            "--resume",
            "--resume-last",
            "--verbose",
            "-v",
            "--quiet",
            "-q",
            "--output",
            "--no-cache",
            "--cache-dir",
            "--tags",
            "--skip-tags",
            "--mask-secrets",
            "--auto-approve",
            "--parallel",
        }),
    },
    "cleanup": {
        "positionals": ("plan",),
        "options": frozenset({"--keep", "--older-than", "--dry-run"}),
    },
    "backfill-costs": {
        "positionals": (),
        "options": frozenset({"--root", "--run-root", "--dry-run", "--codex-pricing-file"}),
    },
    "scaffold": {
        "positionals": ("brief",),
        "options": frozenset({"-o", "--output", "--validate", "--cost-check"}),
    },
    "doctor": {
        "positionals": (),
        "options": frozenset({"--json", "--run-dir"}),
    },
    "ui": {
        "positionals": (),
        "options": frozenset({"--host", "--port", "--no-browser", "--project-root"}),
    },
    "report": {
        "positionals": ("run_path",),
        "options": frozenset({"-o", "--output"}),
    },
    "diff": {
        "positionals": ("run_a", "run_b"),
        "options": frozenset({"--json"}),
    },
    "explain": {
        "positionals": ("plan",),
        "options": frozenset({"--cache-dir", "--json"}),
    },
    "status": {
        "positionals": ("plan",),
        "options": frozenset({"--cache-dir", "--run-dir", "--json"}),
    },
    "eval": {
        "positionals": ("eval_yaml", "run_path"),
        "options": frozenset({"--json", "--verbose", "-v"}),
    },
    "suggest": {
        "positionals": ("plan",),
        "options": frozenset({"--run-dir", "--min-runs", "--json"}),
    },
    "shell": {
        "positionals": (),
        "options": frozenset({"--plan"}),
    },
    "replan": {
        "positionals": ("plan",),
        "options": frozenset({
            "--max-attempts",
            "--model",
            "--auto-approve",
            "--execution-profile",
            "--profile-mode",
            "--mode",
            "--dry-run",
            "--verbose",
            "-v",
            "--quiet",
            "-q",
            "--output",
        }),
    },
}

V1_DOC_SIGNATURE_TEXT = {
    "validate": {"command": "<plan.yaml>", "options": "positional plan"},
    "run": {"command": "[more-plans...]", "options": "--parallel"},
    "cleanup": {"command": "<plan.yaml>", "options": "--older-than"},
    "backfill-costs": {"command": "maestro backfill-costs", "options": "--codex-pricing-file"},
    "scaffold": {"command": "<brief.yaml>", "options": "--cost-check"},
    "doctor": {"command": "maestro doctor", "options": "--run-dir"},
    "ui": {"command": "maestro ui", "options": "--project-root"},
    "report": {"command": "<run-path>", "options": "-o"},
    "diff": {"command": "<run_a> <run_b>", "options": "--json"},
    "explain": {"command": "<plan.yaml>", "options": "--cache-dir"},
    "status": {"command": "<plan.yaml>", "options": "--run-dir"},
    "eval": {"command": "<eval.yaml> <run-path>", "options": "--verbose"},
    "suggest": {"command": "<plan.yaml>", "options": "--min-runs"},
    "shell": {"command": "maestro shell", "options": "--plan"},
    "replan": {"command": "<plan.yaml>", "options": "--max-attempts"},
}

V1_MANIFEST_KEYS = frozenset({
    "plan_name",
    "run_id",
    "run_path",
    "started_at",
    "finished_at",
    "success",
    "execution_profile",
    "task_results",
    "sequential_duration_sec",
    "parallelism_savings_pct",
    "total_cost_usd",
    "total_tokens",
    "budget_exceeded",
})

V1_TASK_RESULT_KEYS = frozenset({
    "task_id",
    "status",
    "exit_code",
    "started_at",
    "finished_at",
    "duration_sec",
    "command",
    "log_path",
    "result_path",
    "message",
    "stdout_tail",
    "cost_usd",
    "token_usage",
    "structured_context",
    "retry_count",
    "failure_history",
    "checkpoint_count",
    "judge_result",
    "handoff_report",
    "context_raw_tokens",
    "context_final_tokens",
    "context_compression_ratio",
    "context_raw_bytes",
    "context_final_bytes",
    "compression_ratio",
    "task_hash",
})

V1_OPTIONAL_TASK_RESULT_KEYS = frozenset({"workspace_brief"})

V1_TOKEN_USAGE_KEYS = frozenset({
    "input_tokens",
    "cached_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "total_tokens",
})

V1_STRUCTURED_CONTEXT_KEYS = frozenset({
    "task_id",
    "status",
    "exit_code",
    "duration_sec",
    "files_changed",
    "decisions",
    "errors",
    "warnings",
    "cost_usd",
    "result_text",
    "summary",
})

V1_WORKSPACE_BRIEF_KEYS = frozenset({
    "brief_text",
    "token_estimate",
    "files_referenced",
})

V1_EVENTS_CONTRACT_BULLETS = (
    "The file exists in each run directory",
    "It is newline-delimited JSON",
    "Each line is a JSON object",
    "Each event object contains `ts` and `event`",
)


def _doc_text() -> str:
    return V1_CONTRACT_DOC.read_text(encoding="utf-8")


def _section_text(doc_text: str, heading: str) -> str:
    lines = doc_text.splitlines()
    start_index: int | None = None
    heading_level = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        marker, _, title = stripped.partition(" ")
        if title == heading:
            start_index = index + 1
            heading_level = len(marker)
            break
    if start_index is None:
        raise AssertionError(f"Missing contract heading: {heading}")

    collected: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if stripped.startswith("#"):
            marker, _, _ = stripped.partition(" ")
            if len(marker) <= heading_level:
                break
        collected.append(line)
    return "\n".join(collected).strip()


def _bullet_items(section_text: str) -> tuple[str, ...]:
    return tuple(
        re.sub(r"\s+", " ", line.strip()[2:]).strip()
        for line in section_text.splitlines()
        if line.strip().startswith("- ")
    )


def _code_bullets(section_text: str) -> tuple[str, ...]:
    items: list[str] = []
    seen_bullets = False
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if seen_bullets:
                break
            continue
        if not stripped.startswith("- "):
            if seen_bullets:
                break
            continue
        seen_bullets = True
        item = re.sub(r"\s+", " ", stripped[2:]).strip()
        match = re.fullmatch(r"`([^`]+)`", item)
        if match is None:
            raise AssertionError(f"Expected a code bullet, got: {item}")
        items.append(match.group(1))
    return tuple(items)


def _table_rows(section_text: str) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "Command" or set(cells[0]) == {"-"}:
            continue
        raw_command = cells[0].strip("`")
        normalized_command = raw_command.removeprefix("maestro ").split(" ", 1)[0]
        rows[normalized_command] = (raw_command, cells[1])
    return rows


def _command_table_surface() -> dict[str, tuple[str, ...]]:
    parser = _build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    surface: dict[str, tuple[str, ...]] = {}
    for command, subparser in subparsers_action.choices.items():
        positionals = tuple(
            action.dest
            for action in subparser._actions
            if not action.option_strings and action.dest != "help"
        )
        options = frozenset(
            option
            for action in subparser._actions
            for option in action.option_strings
            if option not in {"-h", "--help"}
        )
        surface[command] = positionals, tuple(sorted(options))
    return surface


def _write_plan(tmp_path: Path, version: int = 1) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(
        f"version: {version}\nname: contract\n"
        "tasks:\n"
        "  - id: only\n"
        "    command: echo ok\n",
        encoding="utf-8",
    )
    return plan_file


def _fake_execute_task(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    result = TaskResult(
        task_id=task.id,
        status="success",
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=0.01,
        command=f"echo {task.id}",
        log_path=run_path / f"{task.id}.log",
        result_path=run_path / f"{task.id}.result.json",
        message="ok",
        stdout_tail="done",
        cost_usd=1.25,
        token_usage=TokenUsage(
            input_tokens=10,
            cached_tokens=3,
            output_tokens=7,
            cache_creation_tokens=2,
        ),
        structured_context=StructuredContext(
            task_id=task.id,
            status="success",
            exit_code=0,
            duration_sec=0.01,
            files_changed=["src/example.py"],
            decisions=["kept v1"],
            errors=[],
            warnings=[],
            cost_usd=1.25,
            result_text="done",
            summary="completed",
        ),
        workspace_brief=WorkspaceBrief(
            brief_text="focused workspace context",
            token_estimate=42,
            files_referenced=["src/example.py"],
        ),
        task_hash="abc123",
    )
    result.log_path.write_text("status=success\nmessage=ok\n", encoding="utf-8")
    result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def test_v1_schema_version_is_locked_in_runtime_and_contract_doc(tmp_path: Path) -> None:
    assert _CURRENT_SCHEMA_VERSION == 1
    assert _SUPPORTED_SCHEMA_VERSIONS == {1}

    plan_v1 = load_plan(_write_plan(tmp_path, version=1))
    assert plan_v1.version == 1

    with pytest.raises(PlanValidationError, match="supports up to version 1"):
        load_plan(_write_plan(tmp_path, version=2))

    doc_text = _doc_text()
    assert "The authored plan schema version remains `version: 1` throughout `1.x`." in doc_text


def test_v1_cli_contract_doc_matches_golden() -> None:
    doc_text = _doc_text()

    entrypoints = _bullet_items(_section_text(doc_text, "Stable top-level entrypoints"))
    assert entrypoints == tuple(f"`{entrypoint}`" for entrypoint in V1_STABLE_ENTRYPOINTS)

    rows = _table_rows(_section_text(doc_text, "Stable command signatures"))
    assert set(rows) == set(V1_STABLE_COMMANDS)
    for command, fragments in V1_DOC_SIGNATURE_TEXT.items():
        command_cell, options_cell = rows[command]
        assert fragments["command"] in command_cell
        assert fragments["options"] in options_cell
        for option in V1_STABLE_CLI_SURFACE[command]["options"]:
            if len(option) > 2:
                assert option in options_cell


def test_v1_cli_runtime_matches_golden_surface() -> None:
    runtime_surface = _command_table_surface()

    assert set(V1_STABLE_COMMANDS).issubset(runtime_surface)
    for command, expected in V1_STABLE_CLI_SURFACE.items():
        positionals, options = runtime_surface[command]
        assert positionals == expected["positionals"]
        assert expected["options"].issubset(set(options))


def test_v1_run_artifact_contract_doc_matches_golden() -> None:
    doc_text = _doc_text()

    manifest_keys = frozenset(_code_bullets(_section_text(doc_text, "Stable `run_manifest.json` top-level keys")))
    assert manifest_keys == V1_MANIFEST_KEYS

    result_keys = frozenset(_code_bullets(_section_text(doc_text, "Stable `<task-id>.result.json` keys")))
    assert result_keys == V1_TASK_RESULT_KEYS
    assert "`workspace_brief` is also a stable optional key when recursive context is used." in doc_text

    nested_result_section = _section_text(doc_text, "Stable nested result objects")
    nested_result_bullets = _bullet_items(nested_result_section)
    assert any("`token_usage`" in item for item in nested_result_bullets)
    assert any("`structured_context`" in item for item in nested_result_bullets)
    assert any("`workspace_brief`" in item for item in nested_result_bullets)

    events_contract = _bullet_items(_section_text(doc_text, "Stable `events.jsonl` contract"))
    assert events_contract == V1_EVENTS_CONTRACT_BULLETS


def test_v1_run_artifacts_emit_required_files_and_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = PlanSpec(
        version=1,
        name="contract-plan",
        defaults=PlanDefaults(),
        tasks=[TaskSpec(id="contract", command="echo contract")],
        source_path=tmp_path / "plan.yaml",
    )
    plan.source_path.write_text("version: 1\nname: contract-plan\n", encoding="utf-8")

    monkeypatch.setattr("maestro_cli.scheduler.execute_task", _fake_execute_task)
    monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *args, **kwargs: None)

    run_result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
    run_path = run_result.run_path

    assert (run_path / "run_manifest.json").exists()
    assert (run_path / "run_summary.md").exists()
    assert (run_path / "events.jsonl").exists()
    assert (run_path / "contract.log").exists()
    assert (run_path / "contract.result.json").exists()

    manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert V1_MANIFEST_KEYS.issubset(manifest)

    manifest_task = manifest["task_results"]["contract"]
    assert V1_TASK_RESULT_KEYS.issubset(manifest_task)
    assert V1_OPTIONAL_TASK_RESULT_KEYS.issubset(manifest_task)
    assert V1_TOKEN_USAGE_KEYS.issubset(manifest_task["token_usage"])
    assert V1_STRUCTURED_CONTEXT_KEYS.issubset(manifest_task["structured_context"])
    assert V1_WORKSPACE_BRIEF_KEYS.issubset(manifest_task["workspace_brief"])

    task_result = json.loads((run_path / "contract.result.json").read_text(encoding="utf-8"))
    assert V1_TASK_RESULT_KEYS.issubset(task_result)
    assert V1_OPTIONAL_TASK_RESULT_KEYS.issubset(task_result)
    assert V1_TOKEN_USAGE_KEYS.issubset(task_result["token_usage"])
    assert V1_STRUCTURED_CONTEXT_KEYS.issubset(task_result["structured_context"])
    assert V1_WORKSPACE_BRIEF_KEYS.issubset(task_result["workspace_brief"])

    events = [
        json.loads(line)
        for line in (run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events
    assert all({"ts", "event"}.issubset(event) for event in events)
