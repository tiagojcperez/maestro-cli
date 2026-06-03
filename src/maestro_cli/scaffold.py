from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml

from .errors import PlanValidationError
from .models import EngineName, PlanBrief, TaskBrief, TaskType, Topology

# Valid values extracted from Literal types for runtime validation
_VALID_TASK_TYPES: set[str] = {
    "shell", "trivial-fix", "implementation", "complex-implementation",
    "code-review", "qa-verification", "build-verify", "security-audit",
    "branch-setup",
}
_VALID_TOPOLOGIES: set[str] = {"linear", "fan-out", "diamond", "pipeline"}


# ---------------------------------------------------------------------------
# Built-in workflow library catalog
# ---------------------------------------------------------------------------

_WORKFLOW_LIBRARIES: dict[str, dict[str, Any]] = {
    "rest-api": {
        "description": "REST API service (setup + endpoints + tests + review)",
        "goal": "Implement a REST API service with quality gates",
        "topology": "diamond",
        "include_quality_gates": True,
        "include_build_verify": True,
        "tasks": [
            {
                "id": "setup-project",
                "description": "Set up project structure and dependencies",
                "task_type": "shell",
            },
            {
                "id": "implement-models",
                "description": "Implement data models and database schema",
                "task_type": "implementation",
                "depends_on": ["setup-project"],
                "prompt_hint": "Create data models, database migrations, and schema definitions.",
            },
            {
                "id": "implement-endpoints",
                "description": "Implement API endpoints and route handlers",
                "task_type": "implementation",
                "depends_on": ["implement-models"],
                "prompt_hint": "Create REST endpoints with proper HTTP methods, validation, and error handling.",
            },
            {
                "id": "implement-tests",
                "description": "Write API tests",
                "task_type": "qa-verification",
                "depends_on": ["implement-endpoints"],
                "prompt_hint": "Write integration tests for all API endpoints. Cover success and error cases.",
            },
        ],
    },
    "refactor": {
        "description": "Code refactoring (analyse + implement + verify)",
        "goal": "Refactor code for improved quality",
        "topology": "linear",
        "include_quality_gates": True,
        "include_build_verify": True,
        "tasks": [
            {
                "id": "analyse-code",
                "description": "Analyse codebase and identify refactoring targets",
                "task_type": "code-review",
                "prompt_hint": "Identify code smells, duplication, and improvement opportunities. Produce a structured report.",
            },
            {
                "id": "implement-refactor",
                "description": "Apply refactoring changes",
                "task_type": "complex-implementation",
                "depends_on": ["analyse-code"],
                "prompt_hint": "Apply the refactoring changes identified in the analysis. Preserve all existing behaviour.",
            },
            {
                "id": "verify-refactor",
                "description": "Run tests and verify no regressions",
                "task_type": "build-verify",
                "depends_on": ["implement-refactor"],
            },
        ],
    },
    "security-review": {
        "description": "Security audit (scan + audit + remediate + verify)",
        "goal": "Security audit and remediation",
        "topology": "linear",
        "include_quality_gates": False,
        "include_build_verify": True,
        "playbook_ref": "Recipe 3: Security Audit",
        "tasks": [
            {
                "id": "dependency-scan",
                "description": "Scan dependencies for known vulnerabilities",
                "task_type": "shell",
                "prompt_hint": "Run dependency audit tools and report findings.",
            },
            {
                "id": "code-audit",
                "description": "Audit code for security vulnerabilities",
                "task_type": "security-audit",
                "depends_on": ["dependency-scan"],
                "prompt_hint": "Review code for OWASP Top 10, injection, auth bypass, and data exposure risks.",
            },
            {
                "id": "remediate",
                "description": "Fix identified security issues",
                "task_type": "implementation",
                "depends_on": ["code-audit"],
                "prompt_hint": "Apply fixes for all critical and high severity findings from the security audit.",
            },
            {
                "id": "verify-fixes",
                "description": "Verify security fixes and run regression tests",
                "task_type": "build-verify",
                "depends_on": ["remediate"],
            },
        ],
    },
    "bug-fix": {
        "description": "Bug fix (reproduce + fix + regression test)",
        "goal": "Fix a specific bug with regression verification",
        "topology": "linear",
        "include_quality_gates": False,
        "include_build_verify": True,
        "playbook_ref": "Recipe 5: Bug Fix with Regression Check",
        "tasks": [
            {
                "id": "reproduce",
                "description": "Reproduce the bug with a failing test",
                "task_type": "shell",
                "prompt_hint": "Run the failing test or reproduce the bug scenario.",
            },
            {
                "id": "fix",
                "description": "Fix the bug",
                "task_type": "implementation",
                "depends_on": ["reproduce"],
                "prompt_hint": "Fix the bug based on the reproduction output. The failing test should pass after the fix.",
            },
            {
                "id": "regression",
                "description": "Run full test suite to verify no regressions",
                "task_type": "build-verify",
                "depends_on": ["fix"],
            },
        ],
    },
    "test-backfill": {
        "description": "Test coverage backfill (parallel test writing + coverage gate)",
        "goal": "Increase test coverage for under-tested modules",
        "topology": "fan-out",
        "include_quality_gates": False,
        "include_build_verify": True,
        "playbook_ref": "Recipe 4: Test Backfill",
        "tasks": [
            {
                "id": "write-tests",
                "description": "Write tests for under-covered module",
                "task_type": "qa-verification",
                "prompt_hint": "Write comprehensive tests. Cover success, failure, and edge cases. Use tmp_path for file ops.",
            },
            {
                "id": "verify-tests",
                "description": "Run tests and check coverage",
                "task_type": "build-verify",
                "depends_on": ["write-tests"],
            },
        ],
    },
}

# Public constant for discoverability
WORKFLOW_LIBRARY_NAMES: set[str] = set(_WORKFLOW_LIBRARIES.keys())


def list_workflow_libraries() -> list[dict[str, str]]:
    """Return a summary of all built-in workflow libraries."""
    return [
        {"name": name, "description": lib["description"]}
        for name, lib in sorted(_WORKFLOW_LIBRARIES.items())
    ]


def _load_library(name_or_path: str) -> dict[str, Any]:
    """Load a workflow library by built-in name or file path.

    Returns the raw library dict with 'tasks' and optional metadata.
    """
    # Built-in library?
    if name_or_path in _WORKFLOW_LIBRARIES:
        return _WORKFLOW_LIBRARIES[name_or_path]

    # External file?
    lib_path = Path(name_or_path).resolve()
    if not lib_path.exists():
        raise PlanValidationError(
            f"Workflow library '{name_or_path}' not found. "
            f"Built-in libraries: {sorted(WORKFLOW_LIBRARY_NAMES)}"
        )

    try:
        raw = yaml.safe_load(lib_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlanValidationError(f"Invalid workflow library YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlanValidationError("Workflow library root must be an object")
    if "tasks" not in raw or not isinstance(raw["tasks"], list):
        raise PlanValidationError("Workflow library must have a 'tasks' list")

    return raw


def _merge_library_into_brief(
    library: dict[str, Any],
    brief_tasks: list[dict[str, Any]],
    brief_raw: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge library tasks with user brief tasks.

    User brief tasks override library tasks with the same ID.
    Library metadata (goal, topology, etc.) provides defaults that the
    brief can override.
    """
    # Build override index from user tasks
    override_index: dict[str, dict[str, Any]] = {}
    extra_tasks: list[dict[str, Any]] = []
    for task in brief_tasks:
        if not isinstance(task, dict):
            continue
        tid = str(task.get("id", ""))
        lib_ids = {str(t.get("id", "")) for t in library.get("tasks", [])}
        if tid and tid in lib_ids:
            override_index[tid] = task
        else:
            extra_tasks.append(task)

    # Merge: library tasks with overrides applied
    merged_tasks: list[dict[str, Any]] = []
    for lib_task in library.get("tasks", []):
        if not isinstance(lib_task, dict):
            continue
        tid = str(lib_task.get("id", ""))
        if tid in override_index:
            # Merge: library base + user override
            merged = dict(lib_task)
            merged.update(override_index[tid])
            merged_tasks.append(merged)
        else:
            merged_tasks.append(dict(lib_task))

    # Append extra user tasks at the end
    merged_tasks.extend(extra_tasks)

    # Merge metadata: library provides defaults, brief overrides
    merged_meta = dict(brief_raw)
    for key in ("goal", "topology", "include_quality_gates", "include_build_verify"):
        if key not in merged_meta and key in library:
            merged_meta[key] = library[key]

    return merged_tasks, merged_meta


# ---------------------------------------------------------------------------
# Model routing tables (from Agent Policy in CLAUDE.md)
# ---------------------------------------------------------------------------

# task_type -> (engine, model, reasoning_effort)
_MODEL_ROUTING: dict[TaskType, tuple[EngineName | None, str | None, str | None]] = {
    "shell": (None, None, None),
    "branch-setup": (None, None, None),
    "trivial-fix": ("claude", "haiku", None),
    "implementation": ("claude", "sonnet", None),
    "complex-implementation": ("claude", "sonnet", None),
    "code-review": ("claude", "sonnet", None),
    "qa-verification": ("claude", "sonnet", None),
    "build-verify": (None, None, None),
    "security-audit": ("claude", "opus", "high"),
}

# task_type -> default agent name
_AGENT_ROUTING: dict[str, str] = {
    "code-review": "code-reviewer",
    "qa-verification": "qa-engineer",
    "security-audit": "security-engineer",
}

_ANTI_STALLING_PROMPT = (
    "If you have read 5+ files without making any edits, stop analyzing and "
    "either write code or report what is blocking you. "
    "Do not get stuck in analysis loops -- bias toward action."
)

# ---------------------------------------------------------------------------
# Auto-split heuristic (T0.4)
# ---------------------------------------------------------------------------

# Matches common source/config file paths in prompt text.
# The exists() check below handles false positives.
_CODE_FILE_RE = re.compile(
    r"(?:^|[\s\"'`(,])"
    r"((?:[\w./\\-]+/)?[\w.-]+"
    r"\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|cpp|c|h|cs|php|sh|yaml|yml|json|sql|html|css|md|toml|cfg))"
    r"(?=$|[\s\"'`),;:])",
    re.IGNORECASE | re.MULTILINE,
)

_AUTO_SPLIT_LINE_THRESHOLD = 300

_READ_PLAN_PROMPT = """\
## Goal
{goal}

## Files to analyse
{file_list}

Read the file(s) listed above carefully. Produce ONLY a JSON change plan with this structure:

```json
{{
  "files": [
    {{
      "path": "relative/path/to/file.ext",
      "changes": [
        {{
          "location": "function/class name or section description",
          "action": "add|modify|delete",
          "content": "exact replacement code",
          "reason": "why this change is needed"
        }}
      ]
    }}
  ],
  "summary": "one-line description of all changes combined"
}}
```

Do NOT modify any files. Only read, analyse, and output the JSON plan.
"""

_APPLY_CHANGES_PROMPT = """\
## Change Plan (produced by read-plan step)
{{{{ {read_plan_id}.stdout_tail }}}}

## Original Goal
{goal}

Apply the change plan above to the codebase exactly as described. Work file by file. \
Make only the changes specified — do not modify anything else. \
After applying each change, confirm it was made.
"""


def _detect_large_files(
    prompt: str,
    workspace_root: str,
    threshold: int = _AUTO_SPLIT_LINE_THRESHOLD,
) -> list[str]:
    """Return relative paths of files mentioned in *prompt* that exist in
    *workspace_root* and have at least *threshold* lines."""
    root = Path(workspace_root)
    seen: set[str] = set()
    found: list[str] = []
    for match in _CODE_FILE_RE.finditer(prompt):
        raw = match.group(1).replace("\\", "/").strip()
        if raw in seen:
            continue
        seen.add(raw)
        candidate = root / raw
        if not (candidate.exists() and candidate.is_file()):
            continue
        try:
            lines = len(candidate.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        if lines >= threshold:
            found.append(raw)
    return found


def _generate_split_tasks(
    tb: "TaskBrief",  # noqa: F821 — forward ref resolved at call site
    large_files: list[str],
    deps: list[str],
    workdir: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """Return (tasks, final_task_id) for a read-plan + apply-changes pair.

    *final_task_id* is the apply task — quality gates should depend on it.
    """
    from .models import TaskBrief  # local import avoids circular dependency

    goal = tb.prompt_hint or tb.description or tb.id
    file_list = "\n".join(f"- {f}" for f in large_files)
    read_plan_id = f"{tb.id}-read-plan"
    apply_id = f"{tb.id}-apply"

    read_task: dict[str, Any] = {
        "id": read_plan_id,
        "description": f"Analyse {', '.join(large_files)} and produce a change plan",
        "engine": "claude",
        "model": "haiku",
        "timeout_sec": 300,
        "prompt": _READ_PLAN_PROMPT.format(goal=goal, file_list=file_list),
    }
    if deps:
        read_task["depends_on"] = list(deps)
    if workdir:
        read_task["workdir"] = workdir

    apply_task: dict[str, Any] = {
        "id": apply_id,
        "description": f"Apply planned changes to {', '.join(large_files)}",
        "engine": "claude",
        "depends_on": [read_plan_id],
        "append_system_prompt": _ANTI_STALLING_PROMPT,
        "prompt": _APPLY_CHANGES_PROMPT.format(read_plan_id=read_plan_id, goal=goal),
    }
    if workdir:
        apply_task["workdir"] = workdir

    return [read_task, apply_task], apply_id


def _route_model(task_type: TaskType) -> tuple[EngineName | None, str | None, str | None]:
    """Look up engine, model, and reasoning_effort for a task type."""
    return _MODEL_ROUTING.get(task_type, ("claude", "sonnet", None))


def _route_agent(task_type: TaskType, explicit_agent: str | None) -> str | None:
    """Look up the default agent for a task type, or use explicit override."""
    if explicit_agent:
        return explicit_agent
    return _AGENT_ROUTING.get(task_type)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

def _generate_branch_task(branch_name: str, workdir: str | None) -> dict[str, Any]:
    """Generate the Wave 0 branch creation task."""
    cmd = f"git checkout -b {branch_name} 2>/dev/null || git checkout {branch_name}"
    task: dict[str, Any] = {
        "id": "w0-branch",
        "description": f"Create or switch to branch {branch_name}",
        "command": cmd,
    }
    if workdir:
        task["workdir"] = workdir
    return task


def _generate_quality_gates(
    last_impl_ids: list[str],
    workdir: str | None,
    plan_name: str = "",
    plan_goal: str = "",
) -> list[dict[str, Any]]:
    """Generate code-review + qa-verification tasks.

    Uses ``context_mode: map_reduce`` for code review (when >= 3 upstream
    tasks) or ``summarized`` otherwise, giving the AI agent a structured
    synthesis instead of raw output dumps.

    Prompts include goal-backward verification (check work fulfills the
    declared objective) and a simplicity check (flag over-engineering).
    """
    tasks: list[dict[str, Any]] = []

    objective_section = (
        f"## Plan Objective\n"
        f"This plan ({plan_name}) aims to: {plan_goal or 'see task descriptions'}.\n"
        f"Verify that the work fulfills this declared objective.\n\n"
    )

    review_mode = "map_reduce" if len(last_impl_ids) >= 3 else "summarized"
    review: dict[str, Any] = {
        "id": "code-review",
        "description": "Review all implementation changes for quality and correctness",
        "depends_on": list(last_impl_ids),
        "context_from": ["*"],
        "context_mode": review_mode,
        "engine": "claude",
        "agent": "code-reviewer",
        "prompt": (
            "Review all code changes made by the implementation tasks.\n\n"
            + objective_section
            + "## Upstream Context\n"
            "{{ upstream_synthesis }}\n\n"
            "## Review Checklist\n"
            "Check for: correctness, type safety, error handling, test coverage, "
            "code style consistency.\n\n"
            "## Simplicity Check\n"
            "For each significant change, ask: is there a simpler approach that "
            "achieves the same result with less complexity? Flag any over-engineering.\n\n"
            "Provide a structured review with PASS/FAIL verdict.\n"
        ),
    }
    if workdir:
        review["workdir"] = workdir
    tasks.append(review)

    qa: dict[str, Any] = {
        "id": "qa-verification",
        "description": "Verify implementations pass tests and quality gates",
        "depends_on": list(last_impl_ids),
        "context_from": ["*"],
        "context_mode": "summarized",
        "engine": "claude",
        "agent": "qa-engineer",
        "prompt": (
            "Run tests and verify the implementation tasks produced correct results.\n\n"
            + objective_section
            + "## Upstream Task Summaries\n"
            + "".join(
                f"### {{{{ {tid}.summary }}}}\n"
                for tid in last_impl_ids
            )
            + "\nCheck: unit tests pass, no regressions, edge cases handled.\n"
            "Provide a structured QA report with PASS/FAIL verdict.\n"
        ),
    }
    if workdir:
        qa["workdir"] = workdir
    tasks.append(qa)

    return tasks


def _generate_build_verify(
    depends_on: list[str],
    workdir: str | None,
) -> dict[str, Any]:
    """Generate the final build verification task."""
    task: dict[str, Any] = {
        "id": "build-verify",
        "description": "Verify the project builds successfully",
        "depends_on": depends_on,
        "command": "echo 'TODO: replace with actual build command (e.g., npm run build, composer install)'",
        "verify_command": "echo 'TODO: replace with actual verification (e.g., npm test, pytest)'",
    }
    if workdir:
        task["workdir"] = workdir
    return task


# ---------------------------------------------------------------------------
# Main scaffolding
# ---------------------------------------------------------------------------

def scaffold_plan(brief: PlanBrief) -> str:
    """Generate a complete YAML plan from a high-level brief.

    Returns the YAML string, ready to be written to a file or
    passed to ``load_plan()`` for validation.
    """
    # Strict defaults sit comfortably above W20's 900s tight-timeout threshold
    # and pre-arm tasks with progressive backoff so that any ad-hoc retry
    # added by the author silences the warning automatically. Plan-level
    # max_cost_usd and budget_warning_pct match `maestro audit --fix` for
    # SEC001 alignment.
    if brief.strict_defaults:
        timeout_default = 1500
        retry_delay_default: list[float] | None = [60, 120]
    else:
        timeout_default = 600
        retry_delay_default = None

    defaults_block: dict[str, Any] = {
        "env": {"PYTHONUTF8": "1"},
        "timeout_sec": timeout_default,
        "requires_clean_worktree": False,
        "claude": {
            "model": "sonnet",
            "args": ["--dangerously-skip-permissions"],
        },
    }
    if retry_delay_default is not None:
        defaults_block["retry_delay_sec"] = retry_delay_default

    plan: dict[str, Any] = {
        "version": 1,
        "name": brief.name,
        "max_parallel": brief.max_parallel,
        "fail_fast": brief.fail_fast,
        "run_dir": ".maestro-runs",
        "defaults": defaults_block,
        "tasks": [],
    }

    if brief.strict_defaults:
        plan["max_cost_usd"] = 10.0
        plan["budget_warning_pct"] = 0.8

    if brief.workspace_root:
        plan["workspace_root"] = brief.workspace_root

    task_list: list[dict[str, Any]] = []
    impl_task_ids: list[str] = []

    # Wave 0: Branch setup
    if brief.branch_name:
        task_list.append(_generate_branch_task(brief.branch_name, brief.workspace_root))

    # User-defined tasks
    for tb in brief.tasks:
        engine, model, reasoning = _route_model(tb.task_type)
        # Allow explicit engine override from brief
        if tb.engine:
            engine = tb.engine
        agent = _route_agent(tb.task_type, tb.agent)

        # Dependencies: default to w0-branch if no explicit deps and branch exists
        deps = list(tb.depends_on)
        if brief.branch_name and not deps and tb.id != "w0-branch":
            deps = ["w0-branch"]

        # Resolve workdir once — needed by both normal and split paths
        workdir: str | None = tb.workdir or brief.workspace_root or None

        # Auto-split: impl tasks that mention large files → read-plan + apply pair
        if (
            tb.auto_split
            and brief.workspace_root
            and tb.task_type in ("implementation", "complex-implementation")
        ):
            large_files = _detect_large_files(
                tb.prompt_hint or tb.description or "",
                brief.workspace_root,
            )
            if large_files:
                split_tasks, final_id = _generate_split_tasks(tb, large_files, deps, workdir)
                task_list.extend(split_tasks)
                impl_task_ids.append(final_id)
                continue

        task: dict[str, Any] = {
            "id": tb.id,
            "description": tb.description or tb.id,
        }

        if deps:
            task["depends_on"] = deps

        if engine:
            task["engine"] = engine
            if agent:
                task["agent"] = agent
            if model and model != "sonnet":
                task["model"] = model
            if reasoning:
                task["reasoning_effort"] = reasoning

            if tb.prompt_hint:
                task["prompt"] = f"# TODO: Expand this prompt\n{tb.prompt_hint}\n"
            else:
                task["prompt"] = f"# TODO: Write prompt for {tb.id}\n{tb.description or tb.id}\n"

            if tb.task_type in ("implementation", "complex-implementation", "trivial-fix"):
                task["append_system_prompt"] = _ANTI_STALLING_PROMPT
        else:
            task["command"] = f"echo 'TODO: implement {tb.id}'"

        if workdir:
            task["workdir"] = workdir

        task_list.append(task)
        if tb.task_type in ("implementation", "complex-implementation", "trivial-fix"):
            impl_task_ids.append(tb.id)

    # Quality gates
    if brief.include_quality_gates and impl_task_ids:
        gates = _generate_quality_gates(
            impl_task_ids, brief.workspace_root, brief.name, brief.goal,
        )
        task_list.extend(gates)

    # Build verification
    if brief.include_build_verify:
        verify_deps: list[str] = []
        if brief.include_quality_gates and impl_task_ids:
            verify_deps = ["code-review", "qa-verification"]
        elif impl_task_ids:
            verify_deps = list(impl_task_ids)
        elif task_list:
            verify_deps = [task_list[-1]["id"]]
        task_list.append(_generate_build_verify(verify_deps, brief.workspace_root))

    plan["tasks"] = task_list
    return yaml.dump(plan, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Brief loading
# ---------------------------------------------------------------------------

def load_brief(
    path: str | Path,
    *,
    library_override: str | None = None,
    strict_defaults: bool = False,
) -> PlanBrief:
    """Load a plan brief from a YAML file.

    Brief format::

        name: my-feature
        goal: "Add user authentication"
        workspace_root: C:/path/to/project
        branch_name: feature/auth
        library: rest-api          # built-in or path to library YAML
        max_parallel: 3
        tasks:
          - id: implement-endpoints  # overrides library task
            prompt_hint: "Custom prompt for endpoints"
          - id: extra-task           # appended after library tasks
            task_type: implementation

    When ``library`` is set, the library's tasks form the base and the
    brief's tasks can override (by matching ID) or extend (new IDs).
    The ``library_override`` parameter takes precedence over the YAML field.
    """
    brief_path = Path(path).resolve()
    if not brief_path.exists():
        raise PlanValidationError(f"Brief file not found: {brief_path}")

    try:
        raw = yaml.safe_load(brief_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlanValidationError(f"Invalid brief YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlanValidationError("Brief root must be an object")

    name = str(raw.get("name", "")).strip()
    if not name:
        raise PlanValidationError("Brief must have a 'name' field")

    # Resolve library (CLI flag overrides YAML field)
    library_name = library_override or (str(raw["library"]) if raw.get("library") else None)
    library: dict[str, Any] | None = None
    if library_name:
        library = _load_library(library_name)

    tasks_raw = raw.get("tasks", [])
    if not isinstance(tasks_raw, list):
        raise PlanValidationError("Brief tasks must be a list")

    # Merge library tasks with brief tasks if library is set
    if library is not None:
        tasks_raw, raw = _merge_library_into_brief(library, tasks_raw, raw)

    # When library provides tasks and brief has none, allow empty brief tasks
    if not tasks_raw and library is None:
        pass  # will be caught by scaffold_plan if needed

    tasks: list[TaskBrief] = []
    for idx, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise PlanValidationError(f"Brief tasks[{idx}] must be an object")

        task_id = str(item.get("id", "")).strip()
        if not task_id:
            raise PlanValidationError(f"Brief tasks[{idx}].id is required")

        raw_task_type = str(item.get("task_type", "implementation"))
        if raw_task_type not in _VALID_TASK_TYPES:
            raise PlanValidationError(
                f"Brief tasks[{idx}].task_type '{raw_task_type}' is not valid. "
                f"Allowed: {sorted(_VALID_TASK_TYPES)}"
            )

        deps_raw = item.get("depends_on", [])
        if isinstance(deps_raw, str):
            deps_raw = [deps_raw]

        tasks.append(TaskBrief(
            id=task_id,
            description=str(item.get("description", "") or ""),
            task_type=raw_task_type,  # type: ignore[arg-type]
            depends_on=[str(d) for d in deps_raw],
            engine=cast(EngineName, str(item["engine"])) if item.get("engine") else None,
            agent=str(item["agent"]) if item.get("agent") else None,
            workdir=str(item["workdir"]) if item.get("workdir") else None,
            prompt_hint=str(item.get("prompt_hint", "") or ""),
            auto_split=bool(item.get("auto_split", True)),
        ))

    raw_topology = str(raw.get("topology", "pipeline"))
    if raw_topology not in _VALID_TOPOLOGIES:
        raise PlanValidationError(
            f"Brief topology '{raw_topology}' is not valid. "
            f"Allowed: {sorted(_VALID_TOPOLOGIES)}"
        )

    # CLI flag wins over YAML field for opt-in features.
    strict_from_yaml = bool(raw.get("strict_defaults", False))
    return PlanBrief(
        name=name,
        goal=str(raw.get("goal", "") or ""),
        workspace_root=str(raw["workspace_root"]) if raw.get("workspace_root") else None,
        branch_name=str(raw["branch_name"]) if raw.get("branch_name") else None,
        tasks=tasks,
        topology=raw_topology,  # type: ignore[arg-type]
        include_quality_gates=bool(raw.get("include_quality_gates", True)),
        include_build_verify=bool(raw.get("include_build_verify", True)),
        max_parallel=int(raw.get("max_parallel", 3)),
        fail_fast=bool(raw.get("fail_fast", True)),
        library=library_name,
        strict_defaults=strict_defaults or strict_from_yaml,
    )


# ---------------------------------------------------------------------------
# Cost safety validation
# ---------------------------------------------------------------------------

def validate_plan_cost_safety(plan_yaml: str) -> list[str]:
    """Check a plan for cost optimization issues.

    Returns a list of warning strings (empty list = no issues).
    """
    raw = yaml.safe_load(plan_yaml)
    if not isinstance(raw, dict):
        return ["Plan root is not a valid object"]

    warnings: list[str] = []
    tasks = raw.get("tasks", [])
    if not isinstance(tasks, list):
        return ["Plan has no valid tasks list"]

    has_review = False
    has_qa = False
    has_build = False
    opus_tasks: list[str] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        task_id = str(task.get("id", "?"))
        engine = task.get("engine")
        model = str(task.get("model", "") or "")

        if model == "opus":
            opus_tasks.append(task_id)

        if "review" in task_id or task.get("agent") == "code-reviewer":
            has_review = True
        if "qa" in task_id or task.get("agent") == "qa-engineer":
            has_qa = True
        if "build" in task_id or "verify" in task_id:
            has_build = True

        # reasoning_effort on non-Opus claude model
        if (
            engine == "claude"
            and model not in ("opus", "")
            and task.get("reasoning_effort")
        ):
            warnings.append(
                f"Task '{task_id}': reasoning_effort is set but model is '{model}' "
                f"(only effective on Opus)"
            )

    if len(opus_tasks) > 2:
        warnings.append(
            f"Cost concern: {len(opus_tasks)} tasks use Opus model: {opus_tasks}. "
            f"Consider using Sonnet for non-security/architecture tasks."
        )

    if not has_review:
        warnings.append("No code review task found. Consider adding a code-reviewer task.")
    if not has_qa:
        warnings.append("No QA verification task found. Consider adding a qa-engineer task.")
    if not has_build:
        warnings.append("No build verification task found. Consider adding a build step.")

    return warnings
