from __future__ import annotations

import os
from pathlib import Path

from maestro_cli.loader import load_plan
from maestro_cli.models import PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.runners import _load_prompt, _mask_secrets, build_command


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(task: TaskSpec, *, goal: str = "", source_path: Path | None = None) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test-plan",
        goal=goal,
        defaults=PlanDefaults(),
        tasks=[task],
        source_path=source_path,
    )


def test_goal_parsed_from_yaml(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: "Build the feature"
tasks:
  - id: t1
    command: "echo hello"
""",
    ))
    assert plan.goal == "Build the feature"


def test_goal_default_empty(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""",
    ))
    assert plan.goal == ""


def test_goal_in_to_dict() -> None:
    plan = PlanSpec(
        version=1,
        name="test-plan",
        goal="Build the feature",
        defaults=PlanDefaults(),
        tasks=[TaskSpec(id="t1", command="echo hello")],
    )
    data = plan.to_dict()
    assert "goal" in data
    assert data["goal"] == "Build the feature"


def test_goal_template_variable(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: test-plan\n", encoding="utf-8")
    task = TaskSpec(id="t1", engine="claude", prompt="Task: {{ goal }}")
    plan = _make_plan(task, goal="Build the feature", source_path=plan_path)

    resolved = _load_prompt(plan, task, upstream_results=None, context_synthesis="")

    assert "Task: Build the feature" in resolved


def test_goal_prepended_to_engine_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
    task = TaskSpec(id="t1", engine="qwen", prompt="Implement the change")
    plan = _make_plan(task, goal="Build the feature")

    cmd, shell = build_command(plan, task, tmp_path)

    assert not shell
    prompt = cmd[cmd.index("--prompt") + 1]
    assert prompt.startswith("Goal: Build the feature\n\n")
    assert prompt.endswith("Implement the change")


def test_goal_not_prepended_to_command_task(tmp_path: Path) -> None:
    task = TaskSpec(id="t1", command="echo hello")
    plan = _make_plan(task, goal="Build the feature")

    cmd, shell = build_command(plan, task, tmp_path)

    assert cmd == "echo hello"
    assert shell is True


def test_goal_empty_no_prepend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
    task = TaskSpec(id="t1", engine="qwen", prompt="Implement the change")
    plan = _make_plan(task, goal="")

    cmd, shell = build_command(plan, task, tmp_path)

    assert not shell
    prompt = cmd[cmd.index("--prompt") + 1]
    assert not prompt.startswith("Goal:")
    assert prompt == "Implement the change"


def test_goal_preserved_in_matrix_expansion(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: "Build the feature"
tasks:
  - id: build
    engine: claude
    prompt: "Task for {{ matrix.os }} in service of {{ goal }}"
    matrix:
      os: [linux, windows]
""",
    ))

    assert plan.goal == "Build the feature"
    assert {task.id for task in plan.tasks} == {"build.os-linux", "build.os-windows"}
    for task in plan.tasks:
        resolved = _load_prompt(plan, task, upstream_results=None, context_synthesis="")
        assert resolved.startswith("Goal: Build the feature\n\n")
        assert "in service of Build the feature" in resolved


def test_goal_with_template_pattern(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
    task = TaskSpec(id="t1", engine="qwen", prompt="Implement the change")
    goal = "Build {{ task-id.stdout_tail }} feature"
    plan = _make_plan(task, goal=goal, source_path=tmp_path / "plan.yaml")

    cmd, shell = build_command(plan, task, tmp_path)

    assert not shell
    prompt = cmd[cmd.index("--prompt") + 1]
    assert prompt.startswith(f"Goal: {goal}\n\n")
    assert "{{ task-id.stdout_tail }}" in prompt


def test_goal_with_newlines(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: |-
  Line 1
  Line 2
  Line 3
tasks:
  - id: t1
    engine: claude
    prompt: "Implement the change"
""",
    ))

    resolved = _load_prompt(plan, plan.tasks[0], upstream_results=None, context_synthesis="")

    assert plan.goal == "Line 1\nLine 2\nLine 3"
    assert resolved.startswith("Goal: Line 1\nLine 2\nLine 3\n\n")


def test_goal_with_quotes(tmp_path: Path) -> None:
    plan_single = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: 'It''s called "Phase B" with `backticks`'
tasks:
  - id: t1
    engine: claude
    prompt: "Task: {{ goal }}"
""",
    ))
    resolved_single = _load_prompt(plan_single, plan_single.tasks[0], upstream_results=None, context_synthesis="")
    assert plan_single.goal == 'It\'s called "Phase B" with `backticks`'
    assert 'Task: It\'s called "Phase B" with `backticks`' in resolved_single

    plan_double = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: "It’s called 'Phase B' with \\\"quotes\\\" and `backticks`"
tasks:
  - id: t1
    engine: claude
    prompt: "Task: {{ goal }}"
""",
    ))
    resolved_double = _load_prompt(plan_double, plan_double.tasks[0], upstream_results=None, context_synthesis="")
    assert plan_double.goal == 'It’s called \'Phase B\' with "quotes" and `backticks`'
    assert 'Task: It’s called \'Phase B\' with "quotes" and `backticks`' in resolved_double


def test_goal_with_unicode_emoji(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: "Construir o 🚀 sistema de deploy"
tasks:
  - id: t1
    engine: claude
    prompt: "Task: {{ goal }}"
""",
    ))

    resolved = _load_prompt(plan, plan.tasks[0], upstream_results=None, context_synthesis="")

    assert plan.goal == "Construir o 🚀 sistema de deploy"
    assert "Task: Construir o 🚀 sistema de deploy" in resolved


def test_goal_very_long_1000_chars(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
    goal = "x" * 1000
    task = TaskSpec(id="t1", engine="qwen", prompt="Implement the change")
    plan = _make_plan(task, goal=goal, source_path=tmp_path / "plan.yaml")

    cmd, shell = build_command(plan, task, tmp_path)

    assert not shell
    prompt = cmd[cmd.index("--prompt") + 1]
    assert prompt == f"Goal: {goal}\n\nImplement the change"
    assert len(goal) == 1000


def test_goal_with_dollar_signs(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: "Keep cost under $5.00 per run"
tasks:
  - id: t1
    engine: claude
    prompt: "Task: {{ goal }}"
""",
    ))

    resolved = _load_prompt(plan, plan.tasks[0], upstream_results=None, context_synthesis="")

    assert plan.goal == "Keep cost under $5.00 per run"
    assert "Task: Keep cost under $5.00 per run" in resolved


def test_goal_combined_with_append_system_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
    task = TaskSpec(
        id="t1",
        engine="claude",
        prompt="Implement the change",
        append_system_prompt="Always include rollback notes",
    )
    plan = _make_plan(task, goal="Build the feature", source_path=tmp_path / "plan.yaml")

    cmd, shell = build_command(plan, task, tmp_path)

    assert not shell
    assert "--append-system-prompt" in cmd
    system_idx = cmd.index("--append-system-prompt")
    assert cmd[system_idx + 1] == "Always include rollback notes"
    assert cmd[-1].startswith("Goal: Build the feature\n\n")
    assert cmd[-1].endswith("Implement the change")


def test_goal_as_template_variable(tmp_path: Path) -> None:
    goal = "Stress-test and harden the v1.1.0 event callback, --output live, and goal: field before building Phase B TUI"
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: test-plan\n", encoding="utf-8")
    task = TaskSpec(id="t1", engine="claude", prompt="Task: {{ goal }}")
    plan = _make_plan(task, goal=goal, source_path=plan_path)

    resolved = _load_prompt(plan, task, upstream_results=None, context_synthesis="")

    assert f"Task: {goal}" in resolved


def test_goal_in_imports(tmp_path: Path) -> None:
    sub_plan = tmp_path / "sub.yaml"
    sub_plan.write_text(
        """\
version: 1
name: sub-plan
tasks:
  - id: sub-task
    command: "echo sub"
""",
        encoding="utf-8",
    )

    parent_plan = tmp_path / "plan.yaml"
    parent_plan.write_text(
        f"""\
version: 1
name: parent-plan
goal: "Parent goal"
imports:
  - path: {sub_plan.name}
    prefix: lib
tasks:
  - id: main-task
    command: "echo main"
""",
        encoding="utf-8",
    )

    plan = load_plan(parent_plan)

    assert plan.goal == "Parent goal"
    task_ids = {t.id for t in plan.tasks}
    assert "lib/sub-task" in task_ids
    assert "main-task" in task_ids


def test_goal_roundtrip_to_dict() -> None:
    plan = PlanSpec(
        version=1,
        name="test-plan",
        goal="Test",
        defaults=PlanDefaults(),
        tasks=[TaskSpec(id="t1", command="echo hello")],
    )

    d = plan.to_dict()

    assert d["goal"] == "Test"


def test_goal_secrets_masking(tmp_path: Path, monkeypatch: object) -> None:
    secret_value = "ABC123"
    monkeypatch.setenv("MY_TOKEN", secret_value)  # type: ignore[attr-defined]

    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: test-plan\n", encoding="utf-8")
    task = TaskSpec(id="t1", engine="claude", prompt="Deploy with token {{ goal }}")
    plan = PlanSpec(
        version=1,
        name="test-plan",
        goal=f"Deploy with token {secret_value}",
        defaults=PlanDefaults(),
        tasks=[task],
        source_path=plan_path,
        secrets=["MY_TOKEN"],
    )

    prompt_text = _load_prompt(plan, task, upstream_results=None, context_synthesis="")

    # GAP: goal text is not passed through _mask_secrets inside _load_prompt.
    # The secret value appears in the raw prompt returned by _load_prompt.
    assert secret_value in prompt_text

    # _mask_secrets itself works correctly when called explicitly.
    masked = _mask_secrets(prompt_text, {secret_value})
    assert secret_value not in masked
    assert "***" in masked


def test_goal_empty_string_in_yaml(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: ""
tasks:
  - id: t1
    engine: claude
    prompt: "Implement the change"
""",
    ))

    assert plan.goal == ""
    resolved = _load_prompt(plan, plan.tasks[0], upstream_results=None, context_synthesis="")
    assert not resolved.startswith("Goal:")
    assert resolved == "Implement the change"


def test_goal_numeric_coerced_to_string(tmp_path: Path) -> None:
    plan = load_plan(_write_plan(
        tmp_path,
        """\
version: 1
name: test-plan
goal: 42
tasks:
  - id: t1
    command: "echo hello"
""",
    ))

    assert plan.goal == "42"
