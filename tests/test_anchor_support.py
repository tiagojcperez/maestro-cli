from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
import pytest

from maestro_cli.loader import load_plan


def test_yaml_safe_load_handles_anchors() -> None:
    """Verify that yaml.safe_load() resolves anchors and merge keys."""
    yaml_text = """
_defaults: &defaults
  engine: claude
  model: sonnet

data:
  item1:
    <<: *defaults
    name: "Task A"
  item2:
    <<: *defaults
    name: "Task B"
"""
    parsed = yaml.safe_load(yaml_text)
    assert parsed["data"]["item1"]["engine"] == "claude"
    assert parsed["data"]["item1"]["model"] == "sonnet"
    assert parsed["data"]["item2"]["engine"] == "claude"
    assert parsed["data"]["item2"]["model"] == "sonnet"


def test_load_plan_with_anchors(tmp_path: Path) -> None:
    """Verify that load_plan() works with YAML anchors and merge keys."""
    plan_yaml = """
version: 1
name: anchor-test-plan
max_parallel: 2

_impl_defaults: &impl_defaults
  engine: claude
  model: sonnet
  edit_policy: efficient
  max_retries: 1

tasks:
  - id: task-a
    <<: *impl_defaults
    prompt: "Do task A"

  - id: task-b
    <<: *impl_defaults
    prompt: "Do task B"
    depends_on: [task-a]

  - id: qa-check
    engine: claude
    model: opus
    prompt: "Review implementations"
    depends_on: [task-a, task-b]
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(plan_yaml, encoding="utf-8")

    plan = load_plan(plan_file)

    # Verify plan loads correctly
    assert plan.name == "anchor-test-plan"
    assert len(plan.tasks) == 3

    # Verify anchor-resolved fields on task-a
    assert plan.tasks[0].id == "task-a"
    assert plan.tasks[0].engine == "claude"
    assert plan.tasks[0].model == "sonnet"
    assert plan.tasks[0].edit_policy == "efficient"
    assert plan.tasks[0].max_retries == 1

    # Verify anchor-resolved fields on task-b
    assert plan.tasks[1].id == "task-b"
    assert plan.tasks[1].engine == "claude"
    assert plan.tasks[1].model == "sonnet"
    assert plan.tasks[1].max_retries == 1

    # Verify qa-check (no anchor, explicit fields)
    assert plan.tasks[2].id == "qa-check"
    assert plan.tasks[2].engine == "claude"
    assert plan.tasks[2].model == "opus"


def test_load_plan_with_nested_anchors(tmp_path: Path) -> None:
    """Verify that nested anchors and multiple merge keys work."""
    plan_yaml = """
version: 1
name: nested-anchor-plan

_base: &base
  engine: claude
  allow_failure: false

_implementation: &impl
  <<: *base
  edit_policy: efficient
  max_retries: 1

tasks:
  - id: feature
    <<: *impl
    model: sonnet
    prompt: "Implement feature"

  - id: review
    <<: *impl
    model: opus
    reasoning_effort: high
    prompt: "Review feature"
    depends_on: [feature]
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(plan_yaml, encoding="utf-8")

    plan = load_plan(plan_file)

    # Both tasks should inherit from nested anchors
    assert plan.tasks[0].engine == "claude"
    assert plan.tasks[0].allow_failure is False
    assert plan.tasks[0].edit_policy == "efficient"

    assert plan.tasks[1].engine == "claude"
    assert plan.tasks[1].model == "opus"
    assert plan.tasks[1].reasoning_effort == "high"
