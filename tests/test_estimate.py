from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.estimate import (
    PlanEstimate,
    estimate_plan,
    format_estimate,
    format_estimate_json,
)
from maestro_cli.loader import load_plan


def _write_plan(tmp_path: Path, body: str, name: str = "plan.yaml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _write_manifest(
    run_dir: Path, plan_name: str, task_costs: dict[str, float], run_id: str = "r1"
) -> None:
    run = run_dir / f"{run_id}_{plan_name}"
    run.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task_results": {
            tid: {"cost_usd": cost} for tid, cost in task_costs.items()
        }
    }
    (run / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


_BASIC_PLAN = """
version: 1
name: est-demo
defaults:
  claude:
    model: sonnet
tasks:
  - id: design
    engine: claude
    model: opus
    prompt: "Design the authentication module with secure hashing."
  - id: build
    command: ["echo", "build"]
    depends_on: [design]
"""


class TestEstimatePlan:
    def test_engine_task_priced_by_heuristic(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        report = estimate_plan(plan, tmp_path / "runs")

        design = next(t for t in report.task_estimates if t.task_id == "design")
        assert design.source == "heuristic"
        assert design.engine == "claude"
        assert design.model == "opus"
        assert design.cost_usd is not None and design.cost_usd > 0
        assert design.input_tokens > 0
        assert report.heuristic_tasks == 1

    def test_command_task_is_free(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        report = estimate_plan(plan, tmp_path / "runs")

        build = next(t for t in report.task_estimates if t.task_id == "build")
        assert build.kind == "command"
        assert build.source == "shell"
        assert build.cost_usd == 0.0

    def test_total_and_by_engine(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        report = estimate_plan(plan, tmp_path / "runs")

        design = next(t for t in report.task_estimates if t.task_id == "design")
        assert report.total_cost_usd == design.cost_usd
        assert report.by_engine["claude"] == design.cost_usd

    def test_default_model_inherited(self, tmp_path: Path) -> None:
        body = """
version: 1
name: est-demo
defaults:
  claude:
    model: haiku
tasks:
  - id: t1
    engine: claude
    prompt: "do a thing with enough words to count some tokens here please."
"""
        plan = load_plan(str(_write_plan(tmp_path, body)))
        report = estimate_plan(plan, tmp_path / "runs")
        assert report.task_estimates[0].model == "haiku"

    def test_local_engine_zero_cost(self, tmp_path: Path) -> None:
        body = """
version: 1
name: est-demo
tasks:
  - id: t1
    engine: ollama
    model: llama3
    prompt: "lint the code"
"""
        plan = load_plan(str(_write_plan(tmp_path, body)))
        report = estimate_plan(plan, tmp_path / "runs")
        est = report.task_estimates[0]
        assert est.source == "local"
        assert est.cost_usd == 0.0

    def test_subscription_engine_unpriced(self, tmp_path: Path) -> None:
        body = """
version: 1
name: est-demo
tasks:
  - id: t1
    engine: copilot
    model: sonnet
    prompt: "review the change"
"""
        plan = load_plan(str(_write_plan(tmp_path, body)))
        report = estimate_plan(plan, tmp_path / "runs")
        est = report.task_estimates[0]
        assert est.source == "subscription"
        assert est.cost_usd is None
        assert report.unpriced_tasks == 1

    def test_history_overrides_heuristic(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        run_dir = tmp_path / "runs"
        _write_manifest(run_dir, "est-demo", {"design": 0.42})

        report = estimate_plan(plan, run_dir)
        design = next(t for t in report.task_estimates if t.task_id == "design")
        assert design.source == "history"
        assert design.cost_usd == 0.42
        assert design.history_runs == 1
        assert report.history_tasks == 1

    def test_history_averages_multiple_runs(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        run_dir = tmp_path / "runs"
        _write_manifest(run_dir, "est-demo", {"design": 0.20}, run_id="r1")
        _write_manifest(run_dir, "est-demo", {"design": 0.40}, run_id="r2")

        report = estimate_plan(plan, run_dir)
        design = next(t for t in report.task_estimates if t.task_id == "design")
        assert design.cost_usd == pytest.approx(0.30)
        assert design.history_runs == 2

    def test_zero_cost_history_is_ignored(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        run_dir = tmp_path / "runs"
        _write_manifest(run_dir, "est-demo", {"design": 0.0})

        report = estimate_plan(plan, run_dir)
        design = next(t for t in report.task_estimates if t.task_id == "design")
        # A zero/None prior cost is not usable history → falls back to heuristic.
        assert design.source == "heuristic"

    def test_missing_prompt_file_is_unpriced(self, tmp_path: Path) -> None:
        body = """
version: 1
name: est-demo
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt_file: does_not_exist.txt
"""
        plan = load_plan(str(_write_plan(tmp_path, body)))
        report = estimate_plan(plan, tmp_path / "runs")
        est = report.task_estimates[0]
        assert est.source == "unpriced"
        assert est.cost_usd is None

    def test_prompt_file_is_resolved_and_priced(self, tmp_path: Path) -> None:
        (tmp_path / "p.txt").write_text(
            "A long prompt body with enough words to count several tokens here.",
            encoding="utf-8",
        )
        body = """
version: 1
name: est-demo
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt_file: p.txt
"""
        plan = load_plan(str(_write_plan(tmp_path, body)))
        report = estimate_plan(plan, tmp_path / "runs")
        est = report.task_estimates[0]
        assert est.source == "heuristic"
        assert est.cost_usd is not None and est.cost_usd > 0


class TestFormatEstimate:
    def test_text_contains_key_fields(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        report = estimate_plan(plan, tmp_path / "runs")
        text = format_estimate(report)
        assert "est-demo" in text
        assert "design" in text
        assert "Estimated total" in text
        assert "heuristic" in text

    def test_json_is_valid_and_structured(self, tmp_path: Path) -> None:
        plan = load_plan(str(_write_plan(tmp_path, _BASIC_PLAN)))
        report = estimate_plan(plan, tmp_path / "runs")
        payload = json.loads(format_estimate_json(report))
        assert payload["plan_name"] == "est-demo"
        assert "tasks" in payload and len(payload["tasks"]) == 2
        assert "total_cost_usd" in payload

    def test_empty_plan_estimate_to_dict(self) -> None:
        report = PlanEstimate(plan_name="empty")
        d = report.to_dict()
        assert d["plan_name"] == "empty"
        assert d["tasks"] == []
        assert d["total_cost_usd"] == 0.0


class TestEstimateCli:
    def test_cmd_estimate_text(self, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        from maestro_cli.cli import main

        plan_path = _write_plan(tmp_path, _BASIC_PLAN)
        rc = main(["estimate", str(plan_path), "--run-dir", str(tmp_path / "runs")])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Cost estimate: est-demo" in out

    def test_cmd_estimate_json(self, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        from maestro_cli.cli import main

        plan_path = _write_plan(tmp_path, _BASIC_PLAN)
        rc = main(["estimate", str(plan_path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["plan_name"] == "est-demo"
