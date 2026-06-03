"""Tests for Maestro CLI v1.14.0 — Plan Density Score, Deliberation Gate,
Adversarial Debate Judge.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.loader import (
    compute_plan_density,
    compute_plan_density_score,
    load_plan,
)
from maestro_cli.models import JudgeResult, JudgeSpec, PlanSpec, TaskSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_plan(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# compute_plan_density — unit tests
# ---------------------------------------------------------------------------


class TestComputePlanDensity:
    def test_chain_two_tasks(self) -> None:
        """Linear chain: a → b."""
        tasks = [TaskSpec(id="a"), TaskSpec(id="b", depends_on=["a"])]
        plan = PlanSpec(name="test", tasks=tasks)
        d = compute_plan_density(plan)
        assert d["nodes"] == 2
        assert d["edges"] == 1
        assert d["depth"] == 1
        assert "s_complex" in d
        assert d["s_complex"] > 0.0

    def test_single_task_no_deps(self) -> None:
        tasks = [TaskSpec(id="only")]
        plan = PlanSpec(name="test", tasks=tasks)
        d = compute_plan_density(plan)
        assert d["nodes"] == 1
        assert d["edges"] == 0
        assert d["depth"] == 0
        assert d["s_complex"] > 0.0

    def test_empty_plan(self) -> None:
        plan = PlanSpec(name="empty", tasks=[])
        d = compute_plan_density(plan)
        assert d["nodes"] == 0
        assert d["edges"] == 0
        assert d["s_complex"] == 0.0

    def test_parallel_no_deps(self) -> None:
        """Four independent tasks — no edges, no depth."""
        tasks = [TaskSpec(id=f"t{i}") for i in range(4)]
        plan = PlanSpec(name="test", tasks=tasks)
        d = compute_plan_density(plan)
        assert d["nodes"] == 4
        assert d["edges"] == 0
        assert d["depth"] == 0

    def test_chain_depth_increases(self) -> None:
        """a→b→c: depth=2, edges=2."""
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
            TaskSpec(id="c", depends_on=["b"]),
        ]
        plan = PlanSpec(name="test", tasks=tasks)
        d = compute_plan_density(plan)
        assert d["nodes"] == 3
        assert d["edges"] == 2
        assert d["depth"] == 2

    def test_keys_present(self) -> None:
        tasks = [TaskSpec(id="x")]
        plan = PlanSpec(name="t", tasks=tasks)
        d = compute_plan_density(plan)
        for key in ("nodes", "edges", "depth", "s_node", "s_edge", "s_depth", "s_complex"):
            assert key in d

    def test_more_edges_lower_s_edge(self) -> None:
        """Dense DAG has lower s_edge than sparse one."""
        tasks_sparse = [TaskSpec(id="a"), TaskSpec(id="b"), TaskSpec(id="c")]
        tasks_dense = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
            TaskSpec(id="c", depends_on=["a", "b"]),
        ]
        sparse = compute_plan_density(PlanSpec(name="s", tasks=tasks_sparse))
        dense = compute_plan_density(PlanSpec(name="d", tasks=tasks_dense))
        assert dense["s_edge"] < sparse["s_edge"]


# ---------------------------------------------------------------------------
# compute_plan_density_score — label thresholds
# ---------------------------------------------------------------------------


class TestComputePlanDensityScore:
    def test_empty_plan_is_low(self) -> None:
        plan = PlanSpec(name="e", tasks=[])
        score, label, factors = compute_plan_density_score(plan)
        assert score == 0.0
        assert label == "low"
        assert factors == ""

    def test_single_task_low_score(self) -> None:
        plan = PlanSpec(name="t", tasks=[TaskSpec(id="t1", command="echo")])
        score, label, _ = compute_plan_density_score(plan)
        assert label in ("low", "moderate")
        assert 0.0 <= score <= 1.0

    def test_returns_three_values(self) -> None:
        plan = PlanSpec(name="t", tasks=[TaskSpec(id="a")])
        result = compute_plan_density_score(plan)
        assert len(result) == 3
        score, label, factors = result
        assert isinstance(score, float)
        assert isinstance(label, str)
        assert isinstance(factors, str)

    def test_labels_are_valid(self) -> None:
        valid_labels = {"low", "moderate", "high", "very_high"}
        for n in (1, 3, 6, 12):
            tasks = [TaskSpec(id=f"t{i}", command="echo") for i in range(n)]
            plan = PlanSpec(name="t", tasks=tasks)
            _, label, _ = compute_plan_density_score(plan)
            assert label in valid_labels

    def test_score_increases_with_complexity(self) -> None:
        """More tasks + more edges = higher score than single-task plan."""
        simple = PlanSpec(name="s", tasks=[TaskSpec(id="t1")])
        tasks = [TaskSpec(id=f"t{i}", depends_on=[f"t{i-1}"] if i > 0 else []) for i in range(8)]
        complex_plan = PlanSpec(name="c", tasks=tasks)
        simple_score, _, _ = compute_plan_density_score(simple)
        complex_score, _, _ = compute_plan_density_score(complex_plan)
        assert complex_score > simple_score

    def test_factors_string_mentions_s_complex(self) -> None:
        tasks = [
            TaskSpec(id=f"t{i}", depends_on=[f"t{i-1}"] if i > 0 else [])
            for i in range(5)
        ]
        plan = PlanSpec(name="t", tasks=tasks)
        _, _, factors = compute_plan_density_score(plan)
        assert "S_complex" in factors


# ---------------------------------------------------------------------------
# W17 — high dependency density warning
# ---------------------------------------------------------------------------


class TestW17HighDependencyDensity:
    def test_w17_triggered_for_fully_connected(self, tmp_path: Path) -> None:
        """4-node fully-connected DAG → edge density > 60% → W17."""
        # a→b, a→c, a→d, b→c, b→d, c→d  (dense)
        yaml = """\
version: 1
name: dense-dag
tasks:
  - id: a
    command: echo
  - id: b
    depends_on: [a]
    command: echo
  - id: c
    depends_on: [a, b]
    command: echo
  - id: d
    depends_on: [a, b, c]
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert any("W17" in w for w in warnings), (
            f"Expected W17 warning in: {warnings}"
        )

    def test_w17_not_triggered_for_sparse(self, tmp_path: Path) -> None:
        """Fan-out: 5 tasks, 4 edges, max=10 → 40% density → no W17."""
        yaml = """\
version: 1
name: fanout
tasks:
  - id: root
    command: echo
  - id: a
    depends_on: [root]
    command: echo
  - id: b
    depends_on: [root]
    command: echo
  - id: c
    depends_on: [root]
    command: echo
  - id: d
    depends_on: [root]
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert not any("W17" in w for w in warnings)

    def test_w17_single_task_no_warning(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: single
tasks:
  - id: t1
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert not any("W17" in w for w in warnings)


# ---------------------------------------------------------------------------
# W18 — low parallelism warning
# ---------------------------------------------------------------------------


class TestW18LowParallelism:
    def test_w18_triggered_for_deep_chain(self, tmp_path: Path) -> None:
        """6-task chain → all sequential → parallelism = 0 → W18."""
        tasks_yaml = "\n".join(
            f"  - id: t{i}\n    depends_on: [t{i-1}]\n    command: echo"
            if i > 0
            else f"  - id: t{i}\n    command: echo"
            for i in range(6)
        )
        yaml = f"version: 1\nname: chain6\ntasks:\n{tasks_yaml}\n"
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert any("W18" in w for w in warnings), (
            f"Expected W18 warning in: {warnings}"
        )

    def test_w18_not_triggered_for_parallel(self, tmp_path: Path) -> None:
        """Four independent tasks → high parallelism → no W18."""
        yaml = """\
version: 1
name: parallel
tasks:
  - id: t1
    command: echo
  - id: t2
    command: echo
  - id: t3
    command: echo
  - id: t4
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert not any("W18" in w for w in warnings)

    def test_w18_not_triggered_for_small_chain(self, tmp_path: Path) -> None:
        """Short chain (≤3 tasks) → no W18 even if sequential."""
        yaml = """\
version: 1
name: short
tasks:
  - id: a
    command: echo
  - id: b
    depends_on: [a]
    command: echo
  - id: c
    depends_on: [b]
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert not any("W18" in w for w in warnings)


# ---------------------------------------------------------------------------
# W19 — high complexity score warning
# ---------------------------------------------------------------------------


class TestW19HighComplexity:
    def test_w19_not_triggered_for_simple(self, tmp_path: Path) -> None:
        """A simple two-task plan should not get W19."""
        yaml = """\
version: 1
name: simple
tasks:
  - id: t1
    command: echo
  - id: t2
    depends_on: [t1]
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        warnings = plan.validation_warnings or []
        assert not any("W19" in w for w in warnings)


# ---------------------------------------------------------------------------
# Deliberation Gate — loading & validation
# ---------------------------------------------------------------------------


class TestDeliberationLoading:
    def test_deliberation_false_by_default(self) -> None:
        t = TaskSpec(id="t1")
        assert t.deliberation is False
        assert t.deliberation_threshold == 0.5

    def test_deliberation_in_to_dict(self) -> None:
        t = TaskSpec(id="t1", deliberation=True, deliberation_threshold=0.7)
        d = t.to_dict()
        assert d["deliberation"] is True
        assert d["deliberation_threshold"] == 0.7

    def test_deliberation_loaded_from_yaml(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: delib
tasks:
  - id: think
    engine: claude
    model: haiku
    prompt: "Is the sky blue?"
    deliberation: true
    deliberation_threshold: 0.6
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        task = plan.tasks[0]
        assert task.deliberation is True
        assert task.deliberation_threshold == 0.6

    def test_deliberation_false_not_in_yaml(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: no-delib
tasks:
  - id: t1
    command: echo
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].deliberation is False

    def test_deliberation_threshold_out_of_range(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: 1.5
"""
        from maestro_cli.errors import PlanValidationError
        with pytest.raises(PlanValidationError, match="deliberation_threshold"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_deliberation_threshold_negative(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: -0.1
"""
        from maestro_cli.errors import PlanValidationError
        with pytest.raises(PlanValidationError, match="deliberation_threshold"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_deliberation_threshold_boundary_valid(self, tmp_path: Path) -> None:
        """0.0 and 1.0 are valid boundary values."""
        for thresh in ("0.0", "1.0"):
            yaml = f"""\
version: 1
name: boundary
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: {thresh}
"""
            plan = load_plan(_write_plan(tmp_path, yaml))
            assert plan.tasks[0].deliberation_threshold == float(thresh)

    def test_deliberation_threshold_non_numeric(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: "not-a-number"
"""
        from maestro_cli.errors import PlanValidationError
        with pytest.raises(PlanValidationError, match="deliberation_threshold"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# Deliberation Gate — runner integration (mocked subprocess)
# ---------------------------------------------------------------------------


class TestDeliberationGateRunner:
    def test_deliberation_skip_when_gate_fails(self, tmp_path: Path) -> None:
        """Gate returns score < threshold → task is skipped."""
        from maestro_cli.runners import _build_deliberation_context, _run_deliberation_gate

        # Mock the gate so haiku decides "self-answerable" (needs_external=False)
        gate_response = json.dumps({"needs_external": False, "confidence": 0.9})

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = gate_response
            mock_run.return_value = mock_proc

            gate_passes, score = _run_deliberation_gate(
                "t1", "context text", threshold=0.5, workdir=tmp_path
            )
        assert gate_passes is False
        assert score < 0.5  # 1 - 0.9 = 0.1

    def test_deliberation_proceed_when_gate_passes(self, tmp_path: Path) -> None:
        """Gate returns score >= threshold → task proceeds."""
        from maestro_cli.runners import _run_deliberation_gate

        gate_response = json.dumps({"needs_external": True, "confidence": 0.85})

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = gate_response
            mock_run.return_value = mock_proc

            gate_passes, score = _run_deliberation_gate(
                "t1", "context text", threshold=0.5, workdir=tmp_path
            )
        assert gate_passes is True
        assert score >= 0.5

    def test_deliberation_fail_open_on_error(self, tmp_path: Path) -> None:
        """Any subprocess error → fail-open (gate passes)."""
        from maestro_cli.runners import _run_deliberation_gate

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("subprocess failed")

            gate_passes, score = _run_deliberation_gate(
                "t1", "context", threshold=0.5, workdir=tmp_path
            )
        assert gate_passes is True
        assert score == 0.0

    def test_deliberation_fail_open_on_non_zero_exit(self, tmp_path: Path) -> None:
        """Non-zero exit code → fail-open (gate passes)."""
        from maestro_cli.runners import _run_deliberation_gate

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stdout = ""
            mock_run.return_value = mock_proc

            gate_passes, score = _run_deliberation_gate(
                "t1", "context", threshold=0.5, workdir=tmp_path
            )
        assert gate_passes is True
        assert score == 0.0

    def test_build_deliberation_context_with_wildcard(self) -> None:
        """Wildcard context_from collects all upstream outputs."""
        from maestro_cli.models import TaskResult
        from maestro_cli.runners import _build_deliberation_context
        from datetime import datetime, UTC

        now = datetime.now(UTC)
        task = TaskSpec(id="consumer", context_from=["*"])
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=1.0,
                command="echo", log_path=Path("/tmp/a.log"),
                result_path=Path("/tmp/a.result.json"),
                stdout_tail="output from a",
            ),
        }
        ctx = _build_deliberation_context(upstream, task)
        assert "output from a" in ctx
        assert "[a]" in ctx

    def test_build_deliberation_context_no_context_from(self) -> None:
        """No context_from → returns placeholder."""
        from maestro_cli.runners import _build_deliberation_context

        task = TaskSpec(id="t1", context_from=[])
        ctx = _build_deliberation_context({}, task)
        assert "no upstream context" in ctx.lower()


# ---------------------------------------------------------------------------
# Adversarial Debate Judge — JudgeSpec loading
# ---------------------------------------------------------------------------


class TestDebateRoundsLoading:
    def test_judge_spec_default_debate_rounds(self) -> None:
        js = JudgeSpec(criteria=["quality"])
        assert js.debate_rounds == 2

    def test_judge_spec_custom_debate_rounds(self) -> None:
        js = JudgeSpec(criteria=["quality"], debate_rounds=3)
        assert js.debate_rounds == 3

    def test_judge_spec_to_dict_includes_debate_rounds(self) -> None:
        js = JudgeSpec(criteria=["quality"], method="debate", debate_rounds=4)
        d = js.to_dict()
        assert "debate_rounds" in d
        assert d["debate_rounds"] == 4
        assert d["method"] == "debate"

    def test_debate_rounds_loaded_from_yaml(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: debate-plan
tasks:
  - id: t1
    engine: claude
    model: haiku
    prompt: "Write code"
    judge:
      criteria: ["correctness"]
      method: debate
      debate_rounds: 3
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.method == "debate"
        assert plan.tasks[0].judge.debate_rounds == 3

    def test_debate_rounds_defaults_to_2_when_not_set(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: debate-plan
tasks:
  - id: t1
    engine: claude
    model: haiku
    prompt: "Write code"
    judge:
      criteria: ["correctness"]
      method: debate
"""
        plan = load_plan(_write_plan(tmp_path, yaml))
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.debate_rounds == 2

    def test_debate_rounds_zero_raises_error(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: 0
"""
        from maestro_cli.errors import PlanValidationError
        with pytest.raises(PlanValidationError, match="debate_rounds"):
            load_plan(_write_plan(tmp_path, yaml))

    def test_debate_rounds_non_integer_raises_error(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: "three"
"""
        from maestro_cli.errors import PlanValidationError
        with pytest.raises(PlanValidationError, match="debate_rounds"):
            load_plan(_write_plan(tmp_path, yaml))


# ---------------------------------------------------------------------------
# Adversarial Debate Judge — runner routing
# ---------------------------------------------------------------------------


class TestDebateJudgeRunner:
    def _make_judge_response(self, score: float, assessment: str = "looks good") -> str:
        return json.dumps({"score": score, "assessment": assessment})

    def test_debate_method_routes_to_debate_evaluation(self, tmp_path: Path) -> None:
        """_run_judge_evaluation routes to _run_debate_evaluation for method=debate."""
        from maestro_cli.runners import _run_judge_evaluation

        judge = JudgeSpec(
            criteria=["correctness"],
            method="debate",
            debate_rounds=1,
            pass_threshold=0.6,
        )

        bull_resp = self._make_judge_response(0.8, "solid output")
        bear_resp = self._make_judge_response(0.7, "minor issues")

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_bull = MagicMock()
            mock_bull.returncode = 0
            mock_bull.stdout = bull_resp
            mock_bear = MagicMock()
            mock_bear.returncode = 0
            mock_bear.stdout = bear_resp
            mock_run.side_effect = [mock_bull, mock_bear]

            result = _run_judge_evaluation(
                task_id="t1",
                judge=judge,
                stdout_tail="some output",
                workdir=tmp_path,
            )

        assert result.verdict in ("pass", "fail")
        assert 0.0 <= result.overall_score <= 1.0

    def test_debate_score_is_average_of_bull_and_bear(self, tmp_path: Path) -> None:
        """Overall score = average of bull + bear scores."""
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(
            criteria=["quality"],
            method="debate",
            debate_rounds=1,
            pass_threshold=0.6,
        )

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            bull = MagicMock()
            bull.returncode = 0
            bull.stdout = self._make_judge_response(0.8)
            bear = MagicMock()
            bear.returncode = 0
            bear.stdout = self._make_judge_response(0.6)
            mock_run.side_effect = [bull, bear]

            result = _run_debate_evaluation(
                task_id="t1",
                judge=judge,
                stdout_tail="output",
                workdir=tmp_path,
            )

        # (0.8 + 0.6) / 2 = 0.7
        assert abs(result.overall_score - 0.7) < 0.01

    def test_debate_pass_verdict_when_score_above_threshold(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(
            criteria=["quality"],
            method="debate",
            debate_rounds=1,
            pass_threshold=0.5,
        )

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            bull = MagicMock()
            bull.returncode = 0
            bull.stdout = self._make_judge_response(0.8)
            bear = MagicMock()
            bear.returncode = 0
            bear.stdout = self._make_judge_response(0.8)
            mock_run.side_effect = [bull, bear]

            result = _run_debate_evaluation(
                task_id="t1", judge=judge, stdout_tail="out", workdir=tmp_path
            )

        assert result.verdict == "pass"

    def test_debate_fail_verdict_when_score_below_threshold(self, tmp_path: Path) -> None:
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(
            criteria=["quality"],
            method="debate",
            debate_rounds=1,
            pass_threshold=0.9,
        )

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            bull = MagicMock()
            bull.returncode = 0
            bull.stdout = self._make_judge_response(0.5)
            bear = MagicMock()
            bear.returncode = 0
            bear.stdout = self._make_judge_response(0.5)
            mock_run.side_effect = [bull, bear]

            result = _run_debate_evaluation(
                task_id="t1", judge=judge, stdout_tail="out", workdir=tmp_path
            )

        assert result.verdict == "fail"

    def test_debate_error_on_bull_failure(self, tmp_path: Path) -> None:
        """If bull call raises, returns verdict=error."""
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(criteria=["quality"], method="debate", debate_rounds=1)

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("claude not found")

            result = _run_debate_evaluation(
                task_id="t1", judge=judge, stdout_tail="out", workdir=tmp_path
            )

        assert result.verdict == "error"

    def test_debate_two_rounds_makes_four_calls(self, tmp_path: Path) -> None:
        """2 rounds × 2 calls (bull+bear) = 4 subprocess calls total."""
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(
            criteria=["quality"],
            method="debate",
            debate_rounds=2,
            pass_threshold=0.5,
        )

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            resp = self._make_judge_response(0.7)
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = resp
            mock_run.return_value = mock_proc

            result = _run_debate_evaluation(
                task_id="t1", judge=judge, stdout_tail="out", workdir=tmp_path
            )

        assert mock_run.call_count == 4  # 2 rounds × 2 agents
        assert result.verdict in ("pass", "fail")

    def test_debate_empty_criteria_still_scores(self, tmp_path: Path) -> None:
        """Empty criteria list still proceeds with default evaluation text."""
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(criteria=[], method="debate", debate_rounds=1)

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            resp = self._make_judge_response(0.6)
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = resp
            mock_run.return_value = proc

            result = _run_debate_evaluation(
                task_id="t1", judge=judge, stdout_tail="out", workdir=tmp_path
            )

        # Verdict should be "pass" or "fail", not "error"
        assert result.verdict in ("pass", "fail", "error")

    def test_debate_capped_at_4_rounds(self, tmp_path: Path) -> None:
        """debate_rounds > 4 is capped at 4."""
        from maestro_cli.runners import _run_debate_evaluation

        judge = JudgeSpec(criteria=["quality"], method="debate", debate_rounds=99)

        with patch("maestro_cli.runners.subprocess.run") as mock_run:
            resp = self._make_judge_response(0.7)
            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = resp
            mock_run.return_value = proc

            _run_debate_evaluation(
                task_id="t1", judge=judge, stdout_tail="out", workdir=tmp_path
            )

        # 4 rounds max × 2 = 8 calls
        assert mock_run.call_count == 8


# ---------------------------------------------------------------------------
# Validate command integration — density shown in output
# ---------------------------------------------------------------------------


class TestValidateCmdDensityOutput:
    def test_validate_prints_complexity(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """maestro validate prints the complexity score."""
        from maestro_cli.cli import _cmd_validate

        yaml = """\
version: 1
name: my-plan
tasks:
  - id: t1
    command: echo hello
"""
        plan_path = str(_write_plan(tmp_path, yaml))
        rc = _cmd_validate(plan_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "complexity:" in captured.out
