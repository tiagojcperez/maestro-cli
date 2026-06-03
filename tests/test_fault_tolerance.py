from __future__ import annotations

import textwrap

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import CircuitBreakerSpec, PlanSpec
from maestro_cli.runners import _compute_retry_delay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_plan(tmp_path, extra: str = "") -> object:
    """Write a minimal valid plan YAML and return its Path."""
    p = tmp_path / "plan.yaml"
    p.write_text(
        textwrap.dedent(f"""\
            version: 1
            name: test-plan
            tasks:
              - id: t1
                command: echo hi
            {extra}
        """),
        encoding="utf-8",
    )
    return p


def _write_plan_with_circuit_breaker(tmp_path, cb_yaml: str) -> object:
    p = tmp_path / "plan.yaml"
    p.write_text(
        textwrap.dedent(f"""\
            version: 1
            name: test-plan
            circuit_breaker:
            {cb_yaml}
            tasks:
              - id: t1
                command: echo hi
        """),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# CircuitBreakerSpec dataclass
# ---------------------------------------------------------------------------

class TestCircuitBreakerSpec:
    def test_default_values(self) -> None:
        cb = CircuitBreakerSpec()
        assert cb.max_total_failures == 5
        assert cb.action == "fail"

    def test_to_dict(self) -> None:
        cb = CircuitBreakerSpec(max_total_failures=3, action="pause")
        d = cb.to_dict()
        assert d == {"max_total_failures": 3, "action": "pause"}

    def test_plan_spec_includes_circuit_breaker(self) -> None:
        plan = PlanSpec(name="x", tasks=[])
        assert hasattr(plan, "circuit_breaker")
        assert plan.circuit_breaker is None


# ---------------------------------------------------------------------------
# circuit_breaker loader validation
# ---------------------------------------------------------------------------

class TestCircuitBreakerValidation:
    def test_valid_circuit_breaker(self, tmp_path) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent("""\
                version: 1
                name: test-plan
                circuit_breaker:
                  max_total_failures: 3
                  action: pause
                tasks:
                  - id: t1
                    command: echo hi
            """),
            encoding="utf-8",
        )
        plan = load_plan(p)
        assert plan.circuit_breaker is not None
        assert plan.circuit_breaker.max_total_failures == 3
        assert plan.circuit_breaker.action == "pause"

    def test_invalid_max_failures_zero(self, tmp_path) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent("""\
                version: 1
                name: test-plan
                circuit_breaker:
                  max_total_failures: 0
                  action: fail
                tasks:
                  - id: t1
                    command: echo hi
            """),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="E050"):
            load_plan(p)

    def test_invalid_max_failures_negative(self, tmp_path) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent("""\
                version: 1
                name: test-plan
                circuit_breaker:
                  max_total_failures: -1
                  action: fail
                tasks:
                  - id: t1
                    command: echo hi
            """),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="E050"):
            load_plan(p)

    def test_invalid_action(self, tmp_path) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent("""\
                version: 1
                name: test-plan
                circuit_breaker:
                  max_total_failures: 5
                  action: explode
                tasks:
                  - id: t1
                    command: echo hi
            """),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="E050"):
            load_plan(p)

    def test_non_dict_raises(self, tmp_path) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent("""\
                version: 1
                name: test-plan
                circuit_breaker: "enabled"
                tasks:
                  - id: t1
                    command: echo hi
            """),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="E050"):
            load_plan(p)


# ---------------------------------------------------------------------------
# _compute_retry_delay behaviour
# ---------------------------------------------------------------------------

def _make_task(
    retry_delay_sec=None,
    retry_strategy=None,
):
    """Build a minimal TaskSpec-like object via load_plan for _compute_retry_delay."""
    from maestro_cli.models import TaskSpec
    t = TaskSpec(id="t1", command="echo")
    t.retry_delay_sec = retry_delay_sec
    t.retry_strategy = retry_strategy
    return t


class TestRetryStrategy:
    def test_constant_returns_base(self) -> None:
        task = _make_task(retry_delay_sec=3.0, retry_strategy="constant")
        for attempt in range(3):
            assert _compute_retry_delay(task, attempt) == pytest.approx(3.0)

    def test_linear_scales_with_attempt(self) -> None:
        task = _make_task(retry_delay_sec=2.0, retry_strategy="linear")
        assert _compute_retry_delay(task, 0) == pytest.approx(2.0)   # 2 * (0+1)
        assert _compute_retry_delay(task, 1) == pytest.approx(4.0)   # 2 * (1+1)
        assert _compute_retry_delay(task, 2) == pytest.approx(6.0)   # 2 * (2+1)

    def test_exponential_doubles(self) -> None:
        task = _make_task(retry_delay_sec=1.0, retry_strategy="exponential")
        assert _compute_retry_delay(task, 0) == pytest.approx(1.0)   # 1 * 2^0
        assert _compute_retry_delay(task, 1) == pytest.approx(2.0)   # 1 * 2^1
        assert _compute_retry_delay(task, 2) == pytest.approx(4.0)   # 1 * 2^2

    def test_explicit_list_overrides_strategy(self) -> None:
        task = _make_task(retry_delay_sec=[5.0, 10.0, 20.0], retry_strategy="exponential")
        assert _compute_retry_delay(task, 0) == pytest.approx(5.0)
        assert _compute_retry_delay(task, 1) == pytest.approx(10.0)
        assert _compute_retry_delay(task, 2) == pytest.approx(20.0)
        # beyond list bounds → clamp to last
        assert _compute_retry_delay(task, 5) == pytest.approx(20.0)

    def test_default_is_constant(self) -> None:
        task = _make_task(retry_delay_sec=4.0, retry_strategy=None)
        assert _compute_retry_delay(task, 0) == pytest.approx(4.0)
        assert _compute_retry_delay(task, 1) == pytest.approx(4.0)
        assert _compute_retry_delay(task, 2) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# retry_strategy loader validation
# ---------------------------------------------------------------------------

class TestRetryStrategyValidation:
    @pytest.mark.parametrize("strategy", ["constant", "linear", "exponential"])
    def test_valid_strategies(self, tmp_path, strategy: str) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent(f"""\
                version: 1
                name: test-plan
                tasks:
                  - id: t1
                    command: echo hi
                    max_retries: 1
                    retry_strategy: {strategy}
            """),
            encoding="utf-8",
        )
        plan = load_plan(p)
        assert plan.tasks[0].retry_strategy == strategy

    def test_invalid_strategy_raises(self, tmp_path) -> None:
        p = tmp_path / "plan.yaml"
        p.write_text(
            textwrap.dedent("""\
                version: 1
                name: test-plan
                tasks:
                  - id: t1
                    command: echo hi
                    max_retries: 1
                    retry_strategy: fibonacci
            """),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="E051"):
            load_plan(p)
