from __future__ import annotations

import textwrap
from pathlib import Path

from maestro_cli.audit import AuditFinding, audit_plan
from maestro_cli.loader import load_plan
from maestro_cli.models import PlanSpec, TaskSpec
from maestro_cli.runners import _sandbox_observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_plan(*tasks: TaskSpec, max_cost_usd: float | None = 10.0) -> PlanSpec:
    return PlanSpec(name="test-plan", max_cost_usd=max_cost_usd, tasks=list(tasks))


def _engine_task(task_id: str, **kwargs) -> TaskSpec:  # type: ignore[no-untyped-def]
    return TaskSpec(id=task_id, engine="claude", prompt="do something", **kwargs)


def _cmd_task(task_id: str, **kwargs) -> TaskSpec:  # type: ignore[no-untyped-def]
    return TaskSpec(id=task_id, command="echo hi", **kwargs)


def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _sandbox_observation — unit tests
# ---------------------------------------------------------------------------


class TestSandboxObservation:
    def test_basic_wrapping(self) -> None:
        result = _sandbox_observation("task-a", "some output")
        assert result == '<observation source="task-a">\nsome output\n</observation>'

    def test_source_attribute_is_upstream_id(self) -> None:
        result = _sandbox_observation("build-step", "data")
        assert 'source="build-step"' in result

    def test_multiline_content_preserved(self) -> None:
        content = "line1\nline2\nline3"
        result = _sandbox_observation("up", content)
        assert "line1\nline2\nline3" in result

    def test_empty_content(self) -> None:
        result = _sandbox_observation("t", "")
        assert result == '<observation source="t">\n\n</observation>'

    def test_content_with_xml_chars(self) -> None:
        content = "<b>bold</b> & 'quotes'"
        result = _sandbox_observation("src", content)
        assert content in result
        assert 'source="src"' in result

    def test_id_with_hyphens_and_underscores(self) -> None:
        result = _sandbox_observation("my-task_01", "output")
        assert 'source="my-task_01"' in result

    def test_nested_observation_tags_in_content(self) -> None:
        """Nested tags in content must not break the outer wrapper."""
        content = '<observation source="inner">nested</observation>'
        result = _sandbox_observation("outer", content)
        assert result.startswith('<observation source="outer">')
        assert result.endswith("</observation>")
        assert "nested" in result


# ---------------------------------------------------------------------------
# Loader — observation_block field
# ---------------------------------------------------------------------------


class TestObservationBlockLoader:
    def test_defaults_to_false(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: t1
                    command: echo hi
                """,
            )
        )
        assert plan.tasks[0].observation_block is False

    def test_parsed_true(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: t1
                    depends_on: [upstream]
                    context_from: [upstream]
                    observation_block: true
                    command: echo hi
                  - id: upstream
                    command: echo up
                """,
            )
        )
        t1 = next(t for t in plan.tasks if t.id == "t1")
        assert t1.observation_block is True

    def test_parsed_false_explicit(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: t1
                    observation_block: false
                    command: echo hi
                """,
            )
        )
        assert plan.tasks[0].observation_block is False

    def test_observation_block_in_to_dict(self) -> None:
        task = _cmd_task("t1", observation_block=True)
        d = task.to_dict()
        assert d["observation_block"] is True


# ---------------------------------------------------------------------------
# Loader — control_flow_integrity plan field
# ---------------------------------------------------------------------------


class TestControlFlowIntegrityLoader:
    def test_defaults_to_false(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: t1
                    command: echo hi
                """,
            )
        )
        assert plan.control_flow_integrity is False

    def test_parsed_bool_true(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                control_flow_integrity: true
                tasks:
                  - id: t1
                    command: echo hi
                """,
            )
        )
        assert plan.control_flow_integrity is True

    def test_parsed_bool_false(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                control_flow_integrity: false
                tasks:
                  - id: t1
                    command: echo hi
                """,
            )
        )
        assert plan.control_flow_integrity is False

    def test_parsed_string_true(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                control_flow_integrity: "true"
                tasks:
                  - id: t1
                    command: echo hi
                """,
            )
        )
        assert plan.control_flow_integrity is True

    def test_parsed_string_false(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                control_flow_integrity: "false"
                tasks:
                  - id: t1
                    command: echo hi
                """,
            )
        )
        assert plan.control_flow_integrity is False


# ---------------------------------------------------------------------------
# Loader warning — observation_block without context_from
# ---------------------------------------------------------------------------


class TestObservationBlockWarning:
    def test_no_warning_when_observation_block_false(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: t1
                    observation_block: false
                    command: echo hi
                """,
            )
        )
        assert not any("observation_block" in w for w in plan.validation_warnings)

    def test_warning_when_observation_block_true_no_context_from(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: t1
                    observation_block: true
                    command: echo hi
                """,
            )
        )
        matching = [w for w in plan.validation_warnings if "observation_block" in w]
        assert len(matching) == 1
        assert "t1" in matching[0]

    def test_no_warning_when_observation_block_true_with_context_from(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                tasks:
                  - id: upstream
                    command: echo up
                  - id: t1
                    depends_on: [upstream]
                    context_from: [upstream]
                    observation_block: true
                    command: echo hi
                """,
            )
        )
        assert not any("observation_block" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# SEC015 — when expression references unbounded upstream output
# ---------------------------------------------------------------------------


class TestSEC015:
    def test_no_when_no_finding(self) -> None:
        task = _cmd_task("t1")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC015" for f in findings)

    def test_when_references_stdout_tail(self) -> None:
        task = _cmd_task(
            "consumer",
            when="{{ upstream.stdout_tail }} == success",
            depends_on=["upstream"],
        )
        upstream = _cmd_task("upstream")
        plan = _minimal_plan(upstream, task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 1
        assert sec015[0].task_id == "consumer"
        assert sec015[0].severity == "warning"
        assert "stdout_tail" in sec015[0].message

    def test_when_references_log(self) -> None:
        task = _cmd_task(
            "consumer",
            when="{{ upstream.log }}",
            depends_on=["upstream"],
        )
        upstream = _cmd_task("upstream")
        plan = _minimal_plan(upstream, task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 1
        assert "log" in sec015[0].message

    def test_when_references_safe_status_no_finding(self) -> None:
        task = _cmd_task(
            "consumer",
            when="{{ upstream.status }} == success",
            depends_on=["upstream"],
        )
        upstream = _cmd_task("upstream")
        plan = _minimal_plan(upstream, task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC015" for f in findings)

    def test_when_references_exit_code_no_finding(self) -> None:
        task = _cmd_task(
            "consumer",
            when="{{ upstream.exit_code }} == 0",
            depends_on=["upstream"],
        )
        upstream = _cmd_task("upstream")
        plan = _minimal_plan(upstream, task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC015" for f in findings)

    def test_when_references_both_dangerous_fields_two_findings(self) -> None:
        task = _cmd_task(
            "consumer",
            when="{{ upstream.stdout_tail }} and {{ upstream.log }}",
            depends_on=["upstream"],
        )
        upstream = _cmd_task("upstream")
        plan = _minimal_plan(upstream, task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 2

    def test_task_id_is_set_on_finding(self) -> None:
        task = _cmd_task("my-risky-task", when="{{ up.stdout_tail }}", depends_on=["up"])
        up = _cmd_task("up")
        plan = _minimal_plan(up, task)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert sec015[0].task_id == "my-risky-task"

    def test_sec015_via_yaml(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: producer
                command: echo payload
              - id: consumer
                depends_on: [producer]
                when: "{{ producer.stdout_tail }}"
                command: echo done
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec015 = [f for f in findings if f.rule == "SEC015"]
        assert len(sec015) == 1
        assert sec015[0].task_id == "consumer"


# ---------------------------------------------------------------------------
# SEC016 — context_from pulls raw engine output without guard_command
# ---------------------------------------------------------------------------


class TestSEC016:
    def test_no_context_from_no_finding(self) -> None:
        task = _engine_task("t1")
        plan = _minimal_plan(task)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_engine_upstream_without_guard_command(self) -> None:
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 1
        assert sec016[0].task_id == "use"
        assert sec016[0].severity == "warning"
        assert "gen" in sec016[0].message

    def test_engine_upstream_with_guard_command_no_finding(self) -> None:
        upstream = _engine_task("gen", guard_command="cat | grep -v INJECTED")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_command_upstream_no_finding(self) -> None:
        """Non-engine (command) upstreams don't carry LLM injection risk."""
        upstream = _cmd_task("build")
        consumer = _cmd_task(
            "use",
            depends_on=["build"],
            context_from=["build"],
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_context_mode_raw_triggers_finding(self) -> None:
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
            context_mode="raw",
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 1

    def test_context_mode_summarized_no_finding(self) -> None:
        # Refined 2026-04-26 (internal post-mortem): summarized is LLM-mediated
        # via haiku and provides partial injection resistance, so it's exempt
        # like map_reduce/recursive.
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
            context_mode="summarized",
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_context_mode_map_reduce_no_finding(self) -> None:
        """map_reduce synthesises via LLM — lower injection risk."""
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
            context_mode="map_reduce",
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_context_mode_recursive_no_finding(self) -> None:
        """recursive goes through the index→extract→brief pipeline — lower risk."""
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
            context_mode="recursive",
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_wildcard_context_from_no_finding(self) -> None:
        """Wildcard '*' is excluded from enumeration — no per-task assessment."""
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["*"],
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_multiple_upstreams_only_unguarded_flagged(self) -> None:
        guarded = _engine_task("safe-gen", guard_command="validate.sh")
        unguarded = _engine_task("risky-gen")
        consumer = _cmd_task(
            "use",
            depends_on=["safe-gen", "risky-gen"],
            context_from=["safe-gen", "risky-gen"],
        )
        plan = _minimal_plan(guarded, unguarded, consumer)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 1
        assert "risky-gen" in sec016[0].message
        assert "safe-gen" not in sec016[0].message

    def test_unknown_upstream_id_does_not_crash(self) -> None:
        """context_from referencing an unknown ID must not raise."""
        consumer = _cmd_task("use", context_from=["ghost"])
        plan = _minimal_plan(consumer)
        # Should not raise — unknown upstreams are skipped
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)

    def test_finding_message_contains_remediation_hint(self) -> None:
        upstream = _engine_task("gen")
        consumer = _cmd_task("use", depends_on=["gen"], context_from=["gen"])
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        sec016 = next(f for f in findings if f.rule == "SEC016")
        # Message should mention the mitigation options
        assert "guard_command" in sec016.message or "map_reduce" in sec016.message

    def test_sec016_via_yaml(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: generator
                engine: claude
                prompt: Generate a config
              - id: consumer
                depends_on: [generator]
                context_from: [generator]
                command: echo {{ generator.stdout_tail }}
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 1
        assert sec016[0].task_id == "consumer"

    def test_sec016_guarded_upstream_via_yaml(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            max_cost_usd: 5.0
            tasks:
              - id: generator
                engine: claude
                prompt: Generate a config
                guard_command: "cat | python -c 'import sys; sys.exit(0)'"
              - id: consumer
                depends_on: [generator]
                context_from: [generator]
                command: echo {{ generator.stdout_tail }}
            """,
        )
        plan = load_plan(plan_file)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC016" for f in findings)


# ---------------------------------------------------------------------------
# CFI integration — observation_block + control_flow_integrity together
# ---------------------------------------------------------------------------


class TestCFIIntegration:
    def test_observation_block_with_cfi_plan(self, tmp_path: Path) -> None:
        """observation_block: true + control_flow_integrity: true — both fields parsed."""
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: cfi-full
                control_flow_integrity: true
                tasks:
                  - id: producer
                    command: echo artifact
                  - id: consumer
                    depends_on: [producer]
                    context_from: [producer]
                    observation_block: true
                    command: echo hi
                """,
            )
        )
        assert plan.control_flow_integrity is True
        consumer = next(t for t in plan.tasks if t.id == "consumer")
        assert consumer.observation_block is True

    def test_sec015_and_sec016_coexist(self) -> None:
        """A task that violates both SEC015 and SEC016 gets both findings."""
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
            when="{{ gen.stdout_tail }} == ok",
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        rules = {f.rule for f in findings}
        assert "SEC015" in rules
        assert "SEC016" in rules

    def test_observation_block_suppresses_sec016_finding(self) -> None:
        """observation_block: true does NOT suppress SEC016 — the audit is independent
        of runtime sandboxing; the guard_command is still the recommended fix."""
        upstream = _engine_task("gen")
        consumer = _cmd_task(
            "use",
            depends_on=["gen"],
            context_from=["gen"],
            observation_block=True,
        )
        plan = _minimal_plan(upstream, consumer)
        findings = audit_plan(plan)
        # observation_block is a runtime hint; SEC016 still fires to recommend guard_command
        sec016 = [f for f in findings if f.rule == "SEC016"]
        assert len(sec016) == 1

    def test_cfi_false_observation_block_false_no_warnings(self, tmp_path: Path) -> None:
        plan = load_plan(
            _write_plan(
                tmp_path,
                """\
                version: 1
                name: test
                control_flow_integrity: false
                tasks:
                  - id: t1
                    observation_block: false
                    command: echo hi
                """,
            )
        )
        assert not any("observation_block" in w for w in plan.validation_warnings)
