from __future__ import annotations

from pathlib import Path

from maestro_cli.audit import (
    AuditFinding,
    _check_sec002,
    audit_plan,
)
from maestro_cli.models import PlanSpec, TaskSpec


def _plan(*tasks: TaskSpec, **kwargs: object) -> PlanSpec:
    """Build a PlanSpec with a cost budget (avoids unrelated SEC001 noise)."""
    params: dict[str, object] = {"max_cost_usd": 10.0}
    params.update(kwargs)
    return PlanSpec(name="cov-plan", tasks=list(tasks), **params)  # type: ignore[arg-type]


class TestCheckSec002NonTaskSpec:
    """_check_sec002 early-returns when given a non-TaskSpec object."""

    def test_non_taskspec_returns_without_findings(self) -> None:
        findings: list[AuditFinding] = []
        # A plain object is not a TaskSpec -> isinstance check fails -> early return.
        _check_sec002(object(), findings)
        assert findings == []

    def test_dict_is_not_taskspec(self) -> None:
        findings: list[AuditFinding] = []
        _check_sec002({"id": "x", "engine": "claude", "args": ["--yolo"]}, findings)
        assert findings == []


class TestSec017ListCommandUpstream:
    """SEC017 joins an upstream's list-format command when scanning for external-data indicators."""

    def test_list_command_external_indicator_triggers_sec017(self) -> None:
        # Upstream task with a LIST command that contains an external-data
        # indicator ("user input"), no explicit context_trust set.
        upstream = TaskSpec(
            id="fetch",
            command=["curl", "https://example.com/user_input"],
        )
        downstream = TaskSpec(
            id="use",
            command="echo done",
            depends_on=["fetch"],
            context_from=["fetch"],
        )
        plan = _plan(upstream, downstream)
        findings = audit_plan(plan)
        sec017 = [f for f in findings if f.rule == "SEC017"]
        assert sec017, "expected a SEC017 finding for list-command external-data upstream"
        assert any(f.task_id == "use" for f in sec017)
        assert any("context_trust" in f.message for f in sec017)

    def test_list_command_without_indicator_no_sec017(self) -> None:
        # List command with no external-data keywords -> the join still runs
        # but no SEC017 is emitted.
        upstream = TaskSpec(id="fetch", command=["echo", "hello", "world"])
        downstream = TaskSpec(
            id="use",
            command="echo done",
            depends_on=["fetch"],
            context_from=["fetch"],
        )
        plan = _plan(upstream, downstream)
        findings = audit_plan(plan)
        assert all(f.rule != "SEC017" for f in findings)


class TestAuditPackUnresolvablePath:
    """An audit_packs entry that resolve_path() cannot resolve yields PACK001 + continue."""

    def test_empty_pack_ref_yields_pack001(self, tmp_path: Path) -> None:
        # resolve_path returns None for a falsy (empty) ref; the loop appends
        # PACK001 and continues to the next entry.
        plan = _plan(
            TaskSpec(id="t1", command="echo hi"),
            source_path=tmp_path / "plan.yaml",
            audit_packs=[""],
        )
        findings = audit_plan(plan)
        pack001 = [f for f in findings if f.rule == "PACK001"]
        assert pack001, "expected a PACK001 finding for unresolvable pack path"
        assert pack001[0].severity == "error"
        assert "could not be resolved" in pack001[0].message

    def test_unresolvable_continues_to_next_pack(self, tmp_path: Path) -> None:
        # First entry is unresolvable (empty); second is a valid pack that
        # produces a finding -> proves the loop continued past the bad entry.
        pack = tmp_path / "pack.yaml"
        pack.write_text(
            "rules:\n"
            "  - type: glob_exists\n"
            "    glob: definitely-missing-*.nope\n"
            "    rule: PACKX\n"
            "    severity: warning\n"
            "    message: missing file\n",
            encoding="utf-8",
        )
        plan = _plan(
            TaskSpec(id="t1", command="echo hi"),
            source_path=tmp_path / "plan.yaml",
            audit_packs=["", "pack.yaml"],
        )
        findings = audit_plan(plan)
        rules = [f.rule for f in findings]
        assert "PACK001" in rules  # the empty entry
        assert "PACKX" in rules    # the second pack still ran (failing assertion)


class TestAuditPackPassingAssertion:
    """A passing workspace assertion in an audit pack is skipped (continue), producing no finding."""

    def test_passing_assertion_produces_no_finding(self, tmp_path: Path) -> None:
        # Create a file the glob_exists assertion will match -> assertion passes
        # -> the `continue` branch is taken and no finding is appended.
        (tmp_path / "marker.txt").write_text("ok", encoding="utf-8")
        pack = tmp_path / "pack.yaml"
        pack.write_text(
            "rules:\n"
            "  - type: glob_exists\n"
            "    glob: marker.txt\n"
            "    rule: PASSING\n",
            encoding="utf-8",
        )
        plan = _plan(
            TaskSpec(id="t1", command="echo hi"),
            source_path=tmp_path / "plan.yaml",
            audit_packs=["pack.yaml"],
        )
        findings = audit_plan(plan)
        assert all(f.rule != "PASSING" for f in findings)
        # And no spurious pack-loading errors either.
        assert all(not f.rule.startswith("PACK") for f in findings)

    def test_mixed_pass_and_fail_assertions(self, tmp_path: Path) -> None:
        # One passing assertion (skipped via continue) + one failing assertion
        # (emits a finding) -> exercises both branches in one pack.
        (tmp_path / "present.txt").write_text("ok", encoding="utf-8")
        pack = tmp_path / "pack.yaml"
        pack.write_text(
            "rules:\n"
            "  - type: glob_exists\n"
            "    glob: present.txt\n"
            "    rule: PASS_ONE\n"
            "  - type: glob_exists\n"
            "    glob: absent-file.xyz\n"
            "    rule: FAIL_ONE\n"
            "    severity: warning\n",
            encoding="utf-8",
        )
        plan = _plan(
            TaskSpec(id="t1", command="echo hi"),
            source_path=tmp_path / "plan.yaml",
            audit_packs=["pack.yaml"],
        )
        findings = audit_plan(plan)
        rules = [f.rule for f in findings]
        assert "PASS_ONE" not in rules
        assert "FAIL_ONE" in rules
