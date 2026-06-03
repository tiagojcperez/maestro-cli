"""Tests for v1.30.0 features:
- Consistency Groups polish (policy integration, SEC022)
- context_mode: selective (BM25 chunk-level selection)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from maestro_cli.audit import audit_plan
from maestro_cli.loader import load_plan
from maestro_cli.models import CONTEXT_MODES, PlanSpec, TaskSpec
from maestro_cli.runners import (
    _build_selective_context,
    _score_chunk_bm25,
)


def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return p


def _minimal_plan(*tasks: TaskSpec, **kwargs: object) -> PlanSpec:
    return PlanSpec(name="test", tasks=list(tasks), **kwargs)  # type: ignore[arg-type]


# ===========================================================================
# Feature 11: Consistency Groups polish
# ===========================================================================


class TestPolicyIntegration:
    def test_contract_type_in_policy(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            policies:
              - name: require-verify-for-contracts
                rule: "task.contract_type != '' and task.max_retries == 0"
                action: warn
                message: "Tasks producing contracts should have retries"
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                contract_type: api-schema
        """)
        plan = load_plan(p)
        assert plan.policies[0].rule == "task.contract_type != '' and task.max_retries == 0"

    def test_has_consistency_group_in_policy(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            policies:
              - name: cg-approval
                rule: "task.has_consistency_group and not task.requires_approval"
                action: warn
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                consistency_group: [api]
        """)
        plan = load_plan(p)
        assert plan.tasks[0].consistency_group == ["api"]


class TestSEC022:
    def test_consumer_without_verify(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="producer",
                engine="claude",
                prompt="Produce API schema",
                contract_type="api-schema",
            ),
            TaskSpec(
                id="consumer",
                engine="claude",
                prompt="Consume schema",
                depends_on=["producer"],
                consumes_contracts=["producer"],
            ),
        )
        findings = audit_plan(plan)
        sec022 = [f for f in findings if f.rule == "SEC022"]
        assert len(sec022) == 1
        assert "verify_command" in sec022[0].message

    def test_consumer_with_verify_ok(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="producer",
                engine="claude",
                prompt="Produce schema",
                contract_type="api-schema",
            ),
            TaskSpec(
                id="consumer",
                engine="claude",
                prompt="Consume",
                depends_on=["producer"],
                consumes_contracts=["producer"],
                verify_command="echo ok",
            ),
        )
        findings = audit_plan(plan)
        sec022 = [f for f in findings if f.rule == "SEC022"]
        assert len(sec022) == 0

    def test_consumer_with_guard_ok(self) -> None:
        plan = _minimal_plan(
            TaskSpec(
                id="producer",
                engine="claude",
                prompt="Produce",
                contract_type="api-schema",
            ),
            TaskSpec(
                id="consumer",
                engine="claude",
                prompt="Consume",
                depends_on=["producer"],
                consumes_contracts=["producer"],
                guard_command="jq .",
            ),
        )
        findings = audit_plan(plan)
        sec022 = [f for f in findings if f.rule == "SEC022"]
        assert len(sec022) == 0

    def test_no_consumer_no_finding(self) -> None:
        plan = _minimal_plan(
            TaskSpec(id="normal", engine="claude", prompt="stuff"),
        )
        findings = audit_plan(plan)
        sec022 = [f for f in findings if f.rule == "SEC022"]
        assert len(sec022) == 0


# ===========================================================================
# Feature 12: context_mode: selective
# ===========================================================================


class TestContextModeSelective:
    def test_selective_in_context_modes(self) -> None:
        assert "selective" in CONTEXT_MODES

    def test_parse_selective(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "first"
              - id: t2
                engine: claude
                prompt: "analyze security vulnerabilities"
                depends_on: [t1]
                context_from: [t1]
                context_mode: selective
                context_budget_tokens: 2000
        """)
        plan = load_plan(p)
        assert plan.tasks[1].context_mode == "selective"


class TestScoreChunkBM25:
    def test_empty_keywords(self) -> None:
        assert _score_chunk_bm25("some text", set()) == 0.0

    def test_matching_keywords(self) -> None:
        score = _score_chunk_bm25(
            "The security audit found vulnerabilities in auth module",
            {"security", "vulnerabilities", "auth"},
        )
        assert score > 0.0

    def test_no_match(self) -> None:
        score = _score_chunk_bm25(
            "Hello world this is a test",
            {"security", "vulnerability"},
        )
        assert score == 0.0

    def test_tf_saturation(self) -> None:
        single = _score_chunk_bm25("security check", {"security"})
        repeated = _score_chunk_bm25(
            "security security security security",
            {"security"},
        )
        # TF saturation means repeated mentions don't scale linearly
        assert repeated > single
        assert repeated < single * 4  # saturated, not linear

    def test_case_insensitive(self) -> None:
        score = _score_chunk_bm25("Security AUDIT", {"security", "audit"})
        assert score > 0.0


class TestBuildSelectiveContext:
    def test_empty(self) -> None:
        result = _build_selective_context({}, 1000, set())
        assert result == ""

    def test_selects_relevant_chunks(self) -> None:
        texts = {
            "analysis": (
                "The security audit revealed critical issues.\n"
                "SQL injection found in login handler.\n"
                "XSS vulnerability in profile page.\n"
                "The deployment was successful.\n"
                "All tests passed without issues.\n"
                "Performance metrics look good.\n"
            ),
        }
        result = _build_selective_context(
            texts, 500, {"security", "injection", "vulnerability"},
        )
        assert "security" in result.lower()
        assert "analysis" in result  # upstream ID header

    def test_respects_budget(self) -> None:
        big_text = "\n".join(f"line {i} with keyword security" for i in range(100))
        texts = {"upstream": big_text}
        result = _build_selective_context(texts, 50, {"security"})
        assert len(result) < 300  # 50 tokens * 4 chars + overhead

    def test_multiple_upstreams(self) -> None:
        texts = {
            "code-review": "Found security issue in authentication module",
            "test-results": "All tests passed, no failures detected",
        }
        result = _build_selective_context(
            texts, 500, {"security", "authentication"},
        )
        assert "code-review" in result
        # test-results may or may not appear (lower score)

    def test_upstream_boost(self) -> None:
        texts = {
            "high-score": "generic text content here",
            "low-score": "generic text content here",
        }
        result = _build_selective_context(
            texts, 200, {"generic"},
            scores={"high-score": 1.0, "low-score": 0.0},
        )
        # Both should appear but high-score should be prioritized
        assert "high-score" in result

    def test_fallback_to_l0(self) -> None:
        texts = {
            "upstream": "This is some content that doesn't match any keywords at all",
        }
        result = _build_selective_context(texts, 500, {"nonexistent_keyword_xyz"})
        assert "upstream" in result  # should fallback to L0 summary
