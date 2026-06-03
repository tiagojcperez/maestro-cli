from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
    _entity_id,
    build_knowledge_graph,
    extract_entities,
)


# ===========================================================================
# TestEntity
# ===========================================================================


class TestEntity:
    def test_to_dict(self) -> None:
        e = Entity(id="abc", name="foo", entity_type="function", source_task="impl")
        d = e.to_dict()
        assert d["name"] == "foo"
        assert d["entity_type"] == "function"


class TestRelation:
    def test_to_dict(self) -> None:
        r = Relation(source_id="a", target_id="b", relation_type="defines")
        d = r.to_dict()
        assert d["relation_type"] == "defines"
        assert d["confidence"] == pytest.approx(0.95, abs=0.001)


# ===========================================================================
# TestKnowledgeGraph
# ===========================================================================


class TestKnowledgeGraph:
    def test_add_entity(self) -> None:
        g = KnowledgeGraph()
        e = Entity(id="1", name="foo", entity_type="function")
        g.add_entity(e)
        assert "1" in g.entities

    def test_add_relation(self) -> None:
        g = KnowledgeGraph()
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))
        assert len(g.relations) == 1

    def test_get_related_direct(self) -> None:
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        g.add_entity(Entity(id="b", name="B", entity_type="function"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))
        related = g.get_related("a", max_hops=1)
        assert "b" in related
        assert "a" in related

    def test_get_related_multi_hop(self) -> None:
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        g.add_entity(Entity(id="b", name="B", entity_type="function"))
        g.add_entity(Entity(id="c", name="C", entity_type="class"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))
        g.add_relation(Relation(source_id="b", target_id="c", relation_type="mentions"))
        related = g.get_related("a", max_hops=2)
        assert {"a", "b", "c"} == related

    def test_subgraph(self) -> None:
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        g.add_entity(Entity(id="b", name="B", entity_type="function"))
        g.add_entity(Entity(id="c", name="C", entity_type="class"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))
        g.add_relation(Relation(source_id="b", target_id="c", relation_type="mentions"))
        sub = g.subgraph({"a", "b"})
        assert len(sub.entities) == 2
        assert len(sub.relations) == 1  # only a→b, not b→c

    def test_format_context(self) -> None:
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="auth.py", entity_type="file", source_task="impl"))
        g.add_entity(Entity(id="b", name="validate", entity_type="function", source_task="impl"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))
        ctx = g.format_context()
        assert "auth.py" in ctx
        assert "validate" in ctx
        assert "Relationships" in ctx
        assert "[0.95]" in ctx

    def test_format_context_sorts_relations_by_confidence(self) -> None:
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="auth.py", entity_type="file", source_task="impl"))
        g.add_entity(Entity(id="b", name="validate", entity_type="function", source_task="impl"))
        g.add_entity(Entity(id="c", name="token", entity_type="concept", source_task="impl"))
        g.add_relation(Relation(source_id="a", target_id="c", relation_type="mentions"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))
        ctx = g.format_context()
        relationship_lines = [line for line in ctx.splitlines() if line.startswith("- [")]
        assert relationship_lines
        assert "defines" in relationship_lines[0]

    def test_to_dict(self) -> None:
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        d = g.to_dict()
        assert "entities" in d
        assert "relations" in d


# ===========================================================================
# TestExtractEntities
# ===========================================================================


class TestExtractEntities:
    def test_extract_files_from_diff(self) -> None:
        text = "diff --git a/src/auth.py b/src/auth.py\n+++ b/src/auth.py\n"
        entities, _ = extract_entities(text, "impl")
        files = [e for e in entities if e.entity_type == "file"]
        assert any("auth.py" in e.name for e in files)

    def test_extract_backtick_files(self) -> None:
        text = "Modified `src/utils.py` and `lib/helpers.js`."
        entities, _ = extract_entities(text, "impl")
        files = [e for e in entities if e.entity_type == "file"]
        names = {e.name for e in files}
        assert "src/utils.py" in names

    def test_extract_functions(self) -> None:
        text = "def validate_token(token):\n    return True\n"
        entities, _ = extract_entities(text, "impl")
        funcs = [e for e in entities if e.entity_type == "function"]
        assert any(e.name == "validate_token" for e in funcs)

    def test_extract_classes(self) -> None:
        text = "class AuthService:\n    pass\n"
        entities, _ = extract_entities(text, "impl")
        classes = [e for e in entities if e.entity_type == "class"]
        assert any(e.name == "AuthService" for e in classes)

    def test_extract_decisions(self) -> None:
        text = "Decided to use JWT tokens for authentication instead of sessions."
        entities, _ = extract_entities(text, "arch")
        decisions = [e for e in entities if e.entity_type == "decision"]
        assert len(decisions) >= 1

    def test_extract_errors(self) -> None:
        text = "Error: Module 'cryptography' not found in path"
        entities, _ = extract_entities(text, "impl")
        errors = [e for e in entities if e.entity_type == "error"]
        assert len(errors) >= 1

    def test_extract_dependencies(self) -> None:
        text = "import os\nfrom pathlib import Path\n"
        entities, _ = extract_entities(text, "impl")
        deps = [e for e in entities if e.entity_type == "dependency"]
        assert len(deps) >= 1

    def test_file_function_relation(self) -> None:
        text = "diff --git a/src/auth.py b/src/auth.py\n+def validate():\n"
        entities, relations = extract_entities(text, "impl")
        assert len(relations) >= 1
        assert any(r.relation_type == "defines" for r in relations)

    def test_empty_text(self) -> None:
        entities, relations = extract_entities("", "impl")
        assert entities == []
        assert relations == []

    def test_max_entities_cap(self) -> None:
        # Generate lots of functions
        text = "\n".join(f"def func_{i}(): pass" for i in range(100))
        entities, _ = extract_entities(text, "impl")
        assert len(entities) <= 50


# ===========================================================================
# TestBuildKnowledgeGraph
# ===========================================================================


class TestBuildKnowledgeGraph:
    def test_basic_graph_building(self) -> None:
        upstream = {
            "impl": (
                "diff --git a/src/auth.py b/src/auth.py\n"
                "+def validate_token(token):\n"
                "+    return True\n"
            ),
        }
        result = build_knowledge_graph(upstream, 2000)
        assert "validate_token" in result or "auth.py" in result

    def test_empty_upstream(self) -> None:
        assert build_knowledge_graph({}, 1000) == ""

    def test_zero_budget(self) -> None:
        assert build_knowledge_graph({"a": "def foo(): pass"}, 0) == ""

    def test_fallback_no_entities(self) -> None:
        upstream = {"task": "just plain prose with no code"}
        result = build_knowledge_graph(upstream, 1000)
        assert "task" in result  # Falls back to raw truncation

    def test_multiple_upstreams(self) -> None:
        upstream = {
            "task-a": "def alpha(): pass\nclass Beta: pass\n",
            "task-b": "def gamma(): pass\nimport os\n",
        }
        result = build_knowledge_graph(upstream, 5000)
        assert result  # Non-empty


# ===========================================================================
# Loader integration
# ===========================================================================


class TestKnowledgeGraphLoaderIntegration:
    def test_knowledge_graph_accepted(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: kg-test\ntasks:\n"
            "  - id: impl\n    engine: claude\n    prompt: implement\n"
            "  - id: review\n    engine: claude\n    prompt: review\n"
            "    depends_on: [impl]\n    context_from: [impl]\n"
            "    context_mode: knowledge_graph\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        review = [t for t in plan.tasks if t.id == "review"][0]
        assert review.context_mode == "knowledge_graph"

    def test_knowledge_graph_requires_context_from(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: test\ntasks:\n"
            "  - id: review\n    engine: claude\n    prompt: review\n"
            "    context_mode: knowledge_graph\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_from"):
            load_plan(plan_path)

    def test_knowledge_graph_in_context_modes(self) -> None:
        from maestro_cli.models import CONTEXT_MODES

        assert "knowledge_graph" in CONTEXT_MODES


# ===========================================================================
# Meta-Policy Reflexion tests
# ===========================================================================


class TestMetaPolicyReflexion:
    def _make_run_result(self, task_results: dict) -> Any:
        from datetime import datetime, timezone
        from maestro_cli.models import PlanRunResult

        return PlanRunResult(
            plan_name="test",
            run_id="run-1",
            run_path=Path("/tmp/test"),
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            success=False,
            task_results=task_results,
        )

    def test_policy_rule_from_judge_failure(self) -> None:
        from maestro_cli.knowledge import extract_knowledge
        from maestro_cli.models import JudgeResult, TaskResult

        result = self._make_run_result({
            "audit": TaskResult(
                task_id="audit",
                status="failed",
                exit_code=0,
                duration_sec=30.0,
                judge_result=JudgeResult(
                    verdict="fail",
                    overall_score=0.3,
                    reasoning="The code has SQL injection vulnerabilities in the login handler that were not addressed.",
                ),
            ),
        })
        records = extract_knowledge(result)
        policy_rules = [r for r in records if r.kind == "policy_rule"]
        assert len(policy_rules) >= 1
        assert "SQL injection" in policy_rules[0].insight

    def test_policy_rule_from_retry_success(self) -> None:
        from maestro_cli.knowledge import extract_knowledge
        from maestro_cli.models import FailureRecord, TaskResult

        result = self._make_run_result({
            "impl": TaskResult(
                task_id="impl",
                status="success",
                exit_code=0,
                duration_sec=60.0,
                retry_count=2,
                failure_history=[
                    FailureRecord(
                        attempt=1,
                        category="compilation_error",
                        message="SyntaxError: unexpected EOF in module auth.py",
                        exit_code=1,
                    ),
                ],
            ),
        })
        records = extract_knowledge(result)
        policy_rules = [r for r in records if r.kind == "policy_rule"]
        assert len(policy_rules) >= 1
        assert "compilation_error" in policy_rules[0].insight
        assert policy_rules[0].confidence > 0.5  # slightly boosted

    def test_no_policy_rule_for_clean_success(self) -> None:
        from maestro_cli.knowledge import extract_knowledge
        from maestro_cli.models import TaskResult

        result = self._make_run_result({
            "impl": TaskResult(
                task_id="impl",
                status="success",
                exit_code=0,
                duration_sec=10.0,
            ),
        })
        records = extract_knowledge(result)
        policy_rules = [r for r in records if r.kind == "policy_rule"]
        assert len(policy_rules) == 0

    def test_policy_rule_formatted_with_kind_icon(self) -> None:
        from maestro_cli.knowledge import format_knowledge
        from maestro_cli.models import KnowledgeRecord

        records = [
            KnowledgeRecord(
                task_id="audit",
                kind="policy_rule",
                insight="POLICY RULE: Check for injection vulnerabilities.",
                confidence=0.6,
                occurrences=2,
                first_seen="2026-01-01T00:00:00+00:00",
                last_seen="2026-03-25T00:00:00+00:00",
            ),
        ]
        formatted = format_knowledge(records)
        assert "[RULE]" in formatted
        assert "injection" in formatted
