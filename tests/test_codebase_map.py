"""Tests for codebase_map.py — the ``context_mode: codebase_map`` reader that
consumes an Understand-Anything knowledge graph."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.codebase_map import (
    CodebaseGraph,
    GraphEdge,
    GraphNode,
    build_codebase_map_context,
    format_codebase_map,
    load_codebase_graph,
    parse_codebase_graph,
    resolve_graph_path,
)
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import CONTEXT_MODES


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _graph_payload() -> dict[str, Any]:
    """A realistic Understand-Anything ``{nodes, edges}`` payload."""
    return {
        "nodes": [
            {
                "id": "file:src/auth.ts",
                "type": "file",
                "name": "auth.ts",
                "summary": "Authentication module that issues and verifies JWT tokens.",
                "tags": ["auth", "security", "jwt"],
                "complexity": "moderate",
                "filePath": "src/auth.ts",
            },
            {
                "id": "function:src/auth.ts:verifyToken",
                "type": "function",
                "name": "verifyToken",
                "summary": "Validates a JWT and returns the decoded claims.",
                "tags": ["jwt", "validation"],
                "complexity": "simple",
                "filePath": "src/auth.ts",
                "lineRange": [12, 34],
            },
            {
                "id": "file:src/billing.ts",
                "type": "file",
                "name": "billing.ts",
                "summary": "Stripe billing and invoice generation.",
                "tags": ["billing", "stripe"],
                "complexity": "complex",
                "filePath": "src/billing.ts",
            },
        ],
        "edges": [
            {"source": "file:src/auth.ts", "target": "function:src/auth.ts:verifyToken",
             "type": "contains", "direction": "forward", "weight": 1.0},
            {"source": "file:src/billing.ts", "target": "file:src/auth.ts",
             "type": "imports", "direction": "forward", "weight": 0.7},
        ],
    }


def _write_graph(workspace: Path, payload: dict[str, Any]) -> None:
    d = workspace / ".understand-anything"
    d.mkdir(parents=True, exist_ok=True)
    (d / "knowledge-graph.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParse:
    def test_parses_nodes_and_edges(self) -> None:
        g = parse_codebase_graph(_graph_payload())
        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        node = g.nodes["function:src/auth.ts:verifyToken"]
        assert node.name == "verifyToken"
        assert node.type == "function"
        assert node.tags == ["jwt", "validation"]
        assert node.line_range == (12, 34)
        assert node.file_path == "src/auth.ts"

    def test_skips_malformed_nodes_and_edges(self) -> None:
        g = parse_codebase_graph({
            "nodes": [
                {"id": "ok", "type": "file", "name": "ok"},
                "not-a-dict",
                {"type": "file", "name": "no-id"},   # missing id -> skipped
                {"id": "   "},                          # blank id -> skipped
            ],
            "edges": [
                {"source": "a", "target": "b", "type": "calls"},
                {"source": "a"},                        # missing target -> skipped
                42,                                      # not a dict -> skipped
            ],
        })
        assert set(g.nodes) == {"ok"}
        assert len(g.edges) == 1

    def test_tolerates_missing_and_alt_fields(self) -> None:
        g = parse_codebase_graph({
            "nodes": [{"id": "x:y", "file_path": "y.py", "line_range": ["bad", "worse"]}],
            "edges": [{"source": "x:y", "target": "z", "type": "calls", "weight": "nan-ish"}],
        })
        n = g.nodes["x:y"]
        assert n.name == "y"           # derived from id tail when name absent
        assert n.file_path == "y.py"   # snake_case alias honored
        assert n.line_range is None    # malformed lineRange -> None
        assert g.edges[0].weight == 0.5  # non-numeric weight -> default

    def test_empty_payload(self) -> None:
        g = parse_codebase_graph({})
        assert not g.nodes and not g.edges

    def test_degree(self) -> None:
        g = parse_codebase_graph(_graph_payload())
        deg = g.degree()
        assert deg["file:src/auth.ts"] == 2  # contains (out) + imports (in)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoad:
    def test_none_workspace(self) -> None:
        assert resolve_graph_path(None) is None
        assert load_codebase_graph(None) is None

    def test_missing_graph_file(self, tmp_path: Path) -> None:
        assert resolve_graph_path(tmp_path) is None
        assert load_codebase_graph(tmp_path) is None

    def test_loads_present_graph(self, tmp_path: Path) -> None:
        _write_graph(tmp_path, _graph_payload())
        assert resolve_graph_path(tmp_path) is not None
        g = load_codebase_graph(tmp_path)
        assert g is not None and len(g.nodes) == 3

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        d = tmp_path / ".understand-anything"
        d.mkdir()
        (d / "knowledge-graph.json").write_text("{not json", encoding="utf-8")
        assert load_codebase_graph(tmp_path) is None

    def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        d = tmp_path / ".understand-anything"
        d.mkdir()
        (d / "knowledge-graph.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert load_codebase_graph(tmp_path) is None

    def test_empty_nodes_returns_none(self, tmp_path: Path) -> None:
        _write_graph(tmp_path, {"nodes": [], "edges": []})
        assert load_codebase_graph(tmp_path) is None


# ---------------------------------------------------------------------------
# Scoring + formatting
# ---------------------------------------------------------------------------

class TestFormat:
    def test_query_relevant_node_ranks_first(self) -> None:
        g = parse_codebase_graph(_graph_payload())
        out = format_codebase_map(g, "fix the jwt token verification", budget_chars=24000)
        assert "## Codebase map (Understand-Anything knowledge graph)" in out
        # The auth/jwt nodes must appear before the unrelated billing node.
        assert out.index("verifyToken") < out.index("billing.ts")

    def test_node_block_shape(self) -> None:
        g = parse_codebase_graph(_graph_payload())
        out = format_codebase_map(g, "jwt", budget_chars=24000)
        assert "### verifyToken · function · simple  (src/auth.ts:12-34)" in out
        assert "Validates a JWT" in out
        assert "→ contains" in out  # auth.ts file node lists its relationship

    def test_budget_truncates(self) -> None:
        g = parse_codebase_graph(_graph_payload())
        big = format_codebase_map(g, "auth", budget_chars=24000)
        tiny = format_codebase_map(g, "auth", budget_chars=140)
        assert len(tiny) < len(big)
        # header-only (nothing fits) collapses to empty
        assert format_codebase_map(g, "auth", budget_chars=10) == ""

    def test_empty_query_still_gives_overview(self) -> None:
        g = parse_codebase_graph(_graph_payload())
        out = format_codebase_map(g, "", budget_chars=24000)
        # structural prior surfaces file nodes even with no query match
        assert "auth.ts" in out and "billing.ts" in out

    def test_empty_graph(self) -> None:
        assert format_codebase_map(CodebaseGraph(), "x", 24000) == ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_zero_budget(self, tmp_path: Path) -> None:
        _write_graph(tmp_path, _graph_payload())
        assert build_codebase_map_context(tmp_path, "auth", 0) == ""

    def test_no_workspace(self) -> None:
        assert build_codebase_map_context(None, "auth", 6000) == ""

    def test_no_graph(self, tmp_path: Path) -> None:
        assert build_codebase_map_context(tmp_path, "auth", 6000) == ""

    def test_builds_from_graph(self, tmp_path: Path) -> None:
        _write_graph(tmp_path, _graph_payload())
        out = build_codebase_map_context(tmp_path, "jwt verification", 6000)
        assert "Codebase map" in out and "verifyToken" in out


# ---------------------------------------------------------------------------
# Loader integration (context_mode wiring)
# ---------------------------------------------------------------------------

class TestLoaderWiring:
    def test_codebase_map_is_a_valid_mode(self) -> None:
        assert "codebase_map" in CONTEXT_MODES

    def test_plan_with_workspace_root_validates(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\n"
            "name: cbmap\n"
            f"workspace_root: {tmp_path.as_posix()}\n"
            "tasks:\n"
            "  - id: impl\n"
            "    engine: claude\n"
            "    prompt: 'use the codebase'\n"
            "    context_mode: codebase_map\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_path)  # must not raise
        assert plan.tasks[0].context_mode == "codebase_map"

    def test_codebase_map_without_workspace_root_is_e021(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\n"
            "name: cbmap\n"
            "tasks:\n"
            "  - id: impl\n"
            "    engine: claude\n"
            "    prompt: 'use the codebase'\n"
            "    context_mode: codebase_map\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match="E021"):
            load_plan(plan_path)
