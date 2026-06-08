"""Codebase map context mode — consume an Understand-Anything knowledge graph.

``context_mode: codebase_map`` reads the knowledge graph produced by the
Understand-Anything tool (``Lum1104/Understand-Anything``) at
``<workspace_root>/.understand-anything/knowledge-graph.json``, scores its
nodes for relevance to the downstream task prompt, and injects a focused,
budget-bounded codebase map as ``{{ upstream_synthesis }}`` (the same slot the
other synthesis modes use).

This is **interop only** — Maestro reads the documented JSON shape
(``{"nodes": [...], "edges": [...]}``); it does not bundle, vendor, or depend
on Understand-Anything. If the graph is absent the mode degrades to an empty
string (the task simply runs without the extra context).

Zero LLM cost: the graph is pre-built by Understand-Anything; Maestro only
reads it, scores nodes by keyword relevance, and formats them. No external
dependencies — stdlib + JSON only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_GRAPH_REL_PATH = ".understand-anything/knowledge-graph.json"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_MAX_EDGES_PER_NODE = 5

# Relevance weights: a query term hitting a node's name/tags matters more than
# one buried in its prose summary.
_W_NAME = 3.0
_W_TAGS = 2.0
_W_PATH = 1.5
_W_SUMMARY = 1.0

# Structural prior so an unfocused query still yields a sensible overview:
# files/classes anchor a map better than individual functions.
_TYPE_PRIOR: dict[str, float] = {
    "file": 0.6,
    "service": 0.6,
    "class": 0.5,
    "endpoint": 0.5,
    "schema": 0.4,
    "config": 0.4,
}


# ---------------------------------------------------------------------------
# Data model (mirrors the Understand-Anything node/edge schema)
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node: a file, function, class, config, service, endpoint, etc."""

    id: str
    type: str
    name: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    complexity: str = ""
    file_path: str = ""
    line_range: tuple[int, int] | None = None


@dataclass
class GraphEdge:
    """A relationship: imports, calls, inherits, contains, depends_on, ..."""

    source: str
    target: str
    type: str
    weight: float = 0.5


@dataclass
class CodebaseGraph:
    """An in-memory codebase knowledge graph."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)  # id -> node
    edges: list[GraphEdge] = field(default_factory=list)

    def degree(self) -> dict[str, int]:
        """Total (in + out) edge count per node id."""
        deg: dict[str, int] = {}
        for e in self.edges:
            deg[e.source] = deg.get(e.source, 0) + 1
            deg[e.target] = deg.get(e.target, 0) + 1
        return deg


# ---------------------------------------------------------------------------
# Parsing (tolerant — never raises on a malformed graph)
# ---------------------------------------------------------------------------


def _coerce_line_range(value: Any) -> tuple[int, int] | None:
    if isinstance(value, list) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def parse_codebase_graph(data: dict[str, Any]) -> CodebaseGraph:
    """Parse a ``{"nodes": [...], "edges": [...]}`` payload defensively.

    Unknown keys are ignored; malformed nodes/edges are skipped rather than
    raising, so a partially-corrupt graph still yields what it can.
    """
    graph = CodebaseGraph()

    raw_nodes = data.get("nodes")
    if isinstance(raw_nodes, list):
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                continue
            nid = str(raw.get("id") or "").strip()
            if not nid:
                continue
            tags_raw = raw.get("tags")
            tags = (
                [str(t) for t in tags_raw if isinstance(t, (str, int, float))]
                if isinstance(tags_raw, list)
                else []
            )
            graph.nodes[nid] = GraphNode(
                id=nid,
                type=str(raw.get("type") or ""),
                name=str(raw.get("name") or nid.rsplit(":", 1)[-1]),
                summary=str(raw.get("summary") or ""),
                tags=tags,
                complexity=str(raw.get("complexity") or ""),
                file_path=str(raw.get("filePath") or raw.get("file_path") or ""),
                line_range=_coerce_line_range(raw.get("lineRange") or raw.get("line_range")),
            )

    raw_edges = data.get("edges")
    if isinstance(raw_edges, list):
        for raw in raw_edges:
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("source") or "").strip()
            tgt = str(raw.get("target") or "").strip()
            if not src or not tgt:
                continue
            try:
                weight = float(raw.get("weight", 0.5))
            except (TypeError, ValueError):
                weight = 0.5
            graph.edges.append(
                GraphEdge(source=src, target=tgt, type=str(raw.get("type") or "related"), weight=weight)
            )

    return graph


def resolve_graph_path(workspace_root: str | Path | None) -> Path | None:
    """Return the knowledge-graph.json path under *workspace_root*, if present."""
    if not workspace_root:
        return None
    path = Path(workspace_root) / _GRAPH_REL_PATH
    return path if path.is_file() else None


def load_codebase_graph(workspace_root: str | Path | None) -> CodebaseGraph | None:
    """Load + parse the Understand-Anything graph, or ``None`` if unusable."""
    path = resolve_graph_path(workspace_root)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    graph = parse_codebase_graph(data)
    return graph if graph.nodes else None


# ---------------------------------------------------------------------------
# Relevance scoring + formatting (zero LLM cost)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1}


def _score_node(node: GraphNode, query_tokens: set[str], degree: int) -> float:
    """Keyword-overlap relevance + a small structural prior."""
    score = 0.0
    if query_tokens:
        score += _W_NAME * len(query_tokens & _tokens(node.name))
        score += _W_TAGS * len(query_tokens & {t.lower() for t in node.tags})
        score += _W_PATH * len(query_tokens & _tokens(node.file_path))
        score += _W_SUMMARY * len(query_tokens & _tokens(node.summary))
    # Prior keeps the map coherent even when nothing matches the query.
    score += _TYPE_PRIOR.get(node.type, 0.0)
    score += min(degree, 10) * 0.05
    return score


def _format_node(node: GraphNode, edges: list[GraphEdge], nodes: dict[str, GraphNode]) -> str:
    loc = node.file_path
    if node.line_range is not None:
        loc = f"{loc}:{node.line_range[0]}-{node.line_range[1]}"
    head_bits = [node.type or "node"]
    if node.complexity:
        head_bits.append(node.complexity)
    head = f"### {node.name} · {' · '.join(head_bits)}"
    if loc:
        head += f"  ({loc})"
    lines = [head]
    if node.summary:
        lines.append(node.summary)
    rels: list[str] = []
    for e in edges[:_MAX_EDGES_PER_NODE]:
        target = nodes.get(e.target)
        tname = target.name if target else e.target.rsplit(":", 1)[-1]
        rels.append(f"{e.type} {tname}")
    if rels:
        lines.append("→ " + "; ".join(rels))
    return "\n".join(lines) + "\n\n"


def format_codebase_map(graph: CodebaseGraph, query: str, budget_chars: int) -> str:
    """Render the most query-relevant slice of the graph within *budget_chars*."""
    if not graph.nodes or budget_chars <= 0:
        return ""

    query_tokens = _tokens(query)
    degree = graph.degree()

    out_edges: dict[str, list[GraphEdge]] = {}
    for e in graph.edges:
        out_edges.setdefault(e.source, []).append(e)
    for eid in out_edges:
        out_edges[eid].sort(key=lambda x: x.weight, reverse=True)

    ranked = sorted(
        graph.nodes.values(),
        key=lambda n: (_score_node(n, query_tokens, degree.get(n.id, 0)), n.type, n.name),
        reverse=True,
    )

    header = "## Codebase map (Understand-Anything knowledge graph)\n\n"
    parts: list[str] = [header]
    used = len(header)
    for node in ranked:
        block = _format_node(node, out_edges.get(node.id, []), graph.nodes)
        if used + len(block) > budget_chars:
            continue  # try smaller later nodes instead of stopping outright
        parts.append(block)
        used += len(block)

    return "".join(parts) if len(parts) > 1 else ""


def build_codebase_map_context(
    workspace_root: str | Path | None,
    query: str,
    budget_tokens: int = 6000,
) -> str:
    """Load the Understand-Anything graph and format a budget-bounded map.

    Returns an empty string when no usable graph exists at
    ``<workspace_root>/.understand-anything/knowledge-graph.json``.
    """
    if budget_tokens <= 0:
        return ""
    graph = load_codebase_graph(workspace_root)
    if graph is None:
        return ""
    return format_codebase_map(graph, query, budget_tokens * 4)
