"""Knowledge graph context mode — entity extraction and graph-based retrieval.

``context_mode: knowledge_graph`` extracts structured entities (functions,
files, concepts, decisions, errors) from upstream task output and builds a
typed graph.  Downstream tasks receive a focused subgraph of relevant
entities and their relationships instead of raw text.

Inspired by HippoRAG 2 (associative multi-hop retrieval) and MemoRAG
(global memory + clue-guided retrieval).

The graph is lightweight — no external dependencies, stored as JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

EntityType = str  # "file" | "function" | "class" | "concept" | "decision" | "error" | "dependency"
RelationType = str  # "defines" | "modifies" | "depends_on" | "causes" | "resolves" | "mentions"
_RELATION_CONFIDENCE: dict[str, float] = {
    "defines": 0.95,
    "resolves": 0.9,
    "causes": 0.85,
    "depends_on": 0.8,
    "modifies": 0.75,
    "mentions": 0.4,
}


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str  # stable hash
    name: str
    entity_type: EntityType
    source_task: str = ""  # task ID that produced this entity
    properties: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "source_task": self.source_task,
            "properties": self.properties,
        }


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: RelationType
    source_task: str = ""
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.confidence <= 0.0:
            self.confidence = _RELATION_CONFIDENCE.get(self.relation_type, 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "source_task": self.source_task,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class KnowledgeGraph:
    """A lightweight in-memory knowledge graph."""

    entities: dict[str, Entity] = field(default_factory=dict)  # id → Entity
    relations: list[Relation] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def get_related(self, entity_id: str, max_hops: int = 2) -> set[str]:
        """Get entity IDs within max_hops of the given entity."""
        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for eid in frontier:
                for rel in self.relations:
                    if rel.source_id == eid and rel.target_id not in visited:
                        next_frontier.add(rel.target_id)
                    if rel.target_id == eid and rel.source_id not in visited:
                        next_frontier.add(rel.source_id)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def subgraph(self, entity_ids: set[str]) -> KnowledgeGraph:
        """Extract a subgraph containing only the given entities."""
        sub = KnowledgeGraph()
        for eid in entity_ids:
            if eid in self.entities:
                sub.entities[eid] = self.entities[eid]
        sub.relations = [
            r for r in self.relations
            if r.source_id in entity_ids and r.target_id in entity_ids
        ]
        return sub

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relations": [r.to_dict() for r in self.relations],
        }

    def format_context(self, budget_chars: int = 24000) -> str:
        """Format the graph as a human-readable context string."""
        parts: list[str] = []
        used = 0

        # Group entities by type
        by_type: dict[str, list[Entity]] = {}
        for entity in self.entities.values():
            by_type.setdefault(entity.entity_type, []).append(entity)

        for etype, entities in sorted(by_type.items()):
            header = f"## {etype.title()}s\n"
            if used + len(header) > budget_chars:
                break
            parts.append(header)
            used += len(header)

            for entity in entities:
                props = ""
                if entity.properties:
                    props = " — " + ", ".join(f"{k}: {v}" for k, v in entity.properties.items())
                line = f"- **{entity.name}**{props} (from {entity.source_task})\n"
                if used + len(line) > budget_chars:
                    break
                parts.append(line)
                used += len(line)

        # Relations
        if self.relations and used < budget_chars:
            header = "\n## Relationships\n"
            parts.append(header)
            used += len(header)
            # Higher-confidence relations surface first (e.g. DEFINES > MENTIONS).
            # This makes the rendered graph emphasize stronger evidence.
            for rel in sorted(
                self.relations,
                key=lambda r: (r.confidence, r.relation_type, r.source_id, r.target_id),
                reverse=True,
            ):
                src = self.entities.get(rel.source_id)
                tgt = self.entities.get(rel.target_id)
                if src and tgt:
                    line = (
                        f"- [{rel.confidence:.2f}] "
                        f"{src.name} —[{rel.relation_type}]→ {tgt.name}\n"
                    )
                    if used + len(line) > budget_chars:
                        break
                    parts.append(line)
                    used += len(line)

        return "".join(parts)


# ---------------------------------------------------------------------------
# Entity ID generation
# ---------------------------------------------------------------------------


def _entity_id(name: str, entity_type: str) -> str:
    """Generate a stable entity ID."""
    return hashlib.sha256(f"{entity_type}:{name}".encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Entity extraction (regex-based)
# ---------------------------------------------------------------------------

# File paths in diff headers or backtick references
_FILE_RE = re.compile(r"(?:diff --git a/|[+]{3} [ab]/)(\S+)")
_BACKTICK_FILE_RE = re.compile(r"`([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)`")

# Function/class definitions (multi-language, simplified)
_FUNC_RE = re.compile(
    r"(?:def|function|func|fn|pub fn)\s+(\w+)",
)
_CLASS_RE = re.compile(
    r"(?:class|struct|interface|impl|type)\s+(\w+)",
)

# Decision patterns
_DECISION_RE = re.compile(
    r"(?:decided|chose|selected|will use|switching to|adopting)\s+(.{10,80})",
    re.IGNORECASE,
)

# Error patterns
_ERROR_RE = re.compile(
    r"(?:error|Error|ERROR|failed|FAILED|exception|Exception)[:]\s*(.{10,120})",
)

# Dependency/import patterns
_IMPORT_RE = re.compile(
    r"(?:import|require|from|use|include)\s+['\"]?([a-zA-Z0-9_./-]+)",
)

_MAX_ENTITIES_PER_UPSTREAM = 50


def extract_entities(
    text: str,
    source_task: str,
) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations from a task's output text."""
    entities: list[Entity] = []
    relations: list[Relation] = []
    seen_names: set[str] = set()

    def _add(name: str, etype: str, props: dict[str, str] | None = None) -> str:
        if name in seen_names or len(entities) >= _MAX_ENTITIES_PER_UPSTREAM:
            eid = _entity_id(name, etype)
            return eid
        seen_names.add(name)
        eid = _entity_id(name, etype)
        entities.append(Entity(
            id=eid,
            name=name,
            entity_type=etype,
            source_task=source_task,
            properties=props or {},
        ))
        return eid

    # Extract files
    file_ids: list[str] = []
    for m in _FILE_RE.finditer(text[:5000]):
        fid = _add(m.group(1), "file")
        file_ids.append(fid)
    for m in _BACKTICK_FILE_RE.finditer(text[:5000]):
        fid = _add(m.group(1), "file")
        file_ids.append(fid)

    # Extract functions
    for m in _FUNC_RE.finditer(text[:5000]):
        name = m.group(1)
        if len(name) > 2:  # Skip very short names
            fid = _add(name, "function")
            # Relate to nearest file
            if file_ids:
                relations.append(Relation(
                    source_id=file_ids[-1],
                    target_id=fid,
                    relation_type="defines",
                    source_task=source_task,
                ))

    # Extract classes
    for m in _CLASS_RE.finditer(text[:5000]):
        name = m.group(1)
        if len(name) > 2:
            _add(name, "class")

    # Extract decisions
    for m in _DECISION_RE.finditer(text[:5000]):
        decision = m.group(1).strip().rstrip(".")
        if len(decision) > 10:
            _add(decision, "decision")

    # Extract errors
    for m in _ERROR_RE.finditer(text[:3000]):
        error = m.group(1).strip()
        if len(error) > 10:
            _add(error[:80], "error")

    # Extract dependencies
    for m in _IMPORT_RE.finditer(text[:3000]):
        dep = m.group(1).strip("'\"")
        if len(dep) > 1:
            _add(dep, "dependency")

    return entities, relations


# ---------------------------------------------------------------------------
# Graph construction from upstream outputs
# ---------------------------------------------------------------------------


def build_knowledge_graph(
    upstream_texts: dict[str, str],
    budget_tokens: int = 6000,
) -> str:
    """Build a knowledge graph from upstream task outputs and format as context.

    1. Extract entities and relations from each upstream
    2. Build a unified graph
    3. Format within token budget

    Zero LLM cost — all regex-based extraction.
    """
    if not upstream_texts or budget_tokens <= 0:
        return ""

    graph = KnowledgeGraph()

    for task_id, text in upstream_texts.items():
        entities, relations = extract_entities(text, task_id)
        for entity in entities:
            graph.add_entity(entity)
        for relation in relations:
            graph.add_relation(relation)

    if not graph.entities:
        # Fallback: no entities found, return truncated raw text
        parts: list[str] = []
        budget_chars = budget_tokens * 4
        remaining = budget_chars
        for uid, text in upstream_texts.items():
            header = f"--- {uid} ---\n"
            if remaining <= len(header):
                break
            truncated = text[: remaining - len(header)]
            parts.append(header + truncated)
            remaining -= len(header) + len(truncated)
        return "\n\n".join(parts)

    budget_chars = budget_tokens * 4
    return graph.format_context(budget_chars)
