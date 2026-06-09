"""SCIP context mode — consume a pre-built SCIP code-intelligence index (JSON).

``context_mode: scip`` reads a SCIP index in its **JSON** form at
``<workspace_root>/index.scip.json`` (produce it with
``scip print --json index.scip > index.scip.json``), scores its symbols for
relevance to the downstream task prompt, and injects a focused, budget-bounded
structural map as ``{{ upstream_synthesis }}`` (the same slot the other
synthesis modes use).

This is **interop only** — Maestro reads the documented SCIP JSON shape and does
not bundle, vendor, or depend on any SCIP tooling. Because it ingests the JSON
form, it needs **no protobuf library** (true to the zero-dependency core).  It
is multi-language by construction: any SCIP indexer works (``scip-python``,
``scip-typescript``, ``scip-go``, ``scip-java``, ``rust-analyzer``, ...). If the
index is absent the mode degrades to an empty string.

Zero LLM cost: the index is pre-built; Maestro only reads it, scores symbols by
keyword relevance, and formats them. Stdlib + JSON only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Candidate locations for the JSON-rendered SCIP index, in preference order.
_SCIP_REL_PATHS = ("index.scip.json", ".scip/index.json", "scip.index.json")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL_DEFINITION_ROLE = 0x1  # SCIP symbol_roles bit for a definition

# Relevance weights: a query term hitting a symbol's name matters more than one
# buried in its file path or documentation.
_W_NAME = 3.0
_W_FILE = 1.5
_W_DOC = 1.0

# Cap the overview (no query match) so a huge index can't blow the budget on
# arbitrary symbols.
_OVERVIEW_LIMIT = 50


# ---------------------------------------------------------------------------
# Data model (mirrors the SCIP Index/Document/SymbolInformation schema)
# ---------------------------------------------------------------------------


@dataclass
class ScipSymbol:
    """A symbol definition: its display name, file, and documentation."""

    name: str
    file: str
    documentation: str = ""
    symbol_id: str = ""


@dataclass
class ScipIndex:
    """An in-memory slice of a SCIP index (definitions only)."""

    symbols: list[ScipSymbol] = field(default_factory=list)
    tool: str = ""
    document_count: int = 0


# ---------------------------------------------------------------------------
# Parsing (tolerant — never raises on a malformed index; camelCase or snake_case)
# ---------------------------------------------------------------------------


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _symbol_name(symbol: str, display_name: str) -> str:
    """Best-effort readable name from a SCIP symbol string.

    Prefers an explicit ``display_name`` (newer SCIP).  Otherwise parses the
    SCIP symbol grammar — ``<scheme> <manager> <name> <version> <descriptors>``
    — and pulls the trailing identifier(s) from the descriptor portion.
    """
    if display_name:
        return display_name
    if not symbol or symbol.startswith("local "):
        return symbol[:40] or "?"
    parts = symbol.split(" ")
    descriptors = parts[-1] if len(parts) >= 5 else symbol
    tokens = _IDENT_RE.findall(descriptors)
    if not tokens:
        return descriptors[:40] or symbol[:40]
    return ".".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]


def _documentation_text(value: Any) -> str:
    if isinstance(value, list):
        joined = " ".join(str(x) for x in value if isinstance(x, str))
        return " ".join(joined.split())[:200]
    return ""


def parse_scip_index(data: dict[str, Any]) -> ScipIndex:
    """Parse a SCIP JSON payload defensively, collecting symbol definitions."""
    index = ScipIndex()

    meta = data.get("metadata")
    if isinstance(meta, dict):
        tool_info = _get(meta, "tool_info", "toolInfo", default=None)
        if isinstance(tool_info, dict):
            index.tool = str(tool_info.get("name") or "")

    documents = data.get("documents")
    if not isinstance(documents, list):
        return index
    index.document_count = len(documents)

    seen: set[str] = set()
    for document in documents:
        if not isinstance(document, dict):
            continue
        rel_path = str(_get(document, "relative_path", "relativePath", default="") or "")

        symbols = document.get("symbols")
        if isinstance(symbols, list) and symbols:
            for entry in symbols:
                if not isinstance(entry, dict):
                    continue
                symbol = str(entry.get("symbol") or "")
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                index.symbols.append(
                    ScipSymbol(
                        name=_symbol_name(
                            symbol,
                            str(_get(entry, "display_name", "displayName", default="") or ""),
                        ),
                        file=rel_path,
                        documentation=_documentation_text(entry.get("documentation")),
                        symbol_id=symbol,
                    )
                )
            continue

        # Fallback: derive definitions from occurrences when no SymbolInformation.
        occurrences = document.get("occurrences")
        if not isinstance(occurrences, list):
            continue
        for occ in occurrences:
            if not isinstance(occ, dict):
                continue
            roles = _get(occ, "symbol_roles", "symbolRoles", default=0)
            symbol = str(occ.get("symbol") or "")
            if not symbol or symbol in seen:
                continue
            if isinstance(roles, int) and roles & _SYMBOL_DEFINITION_ROLE:
                seen.add(symbol)
                index.symbols.append(
                    ScipSymbol(
                        name=_symbol_name(symbol, ""),
                        file=rel_path,
                        symbol_id=symbol,
                    )
                )

    return index


def resolve_scip_path(workspace_root: str | Path | None) -> Path | None:
    """Return the JSON SCIP index path under *workspace_root*, if present."""
    if not workspace_root:
        return None
    base = Path(workspace_root)
    for rel in _SCIP_REL_PATHS:
        candidate = base / rel
        if candidate.is_file():
            return candidate
    return None


def load_scip_index(workspace_root: str | Path | None) -> ScipIndex | None:
    """Load + parse the SCIP JSON index, or ``None`` if unusable."""
    path = resolve_scip_path(workspace_root)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    index = parse_scip_index(data)
    return index if index.symbols else None


# ---------------------------------------------------------------------------
# Relevance scoring + formatting (zero LLM cost)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1}


def _score_symbol(symbol: ScipSymbol, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    score = 0.0
    score += _W_NAME * len(query_tokens & _tokens(symbol.name))
    score += _W_FILE * len(query_tokens & _tokens(symbol.file))
    score += _W_DOC * len(query_tokens & _tokens(symbol.documentation))
    return score


def _format_symbol(symbol: ScipSymbol) -> str:
    head = f"### {symbol.name}"
    if symbol.file:
        head += f"  ({symbol.file})"
    block = head + "\n"
    if symbol.documentation:
        block += symbol.documentation + "\n"
    return block + "\n"


def format_scip_map(index: ScipIndex, query: str, budget_chars: int) -> str:
    """Render the most query-relevant symbols within *budget_chars*."""
    if not index.symbols or budget_chars <= 0:
        return ""

    query_tokens = _tokens(query)
    scored = sorted(
        index.symbols,
        key=lambda s: (_score_symbol(s, query_tokens), s.file, s.name),
        reverse=True,
    )
    relevant = [s for s in scored if _score_symbol(s, query_tokens) > 0]
    # Fall back to a bounded overview when nothing matches the query, so the
    # mode still yields a useful structural map.
    chosen = relevant if relevant else index.symbols[:_OVERVIEW_LIMIT]

    tool = f" · {index.tool}" if index.tool else ""
    header = f"## Codebase map (SCIP index{tool})\n\n"
    parts: list[str] = [header]
    used = len(header)
    for symbol in chosen:
        block = _format_symbol(symbol)
        if used + len(block) > budget_chars:
            continue  # try smaller later symbols instead of stopping outright
        parts.append(block)
        used += len(block)

    return "".join(parts) if len(parts) > 1 else ""


def build_scip_context(
    workspace_root: str | Path | None,
    query: str,
    budget_tokens: int = 6000,
) -> str:
    """Load the SCIP JSON index and format a budget-bounded structural map.

    Returns an empty string when no usable index exists under *workspace_root*.
    """
    if budget_tokens <= 0:
        return ""
    index = load_scip_index(workspace_root)
    if index is None:
        return ""
    return format_scip_map(index, query, budget_tokens * 4)
