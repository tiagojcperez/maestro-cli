"""AST-based call graph and cross-file dependency analysis for Python.

Upgrades ``context_mode: structural`` from regex chunk scoring to precise
blast-radius-aware context selection for Python files.  Non-Python files
continue using the regex fallback in :mod:`symbols`.

Zero additional dependencies — uses stdlib :mod:`ast` module.

Resolution tiers:
  A (must): direct calls with ``ast.Name`` in the same module, explicit imports.
  B (should): cross-file ``from .X import Y``, ``self.method()`` calls.
  C (out-of-scope): ``getattr``, ``importlib``, dynamic dispatch — flagged
  as uncertain edges.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1
_PARSER_VERSION = 3
_CHARS_PER_TOKEN = 4
_MAX_PY_FILES = 5000
_MIN_CHUNK_SCORE = 0.1
_CHUNK_SIZE = 200  # chars per chunk for scoring
_PAGERANK_DAMPING = 0.85
_PAGERANK_MAX_ITERS = 50
_PAGERANK_TOL = 1e-6

_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".maestro-cache", ".maestro-runs",
    ".maestro-worktrees", ".venv", "venv", ".tox", ".eggs", "dist", "build",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FunctionDef:
    """A function or method extracted from a Python AST."""

    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    is_method: bool = False
    class_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified_name": self.qualified_name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "is_method": self.is_method,
            "class_name": self.class_name,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FunctionDef:
        return FunctionDef(**{k: d[k] for k in FunctionDef.__dataclass_fields__})


@dataclass
class ImportRef:
    """A resolved import statement."""

    module: str
    names: list[str] = field(default_factory=list)
    alias: str = ""
    file_path: str = ""
    is_relative: bool = False
    level: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "file_path": self.file_path,
            "is_relative": self.is_relative,
            "level": self.level,
        }


@dataclass
class CallSite:
    """A function call extracted from the AST."""

    caller: str
    callee_name: str
    file_path: str
    line: int
    uncertain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "callee_name": self.callee_name,
            "file_path": self.file_path,
            "line": self.line,
            "uncertain": self.uncertain,
        }


@dataclass
class CodebaseGraph:
    """In-memory call graph and dependency map for a Python codebase."""

    functions: dict[str, FunctionDef] = field(default_factory=dict)
    imports: list[ImportRef] = field(default_factory=list)
    call_edges: dict[str, set[str]] = field(default_factory=dict)
    reverse_edges: dict[str, set[str]] = field(default_factory=dict)
    file_imports: dict[str, list[str]] = field(default_factory=dict)
    uncertain_calls: list[CallSite] = field(default_factory=list)
    pagerank: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "functions": {k: v.to_dict() for k, v in self.functions.items()},
            "imports": [i.to_dict() for i in self.imports],
            "call_edges": {k: sorted(v) for k, v in self.call_edges.items()},
            "reverse_edges": {k: sorted(v) for k, v in self.reverse_edges.items()},
            "file_imports": self.file_imports,
            "uncertain_calls": [c.to_dict() for c in self.uncertain_calls],
            "pagerank": {k: round(v, 8) for k, v in self.pagerank.items()},
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CodebaseGraph:
        return CodebaseGraph(
            functions={k: FunctionDef.from_dict(v) for k, v in d.get("functions", {}).items()},
            imports=[ImportRef(**i) for i in d.get("imports", [])],
            call_edges={k: set(v) for k, v in d.get("call_edges", {}).items()},
            reverse_edges={k: set(v) for k, v in d.get("reverse_edges", {}).items()},
            file_imports=d.get("file_imports", {}),
            uncertain_calls=[CallSite(**c) for c in d.get("uncertain_calls", [])],
            pagerank={
                str(k): float(v)
                for k, v in d.get("pagerank", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class _ASTVisitor(ast.NodeVisitor):
    """Walk a Python AST to extract functions, imports, and call sites."""

    def __init__(self, module_name: str, file_path: str) -> None:
        self.module = module_name
        self.file_path = file_path
        self.functions: list[FunctionDef] = []
        self.imports: list[ImportRef] = []
        self.calls: list[CallSite] = []
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []

    def _current_qualified(self, name: str) -> str:
        if self._class_stack:
            return f"{self.module}:{self._class_stack[-1]}.{name}"
        return f"{self.module}:{name}"

    def _current_scope(self) -> str:
        if self._func_stack:
            return self._func_stack[-1]
        return f"{self.module}:<module>"

    # -- Functions & methods ------------------------------------------------

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qname = self._current_qualified(node.name)
        self.functions.append(FunctionDef(
            qualified_name=qname,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_method=bool(self._class_stack),
            class_name=self._class_stack[-1] if self._class_stack else "",
        ))
        self._func_stack.append(qname)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    # -- Classes ------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    # -- Imports ------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(ImportRef(
                module=alias.name,
                names=[alias.name.rsplit(".", 1)[-1]],
                alias=alias.asname or "",
                file_path=self.file_path,
                is_relative=False,
                level=0,
            ))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        names = [a.name for a in (node.names or [])]
        self.imports.append(ImportRef(
            module=mod,
            names=names,
            alias="",
            file_path=self.file_path,
            is_relative=node.level > 0,
            level=node.level,
        ))
        self.generic_visit(node)

    # -- Calls --------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        caller = self._current_scope()
        callee_name, uncertain = _extract_call_name(node)
        if callee_name:
            self.calls.append(CallSite(
                caller=caller,
                callee_name=callee_name,
                file_path=self.file_path,
                line=node.lineno,
                uncertain=uncertain,
            ))
        self.generic_visit(node)


def _extract_call_name(node: ast.Call) -> tuple[str, bool]:
    """Extract the callable name from a Call node.  Returns (name, uncertain)."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id, False
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}", False
        return func.attr, True  # chained attribute — uncertain
    # Subscript call, starred, etc.
    return "", True


# ---------------------------------------------------------------------------
# File-level extraction
# ---------------------------------------------------------------------------

def _visit_file(file_path: Path, module_name: str) -> tuple[
    list[FunctionDef], list[ImportRef], list[CallSite],
]:
    """Parse one ``.py`` file and extract symbols, imports, and calls."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, ValueError):
        return [], [], []
    visitor = _ASTVisitor(module_name, str(file_path.as_posix()))
    visitor.visit(tree)
    return visitor.functions, visitor.imports, visitor.calls


def _module_name_from_path(rel_path: str, workspace_root: Path) -> str:
    """Convert ``src/foo/bar.py`` to ``src.foo.bar``."""
    p = Path(rel_path)
    parts = list(p.with_suffix("").parts)
    if parts and parts[-1] == "__init__" and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

def _resolve_import_target(
    imp: ImportRef,
    known_functions: dict[str, FunctionDef],
    *,
    importer_module: str,
    importer_is_package: bool,
    module_exports: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Map imported names to qualified symbols.

    Returns a dict of ``{local_alias: qualified_name}`` for each name the
    import introduces into scope.
    """
    exports = module_exports or {}
    resolved_module = _resolve_import_module(
        imp,
        importer_module=importer_module,
        importer_is_package=importer_is_package,
    )
    result: dict[str, str] = {}
    for name in imp.names:
        if name == "*":
            continue
        exported_target = exports.get(resolved_module, {}).get(name)
        candidate = f"{resolved_module}:{name}" if resolved_module else name
        if exported_target is not None:
            result[name] = exported_target
        elif candidate in known_functions or resolved_module:
            result[name] = candidate
    if imp.alias and imp.names:
        alias_name = imp.names[0]
        result[imp.alias] = result.get(
            alias_name,
            f"{resolved_module}:{alias_name}" if resolved_module else alias_name,
        )
    return result


def _resolve_import_module(
    imp: ImportRef,
    *,
    importer_module: str,
    importer_is_package: bool,
) -> str:
    """Resolve relative import modules to absolute module paths."""
    if not imp.is_relative or imp.level <= 0:
        return imp.module

    importer_parts = [part for part in importer_module.split(".") if part]
    package_parts = importer_parts if importer_is_package else importer_parts[:-1]
    ascend = max(imp.level - 1, 0)
    if ascend >= len(package_parts):
        anchor_parts: list[str] = []
    else:
        anchor_parts = package_parts[: len(package_parts) - ascend]

    if imp.module:
        return ".".join(anchor_parts + imp.module.split("."))
    return ".".join(anchor_parts)


def _resolve_call(
    call: CallSite,
    local_functions: dict[str, str],
    import_aliases: dict[str, str],
    module_name: str,
    class_context: str,
    known_functions: dict[str, FunctionDef],
    module_exports: dict[str, dict[str, str]],
) -> str | None:
    """Resolve a raw call name to a qualified name.

    Returns ``None`` for unresolvable calls (Tier C).
    """
    name = call.callee_name
    if call.uncertain:
        return None

    # self.method() → CurrentClass.method
    if name.startswith("self.") and class_context:
        method = name[5:]
        return f"{module_name}:{class_context}.{method}"

    # Direct local function call
    if name in local_functions:
        return local_functions[name]

    # Imported name
    if name in import_aliases:
        target = import_aliases[name]
        if target in known_functions:
            return target
        if ":" in target:
            mod_part, attr = target.split(":", 1)
            exported = module_exports.get(mod_part, {}).get(attr)
            if exported is not None:
                return exported
        return target if target else None

    # Dotted call: "mod.func" — check if "mod" is an import alias
    if "." in name:
        prefix, attr = name.split(".", 1)
        if prefix in import_aliases:
            base = import_aliases[prefix]
            if ":" in base:
                mod_part = base.split(":")[0]
                exported = module_exports.get(mod_part, {}).get(attr)
                if exported is not None:
                    return exported
                candidate = f"{mod_part}:{attr}"
                if candidate in known_functions:
                    return candidate
                return f"{mod_part}:{attr}"
            return f"{base}:{attr}"

    # Module-level qualified name guess (same module)
    candidate = f"{module_name}:{name}"
    if candidate in local_functions:
        return candidate

    return None


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_py_files(workspace_root: Path) -> list[tuple[Path, str, int]]:
    """Walk workspace and collect ``.py`` files as ``(abs_path, rel_posix, mtime_ns)``."""
    result: list[tuple[Path, str, int]] = []
    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            abs_path = Path(dirpath) / fname
            try:
                stat = abs_path.stat()
            except OSError:
                continue
            rel = abs_path.relative_to(workspace_root).as_posix()
            result.append((abs_path, rel, stat.st_mtime_ns))
            if len(result) >= _MAX_PY_FILES:
                return result
    result.sort(key=lambda t: t[1])
    return result


def _files_snapshot_id(files: list[tuple[Path, str, int]]) -> str:
    """SHA-256 of ``(path, mtime_ns)`` tuples for cache invalidation."""
    h = hashlib.sha256()
    for _, rel, mtime_ns in files:
        h.update(f"{rel}|{mtime_ns}\n".encode())
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _default_cache_dir(workspace_root: Path) -> Path:
    return workspace_root / ".maestro-cache" / "codebase_graph"


def _load_cached_graph(cache_dir: Path, snapshot_id: str) -> CodebaseGraph | None:
    """Load graph from cache if snapshot and versions match."""
    path = cache_dir / f"{snapshot_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("schema_version") != _SCHEMA_VERSION:
        return None
    if data.get("parser_version") != _PARSER_VERSION:
        return None
    if data.get("python_version") != sys.version:
        return None
    if data.get("snapshot_id") != snapshot_id:
        return None
    return CodebaseGraph.from_dict(data)


def _store_graph(
    graph: CodebaseGraph,
    cache_dir: Path,
    snapshot_id: str,
    workspace_root: str,
    file_count: int,
) -> Path:
    """Write graph as JSON with atomic write pattern."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": _SCHEMA_VERSION,
        "parser_version": _PARSER_VERSION,
        "python_version": sys.version,
        "workspace_root": workspace_root,
        "snapshot_id": snapshot_id,
        "file_count": file_count,
        **graph.to_dict(),
    }
    dest = cache_dir / f"{snapshot_id}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_codebase_graph(
    workspace_root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> CodebaseGraph:
    """Build a call graph for all ``.py`` files under *workspace_root*.

    Checks the cache first; rebuilds only if stale or *force_rebuild*.
    """
    ws = Path(workspace_root)
    cdir = Path(cache_dir) if cache_dir else _default_cache_dir(ws)
    py_files = _collect_py_files(ws)
    if not py_files:
        return CodebaseGraph()

    snap_id = _files_snapshot_id(py_files)

    if not force_rebuild:
        cached = _load_cached_graph(cdir, snap_id)
        if cached is not None:
            return cached

    all_functions: dict[str, FunctionDef] = {}
    all_imports: list[ImportRef] = []
    all_calls: list[CallSite] = []
    file_module_info: dict[str, tuple[str, bool]] = {}  # file → (module, is_package_init)

    for abs_path, rel, _ in py_files:
        mod = _module_name_from_path(rel, ws)
        file_module_info[str(abs_path.as_posix())] = (mod, abs_path.stem == "__init__")
        funcs, imps, calls = _visit_file(abs_path, mod)
        for f in funcs:
            all_functions[f.qualified_name] = f
        all_imports.extend(imps)
        all_calls.extend(calls)

    # Build per-file local function lookup and import aliases
    file_local_funcs: dict[str, dict[str, str]] = {}  # file → {name: qname}
    file_import_aliases: dict[str, dict[str, str]] = {}  # file → {alias: qname}

    for qname, fdef in all_functions.items():
        fp = fdef.file_path
        local = file_local_funcs.setdefault(fp, {})
        short = qname.split(":")[-1] if ":" in qname else qname
        local[short] = qname
        # Also register unqualified method name for self.method resolution
        if "." in short:
            local[short.split(".")[-1]] = qname

    # Resolve import aliases per file (pass 1: direct targets)
    for imp in all_imports:
        importer_module, importer_is_package = file_module_info.get(imp.file_path, ("", False))
        aliases = _resolve_import_target(
            imp,
            all_functions,
            importer_module=importer_module,
            importer_is_package=importer_is_package,
        )
        existing = file_import_aliases.setdefault(imp.file_path, {})
        existing.update(aliases)

    # Package __init__.py files can re-export imported symbols.
    module_exports: dict[str, dict[str, str]] = {}
    for file_path, aliases in file_import_aliases.items():
        module_name, is_package_init = file_module_info.get(file_path, ("", False))
        if not module_name or not is_package_init:
            continue
        export_map = module_exports.setdefault(module_name, {})
        for alias_name, target in aliases.items():
            if target in all_functions:
                export_map[alias_name] = target

    # Resolve import aliases again so consumers of package re-exports
    # point directly at the underlying implementation symbol.
    file_import_aliases = {}
    for imp in all_imports:
        importer_module, importer_is_package = file_module_info.get(imp.file_path, ("", False))
        aliases = _resolve_import_target(
            imp,
            all_functions,
            importer_module=importer_module,
            importer_is_package=importer_is_package,
            module_exports=module_exports,
        )
        existing = file_import_aliases.setdefault(imp.file_path, {})
        existing.update(aliases)

    # Resolve call edges
    call_edges: dict[str, set[str]] = {}
    reverse_edges: dict[str, set[str]] = {}
    uncertain_calls: list[CallSite] = []

    for cs in all_calls:
        # Determine class context from caller
        class_ctx = ""
        if cs.caller in all_functions and all_functions[cs.caller].class_name:
            class_ctx = all_functions[cs.caller].class_name

        mod_name = cs.caller.split(":")[0] if ":" in cs.caller else ""
        local_funcs = file_local_funcs.get(cs.file_path, {})
        import_aliases = file_import_aliases.get(cs.file_path, {})

        resolved = _resolve_call(
            cs,
            local_funcs,
            import_aliases,
            mod_name,
            class_ctx,
            all_functions,
            module_exports,
        )
        if resolved is None:
            if cs.callee_name:
                uncertain_calls.append(cs)
            continue

        edges = call_edges.setdefault(cs.caller, set())
        edges.add(resolved)
        rev = reverse_edges.setdefault(resolved, set())
        rev.add(cs.caller)

    # File-level import dependencies
    file_imports: dict[str, list[str]] = {}
    for imp in all_imports:
        deps = file_imports.setdefault(imp.file_path, [])
        importer_module, importer_is_package = file_module_info.get(imp.file_path, ("", False))
        resolved_module = _resolve_import_module(
            imp,
            importer_module=importer_module,
            importer_is_package=importer_is_package,
        )
        if resolved_module:
            deps.append(resolved_module)

    graph = CodebaseGraph(
        functions=all_functions,
        imports=all_imports,
        call_edges=call_edges,
        reverse_edges=reverse_edges,
        file_imports=file_imports,
        uncertain_calls=uncertain_calls,
    )
    graph.pagerank = _compute_pagerank(graph)
    _store_graph(graph, cdir, snap_id, str(ws), len(py_files))
    return graph


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

def blast_radius(
    graph: CodebaseGraph,
    changed_symbols: set[str],
    max_depth: int = 10,
) -> set[str]:
    """BFS over call edges from *changed_symbols*.  Returns all downstream-affected."""
    visited: set[str] = set()
    frontier = set(changed_symbols)
    depth = 0
    while frontier and depth < max_depth:
        visited |= frontier
        next_frontier: set[str] = set()
        for sym in frontier:
            for callee in graph.call_edges.get(sym, set()):
                if callee not in visited:
                    next_frontier.add(callee)
            # Also include callers (upstream impact)
            for caller in graph.reverse_edges.get(sym, set()):
                if caller not in visited:
                    next_frontier.add(caller)
        frontier = next_frontier
        depth += 1
    return visited


def find_clusters(graph: CodebaseGraph) -> list[set[str]]:
    """Tarjan's algorithm for strongly connected components.

    Returns only SCCs with size > 1 (mutually recursive cycles).
    """
    index_counter = [0]
    stack: list[str] = []
    lowlink: dict[str, int] = {}
    index: dict[str, int] = {}
    on_stack: set[str] = set()
    sccs: list[set[str]] = []

    def _strongconnect(v: str) -> None:
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.call_edges.get(v, set()):
            if w not in index:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            component: set[str] = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.add(w)
                if w == v:
                    break
            if len(component) > 1:
                sccs.append(component)

    for node in graph.call_edges:
        if node not in index:
            _strongconnect(node)

    return sccs


def _graph_nodes(graph: CodebaseGraph) -> set[str]:
    """Return the full node set participating in call-graph analytics."""
    nodes = set(graph.functions)
    nodes.update(graph.call_edges)
    nodes.update(graph.reverse_edges)
    for targets in graph.call_edges.values():
        nodes.update(targets)
    for callers in graph.reverse_edges.values():
        nodes.update(callers)
    return nodes


def _compute_pagerank(
    graph: CodebaseGraph,
    *,
    damping: float = _PAGERANK_DAMPING,
    max_iters: int = _PAGERANK_MAX_ITERS,
    tol: float = _PAGERANK_TOL,
) -> dict[str, float]:
    """Compute PageRank over the directed call graph."""
    nodes = _graph_nodes(graph)
    if not nodes:
        return {}

    node_count = len(nodes)
    ranks = {node: 1.0 / node_count for node in nodes}
    outgoing = {
        node: set(graph.call_edges.get(node, set())) & nodes
        for node in nodes
    }
    incoming = {
        node: set(graph.reverse_edges.get(node, set())) & nodes
        for node in nodes
    }

    for _ in range(max_iters):
        sink_mass = sum(ranks[node] for node, edges in outgoing.items() if not edges)
        updated: dict[str, float] = {}
        delta = 0.0

        for node in nodes:
            rank = (1.0 - damping) / node_count
            rank += damping * sink_mass / node_count
            for source in incoming[node]:
                out_degree = len(outgoing[source])
                if out_degree:
                    rank += damping * ranks[source] / out_degree
            updated[node] = rank
            delta += abs(rank - ranks[node])

        ranks = updated
        if delta < tol:
            break

    total = sum(ranks.values())
    if total <= 0:
        return {}
    return {node: rank / total for node, rank in ranks.items()}


# ---------------------------------------------------------------------------
# Integration with context_mode: structural
# ---------------------------------------------------------------------------

def build_ast_structural_context(
    upstream_texts: dict[str, str],
    budget_tokens: int,
    upstream_files_changed: dict[str, list[str]] | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> str:
    """Drop-in enhancement for ``build_structural_context``.

    When *workspace_root* is set and the workspace contains ``.py`` files,
    uses the AST call graph for blast-radius-aware scoring.  Falls back to
    the regex-based ``symbols.build_structural_context`` otherwise.
    """
    from .symbols import build_structural_context as _regex_fallback

    if not upstream_texts or budget_tokens <= 0:
        return ""

    if not workspace_root:
        return _regex_fallback(upstream_texts, budget_tokens, upstream_files_changed)

    ws = Path(workspace_root)
    if not ws.is_dir():
        return _regex_fallback(upstream_texts, budget_tokens, upstream_files_changed)

    try:
        graph = build_codebase_graph(ws)
    except Exception:
        return _regex_fallback(upstream_texts, budget_tokens, upstream_files_changed)

    if not graph.functions:
        return _regex_fallback(upstream_texts, budget_tokens, upstream_files_changed)

    # Collect changed symbols from upstream files
    changed: set[str] = set()
    if upstream_files_changed:
        for uid, files in upstream_files_changed.items():
            for fpath in files:
                if fpath.endswith(".py"):
                    mod = _module_name_from_path(fpath, ws)
                    for qname, fdef in graph.functions.items():
                        if qname.startswith(f"{mod}:"):
                            changed.add(qname)

    # Also extract symbol names from upstream text (regex fallback for names)
    from .symbols import extract_changed_symbols, extract_symbols

    for uid, text in upstream_texts.items():
        diff_syms = extract_changed_symbols(text)
        if diff_syms:
            names = {s.name for s in diff_syms if s.kind != "import"}
        else:
            names = {s.name for s in extract_symbols(text) if s.kind != "import"}
        # Match names against graph functions
        for name in names:
            for qname in graph.functions:
                short = qname.split(":")[-1] if ":" in qname else qname
                if name == short or short.endswith(f".{name}"):
                    changed.add(qname)

    if not changed:
        return _regex_fallback(upstream_texts, budget_tokens, upstream_files_changed)

    blast_set = blast_radius(graph, changed)
    budget_chars = budget_tokens * _CHARS_PER_TOKEN

    # Score chunks using blast radius membership
    scored: list[tuple[float, str, str]] = []
    for uid, text in upstream_texts.items():
        chunks = _split_text_chunks(text)
        for chunk in chunks:
            score = _score_chunk_with_graph(chunk, blast_set, changed, graph)
            if score >= _MIN_CHUNK_SCORE:
                scored.append((score, chunk, uid))

    scored.sort(key=lambda x: -x[0])

    # Greedy selection within budget
    selected: list[str] = []
    used = 0
    seen_uids: set[str] = set()
    for score, chunk, uid in scored:
        header = f"--- {uid} ---\n" if uid not in seen_uids else ""
        entry = header + chunk
        if used + len(entry) > budget_chars:
            continue
        selected.append(entry)
        used += len(entry)
        seen_uids.add(uid)

    return "\n\n".join(selected)


def _split_text_chunks(text: str) -> list[str]:
    """Split text into fixed-size chunks for scoring."""
    chunks: list[str] = []
    lines = text.split("\n")
    current: list[str] = []
    current_len = 0
    for line in lines:
        current.append(line)
        current_len += len(line) + 1
        if current_len >= _CHUNK_SIZE:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append("\n".join(current))
    return chunks


def _score_chunk_with_graph(
    chunk: str,
    blast_set: set[str],
    changed_symbols: set[str],
    graph: CodebaseGraph,
) -> float:
    """Score a chunk by blast-radius symbol membership.

    Changed symbols score 1.5, blast-radius members 1.0, with a small
    PageRank bonus for central symbols in the call graph.
    """
    max_rank = max(graph.pagerank.values(), default=0.0)

    def _bonus(qname: str, base_weight: float) -> float:
        if max_rank <= 0:
            return 0.0
        rank = graph.pagerank.get(qname, 0.0)
        return 0.5 * base_weight * (rank / max_rank)

    score = 0.0
    chunk_lower = chunk.lower()
    for qname in changed_symbols:
        short = qname.split(":")[-1] if ":" in qname else qname
        parts = short.split(".")
        for part in parts:
            if part.lower() in chunk_lower:
                score += 1.5 + _bonus(qname, 1.5)
                break
    for qname in blast_set - changed_symbols:
        short = qname.split(":")[-1] if ":" in qname else qname
        parts = short.split(".")
        for part in parts:
            if part.lower() in chunk_lower:
                score += 1.0 + _bonus(qname, 1.0)
                break
    return score
