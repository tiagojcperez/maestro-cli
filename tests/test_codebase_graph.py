"""Tests for codebase_graph.py — AST-based call graph and dependency analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from maestro_cli.codebase_graph import (
    CallSite,
    CodebaseGraph,
    FunctionDef,
    ImportRef,
    _ASTVisitor,
    _collect_py_files,
    _extract_call_name,
    _files_snapshot_id,
    _module_name_from_path,
    _resolve_call,
    _score_chunk_with_graph,
    _split_text_chunks,
    _visit_file,
    blast_radius,
    build_ast_structural_context,
    build_codebase_graph,
    find_clusters,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _write_py(tmp_path: Path, rel: str, content: str) -> Path:
    """Write a .py file under tmp_path and return its absolute path."""
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# Tier A — Intra-file, direct calls
# ===========================================================================


class TestTierAIntraFile:
    def test_extract_function_defs(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "mod.py", "def foo():\n    pass\ndef bar():\n    pass\ndef baz():\n    pass\n")
        funcs, _, _ = _visit_file(tmp_path / "mod.py", "mod")
        assert len(funcs) == 3
        names = {f.qualified_name for f in funcs}
        assert names == {"mod:foo", "mod:bar", "mod:baz"}

    def test_extract_class_methods(self, tmp_path: Path) -> None:
        code = "class Foo:\n    def __init__(self):\n        pass\n    def run(self):\n        pass\n"
        _write_py(tmp_path, "mod.py", code)
        funcs, _, _ = _visit_file(tmp_path / "mod.py", "mod")
        assert len(funcs) == 2
        for f in funcs:
            assert f.is_method is True
            assert f.class_name == "Foo"
        names = {f.qualified_name for f in funcs}
        assert "mod:Foo.__init__" in names
        assert "mod:Foo.run" in names

    def test_intra_file_call_edges(self, tmp_path: Path) -> None:
        code = "def foo():\n    bar()\n    baz()\ndef bar():\n    pass\ndef baz():\n    pass\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "mod:bar" in graph.call_edges.get("mod:foo", set())
        assert "mod:baz" in graph.call_edges.get("mod:foo", set())

    def test_reverse_edges(self, tmp_path: Path) -> None:
        code = "def foo():\n    bar()\ndef bar():\n    pass\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "mod:foo" in graph.reverse_edges.get("mod:bar", set())

    def test_explicit_import_extraction(self, tmp_path: Path) -> None:
        code = "import os\nfrom pathlib import Path\n"
        _write_py(tmp_path, "mod.py", code)
        _, imports, _ = _visit_file(tmp_path / "mod.py", "mod")
        assert len(imports) == 2
        modules = {i.module for i in imports}
        assert "os" in modules
        assert "pathlib" in modules

    def test_blast_radius_linear_chain(self, tmp_path: Path) -> None:
        code = "def a():\n    b()\ndef b():\n    c()\ndef c():\n    pass\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        result = blast_radius(graph, {"mod:a"})
        assert "mod:a" in result
        assert "mod:b" in result
        assert "mod:c" in result

    def test_blast_radius_respects_max_depth(self, tmp_path: Path) -> None:
        # Chain: a -> b -> c -> d -> e
        code = "def a():\n    b()\ndef b():\n    c()\ndef c():\n    d()\ndef d():\n    e()\ndef e():\n    pass\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        result = blast_radius(graph, {"mod:a"}, max_depth=2)
        assert "mod:a" in result
        assert "mod:b" in result
        # Should stop before reaching the full chain
        assert len(result) < 5


# ===========================================================================
# Tier B — Cross-file, self.method
# ===========================================================================


class TestTierBCrossFile:
    def test_self_method_calls(self, tmp_path: Path) -> None:
        code = "class Foo:\n    def baz(self):\n        self.bar()\n    def bar(self):\n        pass\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "mod:Foo.bar" in graph.call_edges.get("mod:Foo.baz", set())

    def test_cross_file_import(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "utils.py", "def helper():\n    pass\n")
        _write_py(tmp_path, "main.py", "from utils import helper\ndef run():\n    helper()\n")
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        edges = graph.call_edges.get("main:run", set())
        assert any("helper" in e for e in edges)

    def test_relative_import_extraction(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "pkg/__init__.py", "")
        _write_py(tmp_path, "pkg/b.py", "def helper():\n    pass\n")
        _write_py(tmp_path, "pkg/a.py", "from .b import helper\ndef run():\n    helper()\n")
        _, imports, _ = _visit_file(tmp_path / "pkg" / "a.py", "pkg.a")
        assert any(i.is_relative for i in imports)

    def test_module_import_file_deps(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "utils.py", "def helper():\n    pass\n")
        _write_py(tmp_path, "main.py", "import utils\ndef run():\n    utils.helper()\n")
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        main_file = (tmp_path / "main.py").as_posix()
        assert any("utils" in dep for dep in graph.file_imports.get(main_file, []))

    def test_package_init_reexport_resolves_from_import(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "pkg/__init__.py", "from .utils import helper\n")
        _write_py(tmp_path, "pkg/utils.py", "def helper():\n    pass\n")
        _write_py(tmp_path, "main.py", "from pkg import helper\ndef run():\n    helper()\n")
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "pkg.utils:helper" in graph.call_edges.get("main:run", set())

    def test_package_init_reexport_resolves_module_attribute(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "pkg/__init__.py", "from .utils import helper\n")
        _write_py(tmp_path, "pkg/utils.py", "def helper():\n    pass\n")
        _write_py(tmp_path, "main.py", "import pkg\ndef run():\n    pkg.helper()\n")
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "pkg.utils:helper" in graph.call_edges.get("main:run", set())


# ===========================================================================
# Tier C — Uncertain calls
# ===========================================================================


class TestTierCUncertain:
    def test_chained_attribute_flagged_uncertain(self, tmp_path: Path) -> None:
        code = "def foo():\n    a.b.c()\n"
        _write_py(tmp_path, "mod.py", code)
        _, _, calls = _visit_file(tmp_path / "mod.py", "mod")
        uncertain = [c for c in calls if c.uncertain]
        assert len(uncertain) >= 1

    def test_dynamic_call_does_not_crash(self, tmp_path: Path) -> None:
        code = "def foo():\n    func_map = {}\n    func_map['key']()\n"
        _write_py(tmp_path, "mod.py", code)
        # Should not raise
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert isinstance(graph, CodebaseGraph)

    def test_syntax_error_file_skipped(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "good.py", "def ok():\n    pass\n")
        _write_py(tmp_path, "bad.py", "def broken(\n")
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "good:ok" in graph.functions
        assert not any("bad" in k for k in graph.functions)


# ===========================================================================
# Graph algorithms
# ===========================================================================


class TestGraphAlgorithms:
    def test_find_clusters_simple_cycle(self, tmp_path: Path) -> None:
        code = "def a():\n    b()\ndef b():\n    a()\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        clusters = find_clusters(graph)
        assert len(clusters) >= 1
        cycle = clusters[0]
        assert "mod:a" in cycle
        assert "mod:b" in cycle

    def test_find_clusters_no_cycle(self, tmp_path: Path) -> None:
        code = "def a():\n    b()\ndef b():\n    pass\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        clusters = find_clusters(graph)
        assert len(clusters) == 0

    def test_blast_radius_through_cycle(self, tmp_path: Path) -> None:
        code = "def a():\n    b()\ndef b():\n    c()\ndef c():\n    a()\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        result = blast_radius(graph, {"mod:a"})
        assert {"mod:a", "mod:b", "mod:c"} <= result

    def test_pagerank_prefers_shared_helper(self, tmp_path: Path) -> None:
        code = (
            "def helper():\n    pass\n"
            "def a():\n    helper()\n"
            "def b():\n    helper()\n"
            "def c():\n    helper()\n"
        )
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert graph.pagerank["mod:helper"] > graph.pagerank["mod:a"]
        assert graph.pagerank["mod:helper"] > graph.pagerank["mod:b"]
        assert graph.pagerank["mod:helper"] > graph.pagerank["mod:c"]


# ===========================================================================
# Cache
# ===========================================================================


class TestCache:
    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "mod.py", "def foo():\n    pass\n")
        g1 = build_codebase_graph(tmp_path, force_rebuild=True)
        g2 = build_codebase_graph(tmp_path)  # should hit cache
        assert g1.functions.keys() == g2.functions.keys()
        assert g1.call_edges == g2.call_edges

    def test_cache_invalidation_on_change(self, tmp_path: Path) -> None:
        p = _write_py(tmp_path, "mod.py", "def foo():\n    pass\n")
        g1 = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "mod:foo" in g1.functions
        # Modify file
        import time
        time.sleep(0.01)
        p.write_text("def bar():\n    pass\n", encoding="utf-8")
        # Force mtime to differ
        import os
        os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 1))
        g2 = build_codebase_graph(tmp_path)
        assert "mod:bar" in g2.functions

    def test_cache_json_schema(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "mod.py", "def foo():\n    pass\n")
        build_codebase_graph(tmp_path, force_rebuild=True)
        cache_dir = tmp_path / ".maestro-cache" / "codebase_graph"
        jsons = list(cache_dir.glob("*.json"))
        assert len(jsons) == 1
        data = json.loads(jsons[0].read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["parser_version"] == 3
        assert "python_version" in data
        assert "functions" in data
        assert "call_edges" in data
        assert "pagerank" in data

    def test_to_dict_from_dict_roundtrip(self) -> None:
        graph = CodebaseGraph(
            functions={"mod:foo": FunctionDef("mod:foo", "mod.py", 1, 3)},
            call_edges={"mod:foo": {"mod:bar"}},
            reverse_edges={"mod:bar": {"mod:foo"}},
            pagerank={"mod:foo": 0.4, "mod:bar": 0.6},
        )
        d = graph.to_dict()
        g2 = CodebaseGraph.from_dict(d)
        assert g2.functions.keys() == graph.functions.keys()
        assert g2.call_edges == graph.call_edges
        assert g2.pagerank == graph.pagerank


# ===========================================================================
# Integration
# ===========================================================================


class TestIntegration:
    def test_ast_context_with_workspace(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "auth.py", "def validate():\n    check()\ndef check():\n    pass\n")
        upstream = {"task-a": "Modified validate() in auth.py\n+ def validate():\n+     check()"}
        files_changed = {"task-a": ["auth.py"]}
        result = build_ast_structural_context(
            upstream, 500, files_changed, workspace_root=tmp_path,
        )
        assert result  # non-empty
        assert "task-a" in result

    def test_ast_context_no_workspace_falls_back(self) -> None:
        upstream = {"task-a": "def foo(): pass"}
        result = build_ast_structural_context(upstream, 500)
        # Should still produce output via regex fallback
        assert isinstance(result, str)

    def test_ast_context_empty_texts(self, tmp_path: Path) -> None:
        result = build_ast_structural_context({}, 500, workspace_root=tmp_path)
        assert result == ""

    def test_ast_context_zero_budget(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "mod.py", "def foo():\n    pass\n")
        result = build_ast_structural_context({"a": "text"}, 0, workspace_root=tmp_path)
        assert result == ""


# ===========================================================================
# Utilities
# ===========================================================================


class TestUtilities:
    def test_module_name_from_path(self, tmp_path: Path) -> None:
        assert _module_name_from_path("src/foo/bar.py", tmp_path) == "src.foo.bar"
        assert _module_name_from_path("mod.py", tmp_path) == "mod"
        assert _module_name_from_path("pkg/__init__.py", tmp_path) == "pkg"

    def test_collect_py_files_ignores_dirs(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "good.py", "x = 1")
        _write_py(tmp_path, "__pycache__/bad.py", "x = 1")
        _write_py(tmp_path, "node_modules/bad.py", "x = 1")
        files = _collect_py_files(tmp_path)
        paths = {f[1] for f in files}
        assert "good.py" in paths
        assert not any("__pycache__" in p for p in paths)
        assert not any("node_modules" in p for p in paths)

    def test_files_snapshot_id_deterministic(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "a.py", "x = 1")
        files = _collect_py_files(tmp_path)
        s1 = _files_snapshot_id(files)
        s2 = _files_snapshot_id(files)
        assert s1 == s2
        assert len(s1) == 32

    def test_split_text_chunks(self) -> None:
        text = "line1\nline2\nline3\n" * 20
        chunks = _split_text_chunks(text)
        assert len(chunks) >= 2
        reassembled = "\n".join(chunks)
        assert "line1" in reassembled

    def test_score_chunk_with_graph_changed_symbols(self) -> None:
        graph = CodebaseGraph()
        score = _score_chunk_with_graph("validate token check", {"mod:validate"}, {"mod:validate"}, graph)
        assert score >= 1.5

    def test_score_chunk_with_graph_blast_only(self) -> None:
        graph = CodebaseGraph()
        score = _score_chunk_with_graph("check something", {"mod:check"}, set(), graph)
        assert score >= 1.0

    def test_score_chunk_with_graph_uses_pagerank_bonus(self) -> None:
        graph = CodebaseGraph(
            pagerank={
                "mod:core": 0.8,
                "mod:leaf": 0.2,
            }
        )
        core_score = _score_chunk_with_graph("core path hot spot", {"mod:core"}, set(), graph)
        leaf_score = _score_chunk_with_graph("leaf path helper", {"mod:leaf"}, set(), graph)
        assert core_score > leaf_score

    def test_empty_workspace(self, tmp_path: Path) -> None:
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert len(graph.functions) == 0
        assert len(graph.call_edges) == 0

    def test_extract_call_name_simple(self) -> None:
        import ast as _ast
        node = _ast.parse("foo()").body[0].value  # type: ignore[attr-defined]
        name, uncertain = _extract_call_name(node)
        assert name == "foo"
        assert uncertain is False

    def test_extract_call_name_attribute(self) -> None:
        import ast as _ast
        node = _ast.parse("self.bar()").body[0].value  # type: ignore[attr-defined]
        name, uncertain = _extract_call_name(node)
        assert name == "self.bar"
        assert uncertain is False
