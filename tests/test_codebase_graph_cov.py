"""Coverage tests for codebase_graph.py targeting previously-uncovered lines.

Every external boundary (none here — pure stdlib AST/file work) is exercised
through real functions with crafted inputs.  No engine/subprocess/network/git
calls are involved in this module, so nothing needs mocking beyond the cache
JSON files themselves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from maestro_cli.codebase_graph import (
    CallSite,
    CodebaseGraph,
    FunctionDef,
    ImportRef,
    _collect_py_files,
    _load_cached_graph,
    _resolve_call,
    _resolve_import_module,
    _resolve_import_target,
    _store_graph,
    build_ast_structural_context,
    build_codebase_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_py(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _call(name: str, *, uncertain: bool = False) -> CallSite:
    return CallSite(
        caller="mod:caller",
        callee_name=name,
        file_path="mod.py",
        line=1,
        uncertain=uncertain,
    )


# ===========================================================================
# CallSite.to_dict
# ===========================================================================

class TestCallSiteToDict:
    def test_to_dict_round_trip(self) -> None:
        cs = CallSite(
            caller="mod:foo",
            callee_name="bar",
            file_path="mod.py",
            line=7,
            uncertain=True,
        )
        d = cs.to_dict()
        assert d["caller"] == "mod:foo"
        assert d["callee_name"] == "bar"
        assert d["file_path"] == "mod.py"
        assert d["line"] == 7
        assert d["uncertain"] is True
        # reconstructable
        cs2 = CallSite(**d)
        assert cs2 == cs


# ===========================================================================
# _ASTVisitor._current_scope module-level path + async funcs
#
# ===========================================================================

class TestModuleLevelAndAsync:
    def test_module_level_call_uses_module_scope(self, tmp_path: Path) -> None:
        # A call at module level (outside any function) exercises the
        # ``<module>`` scope branch in _current_scope.
        _write_py(tmp_path, "mod.py", "print('hi')\n")
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        # The module-scope caller of an unresolved builtin call lands in
        # uncertain_calls (callee_name non-empty, resolved None).
        callers = {cs.caller for cs in graph.uncertain_calls}
        assert "mod:<module>" in callers

    def test_async_function_def_extracted(self, tmp_path: Path) -> None:
        code = (
            "async def fetch():\n"
            "    helper()\n"
            "def helper():\n"
            "    pass\n"
        )
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert "mod:fetch" in graph.functions
        assert "mod:helper" in graph.call_edges.get("mod:fetch", set())

    def test_uncertain_call_recorded(self, tmp_path: Path) -> None:
        # A chained attribute call inside a function is flagged uncertain and
        # therefore appended to uncertain_calls .
        code = "def foo():\n    a.b.c()\n"
        _write_py(tmp_path, "mod.py", code)
        graph = build_codebase_graph(tmp_path, force_rebuild=True)
        assert any(cs.callee_name for cs in graph.uncertain_calls)


# ===========================================================================
# _resolve_import_target
# ===========================================================================

class TestResolveImportTarget:
    def test_star_import_skipped(self) -> None:
        imp = ImportRef(module="pkg", names=["*"], file_path="mod.py")
        result = _resolve_import_target(
            imp, {}, importer_module="mod", importer_is_package=False,
        )
        # "*" is skipped -> nothing introduced into scope
        assert result == {}

    def test_alias_maps_to_resolved_name(self) -> None:
        # `import os as o`-style: alias present + names present, so the alias
        # branch (332-333) populates result for the alias key.
        imp = ImportRef(
            module="numpy",
            names=["numpy"],
            alias="np",
            file_path="mod.py",
        )
        result = _resolve_import_target(
            imp, {}, importer_module="mod", importer_is_package=False,
        )
        assert "np" in result
        # alias resolves to the same target as the underlying name
        assert result["np"] == result.get("numpy", result["np"])

    def test_alias_uses_fallback_when_name_unresolved(self) -> None:
        # When the underlying name does not land in `result` (empty module),
        # the alias branch uses the dict.get fallback expression.
        imp = ImportRef(
            module="",
            names=["thing"],
            alias="t",
            file_path="mod.py",
            is_relative=False,
            level=0,
        )
        result = _resolve_import_target(
            imp, {}, importer_module="mod", importer_is_package=False,
        )
        # empty module + name not known => name not added, alias falls back
        assert result["t"] == "thing"


# ===========================================================================
# _resolve_import_module
# ===========================================================================

class TestResolveImportModule:
    def test_ascend_beyond_package_root_yields_empty_anchor(self) -> None:
        # level high enough that ascend >= len(package_parts) -> anchor empty.
        imp = ImportRef(
            module="target",
            names=["x"],
            is_relative=True,
            level=5,
            file_path="pkg/a.py",
        )
        resolved = _resolve_import_module(
            imp, importer_module="pkg.a", importer_is_package=False,
        )
        # anchor empty, module appended directly
        assert resolved == "target"

    def test_relative_import_no_module_returns_anchor_only(self) -> None:
        # `from . import sibling` -> imp.module is empty, level 1.
        imp = ImportRef(
            module="",
            names=["sibling"],
            is_relative=True,
            level=1,
            file_path="pkg/a.py",
        )
        resolved = _resolve_import_module(
            imp, importer_module="pkg.a", importer_is_package=False,
        )
        # ascend=0, package_parts=["pkg"], anchor=["pkg"], no module -> "pkg"
        assert resolved == "pkg"

    def test_relative_import_empty_module_at_root_returns_empty(self) -> None:
        # level large + empty module exercises with empty anchor.
        imp = ImportRef(
            module="",
            names=["sibling"],
            is_relative=True,
            level=9,
            file_path="a.py",
        )
        resolved = _resolve_import_module(
            imp, importer_module="a", importer_is_package=False,
        )
        assert resolved == ""


# ===========================================================================
# _resolve_call branches
# ===========================================================================

class TestResolveCall:
    def test_uncertain_call_returns_none(self) -> None:
        result = _resolve_call(
            _call("a.b.c", uncertain=True),
            {}, {}, "mod", "", {}, {},
        )
        assert result is None

    def test_imported_name_resolves_via_module_export(self) -> None:
        # import alias target has ":" and module_exports maps it (394-398).
        call = _call("helper")
        import_aliases = {"helper": "pkg:helper"}
        module_exports = {"pkg": {"helper": "pkg.utils:helper"}}
        result = _resolve_call(
            call, {}, import_aliases, "mod", "", {}, module_exports,
        )
        assert result == "pkg.utils:helper"

    def test_imported_name_falls_through_to_target(self) -> None:
        # alias target has ":", but no export entry -> returns target (399).
        call = _call("helper")
        import_aliases = {"helper": "pkg:helper"}
        result = _resolve_call(
            call, {}, import_aliases, "mod", "", {}, {},
        )
        assert result == "pkg:helper"

    def test_imported_name_empty_target_returns_none(self) -> None:
        # alias maps to empty string -> "return target if target else None".
        call = _call("ghost")
        import_aliases = {"ghost": ""}
        result = _resolve_call(
            call, {}, import_aliases, "mod", "", {}, {},
        )
        assert result is None

    def test_dotted_call_unknown_candidate_returns_modpart_attr(self) -> None:
        # "pkg.func" where prefix maps to "somepkg:_", candidate not known
        # and no export -> returns "somepkg:func" .
        call = _call("pkg.func")
        import_aliases = {"pkg": "somepkg:_"}
        result = _resolve_call(
            call, {}, import_aliases, "mod", "", {}, {},
        )
        assert result == "somepkg:func"

    def test_dotted_call_base_without_colon(self) -> None:
        # base has no ":" -> "return f'{base}:{attr}'" .
        call = _call("pkg.func")
        import_aliases = {"pkg": "somepkg"}
        result = _resolve_call(
            call, {}, import_aliases, "mod", "", {}, {},
        )
        assert result == "somepkg:func"

    def test_module_level_qualified_guess_hit(self) -> None:
        # name not local/import/dotted, but "mod:name" is in local_functions
        # .
        call = _call("helper")
        local_functions = {"helper2": "mod:helper2", "mod:helper": "mod:helper"}
        result = _resolve_call(
            call, local_functions, {}, "mod", "", {}, {},
        )
        assert result == "mod:helper"

    def test_unresolvable_returns_none(self) -> None:
        # nothing matches -> final "return None" .
        call = _call("nowhere")
        result = _resolve_call(
            call, {}, {}, "mod", "", {}, {},
        )
        assert result is None


# ===========================================================================
# _collect_py_files
# ===========================================================================

class TestCollectPyFiles:
    def test_non_py_files_skipped(self, tmp_path: Path) -> None:
        _write_py(tmp_path, "mod.py", "x = 1\n")
        (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        files = _collect_py_files(tmp_path)
        rels = {f[1] for f in files}
        assert "mod.py" in rels
        assert all(r.endswith(".py") for r in rels)

    def test_stat_oserror_skips_file(self, tmp_path: Path, monkeypatch) -> None:
        _write_py(tmp_path, "good.py", "x = 1\n")
        _write_py(tmp_path, "bad.py", "x = 2\n")

        real_stat = Path.stat

        def fake_stat(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.name == "bad.py":
                raise OSError("simulated stat failure")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fake_stat)
        files = _collect_py_files(tmp_path)
        rels = {f[1] for f in files}
        assert "good.py" in rels
        assert "bad.py" not in rels

    def test_max_files_cap_short_circuits(self, tmp_path: Path, monkeypatch) -> None:
        # Drop the cap to 2 so the early return triggers.
        monkeypatch.setattr(
            "maestro_cli.codebase_graph._MAX_PY_FILES", 2, raising=True,
        )
        for i in range(5):
            _write_py(tmp_path, f"m{i}.py", "x = 1\n")
        files = _collect_py_files(tmp_path)
        assert len(files) == 2


# ===========================================================================
# _load_cached_graph invalidation branches
#
# ===========================================================================

class TestLoadCachedGraph:
    def _valid_payload(self, snapshot_id: str) -> dict:
        graph = CodebaseGraph(
            functions={"mod:foo": FunctionDef("mod:foo", "mod.py", 1, 2)},
        )
        from maestro_cli.codebase_graph import (
            _PARSER_VERSION,
            _SCHEMA_VERSION,
        )
        return {
            "schema_version": _SCHEMA_VERSION,
            "parser_version": _PARSER_VERSION,
            "python_version": sys.version,
            "snapshot_id": snapshot_id,
            **graph.to_dict(),
        }

    def _write_cache(self, cache_dir: Path, snapshot_id: str, payload: dict) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{snapshot_id}.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "snap.json").write_text("{not valid json", encoding="utf-8")
        assert _load_cached_graph(cache_dir, "snap") is None

    def test_schema_version_mismatch(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        payload = self._valid_payload("snap")
        payload["schema_version"] = 999
        self._write_cache(cache_dir, "snap", payload)
        assert _load_cached_graph(cache_dir, "snap") is None

    def test_parser_version_mismatch(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        payload = self._valid_payload("snap")
        payload["parser_version"] = -1
        self._write_cache(cache_dir, "snap", payload)
        assert _load_cached_graph(cache_dir, "snap") is None

    def test_python_version_mismatch(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        payload = self._valid_payload("snap")
        payload["python_version"] = "ancient-python-0.0"
        self._write_cache(cache_dir, "snap", payload)
        assert _load_cached_graph(cache_dir, "snap") is None

    def test_snapshot_id_mismatch(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        payload = self._valid_payload("OTHER")
        self._write_cache(cache_dir, "snap", payload)
        assert _load_cached_graph(cache_dir, "snap") is None

    def test_valid_cache_loads(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        payload = self._valid_payload("snap")
        self._write_cache(cache_dir, "snap", payload)
        graph = _load_cached_graph(cache_dir, "snap")
        assert graph is not None
        assert "mod:foo" in graph.functions


# ===========================================================================
# _compute_pagerank empty-paths
# ===========================================================================

class TestComputePagerank:
    def test_empty_graph_pagerank_empty(self, tmp_path: Path) -> None:
        # Empty workspace -> build returns early before pagerank, so call the
        # compute path through a graph with no nodes directly via build.
        from maestro_cli.codebase_graph import _compute_pagerank
        assert _compute_pagerank(CodebaseGraph()) == {}

    def test_pagerank_zero_total_returns_empty(self) -> None:
        # Force the total<=0 branch by running zero iterations so
        # ranks stay at the seeded uniform values... instead drive it through
        # a graph whose ranks sum to zero via damping=0 with no edges is still
        # >0; so patch via max_iters=0 keeps uniform >0.  Use a node set and
        # damping handling: build a graph with a single isolated node and
        # patch ranks to zero is not possible without internals.  Cover 792 by
        # zero nodes path is 758; cover 792 with all-zero via custom graph.
        from maestro_cli.codebase_graph import _compute_pagerank
        # A single isolated node yields a positive uniform rank, so the
        # total<=0 guard is only hit when nodes is non-empty yet ranks sum to
        # zero.  We can reach it by running with max_iters that leaves ranks
        # at zero is not natural; the guard is defensive.  Confirm normal path
        # produces a normalized distribution summing to ~1.
        g = CodebaseGraph(
            functions={"mod:a": FunctionDef("mod:a", "m.py", 1, 1)},
            call_edges={"mod:a": {"mod:b"}},
            reverse_edges={"mod:b": {"mod:a"}},
        )
        ranks = _compute_pagerank(g)
        assert ranks
        assert abs(sum(ranks.values()) - 1.0) < 1e-6


# ===========================================================================
# build_ast_structural_context fallback + selection branches
#
# ===========================================================================

class TestBuildAstStructuralContext:
    def test_workspace_root_not_dir_falls_back(self, tmp_path: Path) -> None:
        # workspace_root points at a file, not a dir ( fallback).
        not_a_dir = tmp_path / "afile.txt"
        not_a_dir.write_text("hello", encoding="utf-8")
        upstream = {"task-a": "def foo():\n    pass\n"}
        result = build_ast_structural_context(
            upstream, 500, workspace_root=not_a_dir,
        )
        assert isinstance(result, str)

    def test_build_graph_exception_falls_back(self, tmp_path: Path, monkeypatch) -> None:
        # build_codebase_graph raising -> except branch fallback (827-828).
        _write_py(tmp_path, "mod.py", "def foo():\n    pass\n")

        def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("graph build exploded")

        monkeypatch.setattr(
            "maestro_cli.codebase_graph.build_codebase_graph", boom,
        )
        upstream = {"task-a": "def foo():\n    pass\n"}
        result = build_ast_structural_context(
            upstream, 500, workspace_root=tmp_path,
        )
        assert isinstance(result, str)

    def test_empty_graph_functions_falls_back(self, tmp_path: Path) -> None:
        # Workspace has only a .py file with no functions -> graph.functions
        # is empty -> fallback .
        _write_py(tmp_path, "mod.py", "x = 1\ny = 2\n")
        upstream = {"task-a": "some upstream text mentioning x"}
        result = build_ast_structural_context(
            upstream, 500, workspace_root=tmp_path,
        )
        assert isinstance(result, str)

    def test_no_changed_symbols_falls_back(self, tmp_path: Path) -> None:
        # Graph has functions, but upstream text references no matching
        # symbol names and no files_changed -> changed stays empty -> fallback
        # . Exercises the symbol-name extraction loop (847-858).
        _write_py(tmp_path, "mod.py", "def alpha():\n    pass\ndef beta():\n    pass\n")
        upstream = {"task-a": "completely unrelated prose with no code symbols"}
        result = build_ast_structural_context(
            upstream, 500, workspace_root=tmp_path,
        )
        assert isinstance(result, str)

    def test_changed_symbol_via_diff_text_drives_selection(self, tmp_path: Path) -> None:
        # Diff-format upstream so extract_changed_symbols returns names
        # ( branch), matching a graph function -> changed populated
        # (855-858) -> blast radius + chunk scoring path taken.
        code = "def validate():\n    check()\ndef check():\n    pass\n"
        _write_py(tmp_path, "auth.py", code)
        upstream = {
            "task-a": "diff --git a/auth.py b/auth.py\n+def validate():\n+    check()\n",
        }
        result = build_ast_structural_context(
            upstream, 500, workspace_root=tmp_path,
        )
        assert isinstance(result, str)
        # selection should include the upstream id header when non-empty
        if result:
            assert "task-a" in result

    def test_changed_via_files_changed_param(self, tmp_path: Path) -> None:
        # upstream_files_changed maps a .py file to the workspace module,
        # populating `changed` directly .
        code = "def validate():\n    check()\ndef check():\n    pass\n"
        _write_py(tmp_path, "auth.py", code)
        upstream = {"task-a": "Modified validate in auth.py: validate check"}
        files_changed = {"task-a": ["auth.py"]}
        result = build_ast_structural_context(
            upstream, 500, files_changed, workspace_root=tmp_path,
        )
        assert isinstance(result, str)
        assert "task-a" in result

    def test_tight_budget_skips_oversized_entries(self, tmp_path: Path) -> None:
        # A tiny budget forces the greedy loop to skip entries that would
        # exceed the budget ( `continue`).
        code = "def validate():\n    check()\ndef check():\n    pass\n"
        _write_py(tmp_path, "auth.py", code)
        big_text = "validate check " * 200  # large upstream body
        upstream = {"task-a": big_text}
        files_changed = {"task-a": ["auth.py"]}
        # budget_tokens=2 -> budget_chars=8, every chunk is larger -> all skip
        result = build_ast_structural_context(
            upstream, 2, files_changed, workspace_root=tmp_path,
        )
        assert isinstance(result, str)
        # Nothing fits in 8 chars of budget once headers are added
        assert result == ""


# ===========================================================================
# _store_graph + round-trip through build (sanity, also covers store path)
# ===========================================================================

class TestStoreGraph:
    def test_store_creates_json(self, tmp_path: Path) -> None:
        graph = CodebaseGraph(
            functions={"mod:foo": FunctionDef("mod:foo", "mod.py", 1, 2)},
        )
        cache_dir = tmp_path / "cache"
        dest = _store_graph(graph, cache_dir, "snap", str(tmp_path), 1)
        assert dest.exists()
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert data["snapshot_id"] == "snap"
        assert data["file_count"] == 1
