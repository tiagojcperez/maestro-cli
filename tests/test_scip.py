from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.scip import (
    ScipIndex,
    ScipSymbol,
    _symbol_name,
    build_scip_context,
    format_scip_map,
    load_scip_index,
    parse_scip_index,
    resolve_scip_path,
)


def _index_payload() -> dict[str, object]:
    return {
        "metadata": {"tool_info": {"name": "scip-python"}},
        "documents": [
            {
                "relative_path": "src/auth/login.py",
                "symbols": [
                    {
                        "symbol": "scip-python python maestro 1.0 `auth.login`/authenticate().",
                        "display_name": "authenticate",
                        "documentation": ["Authenticate a user with password hashing."],
                    },
                    {
                        "symbol": "scip-python python maestro 1.0 `auth.login`/RateLimiter#",
                        "display_name": "RateLimiter",
                        "documentation": ["Token-bucket rate limiter for login."],
                    },
                ],
            },
            {
                "relative_path": "src/utils/io.py",
                "symbols": [
                    {
                        "symbol": "scip-python python maestro 1.0 `utils.io`/read_file().",
                        "documentation": ["Read a file from disk."],
                    }
                ],
            },
        ],
    }


def _write_index(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    (tmp_path / "index.scip.json").write_text(
        json.dumps(payload if payload is not None else _index_payload()),
        encoding="utf-8",
    )
    return tmp_path


class TestSymbolName:
    def test_prefers_display_name(self) -> None:
        assert _symbol_name("scip-python python . . `x`/Y#", "Widget") == "Widget"

    def test_parses_descriptor_when_no_display_name(self) -> None:
        name = _symbol_name("scip-python python maestro 1.0 `pkg`/Bar#method().", "")
        assert name == "Bar.method"

    def test_local_symbol(self) -> None:
        assert _symbol_name("local 0", "") == "local 0"

    def test_empty_symbol(self) -> None:
        assert _symbol_name("", "") == "?"


class TestParseScipIndex:
    def test_extracts_symbols(self) -> None:
        index = parse_scip_index(_index_payload())
        assert index.tool == "scip-python"
        assert index.document_count == 2
        names = {s.name for s in index.symbols}
        # authenticate/RateLimiter use display_name; read_file has none, so it is
        # parsed from the descriptor as module.func ("io.read_file").
        assert names == {"authenticate", "RateLimiter", "io.read_file"}
        auth = next(s for s in index.symbols if s.name == "authenticate")
        assert auth.file == "src/auth/login.py"
        assert "password" in auth.documentation

    def test_camelcase_fields(self) -> None:
        payload = {
            "metadata": {"toolInfo": {"name": "scip-typescript"}},
            "documents": [
                {
                    "relativePath": "src/app.ts",
                    "symbols": [
                        {"symbol": "scip-ts ts app 1 `app`/main().", "displayName": "main"}
                    ],
                }
            ],
        }
        index = parse_scip_index(payload)
        assert index.tool == "scip-typescript"
        assert index.symbols[0].name == "main"
        assert index.symbols[0].file == "src/app.ts"

    def test_occurrence_fallback_for_definitions(self) -> None:
        payload = {
            "documents": [
                {
                    "relative_path": "lib/x.go",
                    "occurrences": [
                        {"symbol": "scip-go go . . `x`/Run().", "symbol_roles": 1},
                        {"symbol": "scip-go go . . `x`/helper().", "symbol_roles": 8},
                    ],
                }
            ]
        }
        index = parse_scip_index(payload)
        # Only the definition (role bit 1) is collected, not the role-8 reference.
        assert [s.file for s in index.symbols] == ["lib/x.go"]
        assert len(index.symbols) == 1

    def test_deduplicates_symbols(self) -> None:
        payload = {
            "documents": [
                {
                    "relative_path": "a.py",
                    "symbols": [
                        {"symbol": "dup", "display_name": "Dup"},
                        {"symbol": "dup", "display_name": "Dup"},
                    ],
                }
            ]
        }
        assert len(parse_scip_index(payload).symbols) == 1

    def test_malformed_is_tolerant(self) -> None:
        assert parse_scip_index({}).symbols == []
        assert parse_scip_index({"documents": "nope"}).symbols == []
        assert parse_scip_index({"documents": [None, 5]}).symbols == []


class TestLoadScipIndex:
    def test_resolve_and_load(self, tmp_path: Path) -> None:
        _write_index(tmp_path)
        assert resolve_scip_path(tmp_path) is not None
        index = load_scip_index(tmp_path)
        assert index is not None
        assert len(index.symbols) == 3

    def test_none_when_absent(self, tmp_path: Path) -> None:
        assert resolve_scip_path(tmp_path) is None
        assert load_scip_index(tmp_path) is None

    def test_none_when_workspace_root_empty(self) -> None:
        assert resolve_scip_path(None) is None
        assert load_scip_index(None) is None

    def test_tolerant_of_bad_json(self, tmp_path: Path) -> None:
        (tmp_path / "index.scip.json").write_text("{not json", encoding="utf-8")
        assert load_scip_index(tmp_path) is None

    def test_none_when_no_symbols(self, tmp_path: Path) -> None:
        (tmp_path / "index.scip.json").write_text(
            json.dumps({"documents": []}), encoding="utf-8"
        )
        assert load_scip_index(tmp_path) is None


class TestFormatScipMap:
    def test_relevance_filters_and_ranks(self) -> None:
        index = parse_scip_index(_index_payload())
        out = format_scip_map(index, "authentication login flow", 6000)
        assert "authenticate" in out
        assert "RateLimiter" in out
        assert "read_file" not in out  # irrelevant to the query
        assert "scip-python" in out

    def test_overview_fallback_when_no_match(self) -> None:
        index = parse_scip_index(_index_payload())
        out = format_scip_map(index, "zzz_nonexistent_term", 6000)
        # No query match → bounded overview (still produces a map).
        assert "authenticate" in out
        assert "read_file" in out

    def test_empty_on_zero_budget(self) -> None:
        index = parse_scip_index(_index_payload())
        assert format_scip_map(index, "auth", 0) == ""

    def test_empty_on_no_symbols(self) -> None:
        assert format_scip_map(ScipIndex(), "auth", 6000) == ""

    def test_budget_limits_output(self) -> None:
        symbols = [
            ScipSymbol(name=f"sym{n}", file=f"f{n}.py", documentation="x" * 50)
            for n in range(100)
        ]
        index = ScipIndex(symbols=symbols)
        out = format_scip_map(index, "sym1 sym2 sym3", 200)
        assert len(out) <= 400  # header + a couple of blocks, not all 100


class TestBuildScipContext:
    def test_end_to_end(self, tmp_path: Path) -> None:
        _write_index(tmp_path)
        out = build_scip_context(str(tmp_path), "authentication and login", 6000)
        assert "authenticate" in out

    def test_empty_without_index(self, tmp_path: Path) -> None:
        assert build_scip_context(str(tmp_path), "auth", 6000) == ""

    def test_empty_on_zero_budget(self, tmp_path: Path) -> None:
        _write_index(tmp_path)
        assert build_scip_context(str(tmp_path), "auth", 0) == ""


class TestScipContextModeWiring:
    def test_loader_accepts_scip_with_workspace_root(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            f"""
version: 1
name: scip-plan
workspace_root: {tmp_path.as_posix()}
tasks:
  - id: t1
    engine: claude
    model: sonnet
    context_mode: scip
    prompt: "Use the codebase map."
""",
            encoding="utf-8",
        )
        plan = load_plan(str(plan_path))
        assert plan.tasks[0].context_mode == "scip"

    def test_loader_rejects_scip_without_workspace_root(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            """
version: 1
name: scip-plan
tasks:
  - id: t1
    engine: claude
    model: sonnet
    context_mode: scip
    prompt: "Use the codebase map."
""",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match="E021"):
            load_plan(str(plan_path))
