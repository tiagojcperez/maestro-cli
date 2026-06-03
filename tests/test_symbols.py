from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.symbols import (
    Symbol,
    build_structural_context,
    detect_language_from_path,
    detect_language_from_text,
    extract_changed_symbols,
    extract_symbols,
)


# ===========================================================================
# TestDetectLanguage
# ===========================================================================


class TestDetectLanguageFromPath:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("src/auth.py", "python"),
            ("app.js", "javascript"),
            ("index.tsx", "typescript"),
            ("main.go", "go"),
            ("lib.rs", "rust"),
            ("handler.php", "php"),
            ("Main.java", "java"),
            ("script.rb", "ruby"),
            ("module.c", "c"),
            ("module.cpp", "cpp"),
            ("unknown.xyz", None),
        ],
    )
    def test_extension_detection(self, path: str, expected: str | None) -> None:
        assert detect_language_from_path(path) == expected


class TestDetectLanguageFromText:
    def test_diff_header(self) -> None:
        text = "diff --git a/src/auth.py b/src/auth.py\n+def foo():\n"
        assert detect_language_from_text(text) == "python"

    def test_code_fence(self) -> None:
        text = "```python\ndef foo():\n    pass\n```"
        assert detect_language_from_text(text) == "python"

    def test_code_fence_go(self) -> None:
        text = "```go\nfunc main() {\n}\n```"
        assert detect_language_from_text(text) == "go"

    def test_shebang(self) -> None:
        text = "#!/usr/bin/env python3\ndef main():\n    pass"
        assert detect_language_from_text(text) == "python"

    def test_keyword_density_python(self) -> None:
        text = "def foo():\n    self.bar()\n    import os\nclass Baz:\n    pass"
        assert detect_language_from_text(text) == "python"

    def test_keyword_density_go(self) -> None:
        text = 'func main() {\n    package main\n    import (\n        "fmt"\n    )\n    type Foo struct {}'
        assert detect_language_from_text(text) == "go"

    def test_no_detection(self) -> None:
        text = "hello world this is just prose"
        assert detect_language_from_text(text) is None


# ===========================================================================
# TestExtractSymbols
# ===========================================================================


class TestExtractSymbolsPython:
    def test_function(self) -> None:
        text = "def validate_token(token: str) -> bool:\n    return True"
        syms = extract_symbols(text, "python")
        assert any(s.name == "validate_token" and s.kind == "function" for s in syms)

    def test_class(self) -> None:
        text = "class AuthService:\n    pass"
        syms = extract_symbols(text, "python")
        assert any(s.name == "AuthService" and s.kind == "class" for s in syms)

    def test_import(self) -> None:
        text = "import os\nfrom pathlib import Path"
        syms = extract_symbols(text, "python")
        imports = [s for s in syms if s.kind == "import"]
        assert len(imports) == 2

    def test_method_inside_class(self) -> None:
        text = "class Foo:\n    def bar(self):\n        pass"
        syms = extract_symbols(text, "python")
        assert any(s.name == "bar" and s.kind == "function" for s in syms)

    def test_auto_detect_language(self) -> None:
        text = "def foo():\n    self.bar()\n    import os\nclass Baz:\n    pass"
        syms = extract_symbols(text)  # No language specified
        assert len(syms) >= 2


class TestExtractSymbolsJavaScript:
    def test_function(self) -> None:
        text = "function handleRequest(req, res) {\n}"
        syms = extract_symbols(text, "javascript")
        assert any(s.name == "handleRequest" and s.kind == "function" for s in syms)

    def test_const_function(self) -> None:
        text = "const validate = function(data) {\n}"
        syms = extract_symbols(text, "javascript")
        assert any(s.name == "validate" and s.kind == "function" for s in syms)

    def test_class(self) -> None:
        text = "class UserController {\n}"
        syms = extract_symbols(text, "javascript")
        assert any(s.name == "UserController" and s.kind == "class" for s in syms)

    def test_export_class(self) -> None:
        text = "export class ApiService {\n}"
        syms = extract_symbols(text, "javascript")
        assert any(s.name == "ApiService" and s.kind == "class" for s in syms)


class TestExtractSymbolsGo:
    def test_function(self) -> None:
        text = "func handleRequest(w http.ResponseWriter, r *http.Request) {\n}"
        syms = extract_symbols(text, "go")
        assert any(s.name == "handleRequest" and s.kind == "function" for s in syms)

    def test_method_receiver(self) -> None:
        text = "func (s *Server) Start() error {\n}"
        syms = extract_symbols(text, "go")
        assert any(s.name == "Start" and s.kind == "function" for s in syms)

    def test_type_struct(self) -> None:
        text = "type Config struct {\n    Port int\n}"
        syms = extract_symbols(text, "go")
        assert any(s.name == "Config" and s.kind == "type" for s in syms)


class TestExtractSymbolsRust:
    def test_function(self) -> None:
        text = "fn process_event(event: &Event) -> Result<()> {\n}"
        syms = extract_symbols(text, "rust")
        assert any(s.name == "process_event" and s.kind == "function" for s in syms)

    def test_pub_function(self) -> None:
        text = "pub fn new() -> Self {\n}"
        syms = extract_symbols(text, "rust")
        assert any(s.name == "new" and s.kind == "function" for s in syms)

    def test_struct(self) -> None:
        text = "pub struct AppState {\n    db: Pool,\n}"
        syms = extract_symbols(text, "rust")
        assert any(s.name == "AppState" and s.kind == "type" for s in syms)

    def test_impl(self) -> None:
        text = "impl AppState {\n    fn connect(&self) {}\n}"
        syms = extract_symbols(text, "rust")
        assert any(s.name == "AppState" and s.kind == "class" for s in syms)


class TestExtractSymbolsPHP:
    def test_function(self) -> None:
        text = "function validateInput($data) {\n}"
        syms = extract_symbols(text, "php")
        assert any(s.name == "validateInput" and s.kind == "function" for s in syms)

    def test_class(self) -> None:
        text = "class UserRepository {\n}"
        syms = extract_symbols(text, "php")
        assert any(s.name == "UserRepository" and s.kind == "class" for s in syms)


class TestExtractSymbolsEmpty:
    def test_empty_text(self) -> None:
        assert extract_symbols("") == []

    def test_unknown_language(self) -> None:
        assert extract_symbols("some code", "brainfuck") == []

    def test_no_symbols_found(self) -> None:
        assert extract_symbols("just plain text\nno code here", "python") == []


# ===========================================================================
# TestExtractChangedSymbols
# ===========================================================================


class TestExtractChangedSymbols:
    def test_diff_with_added_function(self) -> None:
        diff = (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -10,0 +11,3 @@\n"
            "+def validate_token(token: str) -> bool:\n"
            "+    return check(token)\n"
        )
        syms = extract_changed_symbols(diff)
        assert any(s.name == "validate_token" for s in syms)
        assert syms[0].file_path == "src/auth.py"

    def test_diff_ignores_removed_lines(self) -> None:
        diff = (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "-def old_function():\n"
            "+def new_function():\n"
        )
        syms = extract_changed_symbols(diff)
        names = {s.name for s in syms}
        assert "new_function" in names
        assert "old_function" not in names

    def test_diff_multi_file(self) -> None:
        diff = (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "+def validate():\n"
            "diff --git a/src/api.js b/src/api.js\n"
            "+function fetchData() {\n"
        )
        syms = extract_changed_symbols(diff)
        names = {s.name for s in syms}
        assert "validate" in names
        assert "fetchData" in names

    def test_empty_diff(self) -> None:
        assert extract_changed_symbols("") == []


# ===========================================================================
# TestBuildStructuralContext
# ===========================================================================


class TestBuildStructuralContext:
    def test_basic_symbol_filtering(self) -> None:
        upstream = {
            "implement": (
                "diff --git a/src/auth.py b/src/auth.py\n"
                "+def validate_token(token):\n"
                "+    return True\n"
                "\n"
                "The validate_token function checks auth.\n"
                "Other unrelated text that should score lower.\n"
            ),
        }
        result = build_structural_context(upstream, 2000)
        assert "validate_token" in result

    def test_budget_respected(self) -> None:
        upstream = {
            "impl": "def foo():\n    pass\n" * 100,
        }
        result = build_structural_context(upstream, 50)  # Very tight budget
        assert len(result) <= 50 * 4 + 100  # budget_chars + some header overhead

    def test_empty_upstream(self) -> None:
        assert build_structural_context({}, 1000) == ""

    def test_zero_budget(self) -> None:
        assert build_structural_context({"a": "def foo(): pass"}, 0) == ""

    def test_fallback_when_no_symbols(self) -> None:
        upstream = {"task": "just plain prose text without any code"}
        result = build_structural_context(upstream, 1000)
        # Falls back to simple truncation
        assert "task" in result or "prose" in result

    def test_multiple_upstreams(self) -> None:
        upstream = {
            "task-a": "diff --git a/a.py b/a.py\n+def alpha():\n",
            "task-b": "diff --git a/b.py b/b.py\n+def beta():\n",
        }
        result = build_structural_context(upstream, 5000)
        assert "alpha" in result or "beta" in result

    def test_files_changed_adds_symbols(self) -> None:
        upstream = {
            "impl": "Modified the auth module.\nUpdated auth handling.\n",
        }
        files_changed = {"impl": ["src/auth.py", "src/utils.py"]}
        result = build_structural_context(upstream, 2000, files_changed)
        # "auth" should be extracted as a symbol from file path stem
        assert result  # Non-empty result


class TestSymbolToDict:
    def test_roundtrip(self) -> None:
        sym = Symbol(name="foo", kind="function", line_start=10, language="python")
        d = sym.to_dict()
        assert d["name"] == "foo"
        assert d["kind"] == "function"
        assert d["line_start"] == 10


# ===========================================================================
# Integration: loader + models
# ===========================================================================


class TestStructuralContextLoaderIntegration:
    def test_structural_accepted_in_plan(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: struct-test\ntasks:\n"
            "  - id: impl\n    engine: claude\n    prompt: implement\n"
            "  - id: review\n    engine: claude\n    prompt: review\n"
            "    depends_on: [impl]\n    context_from: [impl]\n"
            "    context_mode: structural\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        review = [t for t in plan.tasks if t.id == "review"][0]
        assert review.context_mode == "structural"

    def test_structural_requires_context_from(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        yaml_text = (
            "version: 1\nname: struct-test\ntasks:\n"
            "  - id: review\n    engine: claude\n    prompt: review\n"
            "    context_mode: structural\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_from"):
            load_plan(plan_path)

    def test_structural_in_context_modes(self) -> None:
        from maestro_cli.models import CONTEXT_MODES

        assert "structural" in CONTEXT_MODES
