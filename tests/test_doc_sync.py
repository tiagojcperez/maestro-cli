"""Tests that code artefacts stay in sync with documentation.

Prevents documentation drift by extracting error codes, event types,
SEC rules, warning codes, Literal types, and constants from source
and verifying they appear in CLAUDE.md / CLI_REFERENCE.md.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src" / "maestro_cli"
_CLAUDE_MD = _ROOT / "CLAUDE.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_literal_values(source: str, type_name: str) -> set[str]:
    """Extract string values from a Literal type definition in source."""
    # Match: TypeName = Literal["a", "b", "c"]  (may span multiple lines)
    pattern = re.compile(
        rf'^{type_name}\s*=\s*Literal\[([^\]]+)\]',
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(source)
    if not m:
        return set()
    return set(re.findall(r'"([^"]+)"', m.group(1)))


# ---------------------------------------------------------------------------
# 1. Error codes: every E-code in errors.py must appear in CLAUDE.md
# ---------------------------------------------------------------------------

def test_error_codes_documented() -> None:
    """Every error code defined in errors.py is mentioned in CLAUDE.md."""
    errors_src = _read(_SRC / "errors.py")
    claude_md = _read(_CLAUDE_MD)

    # Pattern: E001 = "E001"  or  code="E030"
    code_pattern = re.compile(r'(E\d{3})\s*=\s*"E\d{3}"')
    defined_codes = set(code_pattern.findall(errors_src))

    assert defined_codes, "No error codes found in errors.py — regex may be wrong"

    missing = {code for code in defined_codes if code not in claude_md}
    assert not missing, (
        f"Error codes defined in errors.py but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 2. Event types: every _emit("event_name", ...) in scheduler.py and
#    event_callback("event_name", ...) in runners.py and
#    _emit(..., "event_name", ...) in watch.py must appear in CLAUDE.md
# ---------------------------------------------------------------------------

def test_event_types_documented() -> None:
    """Every event type emitted in scheduler/runners/watch is in CLAUDE.md."""
    claude_md = _read(_CLAUDE_MD)

    event_types: set[str] = set()

    # scheduler.py: _emit("event_name", ...)
    scheduler_src = _read(_SRC / "scheduler.py")
    for m in re.finditer(r'_emit\(\s*"([a-z_]+)"', scheduler_src):
        event_types.add(m.group(1))

    # runners.py: event_callback("event_name", ...)
    runners_src = _read(_SRC / "runners.py")
    for m in re.finditer(r'event_callback\(\s*"([a-z_]+)"', runners_src):
        event_types.add(m.group(1))

    # watch.py: _emit(event_callback, "event_name", ...)
    watch_src = _read(_SRC / "watch.py")
    for m in re.finditer(r'_emit\(\s*event_callback,\s*"([a-z_]+)"', watch_src):
        event_types.add(m.group(1))

    assert event_types, "No event types found — regex may be wrong"

    missing = {et for et in event_types if et not in claude_md}
    assert not missing, (
        f"Event types emitted in code but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 3. SEC rules: every rule="SECnnn" in audit.py must appear in CLAUDE.md
# ---------------------------------------------------------------------------

def test_sec_rules_documented() -> None:
    """Every SEC rule in audit.py is mentioned in CLAUDE.md."""
    audit_src = _read(_SRC / "audit.py")
    claude_md = _read(_CLAUDE_MD)

    sec_pattern = re.compile(r'rule="(SEC\d{3})"')
    defined_rules = set(sec_pattern.findall(audit_src))

    assert defined_rules, "No SEC rules found in audit.py — regex may be wrong"

    missing = {rule for rule in defined_rules if rule not in claude_md}
    assert not missing, (
        f"SEC rules defined in audit.py but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 4. Warning codes: every numbered W-code in loader.py must appear in
#    CLAUDE.md
# ---------------------------------------------------------------------------

def test_warning_codes_documented() -> None:
    """Every numbered warning code (W1-W99) in loader.py is in CLAUDE.md."""
    loader_src = _read(_SRC / "loader.py")
    claude_md = _read(_CLAUDE_MD)

    # Pattern: W1, W2, ..., W16 in comments or ws.append strings
    w_pattern = re.compile(r'\bW(\d{1,2})\b')
    # Only match in warning-related context lines
    defined_codes: set[str] = set()
    for line in loader_src.splitlines():
        # Lines with W-code comments or ws.append with W-code
        if "W" in line and ("ws.append" in line or "# W" in line or "# --" in line):
            for m in w_pattern.finditer(line):
                code = f"W{m.group(1)}"
                defined_codes.add(code)

    assert defined_codes, "No warning codes found in loader.py — regex may be wrong"

    missing = {code for code in defined_codes if code not in claude_md}
    assert not missing, (
        f"Warning codes in loader.py but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 5. CLI subcommands: every add_parser("cmd") in cli.py must appear in
#    CLI_REFERENCE.md
# ---------------------------------------------------------------------------

def test_cli_subcommands_documented() -> None:
    """Every CLI subcommand in cli.py is documented in CLI_REFERENCE.md."""
    cli_src = _read(_SRC / "cli.py")
    cli_ref = _read(
        Path(__file__).resolve().parent.parent / "docs" / "CLI_REFERENCE.md"
    )

    # Pattern: sub.add_parser("cmd" or subs.add_parser("cmd"
    parser_pattern = re.compile(r'add_parser\(\s*"([a-z_-]+)"')
    defined_cmds = set(parser_pattern.findall(cli_src))

    assert defined_cmds, "No subcommands found in cli.py — regex may be wrong"

    missing = {cmd for cmd in defined_cmds if cmd not in cli_ref}
    assert not missing, (
        f"CLI subcommands in cli.py but missing from CLI_REFERENCE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 6. Engine names: every engine in EngineName Literal must be in CLAUDE.md
# ---------------------------------------------------------------------------

def test_engine_names_documented() -> None:
    """Every engine in the EngineName Literal type is mentioned in CLAUDE.md."""
    models_src = _read(_SRC / "models.py")
    claude_md = _read(_CLAUDE_MD)

    engines = _extract_literal_values(models_src, "EngineName")
    assert engines, "No engines found in EngineName — regex may be wrong"

    missing = {e for e in engines if e not in claude_md}
    assert not missing, (
        f"Engines in EngineName but missing from CLAUDE.md: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 7. Context modes: every mode in ContextMode must be in CLAUDE.md
# ---------------------------------------------------------------------------

def test_context_modes_documented() -> None:
    """Every context mode in ContextMode Literal is mentioned in CLAUDE.md."""
    models_src = _read(_SRC / "models.py")
    claude_md = _read(_CLAUDE_MD)

    modes = _extract_literal_values(models_src, "ContextMode")
    assert modes, "No modes found in ContextMode — regex may be wrong"

    missing = {m for m in modes if m not in claude_md}
    assert not missing, (
        f"Context modes in ContextMode but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 8. Assertion types: every type in ASSERTION_TYPES must be in CLAUDE.md
# ---------------------------------------------------------------------------

def test_assertion_types_documented() -> None:
    """Every judge assertion type in ASSERTION_TYPES is in CLAUDE.md."""
    models_src = _read(_SRC / "models.py")
    claude_md = _read(_CLAUDE_MD)

    # Pattern: ASSERTION_TYPES: set[str] = {"contains", "regex", ...}
    m = re.search(r'ASSERTION_TYPES[^=]*=\s*\{([^}]+)\}', models_src)
    assert m, "ASSERTION_TYPES not found in models.py"

    types = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert types, "No assertion types extracted"

    missing = {t for t in types if t not in claude_md}
    assert not missing, (
        f"Assertion types in ASSERTION_TYPES but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 9. Template variables: every var in _KNOWN_GLOBAL_VARS must be in CLAUDE.md
# ---------------------------------------------------------------------------

def test_template_vars_documented() -> None:
    """Every template variable in _KNOWN_GLOBAL_VARS is in CLAUDE.md."""
    loader_src = _read(_SRC / "loader.py")
    claude_md = _read(_CLAUDE_MD)

    # Extract the set literal contents
    m = re.search(
        r'_KNOWN_GLOBAL_VARS[^=]*=\s*\{([^}]+)\}',
        loader_src,
        re.DOTALL,
    )
    assert m, "_KNOWN_GLOBAL_VARS not found in loader.py"

    variables = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert variables, "No template variables extracted"

    # Check each var appears in CLAUDE.md (with {{ }} or as plain text)
    missing = {
        v for v in variables
        if v not in claude_md and f"{{{{ {v} }}}}" not in claude_md
    }
    assert not missing, (
        f"Template vars in _KNOWN_GLOBAL_VARS but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# 10. Judge presets: every preset name in JUDGE_PRESETS must be in CLAUDE.md
# ---------------------------------------------------------------------------

def test_judge_presets_documented() -> None:
    """Every judge preset in JUDGE_PRESETS dict is mentioned in CLAUDE.md."""
    models_src = _read(_SRC / "models.py")
    claude_md = _read(_CLAUDE_MD)

    # Find the JUDGE_PRESETS block and extract top-level keys.
    # Keys appear on lines like:    "code_quality": {
    presets: set[str] = set()
    in_block = False
    for line in models_src.splitlines():
        if line.startswith("JUDGE_PRESETS"):
            in_block = True
            continue
        if in_block:
            # Top-level keys are indented exactly 4 spaces
            m = re.match(r'    "([a-z_]+)"\s*:', line)
            if m:
                presets.add(m.group(1))
            # End of dict
            if line.startswith("}"):
                break

    assert presets, "No judge presets found in JUDGE_PRESETS"

    missing = {p for p in presets if p not in claude_md}
    assert not missing, (
        f"Judge presets in JUDGE_PRESETS but missing from CLAUDE.md: "
        f"{sorted(missing)}"
    )
