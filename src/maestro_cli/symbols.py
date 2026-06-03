"""Regex-based code symbol extraction for ``context_mode: structural``.

Extracts function, class, method, import, and type definitions from source
code using language-aware regex patterns.  Zero external dependencies — all
extraction is stdlib regex.

Language detection uses file-extension hints from diff headers, code-fence
language tags, shebang lines, and keyword density heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

SymbolKind = str  # "function" | "class" | "method" | "import" | "constant" | "type"


@dataclass
class Symbol:
    """A code symbol extracted from source text."""

    name: str
    kind: SymbolKind
    line_start: int
    line_end: int | None = None
    language: str = ""
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "language": self.language,
            "file_path": self.file_path,
        }


# ---------------------------------------------------------------------------
# Language regex patterns
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".php": "php",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".java": "java",
}

# Each entry: (compiled_regex, symbol_kind, name_group_index)
_LANG_PATTERNS: dict[str, list[tuple[re.Pattern[str], SymbolKind, int]]] = {
    "python": [
        (re.compile(r"^\s*def\s+(\w+)\s*\("), "function", 1),
        (re.compile(r"^\s*class\s+(\w+)"), "class", 1),
        (re.compile(r"^import\s+(\S+)"), "import", 1),
        (re.compile(r"^from\s+(\S+)\s+import"), "import", 1),
    ],
    "javascript": [
        (re.compile(r"^\s*function\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\()"), "function", 1),
        (re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*import\s+"), "import", 0),
    ],
    "typescript": [
        (re.compile(r"^\s*function\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\()"), "function", 1),
        (re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*(?:export\s+)?type\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*import\s+"), "import", 0),
    ],
    "php": [
        (re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+)*function\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*class\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*use\s+(\S+)"), "import", 1),
    ],
    "go": [
        (re.compile(r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)"), "function", 1),
        (re.compile(r"^\s*type\s+(\w+)\s+struct"), "type", 1),
        (re.compile(r"^\s*type\s+(\w+)\s+interface"), "type", 1),
        (re.compile(r"^\s*import\s+"), "import", 0),
    ],
    "rust": [
        (re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*(?:pub\s+)?struct\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*(?:pub\s+)?enum\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*impl\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*use\s+"), "import", 0),
    ],
    "java": [
        (re.compile(r"^\s*(?:public|private|protected|static|\s)*\s+\w+\s+(\w+)\s*\("), "function", 1),
        (re.compile(r"^\s*(?:public\s+|abstract\s+)*class\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*(?:public\s+)?interface\s+(\w+)"), "type", 1),
        (re.compile(r"^\s*import\s+"), "import", 0),
    ],
    "ruby": [
        (re.compile(r"^\s*def\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*class\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*module\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*require\s+"), "import", 0),
    ],
}

# Aliases
_LANG_PATTERNS["c"] = [
    (re.compile(r"^\s*(?:static\s+)?(?:void|int|char|bool|auto|unsigned|long|short|float|double|[\w*]+)\s+(\w+)\s*\("), "function", 1),
    (re.compile(r"^\s*struct\s+(\w+)"), "type", 1),
    (re.compile(r"^\s*typedef\s+.*\s+(\w+)\s*;"), "type", 1),
    (re.compile(r"^\s*#include\s+"), "import", 0),
]
_LANG_PATTERNS["cpp"] = _LANG_PATTERNS["c"] + [
    (re.compile(r"^\s*class\s+(\w+)"), "class", 1),
    (re.compile(r"^\s*namespace\s+(\w+)"), "class", 1),
]


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^(?:diff --git a/|[+]{3} [ab]/|\-{3} [ab]/)(\S+)", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```(\w+)", re.MULTILINE)
_SHEBANG_RE = re.compile(r"^#!\s*/\S+/(?:env\s+)?(\w+)")

_KEYWORD_HINTS: dict[str, list[str]] = {
    "python": ["def ", "import ", "class ", "self.", "__init__"],
    "javascript": ["function ", "const ", "require(", "module.exports"],
    "typescript": ["interface ", "type ", ": string", ": number"],
    "go": ["func ", "package ", "import (", "type "],
    "rust": ["fn ", "let mut ", "impl ", "pub fn"],
    "php": ["<?php", "function ", "namespace ", "$this->"],
    "java": ["public class ", "private ", "System.out"],
    "ruby": ["def ", "require ", "attr_accessor", "end"],
}


def detect_language_from_path(path: str) -> str | None:
    """Detect language from file extension."""
    ext = PurePosixPath(path.replace("\\", "/")).suffix.lower()
    return _EXT_TO_LANG.get(ext)


def detect_language_from_text(text: str) -> str | None:
    """Detect language from text content heuristics."""
    # 1. Diff headers
    for m in _DIFF_FILE_RE.finditer(text[:2000]):
        lang = detect_language_from_path(m.group(1))
        if lang:
            return lang

    # 2. Code fences
    m2 = _CODE_FENCE_RE.search(text[:1000])
    if m2:
        fence_lang = m2.group(1).lower()
        # Map common fence names
        fence_map = {"python": "python", "py": "python", "js": "javascript",
                     "ts": "typescript", "tsx": "typescript", "jsx": "javascript",
                     "go": "go", "rust": "rust", "rs": "rust",
                     "php": "php", "java": "java", "rb": "ruby", "ruby": "ruby",
                     "c": "c", "cpp": "cpp", "cc": "cpp"}
        if fence_lang in fence_map:
            return fence_map[fence_lang]

    # 3. Shebang
    m3 = _SHEBANG_RE.match(text[:200])
    if m3:
        prog = m3.group(1).lower()
        shebang_map = {"python": "python", "python3": "python", "node": "javascript",
                       "ruby": "ruby", "php": "php"}
        if prog in shebang_map:
            return shebang_map[prog]

    # 4. Keyword density
    sample = text[:3000]
    best_lang: str | None = None
    best_count = 0
    for lang, keywords in _KEYWORD_HINTS.items():
        count = sum(1 for kw in keywords if kw in sample)
        if count > best_count:
            best_count = count
            best_lang = lang
    return best_lang if best_count >= 2 else None


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

_MAX_LINES = 500  # Cap to avoid slow scans on huge outputs


def extract_symbols(text: str, language: str | None = None) -> list[Symbol]:
    """Extract code symbols from a text block using regex patterns."""
    if not text:
        return []

    if language is None:
        language = detect_language_from_text(text)
    if language is None or language not in _LANG_PATTERNS:
        return []

    patterns = _LANG_PATTERNS[language]
    symbols: list[Symbol] = []
    lines = text.splitlines()[:_MAX_LINES]

    for line_num, line in enumerate(lines, start=1):
        for pattern, kind, name_group in patterns:
            m = pattern.match(line)
            if m:
                name = m.group(name_group) if name_group > 0 else ""
                if name_group == 0:
                    # Import without named capture — use full match
                    name = line.strip()
                if name:
                    symbols.append(Symbol(
                        name=name,
                        kind=kind,
                        line_start=line_num,
                        language=language,
                    ))
                break  # One match per line

    return symbols


def extract_changed_symbols(diff_text: str) -> list[Symbol]:
    """Extract symbols from added/modified lines in diff output."""
    if not diff_text:
        return []

    symbols: list[Symbol] = []
    current_file = ""
    current_lang: str | None = None

    for line in diff_text.splitlines()[:_MAX_LINES]:
        # Track current file from diff headers
        m = _DIFF_FILE_RE.match(line)
        if m:
            current_file = m.group(1)
            current_lang = detect_language_from_path(current_file)
            continue

        # Only look at added lines (not removed)
        if not line.startswith("+") or line.startswith("+++"):
            continue

        # Strip the leading +
        code_line = line[1:]
        if current_lang and current_lang in _LANG_PATTERNS:
            for pattern, kind, name_group in _LANG_PATTERNS[current_lang]:
                m2 = pattern.match(code_line)
                if m2:
                    name = m2.group(name_group) if name_group > 0 else code_line.strip()
                    if name:
                        symbols.append(Symbol(
                            name=name,
                            kind=kind,
                            line_start=0,
                            language=current_lang,
                            file_path=current_file,
                        ))
                    break

    return symbols


# ---------------------------------------------------------------------------
# Chunk scoring and context building
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 30  # Lines per chunk (same as selective)
_MIN_SCORE = 0.01  # Minimum relevance score to include


def _split_into_chunks(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """Split text into fixed-size line chunks."""
    lines = text.splitlines()
    chunks: list[str] = []
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _score_chunk(chunk: str, target_symbols: set[str]) -> float:
    """Score a chunk by how many target symbols it references."""
    if not target_symbols:
        return 0.0
    chunk_lower = chunk.lower()
    hits = sum(1 for sym in target_symbols if sym.lower() in chunk_lower)
    return hits / len(target_symbols) if target_symbols else 0.0


def build_structural_context(
    upstream_texts: dict[str, str],
    budget_tokens: int,
    upstream_files_changed: dict[str, list[str]] | None = None,
) -> str:
    """Build context by extracting code symbols and filtering by blast radius.

    1. For each upstream, extract symbols from its output (diff or code).
    2. Build the set of "changed symbols" (names that were added/modified).
    3. Score each chunk of upstream text by symbol reference density.
    4. Greedily select highest-scoring chunks within budget.

    Zero LLM cost — all regex and heuristic based.
    """
    if not upstream_texts or budget_tokens <= 0:
        return ""

    budget_chars = budget_tokens * 4

    # Phase 1: Extract changed symbols across all upstreams
    changed_symbols: set[str] = set()
    for uid, text in upstream_texts.items():
        # Try diff-based extraction first (most precise)
        diff_symbols = extract_changed_symbols(text)
        if diff_symbols:
            changed_symbols.update(s.name for s in diff_symbols if s.kind != "import")
        else:
            # Fallback: extract all symbols from text
            all_symbols = extract_symbols(text)
            changed_symbols.update(s.name for s in all_symbols if s.kind != "import")

        # Also extract symbols from files_changed paths
        if upstream_files_changed:
            for fpath in upstream_files_changed.get(uid, []):
                # Extract likely symbol names from file paths (module names)
                stem = PurePosixPath(fpath.replace("\\", "/")).stem
                if stem and stem != "__init__" and not stem.startswith("."):
                    changed_symbols.add(stem)

    if not changed_symbols:
        # No symbols found — fallback to simple truncation
        parts: list[str] = []
        remaining = budget_chars
        for uid, text in upstream_texts.items():
            header = f"--- {uid} ---\n"
            if remaining <= len(header):
                break
            truncated = text[: remaining - len(header)]
            parts.append(header + truncated)
            remaining -= len(header) + len(truncated)
        return "\n\n".join(parts) if parts else ""

    # Phase 2: Score and select chunks
    scored_chunks: list[tuple[float, str, str]] = []  # (score, chunk, upstream_id)
    for uid, text in upstream_texts.items():
        chunks = _split_into_chunks(text)
        for chunk in chunks:
            score = _score_chunk(chunk, changed_symbols)
            if score >= _MIN_SCORE:
                scored_chunks.append((score, chunk, uid))

    # Sort by score descending
    scored_chunks.sort(key=lambda x: -x[0])

    # Phase 3: Greedy selection within budget
    selected: list[str] = []
    used_chars = 0
    seen_uids: set[str] = set()

    for score, chunk, uid in scored_chunks:
        header = f"--- {uid} ---\n" if uid not in seen_uids else ""
        entry = header + chunk
        if used_chars + len(entry) > budget_chars:
            continue
        selected.append(entry)
        used_chars += len(entry)
        seen_uids.add(uid)

    return "\n\n".join(selected)
