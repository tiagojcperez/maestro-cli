"""Release-hygiene / anti-drift linter for the Maestro CLI repo.

Runs a handful of deterministic, dependency-light checks against the
git-tracked tree to catch the kinds of mistakes that block a clean public
release: private codenames / PII leaking out of gitignored notes, doc
placeholders left behind, the internal working-notes directory accidentally
getting tracked, and a missing license declaration.

Stdlib only. Shells out to `git` (via `git ls-files`) to enumerate tracked
files so that gitignored content (e.g. docs/internal/) is never inspected.

Usage:
    python scripts/doc_lint.py

Exit codes:
    0 - all checks passed (prints a short OK summary)
    1 - one or more violations found (prints grouped, actionable messages)
    2 - the linter itself could not run (e.g. not a git repo)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Repo root = parent of the scripts/ directory that holds this file.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Private codenames / PII / internal-run markers that must never appear in
# tracked files. These live only in the gitignored docs/internal/ working notes.
FORBIDDEN_LITERALS = (
    "PRZLAB",
    "facturaportatil",
    "webmaster@safeway",
    "PitLane",
    "c2-followup",
    "c2-reconciliation",
    "servicer.py",
)

# Absolute user-home paths: C:\Users\<name> or /home/<name>. A leaked home path
# usually means a machine-specific note or transcript slipped into the repo.
HOME_PATH_RE = re.compile(r"(?:[A-Za-z]:\\Users\\[^\\\s\"']+|/home/[^/\s\"']+)")

# Documentation placeholders that signal an unfinished doc.
PLACEHOLDER_ORG_RE = re.compile(r"<org>")
ARXIV_PLACEHOLDER_RE = re.compile(r"arxiv\.org/\S*xxxxx", re.IGNORECASE)

# This linter's own source contains the forbidden literals as data; skip it so
# it does not flag itself.
SELF_REL_PATH = "scripts/doc_lint.py"


def _git_ls_files(*args: str) -> list[str]:
    """Return tracked paths (repo-relative, forward slashes) from git ls-files."""
    result = subprocess.run(
        ["git", "ls-files", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-files {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _read_lines(rel_path: str) -> list[str]:
    """Read a tracked file as text lines; best-effort, never raises."""
    abs_path = REPO_ROOT / rel_path
    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return []
    return text.splitlines()


def _is_text_candidate(rel_path: str) -> bool:
    """Skip obvious binary/large-asset files when scanning for literals."""
    binary_suffixes = {
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz",
        ".whl", ".so", ".dll", ".pyd", ".woff", ".woff2", ".ttf", ".eot",
        ".sqlite", ".db", ".bin", ".jpeg",
    }
    return Path(rel_path).suffix.lower() not in binary_suffixes


def check_privacy_leak() -> list[str]:
    """Check 1: no tracked file leaks private codenames, PII, or home paths."""
    violations: list[str] = []
    for rel_path in _git_ls_files():
        if rel_path == SELF_REL_PATH:
            continue
        if not _is_text_candidate(rel_path):
            continue
        for lineno, line in enumerate(_read_lines(rel_path), start=1):
            for literal in FORBIDDEN_LITERALS:
                if literal in line:
                    violations.append(
                        f"{rel_path}:{lineno}: forbidden literal '{literal}' "
                        f"(private codename/PII belongs only in gitignored docs/internal/)"
                    )
            match = HOME_PATH_RE.search(line)
            if match:
                violations.append(
                    f"{rel_path}:{lineno}: leaked absolute user-home path "
                    f"'{match.group(0)}' (use a relative or generic path)"
                )
    return violations


def check_placeholders() -> list[str]:
    """Check 2: no doc placeholders (<org>, arXiv xxxxx, empty License section)."""
    violations: list[str] = []
    for rel_path in _git_ls_files("*.md"):
        if rel_path == SELF_REL_PATH:
            continue
        for lineno, line in enumerate(_read_lines(rel_path), start=1):
            if PLACEHOLDER_ORG_RE.search(line):
                violations.append(
                    f"{rel_path}:{lineno}: literal placeholder '<org>' left in docs"
                )
            if ARXIV_PLACEHOLDER_RE.search(line):
                violations.append(
                    f"{rel_path}:{lineno}: placeholder arXiv URL with 'xxxxx' "
                    f"(replace with a real arXiv ID or remove)"
                )

    # Empty License section: a '## License' heading that is the last non-empty
    # line of README.md means the section has no body.
    readme_lines = _read_lines("README.md")
    non_empty = [
        (idx + 1, ln) for idx, ln in enumerate(readme_lines) if ln.strip()
    ]
    if non_empty:
        last_lineno, last_line = non_empty[-1]
        if last_line.strip().lower() == "## license":
            violations.append(
                f"README.md:{last_lineno}: '## License' is the last non-empty "
                f"line (empty License section - add the license body)"
            )
    return violations


def check_internal_dir_untracked() -> list[str]:
    """Check 3: docs/internal/ must not be tracked by git."""
    tracked = _git_ls_files("docs/internal")
    if tracked:
        listed = ", ".join(tracked[:10])
        suffix = " ..." if len(tracked) > 10 else ""
        return [
            f"docs/internal/ is tracked by git ({len(tracked)} file(s)): "
            f"{listed}{suffix} (it holds private notes and must stay gitignored)"
        ]
    return []


def check_license_declared() -> list[str]:
    """Check 4: LICENSE at repo root + a license declaration in pyproject [project]."""
    violations: list[str] = []

    if not (REPO_ROOT / "LICENSE").is_file():
        violations.append("LICENSE: no LICENSE file at repo root")

    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        violations.append("pyproject.toml: file not found at repo root")
        return violations

    py_lines = pyproject.read_text(encoding="utf-8", errors="replace").splitlines()
    in_project = False
    found_license = False
    license_re = re.compile(r"^\s*license(?:-files)?\s*=")
    for line in py_lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and license_re.match(line):
            found_license = True
            break
    if not found_license:
        violations.append(
            "pyproject.toml: no 'license' declaration found in the [project] table"
        )
    return violations


CHECKS: tuple[tuple[str, "object"], ...] = (
    ("privacy/internal leak", check_privacy_leak),
    ("doc placeholders", check_placeholders),
    ("internal dir untracked", check_internal_dir_untracked),
    ("license declared", check_license_declared),
)


def main() -> None:
    try:
        # Fail fast with a clear message if git is unavailable / not a repo.
        _git_ls_files("--error-unmatch", "README.md")
    except RuntimeError as exc:
        print(f"[doc-lint] cannot run: {exc}", file=sys.stderr)
        sys.exit(2)

    all_violations: list[tuple[str, list[str]]] = []
    for label, check in CHECKS:
        try:
            found = check()  # type: ignore[operator]
        except RuntimeError as exc:
            print(f"[doc-lint] check '{label}' could not run: {exc}", file=sys.stderr)
            sys.exit(2)
        if found:
            all_violations.append((label, found))

    if not all_violations:
        print("[doc-lint] OK - all 4 release-hygiene checks passed.")
        sys.exit(0)

    total = sum(len(items) for _, items in all_violations)
    print(f"[doc-lint] FAIL - {total} violation(s) across "
          f"{len(all_violations)} check(s):\n")
    for label, items in all_violations:
        print(f"## {label} ({len(items)})")
        for item in items:
            print(f"  - {item}")
        print()
    sys.exit(1)


if __name__ == "__main__":
    main()
