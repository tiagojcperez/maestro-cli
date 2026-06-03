from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import (
    DualVerificationResult,
    MergeOverlap,
    MergeReview,
    MergeReviewVerdict,
    WorktreeMergeResult,
)

# ---------------------------------------------------------------------------
# Merge serialization — prevents concurrent git state corruption
# ---------------------------------------------------------------------------

_merge_lock = threading.Lock()

# Track files merged by completed tasks: file_path -> [task_ids]
_merge_ledger: dict[str, list[str]] = {}
_merge_ledger_lock = threading.Lock()


def reset_merge_ledger() -> None:
    """Reset the merge ledger.  Called at the start of each plan run."""
    with _merge_ledger_lock:
        _merge_ledger.clear()


def _record_merged_files(task_id: str, files: list[str]) -> None:
    """Record which files a task merged, for overlap detection."""
    with _merge_ledger_lock:
        for f in files:
            _merge_ledger.setdefault(f, []).append(task_id)


def _get_overlaps(files: list[str]) -> list[MergeOverlap]:
    """Check if any files overlap with previously merged tasks."""
    overlaps: list[MergeOverlap] = []
    with _merge_ledger_lock:
        for f in files:
            if f in _merge_ledger:
                overlaps.append(MergeOverlap(file=f, merged_by=list(_merge_ledger[f])))
    return overlaps


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _sanitize_branch_name(task_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", task_id)


def _branch_name(task_id: str) -> str:
    return f"maestro/{_sanitize_branch_name(task_id)}"


def _run_git(workspace_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=workspace_root,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(f"failed to run git: {exc}") from exc


def get_base_branch(workspace_root: Path) -> str:
    result = _run_git(workspace_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if result.returncode != 0:
        raise ValueError("workspace_root is not a git repository")
    branch = result.stdout.strip()
    if not branch:
        raise ValueError("unable to determine current git branch")
    return branch


# ---------------------------------------------------------------------------
# Worktree lifecycle
# ---------------------------------------------------------------------------

def create_worktree(workspace_root: Path, task_id: str, base_branch: str | None = None) -> Path:
    resolved_base_branch = base_branch or get_base_branch(workspace_root)
    sanitized_task_id = _sanitize_branch_name(task_id)
    branch_name = f"maestro/{sanitized_task_id}"
    worktree_path = workspace_root / ".maestro-worktrees" / task_id
    # Keep .maestro-worktrees/ in .gitignore because these are ephemeral task workspaces.
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[maestro] creating worktree {worktree_path} from {resolved_base_branch}")
    result = _run_git(
        workspace_root,
        ["worktree", "add", str(worktree_path), "-b", branch_name, resolved_base_branch],
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "unknown git error"
        raise RuntimeError(f"failed to create worktree for task '{task_id}': {detail}")
    return worktree_path


def merge_worktree(
    workspace_root: Path,
    task_id: str,
    worktree_path: Path,
    base_branch: str,
    review_model: str = "haiku",
    review_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> WorktreeMergeResult:
    """Merge a worktree branch back to the base branch.

    Thread-safe: all merge operations are serialized via ``_merge_lock``
    to prevent concurrent git state corruption.
    """
    del worktree_path  # kept in signature for API compat
    with _merge_lock:
        return _merge_worktree_locked(
            workspace_root, task_id, base_branch, review_model, review_callback,
        )


def _merge_worktree_locked(
    workspace_root: Path,
    task_id: str,
    base_branch: str,
    review_model: str,
    review_callback: Callable[[str, dict[str, object]], None] | None,
) -> WorktreeMergeResult:
    branch_name = _branch_name(task_id)
    try:
        # Step 1: Get diff stat
        print(f"[maestro] checking worktree diff for {branch_name}")
        diff_result = _run_git(
            workspace_root,
            ["diff", f"{base_branch}...{branch_name}", "--stat"],
        )
        if diff_result.returncode != 0:
            detail = (diff_result.stderr or diff_result.stdout).strip() or "unknown git error"
            return WorktreeMergeResult(status="error", error=detail)

        files_changed = _parse_changed_files(diff_result.stdout.strip())
        if not files_changed:
            return WorktreeMergeResult(status="empty")

        # Step 2: Check overlap with previously merged files
        overlaps = _get_overlaps(files_changed)

        # Step 3: Checkout base and preview merge (no-commit)
        print(f"[maestro] checking out {base_branch} for merge")
        checkout_result = _run_git(workspace_root, ["checkout", base_branch])
        if checkout_result.returncode != 0:
            detail = (checkout_result.stderr or checkout_result.stdout).strip() or "unknown git error"
            return WorktreeMergeResult(status="error", error=detail)

        print(f"[maestro] merging {branch_name} into {base_branch}")
        merge_test = _run_git(
            workspace_root,
            ["merge", "--no-commit", "--no-ff", branch_name],
        )

        if merge_test.returncode == 0:
            # Clean merge preview — commit it
            commit_result = _run_git(
                workspace_root,
                ["commit", "-m", f"maestro: merge {task_id}"],
            )
            if commit_result.returncode != 0:
                _run_git(workspace_root, ["merge", "--abort"])
                detail = (commit_result.stderr or commit_result.stdout).strip() or "commit failed"
                return WorktreeMergeResult(status="error", files_changed=files_changed, error=detail)

            head_result = _run_git(workspace_root, ["rev-parse", "HEAD"])
            merge_commit = head_result.stdout.strip() if head_result.returncode == 0 else None

            # Record in ledger for future overlap detection
            _record_merged_files(task_id, files_changed)

            review = MergeReview(
                verdict="safe" if not overlaps else "resolvable",
                overlapping_files=overlaps,
            )

            return WorktreeMergeResult(
                status="merged",
                files_changed=files_changed,
                merge_commit=merge_commit,
                review=review,
            )

        # Conflict detected — abort the preview
        conflict_files = _parse_conflict_files(merge_test.stdout, merge_test.stderr)
        _run_git(workspace_root, ["merge", "--abort"])

        # Step 4: LLM review for resolution suggestions
        review = _build_conflict_review(
            workspace_root, task_id, branch_name, base_branch,
            files_changed, conflict_files, overlaps, review_model,
        )

        if review_callback:
            try:
                review_callback("worktree_review", {
                    "task_id": task_id,
                    "verdict": review.verdict,
                    "conflict_files": conflict_files,
                    "overlapping_files": [o.file for o in overlaps],
                    "resolution_suggestion": review.resolution_suggestion or "",
                })
            except Exception:
                pass

        return WorktreeMergeResult(
            status="conflict",
            files_changed=files_changed,
            conflict_files=conflict_files,
            review=review,
        )
    except Exception as exc:
        return WorktreeMergeResult(status="error", error=str(exc))


def cleanup_worktree(workspace_root: Path, task_id: str, worktree_path: Path) -> None:
    branch_name = _branch_name(task_id)
    print(f"[maestro] cleaning up worktree {worktree_path}")
    try:
        _run_git(workspace_root, ["worktree", "remove", str(worktree_path), "--force"])
    except Exception:
        pass
    try:
        _run_git(workspace_root, ["branch", "-D", branch_name])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LLM merge review
# ---------------------------------------------------------------------------

_MERGE_REVIEW_TIMEOUT = 60

_MERGE_REVIEW_PROMPT = """\
You are analyzing a git merge conflict to suggest resolution strategies.

Task '{task_id}' (branch: {branch_name}) is being merged into {base_branch}.

Files changed by this task:
{files_changed_list}

Files with merge conflicts:
{conflict_files_list}

Files that overlap with previously merged tasks:
{overlap_list}

Conflict diff:
---
{conflict_diff}
---

Respond ONLY with a JSON object:
{{"resolution_strategy": "additive|non_overlapping|true_conflict", "safe_to_auto_resolve": true|false, "reasoning": "1-3 sentences", "suggestion": "Specific resolution suggestion"}}
"""


def _build_conflict_review(
    workspace_root: Path,
    task_id: str,
    branch_name: str,
    base_branch: str,
    files_changed: list[str],
    conflict_files: list[str],
    overlaps: list[MergeOverlap],
    model: str = "haiku",
) -> MergeReview:
    """Build a MergeReview with LLM-assisted conflict analysis.

    Falls back to a basic review without LLM suggestions on any failure.
    """
    # Get diffs for conflict files
    diff_output = ""
    for cf in conflict_files[:5]:
        diff_result = _run_git(
            workspace_root,
            ["diff", f"{base_branch}...{branch_name}", "--", cf],
        )
        if diff_result.returncode == 0:
            diff_output += f"--- {cf} ---\n{diff_result.stdout[:2000]}\n\n"

    if not diff_output:
        return MergeReview(
            verdict="conflict",
            overlapping_files=overlaps,
            conflict_files=conflict_files,
        )

    overlap_list = "\n".join(
        f"  - {o.file} (also modified by: {', '.join(o.merged_by)})"
        for o in overlaps
    ) or "  (none)"

    prompt = _MERGE_REVIEW_PROMPT.format(
        task_id=task_id,
        branch_name=branch_name,
        base_branch=base_branch,
        files_changed_list="\n".join(f"  - {f}" for f in files_changed),
        conflict_files_list="\n".join(f"  - {f}" for f in conflict_files),
        overlap_list=overlap_list,
        conflict_diff=diff_output[:6000],
    )

    started = time.monotonic()
    try:
        from .runners import _build_safe_env, _resolve_executable

        cmd = _resolve_executable("claude") + [
            "--print", "--model", model,
            "--output-format", "text",
            prompt,
        ]
        proc = subprocess.run(
            cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_MERGE_REVIEW_TIMEOUT,
            env=_build_safe_env({}, {}),
        )
        duration = time.monotonic() - started

        if proc.returncode != 0 or not proc.stdout.strip():
            return MergeReview(
                verdict="conflict",
                overlapping_files=overlaps,
                conflict_files=conflict_files,
                review_model=model,
                review_duration_sec=duration,
            )

        raw = proc.stdout.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            payload = json.loads(raw[start:end])
            suggestion = str(payload.get("suggestion", ""))
            strategy = str(payload.get("resolution_strategy", "true_conflict"))

            verdict: MergeReviewVerdict = "conflict"
            if strategy in ("additive", "non_overlapping"):
                verdict = "resolvable"

            return MergeReview(
                verdict=verdict,
                overlapping_files=overlaps,
                conflict_files=conflict_files,
                resolution_suggestion=suggestion,
                review_model=model,
                review_duration_sec=duration,
            )
    except Exception:
        duration = time.monotonic() - started

    return MergeReview(
        verdict="conflict",
        overlapping_files=overlaps,
        conflict_files=conflict_files,
        review_duration_sec=duration,
    )


# ---------------------------------------------------------------------------
# Dual verification — compare agent claims vs actual git diff
# ---------------------------------------------------------------------------

# Patterns that indicate a file was modified/created/deleted in agent output
_FILE_ACTION_RE = re.compile(
    r"(?:(?:modif|creat|updat|edit|chang|writ|delet|remov|add|fix|refactor|implement)"
    r"(?:ied|ed|ing|e|s)?)"
    r"[:\s]+[`'\"]?"
    r"((?:[\w./\\-]+/)?[\w.-]+\.[\w]+)"
    r"[`'\"]?",
    re.IGNORECASE,
)

# Simpler pattern: file paths that appear with code fence or backtick context
_FILE_MENTION_RE = re.compile(
    r"[`'\"]"
    r"((?:[\w./\\-]+/)?[\w.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|cpp|c|h|cs|yaml|yml|json|toml|md|sql|html|css|sh|cfg))"
    r"[`'\"]",
    re.IGNORECASE,
)


def _extract_claimed_files(stdout: str) -> set[str]:
    """Extract file paths the agent claims to have modified from stdout.

    Uses two heuristics:
    1. Action verbs followed by file paths (\"modified src/foo.py\")
    2. Backtick-quoted file paths in output summaries
    """
    claimed: set[str] = set()
    for match in _FILE_ACTION_RE.finditer(stdout):
        raw = match.group(1).replace("\\", "/").strip()
        # Normalize: strip leading ./ or /
        raw = raw.lstrip("./")
        if raw:
            claimed.add(raw)

    # Second pass: files in backticks near action words
    for match in _FILE_MENTION_RE.finditer(stdout):
        raw = match.group(1).replace("\\", "/").strip().lstrip("./")
        if raw:
            # Only include if there's an action verb nearby (within 200 chars before)
            start = max(0, match.start() - 200)
            context = stdout[start:match.start()].lower()
            action_words = ("modif", "creat", "updat", "edit", "chang", "writ",
                            "delet", "remov", "add", "fix", "refactor", "implement")
            if any(w in context for w in action_words):
                claimed.add(raw)

    return claimed


def _normalize_path(path: str) -> str:
    """Normalize a file path for comparison."""
    return path.replace("\\", "/").strip().lstrip("./")


def verify_worktree_output(
    files_changed: list[str],
    stdout_tail: str,
    *,
    threshold: float = 0.5,
) -> DualVerificationResult:
    """Compare actual git diff files against agent-claimed modifications.

    Args:
        files_changed: Files actually changed in the worktree (from git diff).
        stdout_tail: Agent output (last N lines of stdout).
        threshold: Minimum overlap ratio to consider verified (0.0-1.0).

    Returns:
        DualVerificationResult with verification status and gap details.
    """
    actual = {_normalize_path(f) for f in files_changed}
    claimed = _extract_claimed_files(stdout_tail)

    # Normalize claimed paths to match actual paths (which may be relative)
    # Try basename matching if full path doesn't match
    claimed_normalized: set[str] = set()
    for c in claimed:
        norm = _normalize_path(c)
        claimed_normalized.add(norm)

    # Match by basename if exact path doesn't match
    actual_basenames: dict[str, str] = {}
    for a in actual:
        basename = a.rsplit("/", 1)[-1] if "/" in a else a
        actual_basenames[basename] = a

    matched_claimed: set[str] = set()
    for c in claimed_normalized:
        basename = c.rsplit("/", 1)[-1] if "/" in c else c
        if c in actual or basename in actual_basenames:
            matched_claimed.add(c)

    # Compute overlap
    intersection = actual & claimed_normalized
    # Also count basename matches
    for c in claimed_normalized - intersection:
        basename = c.rsplit("/", 1)[-1] if "/" in c else c
        if basename in actual_basenames:
            intersection.add(actual_basenames[basename])

    union = actual | claimed_normalized
    overlap_ratio = len(intersection) / len(union) if union else 1.0

    unclaimed = sorted(actual - claimed_normalized - {_normalize_path(c) for c in matched_claimed})
    phantom = sorted(claimed_normalized - actual - {_normalize_path(a) for a in intersection})

    verified = overlap_ratio >= threshold or (not actual and not claimed_normalized)

    return DualVerificationResult(
        verified=verified,
        files_in_diff=sorted(actual),
        files_claimed=sorted(claimed_normalized),
        unclaimed_files=unclaimed,
        phantom_files=phantom,
        overlap_ratio=round(overlap_ratio, 3),
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_changed_files(diff_output: str) -> list[str]:
    files_changed: list[str] = []
    for line in diff_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("|", "-", "0 files changed")):
            continue
        if "|" not in line:
            continue
        files_changed.append(line.split("|", 1)[0].strip())
    return files_changed


def _parse_conflict_files(stdout: str, stderr: str) -> list[str]:
    conflict_files: list[str] = []
    combined = f"{stdout}\n{stderr}"
    for line in combined.splitlines():
        match = re.search(r"CONFLICT .* in (.+)$", line.strip())
        if match:
            conflict_files.append(match.group(1).strip())
    return conflict_files
