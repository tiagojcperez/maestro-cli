from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from .models import StructuredContext


def now_utc() -> datetime:
    return datetime.now(UTC)


def resolve_path(base_dir: Path, maybe_relative: str | None) -> Path | None:
    """Resolve *maybe_relative* against *base_dir*, returning an absolute Path.

    Returns ``None`` when *maybe_relative* is falsy.  Absolute paths are
    returned directly; relative paths are joined to *base_dir* and resolved.
    """
    if not maybe_relative:
        return None
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def command_to_string(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


_FENCE_OPEN_RE = re.compile(r"^(`{3,})(\w*)\s*$")
_FENCE_CLOSE_RE = re.compile(r"^(`{3,})\s*$")


def extract_prompt_from_markdown(md_text: str, heading: str) -> str:
    """Extract prompt text from a markdown section.

    Locates a ``## {heading}`` line in *md_text*, then extracts the prompt
    content.  Prefers code-fenced content (especially ` ```text `) when
    available.  Falls back to all prose text under the heading when no code
    fence is found — this lets plan authors write prompts as plain markdown
    without wrapping them in fences.

    Supports nested code fences: the outer fence must use >= N backticks
    (e.g. ```` for 4), and inner fences with fewer backticks are treated
    as literal content.  A closing fence matches only when its backtick
    count equals or exceeds the opening fence's count.

    Args:
        md_text: Full markdown document text.
        heading: Heading text **without** the ``## `` prefix (added automatically).

    Returns:
        The text inside the code fence (or prose content), stripped and
        newline-terminated.

    Raises:
        ValueError: If heading not found, fence is unclosed, or section
            is empty.
    """
    lines = md_text.splitlines()
    heading_line = f"## {heading}".strip()

    start_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == heading_line:
            start_idx = idx
            break

    if start_idx == -1:
        raise ValueError(f"Heading not found in markdown: {heading}")

    # Find the end of this section (next heading of same or higher level).
    section_end = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith(("## ", "# ")):
            section_end = idx
            break

    # Collect all code fences under this heading.
    # Each entry: (line_index, lang_tag, backtick_count)
    fences: list[tuple[int, str, int]] = []
    for idx in range(start_idx + 1, section_end):
        stripped = lines[idx].strip()
        m = _FENCE_OPEN_RE.match(stripped)
        if m and m.group(2):  # must have a language tag (opening fence)
            fences.append((idx, m.group(2).lower(), len(m.group(1))))

    if fences:
        # Prefer the ```text fence (the actual prompt).  Fall back to first.
        fence_start, _, fence_backticks = fences[0]
        for f_idx, f_lang, f_bt in fences:
            if f_lang == "text":
                fence_start = f_idx
                fence_backticks = f_bt
                break

        # Extract content.  Close only when a bare fence line has >= the
        # same number of backticks as the opening fence (CommonMark rule).
        buffer: list[str] = []
        fence_closed = False
        for idx in range(fence_start + 1, len(lines)):
            cm = _FENCE_CLOSE_RE.match(lines[idx].strip())
            if cm and len(cm.group(1)) >= fence_backticks:
                fence_closed = True
                break
            buffer.append(lines[idx])

        if not fence_closed:
            raise ValueError(f"Unclosed code fence under heading: {heading}")

        if not buffer:
            raise ValueError(f"Empty prompt block under heading: {heading}")

        return "\n".join(buffer).strip() + "\n"

    # No code fence found — extract all prose text under the heading.
    prose_lines = lines[start_idx + 1 : section_end]
    prose = "\n".join(prose_lines).strip()

    if not prose:
        raise ValueError(f"Empty section under heading: {heading}")

    return prose + "\n"


_UNSAFE_DIRNAME_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def sanitize_dirname(name: str) -> str:
    return _UNSAFE_DIRNAME_RE.sub("_", name).strip("_") or "unnamed"


_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_\.\-]+)\s*\}\}")


def render_template(text: str, variables: dict[str, str]) -> str:
    """Replace ``{{ variable }}`` placeholders in *text* with values from *variables*.

    Unknown variables are left as-is.  Supported variables:
    ``workspace_root``, ``plan_name``, ``task_id``.
    """
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    return _TEMPLATE_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# Structured context extraction (Feature 3 — deterministic, zero LLM cost)
# ---------------------------------------------------------------------------

_GIT_STATUS_FILE_RE = re.compile(r"^\s*[MADRCU?!]{1,2}\s+(.+)$")
_GIT_DIFF_FILE_RE = re.compile(r"^(?:\+\+\+|---)\s+[ab]/(.+)$")
_ERROR_RE = re.compile(r"(?:error|exception|traceback|fail(?:ed|ure))", re.IGNORECASE)
_WARNING_RE = re.compile(r"(?:warn(?:ing)?)", re.IGNORECASE)

_MAX_LOG_LINES = 500
_MAX_FILES = 100
_MAX_ERRORS = 50
_MAX_WARNINGS = 50
_MAX_DECISIONS = 10
_RESULT_TEXT_CAP = 500


def extract_structured_context(
    log_path: Path,
    task_id: str,
    status: str,
    exit_code: int | None,
    duration_sec: float,
    cost_usd: float | None,
) -> StructuredContext:
    """Extract structured information from a task's log file.

    Parses JSON output lines from engine tasks (Claude/Codex) and scans
    for git status/diff patterns, error lines, and warning lines.
    This is deterministic (regex + JSON parsing) with zero extra cost.
    """
    files_changed: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    result_text = ""

    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return StructuredContext(
            task_id=task_id,
            status=status,
            exit_code=exit_code,
            duration_sec=duration_sec,
            cost_usd=cost_usd,
        )

    lines = raw.splitlines()[-_MAX_LOG_LINES:]

    for line in lines:
        stripped = line.strip()

        # Try to parse JSON lines for engine result
        if '"type"' in stripped and '"result"' in stripped:
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and obj.get("type") == "result":
                    result_text = str(obj.get("result", ""))[:_RESULT_TEXT_CAP]
            except ValueError:
                pass

        # Git status files
        m = _GIT_STATUS_FILE_RE.match(stripped)
        if m:
            files_changed.append(m.group(1).strip())
            continue

        # Git diff files
        m = _GIT_DIFF_FILE_RE.match(stripped)
        if m:
            files_changed.append(m.group(1).strip())
            continue

        # Error lines (only from stderr or lines containing error patterns)
        if stripped.startswith("[stderr]") and _ERROR_RE.search(stripped):
            errors.append(stripped.removeprefix("[stderr]").strip())
        elif not stripped.startswith("[stderr]") and _ERROR_RE.search(stripped):
            # Only capture if it looks like an actual error message, not just
            # code containing the word "error"
            lower = stripped.lower()
            if any(lower.startswith(p) for p in ("error", "traceback", "exception", "fatal")):
                errors.append(stripped)

        # Warning lines
        if stripped.startswith("[stderr]") and _WARNING_RE.search(stripped):
            warnings.append(stripped.removeprefix("[stderr]").strip())

    # Deduplicate and cap
    files_changed = list(dict.fromkeys(files_changed))[:_MAX_FILES]
    errors = list(dict.fromkeys(errors))[:_MAX_ERRORS]
    warnings = list(dict.fromkeys(warnings))[:_MAX_WARNINGS]

    decisions: list[str] = []
    if result_text:
        decisions.append(result_text[:_RESULT_TEXT_CAP])

    return StructuredContext(
        task_id=task_id,
        status=status,
        exit_code=exit_code,
        duration_sec=duration_sec,
        files_changed=files_changed,
        decisions=decisions[:_MAX_DECISIONS],
        errors=errors,
        warnings=warnings,
        cost_usd=cost_usd,
        result_text=result_text,
    )


def format_structured_context(ctx: StructuredContext) -> str:
    """Format a *StructuredContext* into a human-readable text block."""
    parts: list[str] = [f"## Task: {ctx.task_id} [{ctx.status}] ({ctx.duration_sec:.1f}s)"]

    if ctx.files_changed:
        parts.append(f"\n### Files changed ({len(ctx.files_changed)}):")
        for f in ctx.files_changed:
            parts.append(f"- {f}")

    if ctx.decisions:
        parts.append("\n### Key outcomes:")
        for d in ctx.decisions:
            parts.append(f"- {d}")

    if ctx.errors:
        parts.append(f"\n### Errors ({len(ctx.errors)}):")
        for e in ctx.errors:
            parts.append(f"- {e}")

    if ctx.warnings:
        parts.append(f"\n### Warnings ({len(ctx.warnings)}):")
        for w in ctx.warnings:
            parts.append(f"- {w}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Summarization prompts (Features 1 & 2 — LLM-assisted)
# ---------------------------------------------------------------------------

def build_summarization_prompt(
    task_id: str,
    stdout_tail: str,
    structured: StructuredContext,
) -> str:
    """Build a structured 9-section summarization prompt for LLM calls.

    Uses an ``<analysis>`` scratchpad block that improves summary quality
    (the caller strips it from the final output).  The visible output
    follows a fixed 9-section template so downstream consumers can parse
    it reliably.
    """
    parts: list[str] = [
        "Summarize the following task output.",
        "",
        f"Task ID: {task_id}",
        f"Status: {structured.status}",
        f"Duration: {structured.duration_sec:.1f}s",
    ]

    if structured.files_changed:
        parts.append(f"\nFiles changed ({len(structured.files_changed)}):")
        for f in structured.files_changed[:20]:
            parts.append(f"  - {f}")
        if len(structured.files_changed) > 20:
            parts.append(f"  ... and {len(structured.files_changed) - 20} more")

    if structured.errors:
        parts.append(f"\nErrors ({len(structured.errors)}):")
        for e in structured.errors[:5]:
            parts.append(f"  - {e}")

    if structured.warnings:
        parts.append(f"\nWarnings ({len(structured.warnings)}):")
        for w in structured.warnings[:5]:
            parts.append(f"  - {w}")

    if structured.decisions:
        parts.append(f"\nDecisions ({len(structured.decisions)}):")
        for d in structured.decisions[:5]:
            parts.append(f"  - {d}")

    tail = stdout_tail.strip()
    if tail:
        line_count = tail.count("\n") + 1
        parts.append(f"\nTask output (last {line_count} lines):")
        parts.append(tail)

    parts.append(
        "\nFirst, reason inside an <analysis> block (this will be stripped"
        " from the final output).  Then produce the summary using EXACTLY"
        " the 9-section format below.  Omit a section ONLY if it has no"
        " relevant content.\n"
        "\n<analysis>"
        "\n(Your reasoning here — identify what matters, what is noise,"
        " and what downstream tasks need.)"
        "\n</analysis>\n"
        "\n**1. Primary Request:** What this task was asked to do (1 sentence)."
        "\n**2. Key Technical Concepts:** Domain terms, APIs, patterns involved (1-3 bullet points)."
        "\n**3. Files and Code:** Files created, modified, or read — with purpose (1-5 bullet points)."
        "\n**4. Errors and Fixes:** Errors encountered and how they were resolved (bullet points, or 'None')."
        "\n**5. Problem Solving:** Approach taken, alternatives considered (1-2 sentences)."
        "\n**6. Outputs:** Key results, artefacts, or values produced (1-3 bullet points)."
        "\n**7. Pending Issues:** Unresolved problems or TODOs (bullet points, or 'None')."
        "\n**8. Current State:** What the task left behind — final state of files/system (1-2 sentences)."
        "\n**9. Next Steps:** What downstream tasks need to know or do (1-2 bullet points)."
        "\n"
        "\nBe specific and concise. Do NOT repeat raw output — synthesize into actionable context."
    )

    return "\n".join(parts)


def build_reduce_prompt(summaries: dict[str, str]) -> str:
    """Build the reduce-phase prompt that synthesizes individual task summaries.

    Used in ``context_mode: map_reduce`` to combine per-task summaries
    into a single coherent analysis.  Uses the same scratchpad pattern as
    :func:`build_summarization_prompt`.
    """
    parts: list[str] = [
        "Synthesize the following task summaries into a single coherent analysis.",
        "",
    ]

    for tid, summary in summaries.items():
        parts.append(f"### {tid}")
        parts.append(summary)
        parts.append("")

    parts.append(
        "First, reason inside an <analysis> block (this will be stripped).\n"
        "\n<analysis>"
        "\n(Identify cross-cutting concerns, contradictions, and gaps.)"
        "\n</analysis>\n"
        "\nThen produce the synthesis using these sections:"
        "\n**Progress:** Overall completion status (1-2 sentences)."
        "\n**Cross-cutting concerns:** Issues that span multiple tasks (bullet points)."
        "\n**Errors and blockers:** Unresolved problems (bullet points, or 'None')."
        "\n**Key outputs:** Artefacts, files, or values produced (bullet points)."
        "\n**Verdict:** All tasks OK / issues found (1 sentence)."
        "\n"
        "\nBe specific and actionable."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Conditional execution (when expressions)
# ---------------------------------------------------------------------------

_WHEN_RE = re.compile(r"^(.+?)\s*(==|!=)\s*(.+)$")


def evaluate_when_condition(
    expression: str,
    variables: dict[str, str],
) -> tuple[bool, str]:
    """Evaluate a ``when`` condition expression.

    Supports ``{{ var }} == value`` and ``{{ var }} != value`` comparisons.
    Template variables are resolved before evaluation.

    Args:
        expression: The when expression (e.g. ``{{ task.status }} == success``).
        variables: Template variables available for rendering.

    Returns:
        A ``(result, rendered)`` tuple where *result* is the boolean outcome
        and *rendered* is the expression after template substitution.

    Raises:
        ValueError: If the rendered expression doesn't match ``<left> <op> <right>``.
    """
    rendered = render_template(expression, variables)
    m = _WHEN_RE.match(rendered.strip())
    if not m:
        raise ValueError(f"Invalid when expression: {rendered!r}")

    left = m.group(1).strip()
    op = m.group(2)
    right = m.group(3).strip()

    if op == "==":
        return left == right, rendered
    # op == "!="
    return left != right, rendered


def format_duration(duration_sec: float | None) -> str:
    """Format a duration in seconds to a human-readable string."""
    if duration_sec is None:
        return "--"
    if duration_sec < 60:
        return f"{duration_sec:.1f}s"
    minutes, seconds = divmod(int(duration_sec), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_cost(cost_usd: float | None) -> str:
    """Format a cost in USD to a human-readable string."""
    if cost_usd is None:
        return "--"
    return f"${cost_usd:.2f}"


_HUMANIZE_MAX_LEN = 200


def _truncate(text: str, max_len: int = _HUMANIZE_MAX_LEN) -> str:
    """Truncate text with ellipsis if it exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def humanize_output_line(line: str, max_len: int = _HUMANIZE_MAX_LEN) -> str:
    """Extract human-readable text from engine JSON output lines.

    Codex ``--json`` emits structured events (``item.completed``,
    ``item.started``, etc.) with nested payloads.  This helper pulls the
    useful text out so the TUI/live display shows something readable instead
    of raw JSON.  Non-JSON lines are returned unchanged.
    """
    stripped = line.strip()
    if not stripped.startswith("{"):
        return line
    try:
        obj = json.loads(stripped)
    except ValueError:
        return line
    if not isinstance(obj, dict):
        return line  # pragma: no cover

    # --- Codex JSON events ---
    item = obj.get("item") if isinstance(obj.get("item"), dict) else None
    event_type = obj.get("type", "")
    if item:
        item_type = item.get("type", "")
        if item_type == "agent_message":
            text = item.get("text", "")
            if text:
                first = text.split("\n")[0].strip().replace("**", "")
                return _truncate(first, max_len)
        elif item_type == "command_execution":
            cmd = item.get("command", "")
            if isinstance(cmd, str) and cmd:
                normalized = cmd.replace("\\\\", "/").replace("\\", "/")
                try:
                    parts = shlex.split(normalized, posix=True)
                except ValueError:
                    parts = normalized.split()
                if parts:
                    binary = parts[0].rsplit("/", 1)[-1]
                    rest = " ".join(parts[1:3])
                    label = f"{binary} {rest}".strip()
                    if event_type == "item.started":
                        return f"$ {_truncate(label, 100)}"
                    return f"cmd done: {_truncate(label, 100)}"
            if event_type == "item.started":
                return "running command..."
            return "command completed"
        elif item_type == "reasoning":
            text = item.get("text", "")
            if text:
                first = text.split("\n")[0].strip()
                return f"thinking: {_truncate(first, 100)}"

    # --- Claude stream-json events ---
    if event_type in ("system", "user", "assistant", "result"):
        if event_type == "result":
            result_text = obj.get("result", "")
            if isinstance(result_text, str) and result_text.strip():
                first = result_text.strip().split("\n")[0].strip()
                return _truncate(first, max_len)
            return ""
        if event_type == "system":
            return ""
        if event_type == "assistant":
            msg = obj.get("message", {})
            if isinstance(msg, dict):
                for part in msg.get("content", []):
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            if text and text.strip():
                                first = text.strip().split("\n")[0].strip()
                                return _truncate(first, max_len)
                        elif part.get("type") == "tool_use":
                            return f"tool: {part.get('name', 'unknown')}"
            return ""
        if event_type == "user":
            return ""

    # --- Noise events: suppress ---
    _SUPPRESS_EVENTS = {
        "rate_limit_event", "rate.limit", "rate_limit",
        "response.completed", "turn.completed",
        "response.started", "turn.started",
        "item.started",  # codex internal lifecycle
    }
    if event_type in _SUPPRESS_EVENTS:
        return ""

    # --- Generic fallback for known event types ---
    if event_type:
        return str(event_type).replace(".", " ").replace("_", " ")

    return line
