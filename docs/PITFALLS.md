# Maestro CLI -- Pitfalls & Gotchas

Common mistakes and their solutions, collected from real usage. Covers plan authoring, cross-platform issues, Textual TUI gotchas, and engine integration.

See also: [PLAYBOOK.md](PLAYBOOK.md) for curated recipes that avoid these pitfalls.

## How to surface these in your plan

Run `maestro check <plan.yaml>` — it executes `validate` + `audit` in one pass and exits non-zero on any audit error. Add `--strict` to also fail on validation warnings (for CI gates), `--with-suggest` to include optimization hints from prior runs, or `--json` for a structured report. This is the canonical entry point as of 2026-04-27 — running `validate` and `audit` separately is no longer recommended.

For first-run plans, `maestro scaffold <brief.yaml> --strict-defaults` injects sane baselines (`timeout_sec=1500`, `retry_delay_sec=[60, 120]`, `max_cost_usd=10.0`, `budget_warning_pct=0.8`) that pre-empt several of the warnings catalogued below.

---

## P1. Windows Shell Execution

> **Maestro Prevention: 90%** — Validation warns when string commands use `shell=True` on Windows and detects `Git\usr\bin\bash.exe` (wrong bash) in list commands, pre_command, verify_command, and guard_command. Also detects bash-only syntax (heredocs, process substitution) in string commands, and warns when pipes (`|`) appear in string commands.

### Problem: `shell: true` invokes `cmd.exe`, not Bash

On Windows, `shell=True` in Python's `subprocess` uses `cmd.exe`.
Writing `bash -c "..."` with `shell: true` often resolves to WSL's bash (if installed), not Git Bash.

### Solution: Use list-format commands pointing to Git Bash

```yaml
# CORRECT -- bypasses cmd.exe entirely (shell=False for list commands)
command:
  - "C:\\Program Files\\Git\\bin\\bash.exe"
  - "-lc"
  - |-
    cd /c/xampp/htdocs/myproject &&
    echo "Running in Git Bash"
```

```yaml
# WRONG -- hits cmd.exe, then maybe WSL bash
command: bash -c "echo hello"
shell: true
```

```yaml
# WRONG -- raw MSYS2 binary, no PATH setup (mkdir/cp/wc not found, exit 127)
command:
  - "C:\\Program Files\\Git\\usr\\bin\\bash.exe"
  - "-c"
  - "echo hello"
```

**Key**: Use `Git\bin\bash.exe` (NOT `Git\usr\bin\bash.exe`).

---

## P2. Markdown Prompt Extraction

> **Maestro Prevention: 98%** — `extract_prompt_from_markdown()` validates exact heading match, prefers code-fenced content (` ```text `), falls back to prose text when no fence is found. `validate_plan()` enforces `prompt_md_file` + `prompt_md_heading` pairing. Validation warns when `prompt_md_heading` starts with `#` (common mistake since the loader prepends `## ` automatically).

### Problem: `prompt_md_heading` not found

The `extract_prompt_from_markdown()` function in `utils.py` has specific requirements:

### Rule 1: Do NOT include `##` in the heading

The loader **prepends** `## ` automatically. If your YAML has `"## My Heading"`, it becomes `"## ## My Heading"` -- which won't match.

```yaml
# CORRECT
prompt_md_heading: "My Task Heading"

# WRONG -- becomes "## ## My Task Heading"
prompt_md_heading: "## My Task Heading"
```

### Rule 2: Code fences are preferred but not required

The extractor prefers content inside a ` ```text ` (or any ` ``` `) code fence. If no fence is found, **all prose text** under the heading is extracted instead (up to the next heading of same or higher level).

```markdown
## My Task Heading

```text
This is the actual prompt content that gets extracted.
It can be multiple lines.
All of this becomes the prompt.
` ` `
```

```markdown
## My Task Heading

This plain prose text IS extracted as the prompt.
Multiple paragraphs work fine.
```

### Rule 3: Prefer ` ```text ` over other fence types

If multiple code fences exist under a heading, the extractor prefers ` ```text `.
If no ` ```text ` fence is found, it falls back to the **first** fence.

### Rule 4: Code fences must be closed

An unclosed fence raises: "Unclosed code fence under heading: ..."

### Rule 5: Heading match is exact (after `## ` prepend)

The heading in the YAML must match the markdown heading **exactly** (case-sensitive, whitespace-sensitive).

---

## P3. Unicode in YAML

> **Maestro Prevention: 55%** — Validation warns when `prompt_md_heading` contains non-ASCII characters (em-dashes, arrows, curly quotes). Warning suggests ASCII equivalents.

### Problem: Special characters cause encoding mismatches

Unicode characters like em-dashes (`--`), arrows (`->`), and other non-ASCII symbols can cause:
- Heading mismatch between YAML `prompt_md_heading` and `.md` file headings
- YAML parsing errors on some systems

### Solution: Use ASCII equivalents

| Character | Avoid | Use Instead |
|-----------|-------|-------------|
| Em-dash | `--` (U+2014) | `--` |
| Arrow | `->` (U+2192) | `->` |
| Ellipsis | `...` (U+2026) | `...` |
| Curly quotes | `""` | `""` |

```yaml
# CORRECT
prompt_md_heading: "W2: Overlays -- Drawer, Loading"

# RISKY -- em-dash may cause encoding mismatch
prompt_md_heading: "W2: Overlays -- Drawer, Loading"
```

---

## P4. Path Formatting

> **Maestro Prevention: 75%** — Validation warns when path fields (`workspace_root`, `run_dir`, `workdir`, `prompt_file`, `prompt_md_file`) contain backslashes. `resolve_path()` also normalizes at runtime.

### Problem: Backslashes in YAML paths

YAML interprets `\` as escape characters. Use forward slashes or double backslashes.

```yaml
# CORRECT -- forward slashes everywhere (silences W4)
prompt_md_file: "C:/xampp/htdocs/app/docs/prompts.md"
workspace_root: C:/xampp/htdocs/app
workdir: "C:/xampp/htdocs/app"

# WORKS BUT TRIGGERS W4 -- escaped backslashes parse but the validator
# still flags backslashes in path fields regardless of escaping
workdir: "C:\\xampp\\htdocs\\app"

# WRONG -- unescaped backslashes (YAML rejects \x as a bad escape)
prompt_md_file: "C:\xampp\htdocs\app\docs\prompts.md"
```

Forward slashes are the simplest, cheapest answer for every path field.
`Path` accepts forward slashes on Windows transparently; `resolve_path()`
normalises at runtime. Reach for double-backslashes only when interpolating
into a non-Path consumer that demands native Windows separators (rare).

**Note**: Inside bash scripts (list-format commands), always use forward slashes:
```yaml
command:
  - "C:\\Program Files\\Git\\bin\\bash.exe"
  - "-c"
  - |-
    cd /c/xampp/htdocs/app       # forward slashes inside bash
    /c/xampp/mysql/bin/mysql -u root  # full path for mysql
```

---

## P5. MySQL/MariaDB on XAMPP

> **Maestro Prevention: 0%** — External tool configuration, impossible to prevent programmatically. Only documentation helps.

### Problem: `mysql` command not found

XAMPP's MySQL binaries are NOT on the system PATH by default.

### Solution: Use full path in bash scripts

```yaml
command:
  - "C:\\Program Files\\Git\\bin\\bash.exe"
  - "-c"
  - |-
    /c/xampp/mysql/bin/mysql -u root -e "SELECT 1"
    /c/xampp/mysql/bin/mysqldump -u root mydb > backup.sql
```

---

## P6. Task Dependencies & fail_fast

> **Maestro Prevention: 85%** — Full validation: cycle detection (DFS), invalid refs, self-dependencies, `context_from` must be in `depends_on`. `allow_failure` and `fail_fast` well documented. The 15% gap is behavioural surprise from `fail_fast: true` killing independent branches.

### Problem: Tasks skipped unexpectedly

When `fail_fast: true` (default), a single task failure causes ALL pending/unstarted tasks to be skipped -- even tasks in parallel branches with no dependency on the failed task.

### Solution: Use `allow_failure: true` for non-critical tasks

```yaml
tasks:
  - id: audit-grep
    description: "Search for leftover references (informational)"
    allow_failure: true  # won't block other tasks
    command: "grep -r 'old_name' . || echo 'Clean'"

  - id: critical-build
    description: "This must succeed"
    command: "npm run build"
```

Or use `fail_fast: false` at plan level if you want maximum independence.

---

## P7. Dry Run vs Real Run Differences

> **Maestro Prevention: 55%** — Dry-run clearly separated from real run. After dry-run, Maestro prints a checklist of items NOT validated (CLI tools, workdir, network, git state).

### Problem: Dry run passes but real run fails

The dry run validates:
- Plan YAML structure
- Dependency graph (no cycles)
- Prompt extraction (heading + code fence)
- Command construction

The dry run does **NOT** validate:
- Whether `codex`/`claude` CLI tools are on PATH
- Whether `workdir` directory exists (only checked at execution time)
- Whether shell commands would succeed
- Network availability
- Git worktree state

### Solution: After dry-run passes, do a quick sanity check

```powershell
# Verify CLIs are available
claude --version
codex --version

# Verify workdir exists
ls C:\xampp\htdocs\app
```

---

## P8. Timeout Behavior

> **Maestro Prevention: 75%** — `timeout_sec` type-validated in loader. Validation warns when tasks have no explicit `timeout_sec` and no `defaults.timeout_sec` (falling back to hardcoded 1800s default). Timed-out tasks get `exit_code: 124`. The retry-design check is now folded into the unified W20 (see Pitfall 13) — W20 fires only when retries lack any escape valve at all, not just because timeout + retries co-exist.

### Problem: Tasks silently killed after default timeout

If no `timeout_sec` is set at task or plan level, the default is **1800 seconds (30 minutes)**.
Timed-out tasks get `exit_code: 124`.

### Solution: Set explicit timeouts

```yaml
defaults:
  timeout_sec: 600  # 10 min default

tasks:
  - id: quick-task
    timeout_sec: 60   # 1 min for simple tasks

  - id: complex-build
    timeout_sec: 1800  # 30 min for heavy work
```

---

## P9. Environment Variable Isolation

> **Maestro Prevention: 80%** — Robust `_ENV_ALLOWLIST` system, `defaults.env` and `task.env` well documented. Clean environment is deterministic. Validation now detects `$VAR` and `${VAR}` references in string commands and warns when the variable is not in the env allowlist or task/plan env.

### Problem: Task can't see my custom env vars

Maestro builds a **clean environment** from an allowlist. Only these system vars are inherited:

```
# System / shell
PATH, HOME, USER, LOGNAME, SHELL, LANG, LC_ALL, TERM

# Windows
USERPROFILE, SYSTEMROOT, SYSTEMDRIVE, COMSPEC, PATHEXT,
TEMP, TMP, APPDATA, LOCALAPPDATA, PROGRAMFILES,
PROGRAMFILES(X86), WINDIR, HOMEDRIVE, HOMEPATH

# Python
PYTHONUTF8, PYTHONIOENCODING

# Engine auth
GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_APPLICATION_CREDENTIALS,
GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI,
COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN,
COPILOT_MODEL, COPILOT_ALLOW_ALL,
DASHSCOPE_API_KEY,
OLLAMA_HOST, LLAMA_MODEL_DIR
```

All other vars must be explicitly set in the plan.

### Solution: Use `env` at plan or task level

```yaml
defaults:
  env:
    PYTHONUTF8: "1"
    MY_API_KEY: "xxx"

tasks:
  - id: deploy
    env:
      DEPLOY_TARGET: "production"
    command: "deploy.sh"
```

---

## P10. Resume After Failure

> **Maestro Prevention: 90%** — `--resume` and `--only` fully implemented. Clear CLI docs. Skip logic for already-succeeded tasks works reliably.

### Problem: Plan failed at task 8 of 12, don't want to re-run 1-7

### Solution: Use `--resume` or `--resume-last`

```powershell
# First run fails
maestro run plan.yaml --execution-profile yolo
# Output: manifest at .maestro-runs/20260221_120000_.../

# Resume from where it left off (skip succeeded tasks)
maestro run plan.yaml --execution-profile yolo --resume .maestro-runs/20260221_120000_.../

# Or let Maestro find the most recent run automatically
maestro run plan.yaml --execution-profile yolo --resume-last
```

Or use `--only` to run specific tasks:

```powershell
# Only run the failed task and its dependents
maestro run plan.yaml --only w2-overlays --execution-profile yolo
```

---

## P11. Recursive Context Pipeline

> **Maestro Prevention: 85%** — E021 validation catches `context_mode: recursive` without `workspace_root`. Built-in `_DEFAULT_EXCLUDES` prevent indexing noise. Max 5000 files cap prevents monorepo blowout. All pipeline errors (E108-E110) are caught gracefully with fallback to empty context.

### Problem: `context_mode: recursive` requires `workspace_root`

The recursive context pipeline needs to know which directory to index. Without `workspace_root` (at plan level or via `workdir`), validation fails with E021.

```yaml
# WRONG -- E021: no workspace_root
tasks:
  - id: implement
    engine: claude
    context_mode: recursive
    prompt: "Implement feature"

# CORRECT -- workspace_root is set
workspace_root: "C:/py/my-project"
tasks:
  - id: implement
    engine: claude
    context_mode: recursive
    prompt: "{{ workspace_brief }} Implement feature"
```

### Problem: Large workspaces cause slow indexing

The first index build reads and hashes files (first 8KB each). For large workspaces, use `workspace_index_exclude` to skip irrelevant directories.

```yaml
defaults:
  workspace_index_exclude:
    - "node_modules/**"
    - "dist/**"
    - "*.generated.*"

tasks:
  - id: implement
    engine: claude
    context_mode: recursive
    workspace_index_exclude:      # Additional task-level excludes
      - "tests/fixtures/**"
    prompt: "{{ workspace_brief }} Implement feature"
```

### Problem: Stale workspace index

The workspace index is cached in `.maestro-cache/index/` and validated via stat-only quick scan (file sizes + mtimes). If file contents change but size and mtime remain the same (rare edge case), the cache may be stale. Delete `.maestro-cache/index/` to force a rebuild.

### Problem: `{{ workspace_brief }}` is empty

If the recursive pipeline fails (e.g., haiku call errors), the pipeline returns empty context gracefully (E108/E109/E110 logged as warnings). The task still executes, but without workspace context. Check the run's `events.jsonl` for error details.

---

## P12. Heredoc / Bash-only Syntax in verify_command

> **Maestro Prevention: 80%** -- Validation warns about string commands using `shell=True` on Windows and now detects bash-only syntax (heredocs `<<`, process substitution `<()` / `>()`) and pipes (`|`) in string commands, pre_command, verify_command, and guard_command.

### Problem: Heredoc `<< 'EOF'` fails on Windows cmd.exe

String-format `verify_command` (and `command`, `pre_command`) runs with `shell=True`, which invokes `cmd.exe` on Windows. Bash-only syntax like heredocs (`<< 'PYEOF'`), pipes (`|`), and process substitution (`<()`) will fail with cryptic errors like `<< was unexpected at this time`.

```yaml
# WRONG -- heredoc is bash-only, cmd.exe can't parse it
verify_command: |
  cd C:/py/project && py << 'PYEOF'
  import sys
  print("hello")
  PYEOF
```

### Solution: Use list-format with `py -c`

List-format commands bypass the shell entirely (`shell=False`), so they work on all platforms.

```yaml
# CORRECT -- list format, no shell involved
verify_command:
  - "py"
  - "-c"
  - |
    import sys
    print("hello")
```

For commands that need `pytest` or other modules:
```yaml
verify_command:
  - "py"
  - "-m"
  - "pytest"
  - "tests/test_module.py"
  - "-v"
  - "--tb=short"
```

### Problem: Pipes `|` in string commands

Pipes (`cmd1 | cmd2`) also require a shell. On Windows, `cmd.exe` handles pipes differently from bash. Avoid them in verify_commands.

```yaml
# WRONG -- pipe needs shell, behaviour differs on Windows
verify_command: "py -m pytest tests/ -q 2>&1 | tail -5"

# CORRECT -- just run pytest directly, skip the tail
verify_command:
  - "py"
  - "-m"
  - "pytest"
  - "tests/"
  - "-q"
```

---

## P13. Retry Anti-Patterns

> **Maestro Prevention: 90%** — One unified warning (W20) fires when retries have no escape valve. The legacy W21 (timeout + retries) and W-timeout-retry-futility were folded into W20 on 2026-04-26 after an internal post-mortem showed they contradicted each other.

### verify_command without retries (still its own warning)

Setting `verify_command` without `max_retries` means a verify failure immediately fails the task with no opportunity to self-correct. This is usually unintentional — the whole point of verify is to gate retry.

```yaml
# WRONG -- verify fails immediately, no retry possible
- id: implement
  engine: claude
  verify_command: ["py", "-m", "pytest", "tests/"]
  max_retries: 0   # default

# CORRECT -- give the agent a chance to fix failures
- id: implement
  engine: claude
  verify_command: ["py", "-m", "pytest", "tests/"]
  max_retries: 2
```

### judge retry without max_iterations

When `judge.on_fail: retry`, each judge failure triggers a retry. Without `max_iterations`, this can spiral indefinitely if the agent keeps producing output that barely misses the threshold.

```yaml
# WRONG -- no hard cap, potential infinite retry spiral
- id: implement
  engine: claude
  judge:
    criteria: ["All tests pass", "No regressions"]
    on_fail: retry
  max_retries: 3

# CORRECT -- hard cap on total attempts
- id: implement
  engine: claude
  judge:
    criteria: ["All tests pass", "No regressions"]
    on_fail: retry
  max_retries: 3
  max_iterations: 5
```

### W20: retries without an escape valve

A retry only helps if **something differs between attempts**. If `max_retries > 0` but the task has no feedback signal, no progressive backoff, no model escalation, and no fallback engine, retries reproduce identical conditions and likely fail the same way.

W20 fires when **none** of the following escape valves are present:

- `verify_command` / `guard_command` / `assert` / `judge` — retries get feedback
- `escalation: [haiku, sonnet, opus]` — next retry uses a stronger model (engine tasks only)
- `fallback_engine: codex` — engine-level swap on infra failures (engine tasks only)
- `retry_delay_sec: [60, 120]` — list form signals progressive backoff intent (helps with rate limits / transient errors); a positive scalar also counts

```yaml
# WRONG -- W20 fires: 2 retries with nothing to differentiate them
- id: heavy-build
  engine: claude
  timeout_sec: 120
  max_retries: 2

# CORRECT (option A) -- verify_command supplies retry feedback
- id: heavy-build
  engine: claude
  timeout_sec: 600
  max_retries: 1
  verify_command: ["py", "-m", "pytest", "tests/"]

# CORRECT (option B) -- progressive backoff for rate-limit recovery
- id: heavy-build
  engine: claude
  timeout_sec: 600
  max_retries: 2
  retry_delay_sec: [60, 120]

# CORRECT (option C) -- escalate model on retry
- id: heavy-build
  engine: claude
  timeout_sec: 600
  max_retries: 2
  escalation: [sonnet, opus]

# ALSO CORRECT -- best-effort first-run, accept that retries don't help
- id: heavy-build
  engine: claude
  timeout_sec: 600
  max_retries: 0
```

W20 is a **design choice** signal, not a bug. If your task is one-shot best-effort, `max_retries: 0` is the cheapest valid response and silences the warning.

---

## P14. Context Budget Gaps

> **Maestro Prevention: 85%** — Validation warns when context_from is set without context_budget_tokens, preventing uncontrolled context costs.

### Problem: `context_from` without `context_budget_tokens`

When upstream tasks produce large outputs (long logs, full file contents), injecting all of it into the next task's prompt can silently exceed the model's context window or inflate costs dramatically.

```yaml
# RISKY -- no budget cap, upstream log could be 100K+ tokens
- id: review
  engine: claude
  context_from: [long-build-task]

# CORRECT -- cap the upstream context
- id: review
  engine: claude
  context_from: [long-build-task]
  context_budget_tokens: 8000
```

You can also set a plan-level default to cover all tasks at once:

```yaml
defaults:
  context_budget_tokens: 8000  # applies to all tasks unless overridden
```

---

## P15. Judge Typed Assertions on Engine Tasks

> **Maestro Prevention: 95%** — Validation warns when judge typed assertions (`contains`, `regex`) are used on ANY engine task, since judge evaluates engine stdout (JSON output), not file contents. Use `verify_command` + `guard_command` for deterministic checks.

### Problem: `type: contains` / `type: regex` check engine stdout, not generated files

This applies to **ALL engines** (Claude, Codex, Gemini, Copilot, Qwen, Ollama, Llama) — not just Codex. The judge receives `stdout_tail` from the engine command, which is always JSON metadata or conversational output, never the raw file contents the agent edited. A `contains` assertion like `"contains 'def my_function'"` will fail because that string appears in the generated file, not in the engine's stdout.

```yaml
# WRONG -- 'contains' checks engine stdout (JSON), not the files it edited
- id: implement
  engine: claude   # also fails on codex, gemini, etc.
  judge:
    criteria:
      - type: contains
        value: "def my_function"
      - type: regex
        pattern: "class\\s+MyClass"

# CORRECT -- use verify_command for deterministic file checks
- id: implement
  engine: claude
  verify_command:
    - "py"
    - "-c"
    - |
      code = open('src/mod.py', encoding='utf-8').read()
      assert 'def my_function' in code, 'function not found'
      print('OK')

  # Optional: guard_command receives engine stdout via stdin
  guard_command:
    - "py"
    - "-c"
    - |
      import sys
      code = open('src/mod.py', encoding='utf-8').read()
      assert 'class MyClass' in code, 'class not found'
      print('guard OK')

  # Judge should only use LLM-evaluated criteria
  judge:
    criteria:
      - type: llm-rubric
        value: "Implementation follows project conventions"
      - type: rubric
        name: completeness
        levels:
          - score: 1
            description: "Missing most requirements"
          - score: 5
            description: "All requirements met"
```

### Safe judge criteria for engine tasks

| Criterion type | Safe? | Notes |
|---------------|-------|-------|
| `type: contains` | **NO** | Checks engine stdout, not files |
| `type: regex` | **NO** | Checks engine stdout, not files |
| `type: llm-rubric` | Yes | LLM interprets stdout context qualitatively |
| `type: rubric` (Likert) | Yes | LLM scores on rubric scale |
| `type: is-json` | Yes | Checks if stdout is valid JSON |
| `type: json-schema` | Yes | Validates stdout against an inline `schema:` or `schema_file:`. Pairs naturally with `output_schema:` for typed output validation |
| `type: cost_under` | Yes | Checks task cost metadata |
| `type: duration_under` | Yes | Checks task duration metadata |
| Plain string | Yes | LLM-evaluated, qualitative |

---

## P16. Cross-Platform CI Naming Drift

> **Maestro Prevention: 40%** -- Generated CI examples can already cover multiple platforms, but docs drift still happens if the narrative uses `ubuntu` in one place and `linux` in another.

### Problem: Platform guidance uses inconsistent names

Cross-platform validation and migration docs become brittle when the repo mixes:

- `linux` in one file
- `ubuntu` in another
- `macOS` prose without the exact `macos` token used by generated examples

That inconsistency is easy for humans to gloss over, but it breaks deterministic validation and confuses users reading the examples.

### Solution: Keep the public guidance aligned on the same platform vocabulary

- Use `linux`, `windows`, and `macos` explicitly in user-facing docs that describe generated CI coverage.
- It is fine for provider-specific YAML to still contain runner names such as `ubuntu-latest`, `windows-latest`, and `macos-latest`, but the surrounding docs should still explain the coverage in the normalized `linux/windows/macos` terms.
- For GitLab CI, document that `linux` is the default lane and `windows` plus `macos` are opt-in lanes that need matching tagged runners.

This keeps `verify_command` checks cheap and deterministic while preserving clear platform expectations for users.

---

## P17. Textual 8.x DataTable Is Fundamentally Broken

> **Maestro Prevention: 100%** -- Maestro's DAGPanel uses `Static` + Rich `Table` instead of `DataTable`, avoiding all these issues entirely.

### Problem: DataTable crashes on cursor, row indexing, and reactive properties

Textual 8.x's `DataTable` has cascading bugs that make it unreliable for
real-time updating widgets:

1. **`cursor_type` reactive watcher**: Setting `cursor_type` (even in `__init__`
   or `on_mount`) triggers `watch_cursor_type` → `_scroll_cursor_into_view()` →
   `_get_fixed_offset()` → `self.rows[row_key]` with `row_key=None`, causing
   `KeyError: None`. This happens regardless of when or how cursor_type is set.

2. **`add_row()` triggers `_highlight_cursor()`**: Every `add_row()` call
   triggers cursor highlighting, which hits the same `rows[None]` bug.

3. **`_update_dimensions` on idle**: After rows are added, the idle handler
   calls `_update_dimensions` → `get_row_at(row_index)` with stale indices,
   causing `RowDoesNotExist` errors.

All three bugs cascade from the same root: DataTable assumes a fully-rendered,
stable state that doesn't exist when rows are being added programmatically.

### Solution: Use `Static` + Rich `Table` instead

```python
from rich.table import Table
from textual.widgets import Static

class DAGPanel(Static):
    """Render a Rich Table inside a Static widget — no DataTable bugs."""

    def _render_table(self) -> Table:
        table = Table(box=None, show_header=True, expand=True)
        table.add_column("", width=4)
        table.add_column("Task", ratio=3)
        # ... add columns ...
        for task in self._plan.tasks:
            table.add_row(...)  # build from internal state dict
        return table

    def _refresh_table(self) -> None:
        self.update(self._render_table())  # full re-render, always safe
```

This approach re-renders the entire table on each update. For tables under ~100
rows this is instant and avoids all DataTable internal state issues.

---

## P18. Textual Worker Thread Race Condition

> **Maestro Prevention: 100%** -- Maestro's `Static`-based DAGPanel accepts updates at any time (before or after mount), since `Static.update()` is always safe to call.

### Problem: Worker thread events arrive before widget `on_mount()`

When using `run_worker(thread=True)` to run a background task that fires events
via `call_from_thread()`, those events can arrive at widgets before `on_mount()`
has finished. With `DataTable`, this causes crashes. With `Static`, the update is
silently buffered.

The race is real in production — not just a test timing issue. The worker thread
starts immediately, and `call_from_thread()` posts messages to the Textual event
loop, which may process them before all widgets have mounted.

### Solution A (preferred): Use `Static` + state dict

Store widget state in a plain dict and re-render on each update. `Static.update()`
works at any point in the widget lifecycle — before mount, during mount, or after.

```python
class DAGPanel(Static):
    def __init__(self, plan: PlanSpec) -> None:
        super().__init__()
        self._states: dict[str, TaskState] = {
            task.id: TaskState(task_id=task.id) for task in plan.tasks
        }

    def update_task_start(self, payload: dict[str, object]) -> None:
        state = self._states.get(str(payload.get("task_id", "")))
        if state:
            state.status = "running"
            self.update(self._render_table())  # always safe
```

### Solution B (if using DataTable): Event queue with replay

```python
class MyTable(DataTable):
    def __init__(self) -> None:
        super().__init__()
        self._mounted = False
        self._pending: list[tuple[str, dict]] = []

    def on_mount(self) -> None:
        # ... add columns and rows ...
        self._mounted = True
        for event_name, payload in self._pending:
            self._apply(event_name, payload)
        self._pending.clear()

    def handle_event(self, name: str, payload: dict) -> None:
        if not self._mounted:
            self._pending.append((name, payload))
        else:
            self._apply(name, payload)
```

Solution A is simpler and avoids all DataTable bugs. Solution B works but still
requires workarounds for the cursor/row-indexing issues described in Pitfall #17.

---

## P19. CLAUDECODE Env Var Blocks Engine Tasks

> **Maestro Prevention: 90%** — `_preflight_checks()` detects `CLAUDECODE` env var and warns when the plan uses `engine: claude`. `_build_safe_env()` strips it from subprocess environments (not in allowlist). Stderr surfacing now shows the actual error message in `TaskResult.message`.

### Problem: Silent engine task failures inside Claude Code

When `maestro run` is executed from a terminal spawned by Claude Code IDE extension, the `CLAUDECODE` environment variable is set. Claude CLI detects this and refuses to start nested sessions:

```
Error: Claude Code cannot be launched inside another Claude Code session.
Nested sessions share runtime resources and will crash all active sessions.
```

All `engine: claude` tasks fail instantly with exit code 1 and **no stdout output**. Before stderr surfacing, the only diagnostic was `(no log output)` in the TUI.

### Solution: Run from a clean terminal

```powershell
# Option A: Use a fresh terminal (not spawned by Claude Code)
maestro run plan.yaml

# Option B: Unset the variable in the terminal you launch maestro from.
# Choose the line for your shell:
#   cmd.exe       :  set CLAUDECODE=
#   PowerShell    :  $env:CLAUDECODE = ""        (or Remove-Item Env:CLAUDECODE)
#   bash / zsh    :  unset CLAUDECODE
maestro run plan.yaml
```

Maestro now shows a preflight warning and includes stderr in failure messages for diagnosis.

---

## P20. UNC Paths Break verify_command on Windows

> **Maestro Prevention: 80%** — `_preflight_checks()` detects UNC paths in task workdirs on Windows and warns. Validation warns about string-format commands using `shell=True`.

### Problem: CMD.EXE rejects UNC paths as working directory

When `workspace_root` points to a network-mounted drive that resolves to a UNC path (`\\SERVER\SHARE\...`), `CMD.EXE` (used for string-format commands with `shell=True`) fails:

```
'\\SERVER\SHARE\project'
CMD.EXE was started with the above path as the current directory.
UNC paths are not supported. Defaulting to Windows directory.
```

This causes `verify_command`, `pre_command`, and `guard_command` (string format) to run in the wrong directory, leading to false failures.

### Solution: Map the UNC path to a drive letter

```powershell
# Map to a drive letter
net use S: \\SERVER\SHARE

# Or use list-format commands (bypasses CMD.EXE)
```

```yaml
# Use a mapped drive letter
workspace_root: S:/project

# Or use list-format commands with Git Bash
verify_command:
  - "C:\\Program Files\\Git\\bin\\bash.exe"
  - "-lc"
  - "cd /s/project && python -m pytest tests/"
```

---

## P21. prompt_md_file Relative Path Resolution

> **Maestro Prevention: 95%** — `_resolve_prompt_path()` tries `workspace_root` first, then falls back to `plan.source_dir`. Validation error messages include the searched path.

### Problem: `prompt_md_file` resolved relative to plan directory, not workspace

When a plan file lives outside the workspace (e.g. `C:/projects/myapp/plans/plan.yaml`) and references `prompt_md_file: docs/prompts.md`, the file is searched at `C:/projects/myapp/plans/docs/prompts.md` instead of `{workspace_root}/docs/prompts.md`.

### Solution: Automatic since v1.10.1

Maestro now resolves `prompt_file` and `prompt_md_file` relative to `workspace_root` first (when set), falling back to the plan's directory. No YAML changes needed.

For explicit control, use absolute paths:

```yaml
prompt_md_file: "S:/xampp/htdocs/app/docs/prompts.md"
```

## P22. Judge Timeout on Multi-Call Methods

> **Maestro Prevention: 95%** — Since v1.19.0, `_compute_judge_timeout()` auto-scales the default timeout based on method, criteria count, and quorum. W22 warns when an explicit `timeout_sec` is too low.

### Problem: g_eval / debate judges time out at the default 60s

The default `judge.timeout_sec` of 60s is per-LLM-call, but multi-call methods
make several sequential calls. With the default timeout, individual calls get
killed before finishing -- especially with many criteria (larger prompts):

| Method | LLM Calls | Default 60s Enough? |
|--------|-----------|---------------------|
| `direct` | 1 | Usually yes |
| `g_eval` | 2 (steps + score) | Risky with 5+ criteria |
| `reflection` | 2 (critique + score) | Risky with 5+ criteria |
| `debate` (2 rounds) | 4 (bull + bear x2) | No |
| Any + `quorum: 3` | x3 | Rarely |

Real failure: `g_eval` with 8 criteria timed out at 60s -- the task
itself succeeded but the judge couldn't finish scoring, marking the task as failed.

### Solution: Auto-scaled since v1.19.0

Maestro now computes a sensible default when `timeout_sec` is not set:

- **g_eval**: 120s base + 15s per criterion over 4
- **debate**: 60s x rounds x 2 + criteria scaling
- **quorum**: multiplied by quorum count

```yaml
# Before v1.19.0 -- manual workaround needed:
judge:
  method: g_eval
  criteria: [...]   # 8 criteria
  timeout_sec: 180  # Had to guess

# After v1.19.0 -- auto-scaled to 180s (120 + 4*15):
judge:
  method: g_eval
  criteria: [...]   # 8 criteria
  # timeout_sec omitted -- auto-computed
```

You can still set `timeout_sec` explicitly to override auto-scaling. W22 warns
if your explicit value is below the recommended minimum.

---

## P23. Workflow Library Override -- Field-Level Merge, Not Replace

> **Maestro Prevention: 30%** -- Validation catches type mismatches, but the shallow merge semantics are by design and not flagged as a warning.

### Problem: Brief overrides a library task, but unexpected fields persist

**Symptom**: Brief overrides a library task, but unexpected fields from the library persist.

**Cause**: `_merge_library_into_brief()` does a shallow dict merge -- the brief's fields are applied ON TOP of the library task's fields. Fields not specified in the brief are inherited from the library.

**Example**: Library `rest-api` defines `implement-endpoints` with `task_type: implementation`. Brief overrides only `prompt_hint`. The task keeps `task_type: implementation` from the library, even if you expected to change it.

### Solution: Specify ALL fields you want in the brief

To fully override a library task, specify ALL fields you want in the brief. To add a completely new task, use a new `id` not present in the library.

```yaml
# WRONG -- only overrides prompt_hint, inherits everything else from library
tasks:
  - id: implement-endpoints
    prompt_hint: "Use FastAPI instead of Flask"

# CORRECT -- specify all fields you want to control
tasks:
  - id: implement-endpoints
    task_type: api
    prompt_hint: "Use FastAPI instead of Flask"
    tags: [api, fastapi]
```

**Cost**: Low -- usually caught at validation, not at runtime.

---

## P24. CWE Presets Use `aggregation: min` -- Stricter Than Expected

> **Maestro Prevention: 60%** -- Judge results clearly show per-criterion scores, but the `min` aggregation behaviour may surprise users expecting an average.

### Problem: Task fails judge evaluation even though most criteria scored well

**Symptom**: Task fails judge evaluation even though most criteria scored well.

**Cause**: All 4 CWE presets (`cwe_injection`, `cwe_auth`, `cwe_data_exposure`, `cwe_top_25`) use `aggregation: min`. This means EVERY criterion must individually score above `min_score`. One weak area fails the entire evaluation.

**Example**: `cwe_injection` scores: SQL=0.9, Command=0.9, XSS=0.8, Path Traversal=0.3. Overall: FAIL (min=0.3 < threshold 0.8).

### Solution: Override the aggregation if strict evaluation is too aggressive

```yaml
# Override min with weighted average
judge:
  preset: cwe_injection
  aggregation: weighted_mean
```

Or use `security_audit` preset (uses default `mean` aggregation) for general reviews.

**Cost**: Wasted retries/escalations when one CWE category is weak but others pass.

---

## P25. `output_scope` Glob Matching -- `**` Needed for Nested Directories

> **Maestro Prevention: 50%** -- `scope_violation` events are emitted with full details (expected vs actual paths), but the glob syntax itself is not validated for common mistakes.

### Problem: `scope_violation` fires for files within the intended directory

**Symptom**: `scope_violation` event fires even though the task modified files within the intended directory.

**Cause**: `output_scope: ["src/*.py"]` only matches files directly in `src/`, not in subdirectories. Use `src/**/*.py` for recursive matching.

**Example**: Task modifies `src/auth/login.py`. Scope is `["src/*.py"]`. Violation detected because `src/auth/login.py` doesn't match `src/*.py`.

### Solution: Use `**` for recursive glob matching

```yaml
# WRONG -- only matches src/foo.py, not src/auth/login.py
output_scope: ["src/*.py"]

# CORRECT -- matches all .py files under src/ at any depth
output_scope: ["src/**/*.py", "tests/**/*.py"]
```

**Cost**: False violation events; potential confusion in event logs.

---

## P26. Dual Verification -- Low Overlap Doesn't Mean Failure

> **Maestro Prevention: 70%** -- `worktree_verification` events include full detail (overlap_ratio, unclaimed_files, phantom_files), but interpretation requires understanding agent behaviour.

### Problem: Low `overlap_ratio` in worktree verification event

**Symptom**: `worktree_verification` event shows low `overlap_ratio` but the task was actually correct.

**Cause**: `verify_worktree_output()` compares files mentioned in agent stdout against actual git diff. Agents often don't list every file they touched -- especially auto-generated files, `__init__.py`, or config changes.

**Example**: Agent modifies `src/auth.py` and `src/__init__.py`. stdout mentions only `auth.py`. Overlap ratio: 0.5. `unclaimed_files: ["src/__init__.py"]`.

### Solution: Low overlap is informational, not a failure

Tasks aren't automatically failed by low overlap. Investigate `unclaimed_files` and `phantom_files` in the event payload to understand the discrepancy.

```yaml
# Agents rarely list 100% of modified files
# Common unclaimed files: __init__.py, config changes, auto-generated imports
# Common phantom files: files the agent discussed but didn't actually modify
```

**Cost**: None (informational only), but may cause confusion in event analysis.

---

## P27. Multi-Dimensional Eval -- ALL Dimensions Must Pass

> **Maestro Prevention: 80%** -- `EvalSuiteResult` clearly shows per-dimension pass/fail status, and the AND logic is documented. But users expecting weighted averages across dimensions may be surprised.

### Problem: Eval reports FAIL even though most dimensions passed

**Symptom**: Eval reports FAIL even though most dimensions passed with high scores.

**Cause**: `EvalSuiteResult.overall_pass` uses AND logic -- ALL dimensions must individually pass. There is no weighted average across dimensions.

**Example**: 3 dimensions: correctness (PASS, 0.9), efficiency (PASS, 0.85), security (FAIL, 0.4). Overall: FAIL.

### Solution: This is by design -- run separate eval specs for OR logic

Security can't be compensated by correctness. If you need OR logic, run dimensions as separate eval specs.

```yaml
# Single eval with AND logic (default)
dimensions:
  - name: correctness
    criteria: [...]
  - name: security
    criteria: [...]
# Result: FAIL if ANY dimension fails

# For OR logic: run as separate eval files
# eval_correctness.yaml -- evaluate correctness only
# eval_security.yaml -- evaluate security only
```

**Cost**: Unexpected FAIL verdicts when one dimension is weak.

---

## P28. Engine Generates Code for Wrong Language Version

**Maestro Prevention**: 80% — verify_command catches syntax errors; max_retries auto-fixes

**Symptom**: Parse error on valid-looking code. verify_command catches it but task fails.

**Cause**: The engine (Claude, Codex, etc.) defaults to the latest language version in its training data. If your project uses an older version (PHP 8.2, Python 3.9, Node 16), the engine may generate features from newer versions (PHP 8.3 typed constants, Python 3.10 match/case, Node 20 Array.groupBy).

**Example**: Claude Sonnet generated `public const int MAX_MESSAGE_LENGTH = 20000;` (PHP 8.3 typed constants) in a PHP 8.2 project. verify_command (`php -l`) caught the parse error. With `max_retries: 0`, the task failed. With `fail_fast: false`, 4 downstream tasks were dependency-skipped.

**Solution**:
1. **Specify the language version in the prompt**: "Use PHP 8.2 syntax only. Do NOT use typed constants, readonly classes, or other 8.3+ features."
2. **Set `max_retries: 1`**: The engine receives the parse error output and almost always self-corrects on retry.
3. **Use `pre_command`** to extract version from project config: `pre_command: "php -r \"echo PHP_MAJOR_VERSION.'.'.PHP_MINOR_VERSION;\""` and reference `{{ pre-task.stdout_tail }}` in the prompt.
4. **Add a `reminders` trigger**: `{trigger: "Parse error", message: "Check that generated code is compatible with PHP 8.2. No typed constants, no readonly classes."}`.

**Cost**: $3.79 wasted in real run (16/24 tasks, 4 dependency-skipped). A single retry ($0.50-1.50) would have fixed it automatically.

---

## P29. `file_contains` / `file_not_contains` False Positives on Delegated Logic

**Maestro Prevention**: 0% — assertions are working as designed; the problem is semantic mismatch

**Symptom**: Task fails with assert `file_contains` or `file_not_contains` even though the code correctly implements the requirement. The assertion checks for a literal substring, but the logic was delegated to another layer or the string is used in a different context than expected.

**Cause**: `file_contains` and `file_not_contains` are literal substring checks. They cannot understand:
- **Delegation**: Controller calls `$repo->findFiltered()` which handles the filter internally — the filter keyword doesn't appear in the controller
- **Context**: `storage_path` is used internally to read files from disk (safe) but `file_not_contains` rejects ANY occurrence, including safe internal use
- **Mapping**: View receives pre-mapped data (`$msgType === 'note'`) instead of raw column names (`is_internal`)

**Example** (3 real false positives, $3.25 wasted):
```yaml
# BAD: Checks for literal "is_internal" in controller
# but controller delegates to repository
assert:
  - type: file_contains
    path: src/Controllers/TicketController.php
    pattern: "is_internal"

# BAD: Rejects ANY "storage_path" usage
# but controller needs it internally for file streaming
assert:
  - type: file_not_contains
    path: src/Controllers/TicketApiController.php
    pattern: "storage_path"
```

**Solution**:
1. **Use `file_contains` only for structural checks**: class names, `extends`, `use` statements, interface implementations — things that MUST appear literally
2. **For semantic security checks, use `verify_command`** with a dedicated script:
   ```yaml
   verify_command: |
     php -r "
       \$c = file_get_contents('src/Controllers/TicketApiController.php');
       // Check storage_path not in JSON response methods
       preg_match_all('/(?:jsonResponse|apiOk)\s*\([^)]*storage_path/s', \$c, \$m);
       exit(count(\$m[0]) > 0 ? 1 : 0);
     "
   ```
3. **For delegation patterns, check the delegated layer** instead:
   ```yaml
   assert:
     - type: file_contains
       path: src/Repositories/MessageRepository.php
       pattern: "is_internal"
   ```
4. **Use `llm-rubric` for semantic validation** when substring checks are insufficient:
   ```yaml
   judge:
     criteria:
       - type: llm-rubric
         value: "The controller must filter internal notes. Check if the code calls a method that handles this filtering, even if delegated to a repository."
   ```

**Scenario 4 — Agent follows codebase pattern instead of prompt literal**:
```yaml
assert:
  - type: file_contains
    path: resources/views/backoffice/dashboard/index.php
    pattern: "ob_start"
```
The prompt said "use `ob_start()` / `ob_get_clean()` + require layout". The agent explored the codebase, found that existing views use `head.php`/`foot.php` includes (not `ob_start`), and followed the real pattern. The code is correct and consistent — but the assert fails because the literal `ob_start` is absent.

**Cost**: $3.25+ wasted on 4 false positives + dependency-skipped tasks in a 24-task plan. Correct code was discarded because assertions couldn't understand delegation patterns, internal usage, data mapping, or codebase conventions.

---

## P30. `UnicodeDecodeError` in `verify_command` on Windows

**Severity**: HIGH — cascades via `fail_fast`
**First seen**: v1.25.0 real-engine run (phase-2-ta-backtesting)

### Problem: `open()` without `encoding=` uses cp1252 on Windows

When a `verify_command` script uses `open(path).read()` in Python on Windows, the default codec is `cp1252` (not UTF-8). If the engine generates files with UTF-8 characters (arrows `←`, em-dashes `—`, smart quotes, emoji), the verify script crashes:

```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x90 in position 11719
```

This kills the task, and `fail_fast` cascades to all dependents.

### Solution

**Maestro v1.31.0+ auto-injects `PYTHONUTF8=1`** into the environment on Windows, so `open()` defaults to UTF-8 everywhere. No plan changes needed.

For older versions, or for non-Python verify scripts:

```yaml
# Option 1: explicit encoding in verify_command
verify_command: ["python", "-c", "open('file.tsx', encoding='utf-8').read()"]

# Option 2: set PYTHONUTF8 in task env
env:
  PYTHONUTF8: "1"
```

### Why it matters

Engine-generated code frequently contains UTF-8 characters (Unicode arrows in React, em-dashes in comments, non-ASCII identifiers). This is not a bug in the engine — it's a Windows codec mismatch that the plan author needs to handle.

---

## P31. Codex `reasoning_effort` override from user config

**Severity**: HIGH — silent failure, all tasks fail
**First seen**: v1.25.0 real-engine run (phase-2-ta-backtesting)

### Problem: `~/.codex/config.toml` injects incompatible `reasoning_effort`

When a Maestro plan does **not** set `reasoning_effort` (neither on the task nor in `defaults.codex.reasoning_effort`), the Codex CLI falls back to the user's `~/.codex/config.toml`. If that file has `model_reasoning_effort = "xhigh"` and the task uses `gpt-5-codex-mini` (which only supports `low`/`medium`/`high`), every task fails immediately:

```
Unsupported value: 'xhigh' is not supported with the 'gpt-5-codex-mini' model.
```

### Solution

**Always set `reasoning_effort` explicitly** for codex tasks:

```yaml
defaults:
  codex:
    model: "5-mini"
    reasoning_effort: low    # ← prevents user config override

tasks:
  - id: scaffold
    engine: codex
    # reasoning_effort inherited from defaults — safe
```

**Maestro v1.31.0+ emits W23** when codex tasks lack explicit `reasoning_effort`.

### Why it matters

The Codex CLI has its own config file (`~/.codex/config.toml`) that can override Maestro's settings. Unlike Claude CLI (where effort is per-env-var and Maestro controls it), the Codex CLI merges its config with Maestro's flags — and the config wins for unset values.

---

## P32. Structure-Only Verification Misses Integration Bugs

**Severity**: CRITICAL — all tasks passed, system non-functional
**First seen**: an internal post-mortem on a large multi-task run

### Problem: Per-file keyword checks pass but the system doesn't work

When every `verify_command` follows this pattern:

```python
content = open('path/to/file.ts', encoding='utf-8').read()
assert 'SomeClassName' in content, 'Missing class'
assert 'someMethod' in content, 'Missing method'
print('OK')
```

It answers "did the model create a file with roughly the right shape?" but cannot catch:

| Bug Class | Why Keyword Checks Miss It |
|-----------|---------------------------|
| Interface mismatch (form sends wrong payload shape) | Checks file A and B independently |
| Response wrapper disagreement (`{items: [...]}` vs `[...]`) | Each file is internally consistent |
| Missing DB column (query uses `t.updated_at` but migration lacks it) | Migration check ≠ query check |
| DI bypass (controller accesses `this.db` directly) | "Controller has class" ≠ "follows architecture" |
| Constructor signature mismatch across implementations | Each class file looks correct alone |
| HTTP protocol mismatch (empty POST body rejected) | Only testable at runtime |

**Real impact**: most of the issues in that run were cross-task interface mismatches invisible to single-file verification. Every API endpoint and UI interaction was broken despite all verify_commands passing.

### Solution: Layer your verification

**Level 1 — Compiler checks (near-zero cost):**
```yaml
# After each wave, check that everything compiles together
verify_command: "npx tsc --noEmit --project apps/api/tsconfig.json"
```
This alone catches ~4/9 critical bugs: missing columns, response type mismatches, constructor arity errors, and import failures.

**Level 2 — Architectural grep checks (zero cost):**
```yaml
guard_command:
  - py
  - -c
  - "import sys; c = sys.stdin.read(); assert '.db.' not in c, 'Controller must not access DB directly'; print('OK')"
```

**Level 3 — Cross-file integration checks:**
```yaml
# A dedicated task that checks multiple files together
- id: integration-check
  depends_on: [create-controller, create-dashboard]
  command:
    - py
    - -c
    - |
      import re
      ctrl = open('controller.ts', encoding='utf-8').read()
      page = open('page.tsx', encoding='utf-8').read()
      ctrl_keys = set(re.findall(r'return\s*\{\s*(\w+):', ctrl))
      page_keys = set(re.findall(r'data\.(\w+)', page))
      assert ctrl_keys & page_keys, f'No overlap: controller returns {ctrl_keys}, dashboard reads {page_keys}'
      print('OK')
```

> Since 2026-04-26, SEC016 only fires on `context_mode: raw`. If your downstream consumes the upstream via `summarized` / `map_reduce` / `recursive` / `layered` / `selective` / `structural` / `council` / `knowledge_graph`, the audit treats the LLM-mediation or heuristic extraction as partial injection resistance and exempts the upstream — adding `guard_command` on top is no longer required to silence SEC016.

**Level 4 — Review task as a gate (not a suggestion):**
```yaml
- id: review
  judge:
    on_fail: fail    # NOT warn — broken code must not pass
```

### The golden rule

**If two tasks produce code that must agree on an interface, at least one verify_command must check both files together.** Per-file checks are necessary but never sufficient for multi-file codegen.

See Recipe 8 in [PLAYBOOK.md](PLAYBOOK.md) for a complete pattern.

---

## P33. Review Task with `on_fail: warn` Reports SUCCESS on Broken Code

**Severity**: HIGH — false confidence, broken deliverable
**First seen**: an internal post-mortem on a large multi-task run

### Problem: Review says FAIL, run says SUCCESS

A review task with `on_fail: warn` correctly identified that the generated code was broken (judge score 0.42, threshold 0.7, quorum 0/3 pass). But because `on_fail: warn` means "log it and move on", the overall run reported SUCCESS.

The user sees:
```
[maestro] run complete: all tasks passed  ← GREEN
```

The review log says:
```
judge score: 0.42/1.0 (FAIL)
findings: C1-C6, W1-W9
```

### When to use `on_fail: warn` vs `fail`

| Scenario | `on_fail` | Reason |
|----------|-----------|--------|
| Code review on implementation | `fail` | Broken code should not pass |
| Optional style check | `warn` | Style issues don't block delivery |
| Security audit | `fail` | Security gaps must be fixed |
| Performance benchmark | `warn` | Informational, not blocking |
| Final integration review | `fail` | Last gate before delivery |

### Solution

```yaml
# BAD — review correctly identifies 9 bugs, run reports SUCCESS
- id: review
  judge:
    on_fail: warn    # ← reviewer's findings have no effect

# GOOD — review findings block the run
- id: review
  judge:
    on_fail: fail    # ← broken code fails the run

# BETTER — review findings trigger a fix attempt
- id: review
  judge:
    on_fail: retry   # ← re-runs dependent tasks with review context
  max_retries: 1
  max_iterations: 3
```

### Rule

If you're paying for a review task, make its verdict count. `on_fail: warn` means "I want to know but don't care enough to act." For any plan where correctness matters, use `on_fail: fail` or `on_fail: retry`.

---

## P34. Unknown Assert Fields Silently Invert Logic

**Severity**: MEDIUM — false failure or false pass depending on assertion type
**First seen**: crud-modules plan (2026-03-24)
**Maestro Prevention**: 100% — E018 rejects unknown fields since v1.31.1

### Problem: `negate: true` is not a valid field

A plan author tried to negate a `file_contains` assertion:

```yaml
assert:
  - type: file_contains
    path: views/salespeople/index.php
    pattern: "$brands"
    negate: true    # ← NOT a valid field
```

The `negate` field was silently ignored. The assertion became a positive check for `$brands`. Since the file correctly did NOT contain `$brands`, the positive `file_contains` assertion FAILED — the opposite of the intended behavior.

### Solution

Use the correct negative assertion type instead of inventing fields:

```yaml
# WRONG — negate: true is silently ignored (pre-v1.31.1) or rejected (v1.31.1+)
- type: file_contains
  path: file.php
  pattern: "$brands"
  negate: true

# CORRECT — use the dedicated negative type
- type: file_not_contains
  path: file.php
  pattern: "$brands"
```

### Valid assert fields

All assertion rules accept: `type` (required), type-specific fields (`path`, `pattern`, `glob`, `json_path`, `package`, `count`, `min_count`), and optional metadata (`message`, `severity`, `rule`/`id`, `task_id`). Any other field triggers E018.

### Negation types

| Positive | Negative |
|----------|----------|
| `file_contains` | `file_not_contains` |
| `file_regex` | `file_regex_absent` |

---

## P35. Alpine `:href` Without `x-data` Leaves PDF Links Dead

**Severity**: MEDIUM — valid HTML/PHP, broken runtime behaviour
**First seen**: report-pdf-generation post-mortem (2026-03-27)
**Maestro Prevention**: 0% — static PHP checks and substring assertions do not understand Alpine scope semantics

### Problem: The generated link looks dynamic but never gets a real `href`

An engine updates a Blade/PHP/HTML view and emits:

```html
<a :href="`/reports/sales/pdf?date_from=${dateFrom}&date_to=${dateTo}`">
  Export PDF
</a>
```

The markup is syntactically valid. `php -l` passes. The file contains the expected `/pdf` route. But Alpine never evaluates `:href` because there is no `x-data` scope on the link or any ancestor.

### Why it happens

Low-effort edit tasks often learn the surface pattern "`:href` means dynamic link" but miss the framework contract: **Alpine bindings only work inside an `x-data` scope**.

This usually shows up in report/download buttons where the prompt says "preserve current filters in the PDF URL". The model emits a plausible Alpine expression, but the page already has the needed variables available in server-side PHP and didn't need Alpine at all.

### Symptoms

- Link renders without a usable `href`
- Browser shows the literal element but clicking does nothing useful
- Manual review finds `:href=` but no nearby `x-data`
- Mixed results across similar files: some views work, others silently fail

### Safer patterns

1. Prefer server-side URL building when the variables are already in template scope:

```php
<?php
$pdfQuery = http_build_query([
    'date_from' => $dateFrom,
    'date_to' => $dateTo,
]);
?>
<a href="/reports/sales/pdf?<?= $pdfQuery ?>">Export PDF</a>
```

2. If Alpine is required, make the scope explicit:

```html
<div x-data="{ dateFrom: '2026-03-01', dateTo: '2026-03-31' }">
  <a :href="`/reports/sales/pdf?date_from=${dateFrom}&date_to=${dateTo}`">
    Export PDF
  </a>
</div>
```

### Planning rule

For frontend edits involving Alpine, Vue, React, or similar frameworks, do not treat them as "mechanical low-effort substitutions". If the task depends on framework semantics, use a stronger prompt and at least medium reasoning effort.

---

## P36. Retry Without Context Repeats the Same Mistake 8x

**Discovered**: test-improvement-loop runs — 4 tasks failed with identical tracebacks across 8 consecutive runs, burning all retries every time.

### What happened

A plan had `max_retries: 3` but no `context_from` linking the task to its verify_command output. Each retry sent the exact same prompt without the error traceback, so the agent made the exact same mistake every time.

### Root cause

Maestro auto-injects verify_command failure feedback into retry prompts — but only when the task has `verify_command` AND `max_retries > 0` on the **same task**. If the feedback injection is working but the prompt is too vague ("Fix the function"), the agent lacks enough context to know *what specifically* failed.

### Fix

1. Make prompts specific: include file paths, function signatures, expected behaviour
2. Use `context_from` from an earlier analysis/test task so the agent has error context
3. Set `max_retries: 1` on first runs to avoid burning 3 retries on a fundamentally broken prompt
4. Check logs after the first failure — if the traceback is identical across retries, the prompt needs rewriting, not more retries

### Planning rule

If a task fails with the same error 2+ times in a row, the problem is the prompt, not the retry count. The `repeated_error` built-in reminder trigger (v1.24.0) detects this pattern and injects a hint, but it cannot fix a fundamentally underspecified prompt.

---

## P37. `--set` with Non-Deterministic Values Breaks Caching

**Discovered**: `--set` feature design review.

### What happens

`maestro run plan.yaml --set timestamp=$(date +%s)` injects a different value every run. The cache hash includes `extra_template_vars`, so every run is a cache miss even if the plan logic is identical.

### Fix

Only use `--set` for stable values (environment names, region codes, feature flags). For values that change per-run (timestamps, UUIDs), use `{{ task_id }}` or generate them inside the task prompt instead of injecting from the CLI.

### Planning rule

`--set` values become part of the cache key. Treat them like function arguments: same inputs should produce same outputs.

## P38. `run_summary.md` Waves Are DAG Levels, Not Runtime Slots

**Discovered**: an internal post-mortem on a backfill task (2026-04-26).

### What happens

A plan with `max_parallel: 3` and 6 read-only tasks (all with `depends_on: []`) shows all 6 tasks in **Wave 0** of `run_summary.md`, even though only 3 of them ran simultaneously at any moment. Authors looking at the timeline conclude that Maestro silently exceeded `max_parallel`, when it didn't.

### Why

`_compute_waves()` in `scheduler.py` computes **DAG topological levels** — a task lands in wave N when all of its `depends_on` are satisfied by waves < N. This is the *structural* parallelism available in the plan, not the runtime schedule. The actual scheduler respects `max_parallel` by running the wave in back-to-back batches under the hood.

So:

- "Wave 0 has 6 tasks" → 6 tasks could in principle run in parallel
- "max_parallel: 3" → at runtime, at most 3 ran concurrently
- The scheduler ran the wave as 3 + 3 (or 3 + 2 + 1 etc., depending on per-task durations)

### Fix

Since 2026-04-27, `run_summary.md` adds an inline note when any wave has more tasks than `max_parallel`:

> _Waves are DAG topological levels, not runtime slots. `max_parallel: N` was respected at runtime even when a wave lists more tasks._

For pre-2.5 runs the note is absent; the wave line is still correct as a *structural* description of the DAG.

### Planning rule

Use `events.jsonl` (`task_start` / `task_complete` timestamps) if you want the actual runtime concurrency profile. The wave timeline is a topology summary, not a Gantt chart.

---

## P39. Council Topology Misconfiguration (E072, W28)

**Discovered**: v2.1.0 chain + graph topology rollout.

### What happens

A plan sets `topology: graph` on a `council:` block but doesn't supply a
complete `connections` map. E072 fires at validation time. Or, the author
sets `topology: chain` with `rounds: 3` expecting a 3-pass refinement loop
and instead gets 3 identical chain re-runs (W28, silent).

### Why

Three council topologies, three different invariants:

| Topology | What participants see | What `connections` does | What `rounds > 1` does |
|---|---|---|---|
| `star` (default) | Every participant sees all peers' previous-round responses | Ignored (W28 if set) | Each round repeats with the previous round's full transcript |
| `chain` | Each participant sees only the immediate predecessor's response | Ignored (W28 if set) | Re-runs the entire chain from scratch (W28 if `rounds > 1`) |
| `graph` | Each participant sees only the IDs listed in their `connections` entry | **Required**; missing or empty fires E072 | Each round respects the visibility graph |

### Fix

For `graph` topology, supply a complete map keyed by every participant ID:

```yaml
council:
  topology: graph
  rounds: 2
  connections:
    pragmatist: [theoretician]
    safety:     [theoretician]
    theoretician: [pragmatist, safety]
  participants:
    - { id: pragmatist,   model: sonnet }
    - { id: safety,       model: sonnet }
    - { id: theoretician, model: opus }
```

For `chain`, keep `rounds: 1` — extra rounds re-pipeline the chain with no
benefit. For `star`, omit `connections` entirely; it's ignored.

### Planning rule

Reach for `chain` when the task is genuinely a draft → critique → polish
pipeline (cheap, single-pass). Reach for `star` when participants need to
react to each other (default, sweet-spot for 2 rounds). Reach for `graph`
only when you genuinely want to model a structured discussion topology;
otherwise the `connections` plumbing is over-engineering.
