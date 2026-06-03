---
name: debug-run
description: Debug a failed Maestro CLI plan run by analyzing the output in .maestro-runs/. Use when a maestro run command fails and the user needs help understanding what went wrong.
tags: debugging, runs, observability
triggers: failed run, run failure, .maestro-runs, run manifest, debug run
recommended-when: Use when a Maestro run already failed and the next step is root-cause analysis from manifests, logs, and events.
recommended-chain: debug-run -> write-tests
---

Debug the failed Maestro CLI run. $ARGUMENTS

## 1. Locate the Run Directory
```powershell
# List recent runs (newest first)
ls .maestro-runs/ | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

## 2. Read the Manifest
```powershell
# Check overall success and find failed tasks
cat .maestro-runs/<run-dir>/run_manifest.json
```

Also check `events.jsonl` for a timestamped event log:
```powershell
cat .maestro-runs/<run-dir>/events.jsonl
```

Key fields in `run_manifest.json`:
- `success` -- Overall pass/fail
- `execution_profile` -- Which safety mode was used
- `task_results.<id>.status` -- Per-task status: `success`, `failed`, `soft_failed`, `skipped`, `dry_run`

## 3. Find the First Failure
Look for the first task with `"status": "failed"` in the manifest. This is the root cause -- downstream tasks will be `"skipped"`.

Precision-first rule:
- Read the manifest and logs before theorizing.
- If you later involve `code-reviewer`, `qa-engineer`, or `security-engineer`,
  hand them concrete evidence first instead of a broad persona prompt.

## 4. Read the Failed Task's Logs
```powershell
# Structured result
cat .maestro-runs/<run-dir>/<task-id>.result.json

# Full execution transcript
cat .maestro-runs/<run-dir>/<task-id>.log
```

Key fields in `<task-id>.result.json`:
- `exit_code` -- Process exit code (124 = timeout)
- `command` -- The exact command that was executed
- `message` -- Error description
- `duration_sec` -- How long it ran

## 5. Common Failure Causes

| Symptom | Likely Cause |
|---------|-------------|
| `exit_code: 124` | Task timed out -- increase `timeout_sec` |
| `exit_code: 1` + pre_command in log | `pre_command` failed -- check setup step |
| `"Working tree is not clean"` | `requires_clean_worktree` gate -- commit or stash changes |
| `"Workdir does not exist"` | Bad `workdir` path -- check plan YAML |
| `"prompt_file not found"` | Wrong `prompt_file` path -- check relative to plan location |
| `"Heading not found in markdown"` | `prompt_md_heading` doesn't match any `## Heading` |
| `"No ```text prompt block"` | Missing code fence under the markdown heading |
| `"status": "skipped"` | Dependency failed -- fix the upstream task first |
| `codex: command not found` | `codex` CLI not on PATH |
| `claude: command not found` | `claude` CLI not on PATH |
| `gemini: command not found` / `copilot: command not found` / `qwen: command not found` / `ollama: command not found` | Required engine CLI not on PATH |
| `"verify_command failed"` in message | `verify_command` check failed -- fix the verification or the code |
| `retry_count > 0` in result.json | Task retried N times before final status -- check all retry logs |
| `"context_mode"` errors in log | `context_from` references missing/failed upstream -- check deps |
| `"webhook"` errors in log | Webhook delivery failed -- check `webhook_url` and network connectivity |
| `"Budget exceeded"` | Budget cap stopped new tasks dispatching -- reduce dependency cone, raise budget, or use a cheaper slice |
| Run says `SUCCESS` but target task is `skipped` | Selection/dependency logic finished cleanly, but the requested target never actually ran |

## 5b. Retry & Verification Debugging

- If `retry_count > 0` in result.json: the task failed initially but may have recovered. Read the full log for `[retry N/M]` markers to trace each attempt.
- If `"verify_command failed"` in message: the main command succeeded but post-verification caught an issue. Look for the `[verify_command]` section in the log.
- If status is `soft_failed`: the task had `allow_failure: true` -- downstream tasks still ran but used degraded upstream output.
- Check `max_retries` in the plan YAML to understand how many attempts were allowed.

## 6. Fix and Re-run
After fixing the issue:
```powershell
# Re-run only the failed task and its dependents
maestro run plan.yaml --only <failed-task-id>

# Re-run only a tagged slice if the dependency cone is too expensive
maestro run plan.yaml --tags <tag>

# Or re-run the entire plan
maestro run plan.yaml
```

## Checklist
- [ ] Run directory located
- [ ] Manifest read, overall status checked
- [ ] First failed task identified
- [ ] Task log and result.json analyzed
- [ ] Root cause identified
- [ ] Fix applied (plan YAML, prompt file, or environment)
- [ ] Re-run successful
