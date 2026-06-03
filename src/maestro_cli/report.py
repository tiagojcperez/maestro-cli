from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

_SUCCESS_LIKE = {"success", "soft_failed", "dry_run"}
_MODEL_FLAG_CLAUDE = re.compile(r"(?:^|\s)--model(?:=|\s+)(\"[^\"]+\"|'[^']+'|[^\s]+)")
_MODEL_FLAG_CODEX = re.compile(r"(?:^|\s)-m\s+(\"[^\"]+\"|'[^']+'|[^\s]+)")


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return None
    return None


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _duration_from_timestamps(started_at: object, finished_at: object) -> float | None:
    start_dt = _parse_iso(started_at)
    finish_dt = _parse_iso(finished_at)
    if start_dt is None or finish_dt is None:
        return None
    return max(0.0, (finish_dt - start_dt).total_seconds())


def _clean_cli_token(value: str) -> str:
    token = value.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1]
    return token.strip()


def _infer_engine_label(task_result: dict[str, Any]) -> str:
    command = task_result.get("command")
    if not isinstance(command, str) or not command.strip():
        return "shell"

    match = _MODEL_FLAG_CLAUDE.search(command)
    if match:
        model = _clean_cli_token(match.group(1))
        if model:
            return f"claude:{model}"
    match = _MODEL_FLAG_CODEX.search(command)
    if match:
        model = _clean_cli_token(match.group(1))
        if model:
            return f"codex:{model}"

    lowered = command.lower()
    if "codex" in lowered:
        return "codex"
    if "claude" in lowered:
        return "claude"
    if "gemini" in lowered:
        return "gemini"
    return "shell"


def _normalize_task(task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    status = str(raw.get("status", "pending")).lower()

    duration_sec = _coerce_float(raw.get("duration_sec"))
    if duration_sec is None:
        duration_sec = _duration_from_timestamps(raw.get("started_at"), raw.get("finished_at"))
    if duration_sec is None:
        duration_sec = 0.0

    cost_usd = _coerce_float(raw.get("cost_usd"))

    token_usage = raw.get("token_usage")
    tokens: int | None = None
    if isinstance(token_usage, dict):
        tokens = _coerce_int(token_usage.get("total_tokens"))
        if tokens is None:
            in_tok = _coerce_int(token_usage.get("input_tokens")) or 0
            cached_tok = _coerce_int(token_usage.get("cached_tokens")) or 0
            out_tok = _coerce_int(token_usage.get("output_tokens")) or 0
            if in_tok or cached_tok or out_tok:
                tokens = in_tok + cached_tok + out_tok

    return {
        "task_id": task_id,
        "status": status,
        "duration_sec": duration_sec,
        "cost_usd": cost_usd,
        "tokens": tokens,
        "engine": _infer_engine_label(raw),
        "command": str(raw.get("command", "")),
        "stdout_tail": str(raw.get("stdout_tail", "")),
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "message": str(raw.get("message", "")),
    }


def _status_counts(tasks: list[dict[str, Any]]) -> tuple[int, int, int]:
    ok_count = 0
    failed_count = 0
    skipped_count = 0
    for task in tasks:
        status = task["status"]
        if status in _SUCCESS_LIKE:
            ok_count += 1
        elif status == "failed":
            failed_count += 1
        elif status == "skipped":
            skipped_count += 1
    return ok_count, failed_count, skipped_count


def _total_cost(manifest: dict[str, Any], tasks: list[dict[str, Any]]) -> float | None:
    manifest_total = _coerce_float(manifest.get("total_cost_usd"))
    if manifest_total is not None:
        return manifest_total
    values: list[float] = [t["cost_usd"] for t in tasks if t["cost_usd"] is not None]
    if not values:
        return None
    return float(sum(values))


def _total_tokens(manifest: dict[str, Any], tasks: list[dict[str, Any]]) -> int | None:
    manifest_total = _coerce_int(manifest.get("total_tokens"))
    if manifest_total is not None:
        return manifest_total
    values: list[int] = [t["tokens"] for t in tasks if t["tokens"] is not None]
    if not values:
        return None
    return int(sum(values))


def _run_duration(manifest: dict[str, Any], tasks: list[dict[str, Any]]) -> float | None:
    duration = _duration_from_timestamps(manifest.get("started_at"), manifest.get("finished_at"))
    if duration is not None:
        return duration
    if not tasks:
        return None
    return float(sum(t["duration_sec"] for t in tasks))


def _status_label(manifest: dict[str, Any], tasks: list[dict[str, Any]]) -> tuple[str, str]:
    success = manifest.get("success")
    if isinstance(success, bool):
        return ("SUCCESS", "success") if success else ("FAILED", "failed")
    if any(task["status"] == "failed" for task in tasks):
        return "FAILED", "failed"
    if tasks:
        return "SUCCESS", "success"
    return "UNKNOWN", "pending"


def _prepare_report_data(manifest: dict[str, Any], run_path: Path) -> dict[str, Any]:
    task_results = manifest.get("task_results")
    if not isinstance(task_results, dict):
        task_results = {}

    tasks = [_normalize_task(task_id, raw) for task_id, raw in task_results.items() if isinstance(raw, dict)]
    tasks.sort(
        key=lambda item: (
            _parse_iso(item["started_at"]) or datetime.max,
            item["task_id"],
        )
    )

    ok_count, failed_count, skipped_count = _status_counts(tasks)
    total_cost_usd = _total_cost(manifest, tasks)
    total_tokens = _total_tokens(manifest, tasks)
    duration_sec = _run_duration(manifest, tasks)
    status_text, status_kind = _status_label(manifest, tasks)

    return {
        "plan_name": str(manifest.get("plan_name") or "(unknown plan)"),
        "run_id": str(manifest.get("run_id") or run_path.name),
        "status_text": status_text,
        "status_kind": status_kind,
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "duration_sec": duration_sec,
        "ok_count": ok_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "task_count": len(tasks),
        "total_cost_usd": total_cost_usd,
        "total_tokens": total_tokens,
        "tasks": tasks,
    }


def _format_cost(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.2f}"


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def _format_duration(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 60:
        return f"{value:.1f}s"
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes}m {seconds}s"


def _json_for_script(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return payload.replace("</", "<\\/")


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Maestro Report - __TITLE__</title>
  <style>
    :root { --bg-0:#08090d; --bg-1:#0e1117; --bg-2:#151921; --bg-3:#1c2030; --text-0:#f0f2f5; --text-1:#c4c9d4; --text-2:#8891a4; --text-3:#545d72; --accent:#c0785a; --success:#3ddc84; --success-bg:rgba(61,220,132,0.1); --failed:#ff5555; --failed-bg:rgba(255,85,85,0.1); --warning:#f5a623; --warning-bg:rgba(245,166,35,0.1); --running:#5b9bf5; --running-bg:rgba(91,155,245,0.1); --skipped:#6b7280; --skipped-bg:rgba(107,114,128,0.1); --border:rgba(255,255,255,0.06); --radius:10px; --radius-lg:16px; --mono:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; --font:-apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:var(--font); background:var(--bg-0); color:var(--text-1); line-height:1.5; }
    .container { max-width:1200px; margin:0 auto; padding:24px 20px 36px; }
    .run-header { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; margin-bottom:14px; }
    .run-header h1 { margin:0; display:flex; flex-wrap:wrap; align-items:center; gap:10px; color:var(--text-0); font-size:1.4rem; letter-spacing:-0.02em; }
    .run-meta { margin-top:8px; display:flex; flex-wrap:wrap; gap:14px; color:var(--text-2); font-size:0.84rem; }
    .run-meta code { font-family:var(--mono); font-size:0.78rem; background:var(--bg-3); border:1px solid var(--border); border-radius:5px; padding:2px 6px; color:var(--text-1); }
    .run-stats { display:flex; gap:20px; flex-wrap:wrap; }
    .stat { min-width:105px; text-align:right; }
    .stat-label { color:var(--text-3); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px; }
    .stat-value { color:var(--text-0); font-size:1rem; font-weight:600; font-family:var(--mono); }
    .badge { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; white-space:nowrap; }
    .badge::before { content:""; width:6px; height:6px; border-radius:50%; }
    .badge-success { background:var(--success-bg); color:var(--success); } .badge-success::before { background:var(--success); }
    .badge-failed { background:var(--failed-bg); color:var(--failed); } .badge-failed::before { background:var(--failed); }
    .badge-soft_failed { background:var(--warning-bg); color:var(--warning); } .badge-soft_failed::before { background:var(--warning); }
    .badge-skipped { background:var(--skipped-bg); color:var(--skipped); } .badge-skipped::before { background:var(--skipped); }
    .badge-dry_run { background:var(--warning-bg); color:var(--warning); } .badge-dry_run::before { background:var(--warning); }
    .badge-running { background:var(--running-bg); color:var(--running); } .badge-running::before { background:var(--running); }
    .badge-pending { background:var(--skipped-bg); color:var(--skipped); } .badge-pending::before { background:var(--skipped); opacity:0.6; }
    .stats-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:16px 0; }
    .stat-card { background:var(--bg-2); border:1px solid var(--border); border-radius:var(--radius-lg); padding:14px; }
    .stat-card-label { color:var(--text-3); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
    .stat-card-value { color:var(--text-0); font-family:var(--mono); font-size:1.22rem; font-weight:700; line-height:1.2; margin-bottom:2px; }
    .stat-card-detail { color:var(--text-2); font-size:0.79rem; }
    .card { background:var(--bg-2); border:1px solid var(--border); border-radius:var(--radius-lg); padding:16px; margin-bottom:16px; }
    .card-header { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:10px; }
    .card-title { margin:0; color:var(--text-3); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; }
    .card-subtitle { color:var(--text-2); font-size:0.78rem; }
    table { width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; }
    th, td { border-bottom:1px solid var(--border); padding:9px 10px; text-align:left; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    th { color:var(--text-3); font-size:0.69rem; text-transform:uppercase; letter-spacing:0.06em; font-weight:600; }
    td { color:var(--text-1); font-size:0.82rem; }
    tbody tr:hover { background:rgba(255,255,255,0.03); }
    .mono { font-family:var(--mono); font-size:0.77rem; }
    .sortable { cursor:pointer; user-select:none; position:relative; padding-right:20px; }
    .sortable::after { content:"\\2195"; position:absolute; right:8px; top:50%; transform:translateY(-50%); opacity:0.4; font-size:0.64rem; color:var(--text-3); }
    .sortable.sort-active::after { content:"\\25B2"; opacity:0.95; color:var(--text-2); } .sortable.sort-active[data-order="desc"]::after { content:"\\25BC"; }
    .gantt-axis { position:relative; height:24px; margin:0 0 8px 145px; } .gantt-tick { position:absolute; transform:translateX(-50%); color:var(--text-3); font-family:var(--mono); font-size:0.67rem; }
    .gantt-row { display:flex; align-items:center; height:30px; margin-bottom:3px; } .gantt-label { width:145px; flex-shrink:0; text-align:right; padding-right:10px; color:var(--text-1); font-family:var(--mono); font-size:0.76rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .gantt-track { flex:1; height:22px; background:var(--bg-1); border-radius:4px; position:relative; overflow:hidden; } .gantt-bar { position:absolute; top:0; height:22px; border-radius:4px; min-width:2px; opacity:0.82; }
    .gantt-bar-success { background:var(--success); } .gantt-bar-failed { background:var(--failed); } .gantt-bar-soft_failed { background:var(--warning); } .gantt-bar-skipped { background:var(--skipped); opacity:0.58; } .gantt-bar-dry_run { background:var(--warning); opacity:0.64; } .gantt-bar-running { background:var(--running); } .gantt-bar-pending { background:var(--skipped); opacity:0.38; }
    .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; } .bar-chart { display:flex; flex-direction:column; gap:8px; }
    .metric-row { display:grid; grid-template-columns:160px 1fr auto; gap:10px; align-items:center; min-height:22px; }
    .metric-label { color:var(--text-1); font-size:0.78rem; font-family:var(--mono); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .metric-track { height:14px; background:var(--bg-1); border-radius:999px; overflow:hidden; position:relative; }
    .metric-fill { position:absolute; inset:0 auto 0 0; border-radius:999px; } .metric-fill-cost { background:linear-gradient(90deg,var(--accent),#8b5e3c); } .metric-fill-tokens { background:linear-gradient(90deg,var(--running),#4a80cf); }
    .metric-value { color:var(--text-1); font-size:0.78rem; font-family:var(--mono); white-space:nowrap; }
    details.task-detail { border:1px solid var(--border); border-radius:var(--radius); background:var(--bg-1); margin-bottom:8px; overflow:hidden; } details.task-detail[open] { border-color:rgba(255,255,255,0.1); }
    details.task-detail > summary { list-style:none; cursor:pointer; padding:10px 12px; display:flex; justify-content:space-between; align-items:center; gap:10px; } details.task-detail > summary::-webkit-details-marker { display:none; }
    .task-summary-left { display:flex; align-items:center; gap:10px; min-width:0; } .task-summary-id { color:var(--text-0); font-family:var(--mono); font-size:0.8rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .task-summary-right { color:var(--text-2); font-family:var(--mono); font-size:0.75rem; display:flex; gap:12px; flex-shrink:0; }
    .task-detail-body { border-top:1px solid var(--border); padding:12px; display:grid; gap:12px; }
    .detail-section-title { margin:0 0 6px; color:var(--text-3); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; }
    pre.detail-block { margin:0; background:var(--bg-0); border:1px solid var(--border); border-radius:7px; padding:10px; color:var(--text-1); font-size:0.75rem; line-height:1.55; white-space:pre-wrap; word-break:break-word; font-family:var(--mono); max-height:260px; overflow:auto; }
    .empty { color:var(--text-3); font-size:0.82rem; padding:8px 0; }
    .muted { color:var(--text-3); font-size:0.8rem; }
    @media (max-width:980px) { .stats-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .chart-row { grid-template-columns:1fr; } .metric-row { grid-template-columns:120px 1fr auto; } .gantt-label { width:105px; } .gantt-axis { margin-left:105px; } }
    @media (max-width:720px) { .run-header { flex-direction:column; } .stat { text-align:left; } .stats-grid { grid-template-columns:1fr; } .metric-row { grid-template-columns:1fr; gap:6px; } .gantt-label { width:84px; font-size:0.69rem; } .gantt-axis { margin-left:84px; } th,td { padding:8px 6px; } }
  </style>
</head>
<body>
  <div class="container">
    <section class="run-header">
      <div>
        <h1><span id="plan-name">__PLAN_NAME__</span><span class="badge badge-__HEADER_BADGE__" id="run-status-badge">__STATUS_TEXT__</span></h1>
        <div class="run-meta">
          <span>Run ID: <code id="run-id">__RUN_ID__</code></span>
          <span>Started: <span id="started-at">__STARTED_AT__</span></span>
          <span>Finished: <span id="finished-at">__FINISHED_AT__</span></span>
        </div>
      </div>
      <div class="run-stats">
        <div class="stat"><div class="stat-label">Duration</div><div class="stat-value">__HEADER_DURATION__</div></div>
        <div class="stat"><div class="stat-label">Tasks</div><div class="stat-value">__HEADER_TASKS__</div></div>
      </div>
    </section>
    <section class="stats-grid">
      <article class="stat-card"><div class="stat-card-label">Duration</div><div class="stat-card-value">__HEADER_DURATION__</div><div class="stat-card-detail">Wall-clock runtime</div></article>
      <article class="stat-card"><div class="stat-card-label">Tasks</div><div class="stat-card-value">__HEADER_TASKS__</div><div class="stat-card-detail">__TASK_SPLIT__</div></article>
      <article class="stat-card"><div class="stat-card-label">Cost</div><div class="stat-card-value">__TOTAL_COST__</div><div class="stat-card-detail">Total spend</div></article>
      <article class="stat-card"><div class="stat-card-label">Tokens</div><div class="stat-card-value">__TOTAL_TOKENS__</div><div class="stat-card-detail">Total token usage</div></article>
    </section>
    <section class="card">
      <header class="card-header"><h2 class="card-title">Task Table</h2><div class="card-subtitle">Click headers to sort</div></header>
      <div style="overflow:auto;">
        <table>
          <thead><tr><th class="sortable sort-active" data-sort="task_id" data-order="asc">Task ID</th><th class="sortable" data-sort="status" data-order="asc">Status</th><th class="sortable" data-sort="duration_sec" data-order="desc">Duration</th><th class="sortable" data-sort="cost_usd" data-order="desc">Cost</th><th class="sortable" data-sort="tokens" data-order="desc">Tokens</th><th class="sortable" data-sort="engine" data-order="asc">Engine</th></tr></thead>
          <tbody id="task-table-body"></tbody>
        </table>
      </div>
    </section>
    <section class="card">
      <header class="card-header"><h2 class="card-title">Execution Timeline</h2><div class="card-subtitle" id="timeline-total"></div></header>
      <div id="timeline-body"></div>
    </section>
    <section class="chart-row">
      <article class="card"><header class="card-header"><h2 class="card-title">Cost Breakdown</h2><div class="card-subtitle">Per task</div></header><div class="bar-chart" id="cost-chart"></div></article>
      <article class="card"><header class="card-header"><h2 class="card-title">Token Breakdown</h2><div class="card-subtitle">Per task</div></header><div class="bar-chart" id="token-chart"></div></article>
    </section>
    <section class="card">
      <header class="card-header"><h2 class="card-title">Task Details</h2><div class="card-subtitle">Command and stdout tail</div></header>
      <div id="task-details"></div>
    </section>
  </div>
  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const reportData = JSON.parse(document.getElementById("report-data").textContent);
    const STATUS_ORDER = { failed:0, soft_failed:1, running:2, pending:3, skipped:4, dry_run:5, success:6 };
    const escapeHtml = (value) => { const div = document.createElement("div"); div.textContent = value == null ? "" : String(value); return div.innerHTML; };
    const simpleMarkdown = (text) => { if (!text) return ""; return escapeHtml(text).split("\\n").map((l) => { const hm = l.match(/^(#{2,4})\\s+(.+)$/); if (hm) return `<strong style="color:var(--text-1)">${hm[2]}</strong>`; return l.replace(/\\*\\*(.+?)\\*\\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, '<code style="background:var(--bg-2);padding:0.1em 0.35em;border-radius:3px;font-size:0.95em">$1</code>'); }).join("\\n"); };
    const formatDuration = (sec) => { if (sec == null || !Number.isFinite(sec)) return "—"; if (sec < 1) return `${Math.round(sec * 1000)}ms`; if (sec < 60) return `${sec.toFixed(1)}s`; const m = Math.floor(sec / 60); const s = Math.round(sec % 60); return `${m}m ${s}s`; };
    const formatCost = (cost) => (cost == null || !Number.isFinite(cost)) ? "—" : `$${cost.toFixed(2)}`;
    const formatTokens = (tokens) => (tokens == null || !Number.isFinite(tokens)) ? "—" : Math.round(tokens).toLocaleString();
    const formatDate = (iso) => { if (!iso || typeof iso !== "string") return "—"; const dt = new Date(iso); return Number.isNaN(dt.getTime()) ? iso : dt.toLocaleString(); };
    const badge = (status) => { const s = (status || "pending").toLowerCase(); return `<span class="badge badge-${s}">${escapeHtml(s)}</span>`; };
    document.getElementById("started-at").textContent = formatDate(reportData.started_at);
    document.getElementById("finished-at").textContent = formatDate(reportData.finished_at);
    let currentSort = { key: "task_id", order: "asc" };
    const taskSortValue = (task, key) => { if (key === "status") return STATUS_ORDER[task.status] ?? 99; if (key === "duration_sec") return Number(task.duration_sec || 0); if (key === "cost_usd") return Number(task.cost_usd || 0); if (key === "tokens") return Number(task.tokens || 0); const raw = task[key]; return raw == null ? "" : String(raw).toLowerCase(); };
    const sortedTasks = () => [...reportData.tasks].sort((a, b) => { const av = taskSortValue(a, currentSort.key); const bv = taskSortValue(b, currentSort.key); if (av < bv) return currentSort.order === "asc" ? -1 : 1; if (av > bv) return currentSort.order === "asc" ? 1 : -1; return a.task_id.localeCompare(b.task_id); });
    const updateSortHeaders = () => document.querySelectorAll(".sortable").forEach((th) => { const active = th.dataset.sort === currentSort.key; th.classList.toggle("sort-active", active); th.dataset.order = active ? currentSort.order : "asc"; });
    const renderTaskTable = () => { const rows = sortedTasks().map((task) => `<tr><td class="mono">${escapeHtml(task.task_id)}</td><td>${badge(task.status)}</td><td class="mono">${formatDuration(task.duration_sec)}</td><td class="mono">${formatCost(task.cost_usd)}</td><td class="mono">${formatTokens(task.tokens)}</td><td class="mono">${escapeHtml(task.engine || "shell")}</td></tr>`); document.getElementById("task-table-body").innerHTML = rows.join(""); };
    document.querySelectorAll(".sortable").forEach((header) => header.addEventListener("click", () => { const key = header.dataset.sort; if (!key) return; if (currentSort.key === key) { currentSort.order = currentSort.order === "asc" ? "desc" : "asc"; } else { currentSort.key = key; currentSort.order = key === "task_id" || key === "engine" ? "asc" : "desc"; } updateSortHeaders(); renderTaskTable(); }));
    const renderTimeline = () => { const wrapper = document.getElementById("timeline-body"); const entries = reportData.tasks.filter((task) => task.started_at && task.finished_at).map((task) => ({ ...task, startMs: new Date(task.started_at).getTime(), endMs: new Date(task.finished_at).getTime() })).filter((task) => Number.isFinite(task.startMs) && Number.isFinite(task.endMs) && task.endMs >= task.startMs); if (!entries.length) { wrapper.innerHTML = '<div class="empty">No task timestamp data available.</div>'; return; } entries.sort((a, b) => a.startMs - b.startMs); const globalStart = Math.min(...entries.map((e) => e.startMs)); const globalEnd = Math.max(...entries.map((e) => e.endMs)); const totalMs = Math.max(1, globalEnd - globalStart); const axis = ['<div class="gantt-axis">']; for (let i = 0; i <= 5; i += 1) { const pct = (i / 5) * 100; axis.push(`<span class="gantt-tick" style="left:${pct.toFixed(2)}%">${formatDuration((totalMs / 1000) * (i / 5))}</span>`); } axis.push("</div>"); const rows = entries.map((task) => { const leftPct = ((task.startMs - globalStart) / totalMs) * 100; const widthPct = Math.max(0.5, ((task.endMs - task.startMs) / totalMs) * 100); const title = `${task.task_id}: ${formatDuration(task.duration_sec)} (${task.status})`; return `<div class="gantt-row"><div class="gantt-label">${escapeHtml(task.task_id)}</div><div class="gantt-track"><div class="gantt-bar gantt-bar-${escapeHtml(task.status)}" style="left:${leftPct.toFixed(3)}%;width:${widthPct.toFixed(3)}%" title="${escapeHtml(title)}"></div></div></div>`; }); wrapper.innerHTML = axis.join("") + rows.join(""); document.getElementById("timeline-total").textContent = `${formatDuration(totalMs / 1000)} total`; };
    const renderMetricChart = (containerId, metricKey, valueFormat, fillClass, emptyText) => { const items = reportData.tasks.filter((task) => task[metricKey] != null && Number(task[metricKey]) > 0).map((task) => ({ task_id: task.task_id, value: Number(task[metricKey]) })).sort((a, b) => b.value - a.value); const container = document.getElementById(containerId); if (!items.length) { container.innerHTML = `<div class="empty">${emptyText}</div>`; return; } const max = Math.max(...items.map((x) => x.value), 1); container.innerHTML = items.map((item) => `<div class="metric-row"><div class="metric-label">${escapeHtml(item.task_id)}</div><div class="metric-track"><div class="metric-fill ${fillClass}" style="width:${((item.value / max) * 100).toFixed(3)}%"></div></div><div class="metric-value">${escapeHtml(valueFormat(item.value))}</div></div>`).join(""); };
    const renderTaskDetails = () => { const blocks = reportData.tasks.map((task) => { const commandText = task.command && task.command.trim() ? task.command : "(no command)"; const tailText = task.stdout_tail && task.stdout_tail.trim() ? task.stdout_tail : "(empty)"; const meta = [formatDuration(task.duration_sec), formatCost(task.cost_usd), formatTokens(task.tokens), task.engine || "shell"].join(" · "); return `<details class="task-detail"><summary><div class="task-summary-left">${badge(task.status)}<span class="task-summary-id">${escapeHtml(task.task_id)}</span></div><div class="task-summary-right">${escapeHtml(meta)}</div></summary><div class="task-detail-body"><div><h3 class="detail-section-title">Command</h3><pre class="detail-block">${escapeHtml(commandText)}</pre></div><div><h3 class="detail-section-title">stdout_tail</h3><pre class="detail-block">${simpleMarkdown(tailText)}</pre></div>${task.message ? `<div class="muted">${escapeHtml(task.message)}</div>` : ""}</div></details>`; }); document.getElementById("task-details").innerHTML = blocks.length ? blocks.join("") : '<div class="empty">No task details available.</div>'; };
    updateSortHeaders(); renderTaskTable(); renderTimeline(); renderMetricChart("cost-chart", "cost_usd", (v) => `$${v.toFixed(2)}`, "metric-fill-cost", "No per-task cost data."); renderMetricChart("token-chart", "tokens", (v) => `${Math.round(v).toLocaleString()}`, "metric-fill-tokens", "No per-task token data."); renderTaskDetails();
  </script>
</body>
</html>
"""


def build_report_html(manifest: dict[str, Any], run_path: Path) -> str:
    data = _prepare_report_data(manifest, run_path)
    html_out = _REPORT_TEMPLATE
    html_out = html_out.replace("__TITLE__", escape(f"{data['plan_name']} ({data['run_id']})"))
    html_out = html_out.replace("__PLAN_NAME__", escape(str(data["plan_name"])))
    html_out = html_out.replace("__RUN_ID__", escape(str(data["run_id"])))
    html_out = html_out.replace("__STATUS_TEXT__", escape(str(data["status_text"])))
    html_out = html_out.replace("__HEADER_BADGE__", escape(str(data["status_kind"])))
    html_out = html_out.replace("__STARTED_AT__", escape(str(data["started_at"] or "—")))
    html_out = html_out.replace("__FINISHED_AT__", escape(str(data["finished_at"] or "—")))
    html_out = html_out.replace("__HEADER_DURATION__", escape(_format_duration(data["duration_sec"])))
    html_out = html_out.replace("__HEADER_TASKS__", escape(str(data["task_count"])))
    html_out = html_out.replace(
        "__TASK_SPLIT__",
        escape(f"{data['ok_count']} ok / {data['failed_count']} failed / {data['skipped_count']} skipped"),
    )
    html_out = html_out.replace("__TOTAL_COST__", escape(_format_cost(data["total_cost_usd"])))
    html_out = html_out.replace("__TOTAL_TOKENS__", escape(_format_tokens(data["total_tokens"])))
    html_out = html_out.replace("__REPORT_DATA__", _json_for_script(data))
    return html_out


def generate_report(run_path: Path, output_path: Path | None = None) -> Path:
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run_manifest.json not found in {run_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {manifest_path}: {exc}") from exc

    html_out = build_report_html(manifest, run_path)
    target = output_path if output_path is not None else run_path / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_out, encoding="utf-8")
    return target
