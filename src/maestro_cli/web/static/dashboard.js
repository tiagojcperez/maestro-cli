/* Maestro UI — Dashboard logic */

// ---------------------------------------------------------------------------
// Stats Cards + Charts
// ---------------------------------------------------------------------------

let statsCharts = { donut: null, durationTrend: null, costTrend: null, costByModel: null, tokenTrend: null, tokenByModel: null };

async function loadStats() {
    try {
        const stats = await apiFetch('/runs/stats');
        renderStatsCards(stats);
        renderDashboardCharts(stats);
    } catch (err) {
        console.error('Error loading stats:', err);
        // Ensure charts row stays hidden on failure
        const chartRow = document.getElementById('dashboard-charts');
        if (chartRow) chartRow.classList.add('hidden');
    }
}

function renderStatsCards(stats) {
    const container = document.getElementById('stats-grid');
    if (!container) return;

    const successRate = stats.total_runs > 0
        ? ((stats.success_count / stats.total_runs) * 100).toFixed(0)
        : '—';

    const totalCost = stats.total_cost_usd != null
        ? `$${stats.total_cost_usd.toFixed(2)}`
        : '—';

    const avgCost = (stats.total_cost_usd != null && stats.total_runs > 0)
        ? `avg $${(stats.total_cost_usd / stats.total_runs).toFixed(2)} per run`
        : '';

    const totalTokensText = stats.total_tokens != null ? formatTokenCount(stats.total_tokens) : '—';
    const avgTokensText = stats.avg_tokens_per_run != null
        ? `avg ${formatTokenCount(stats.avg_tokens_per_run)} per run`
        : '';

    const tokensCard = stats.total_tokens != null ? `
        <div class="stat-card">
            <div class="stat-card-label">Total Tokens</div>
            <div class="stat-card-value">${totalTokensText}</div>
            <div class="stat-card-detail">${avgTokensText}</div>
        </div>` : '';

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-card-label">Total Runs</div>
            <div class="stat-card-value">${stats.total_runs}</div>
            <div class="stat-card-detail">${stats.success_count} success, ${stats.failed_count} failed</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Success Rate</div>
            <div class="stat-card-value">${successRate}${successRate !== '—' ? '%' : ''}</div>
            <div class="stat-card-detail-bar">
                <div class="mini-bar">
                    <div class="mini-bar-fill" style="width: ${successRate === '—' ? 0 : successRate}%"></div>
                </div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Total Cost</div>
            <div class="stat-card-value">${totalCost}</div>
            <div class="stat-card-detail">${avgCost}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Avg Duration</div>
            <div class="stat-card-value">${formatDuration(stats.avg_duration_sec)}</div>
            <div class="stat-card-detail">${stats.total_runs} runs measured</div>
        </div>
        ${tokensCard}
    `;
}

function renderDashboardCharts(stats) {
    // Destroy existing charts
    if (statsCharts.donut) { statsCharts.donut.destroy(); statsCharts.donut = null; }
    if (statsCharts.durationTrend) { statsCharts.durationTrend.destroy(); statsCharts.durationTrend = null; }
    if (statsCharts.costTrend) { statsCharts.costTrend.destroy(); statsCharts.costTrend = null; }
    if (statsCharts.costByModel) { statsCharts.costByModel.destroy(); statsCharts.costByModel = null; }
    if (statsCharts.tokenTrend) { statsCharts.tokenTrend.destroy(); statsCharts.tokenTrend = null; }
    if (statsCharts.tokenByModel) { statsCharts.tokenByModel.destroy(); statsCharts.tokenByModel = null; }

    const chartRow = document.getElementById('dashboard-charts');
    if (!chartRow) return;

    const hasStatus = Object.keys(stats.status_distribution || {}).length > 0;
    const hasRecent = (stats.recent_runs || []).length > 0;
    const hasCost = (stats.cost_by_run || []).length > 0;
    const hasModelCost = (stats.cost_by_model || []).length > 0;

    // Main chart row: donut + duration trend
    if (!hasStatus && !hasRecent) {
        chartRow.classList.add('hidden');
    } else {
        chartRow.classList.remove('hidden');

        // Status donut
        const donutCard = document.getElementById('chart-status-donut');
        if (hasStatus && donutCard) {
            donutCard.classList.remove('hidden');
            statsCharts.donut = createStatusDonut('canvas-status-donut', stats.status_distribution);
        } else if (donutCard) {
            donutCard.classList.add('hidden');
        }

        // Duration trend
        const durationCard = document.getElementById('chart-duration-trend');
        if (hasRecent && durationCard) {
            durationCard.classList.remove('hidden');
            statsCharts.durationTrend = createDurationTrend('canvas-duration-trend', stats.recent_runs);
        } else if (durationCard) {
            durationCard.classList.add('hidden');
        }
    }

    // Cost chart row (separate, only if at least one cost dataset exists)
    const costRow = document.getElementById('dashboard-cost-charts');
    const costTrendCard = document.getElementById('chart-cost-trend');
    const costByModelCard = document.getElementById('chart-cost-by-model');
    const visibleCostCards = (hasCost ? 1 : 0) + (hasModelCost ? 1 : 0);

    if (!costRow) return;
    if (visibleCostCards === 0) {
        costRow.classList.add('hidden');
        return;
    }

    costRow.classList.remove('hidden');
    costRow.classList.toggle('chart-row-one-col', visibleCostCards === 1);

    if (hasCost && costTrendCard) {
        costTrendCard.classList.remove('hidden');
        statsCharts.costTrend = createCostTrend('canvas-cost-trend', stats.cost_by_run);
    } else if (costTrendCard) {
        costTrendCard.classList.add('hidden');
    }

    if (hasModelCost && costByModelCard) {
        costByModelCard.classList.remove('hidden');
        statsCharts.costByModel = createCostByModel('canvas-cost-by-model', stats.cost_by_model);
    } else if (costByModelCard) {
        costByModelCard.classList.add('hidden');
    }

    // Token chart row
    const tokenRow = document.getElementById('dashboard-token-charts');
    const tokenTrendCard = document.getElementById('chart-token-trend');
    const tokenByModelCard = document.getElementById('chart-token-by-model');
    const hasTokenTrend = (stats.recent_runs || []).some(r => r.total_tokens != null);
    const hasTokenByModel = (stats.tokens_by_model || []).length > 0;
    const visibleTokenCards = (hasTokenTrend ? 1 : 0) + (hasTokenByModel ? 1 : 0);

    if (!tokenRow) return;
    if (visibleTokenCards === 0) {
        tokenRow.classList.add('hidden');
        return;
    }

    tokenRow.classList.remove('hidden');
    tokenRow.classList.toggle('chart-row-one-col', visibleTokenCards === 1);

    if (hasTokenTrend && tokenTrendCard) {
        tokenTrendCard.classList.remove('hidden');
        statsCharts.tokenTrend = createTokenTrend('canvas-token-trend', stats.recent_runs);
    } else if (tokenTrendCard) {
        tokenTrendCard.classList.add('hidden');
    }

    if (hasTokenByModel && tokenByModelCard) {
        tokenByModelCard.classList.remove('hidden');
        statsCharts.tokenByModel = createTokenByModel('canvas-token-by-model', stats.tokens_by_model);
    } else if (tokenByModelCard) {
        tokenByModelCard.classList.add('hidden');
    }
}

// ---------------------------------------------------------------------------
// Runs Table (with filtering + sorting)
// ---------------------------------------------------------------------------

let allRuns = [];
let runsSortKey = 'started_at';
let runsSortAsc = false;
const RUNS_PER_PAGE = 15;
let currentPage = 1;
const runSummaryCache = new Map();
const runSummaryInflight = new Set();

async function loadRuns() {
    const container = document.getElementById('runs-list');
    if (!container) return;

    try {
        allRuns = await apiFetch('/runs');
        applyRunSummaryCache(allRuns);
        renderRunsTable();
        hydrateMissingRunSummaries(allRuns);
    } catch (err) {
        container.innerHTML = `<p class="text-failed">Error loading runs: ${escapeHtml(err.message)}</p>`;
    }
}

async function loadRunRoots() {
    const container = document.getElementById('runs-roots');
    if (!container) return;

    try {
        const data = await apiFetch('/runs/roots');
        const projectRoots = Array.isArray(data?.project_roots)
            ? data.project_roots
            : (data?.project_root ? [data.project_root] : []);
        const roots = Array.isArray(data?.run_roots) ? data.run_roots : [];

        if (projectRoots.length === 0 && roots.length === 0) {
            container.classList.add('hidden');
            container.innerHTML = '';
            return;
        }

        let html = '';
        if (projectRoots.length > 0) {
            const label = projectRoots.length === 1
                ? 'Project root:'
                : `Project roots (${projectRoots.length}):`;
            const items = projectRoots
                .map((root) => `<code>${escapeHtml(String(root))}</code>`)
                .join('');
            html += `<div class="runs-roots-line"><span class="runs-roots-label">${escapeHtml(label)}</span>${items}</div>`;
        }

        if (roots.length > 0) {
            const label = roots.length === 1 ? 'Run root:' : `Run roots (${roots.length}):`;
            const items = roots
                .map((root) => `<code>${escapeHtml(String(root))}</code>`)
                .join('');
            html += `<div class="runs-roots-line"><span class="runs-roots-label">${escapeHtml(label)}</span>${items}</div>`;
        }

        container.innerHTML = html;
        container.classList.remove('hidden');
    } catch (err) {
        console.error('Error loading run roots:', err);
        container.classList.add('hidden');
        container.innerHTML = '';
    }
}

function applyRunSummaryCache(runs) {
    if (!Array.isArray(runs)) return;
    for (const run of runs) {
        const cached = runSummaryCache.get(run.run_id);
        if (!cached) continue;
        Object.assign(run, cached);
    }
}

function shouldHydrateRunSummary(run) {
    if (!run || run.active) return false;
    // Always confirm completed runs once via detail endpoint so mode/cost/duration
    // remain accurate even when list endpoint is from an older backend build.
    return !runSummaryCache.has(run.run_id);
}

function deriveRunSummary(detail) {
    if (!detail || typeof detail !== 'object') return null;

    const out = {};
    if (typeof detail.execution_profile === 'string') {
        out.execution_profile = detail.execution_profile;
    }
    if (typeof detail.finished_at === 'string') {
        out.finished_at = detail.finished_at;
    }

    if (typeof detail.duration_sec === 'number' && Number.isFinite(detail.duration_sec)) {
        out.duration_sec = Math.max(0, detail.duration_sec);
    } else if (detail.started_at && detail.finished_at) {
        const dur = (new Date(detail.finished_at) - new Date(detail.started_at)) / 1000;
        if (Number.isFinite(dur)) out.duration_sec = Math.max(0, dur);
    }

    if (typeof detail.total_cost_usd === 'number' && Number.isFinite(detail.total_cost_usd)) {
        out.total_cost_usd = detail.total_cost_usd;
    } else if (detail.task_results && typeof detail.task_results === 'object') {
        const costs = Object.values(detail.task_results)
            .map(tr => tr?.cost_usd)
            .filter(v => typeof v === 'number' && Number.isFinite(v));
        out.total_cost_usd = costs.length > 0
            ? costs.reduce((acc, n) => acc + n, 0)
            : null;
    }

    if (typeof detail.dry_run === 'boolean') {
        out.dry_run = detail.dry_run;
    } else if (detail.task_results && typeof detail.task_results === 'object') {
        const statuses = Object.values(detail.task_results)
            .map(tr => tr?.status)
            .filter(s => typeof s === 'string');
        if (statuses.length > 0) {
            const hasDryRun = statuses.includes('dry_run');
            const hasRealExec = statuses.some(s => ['success', 'failed', 'soft_failed'].includes(s));
            out.dry_run = hasDryRun && !hasRealExec;
        }
    }

    if (detail.collaboration_summary && typeof detail.collaboration_summary === 'object') {
        out.collaboration_summary = detail.collaboration_summary;
    }

    return Object.keys(out).length > 0 ? out : null;
}

function renderRunCollaborationSummary(summary) {
    if (!summary || typeof summary !== 'object') {
        return '<span class="text-2">—</span>';
    }

    const ownerCount = Number.isFinite(summary.owner_count) ? summary.owner_count : 0;
    const blockedCount = Number.isFinite(summary.blocked_count) ? summary.blocked_count : 0;
    const topOwners = Array.isArray(summary.top_owners) ? summary.top_owners.slice(0, 2) : [];
    const pills = [];

    pills.push(`<span class="pill pill-neutral">${ownerCount} owners</span>`);
    if (blockedCount > 0) {
        pills.push(`<span class="pill pill-blocked">${blockedCount} blocked</span>`);
    }
    for (const owner of topOwners) {
        pills.push(`<span class="pill pill-owner">${escapeHtml(owner)}</span>`);
    }

    return `<div class="run-collab-summary">${pills.join('')}</div>`;
}

function updateRunWithSummary(runId, patch) {
    if (!patch) return false;
    let changed = false;
    for (const run of allRuns) {
        if (run.run_id !== runId) continue;
        for (const [key, value] of Object.entries(patch)) {
            if (run[key] !== value) {
                run[key] = value;
                changed = true;
            }
        }
        break;
    }
    return changed;
}

async function hydrateMissingRunSummaries(runs) {
    if (!Array.isArray(runs)) return;

    const candidates = runs.filter(shouldHydrateRunSummary).slice(0, 25);
    for (const run of candidates) {
        const runId = run.run_id;
        if (!runId || runSummaryInflight.has(runId)) continue;
        if (runSummaryCache.has(runId)) continue;

        runSummaryInflight.add(runId);
        try {
            const detail = await apiFetch(`/runs/${encodeURIComponent(runId)}`);
            const patch = deriveRunSummary(detail);
            if (!patch) continue;
            runSummaryCache.set(runId, patch);
            if (updateRunWithSummary(runId, patch)) {
                renderRunsTable();
            }
        } catch {
            // ignore; list endpoint remains source of truth
        } finally {
            runSummaryInflight.delete(runId);
        }
    }
}

function runDurationSeconds(run) {
    if (typeof run.duration_sec === 'number' && Number.isFinite(run.duration_sec)) {
        return Math.max(0, run.duration_sec);
    }
    if (run.started_at && run.finished_at) {
        const seconds = (new Date(run.finished_at) - new Date(run.started_at)) / 1000;
        if (Number.isFinite(seconds)) return Math.max(0, seconds);
    }
    return null;
}

function getRunSortValue(run, key) {
    if (!run || !key) return null;

    if (key === 'started_at') {
        if (!run.started_at) return null;
        const t = new Date(run.started_at).getTime();
        return Number.isFinite(t) ? t : null;
    }

    if (key === 'duration_sec') {
        return runDurationSeconds(run);
    }

    if (key === 'total_cost_usd') {
        if (typeof run.total_cost_usd === 'number' && Number.isFinite(run.total_cost_usd)) {
            return run.total_cost_usd;
        }
        return null;
    }

    if (key === 'run_id' || key === 'plan_name') {
        return String(run[key] || '').toLowerCase();
    }

    return null;
}

function sortRuns(runs) {
    const sorted = [...runs];
    sorted.sort((a, b) => {
        const va = getRunSortValue(a, runsSortKey);
        const vb = getRunSortValue(b, runsSortKey);

        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;

        let cmp = 0;
        if (typeof va === 'number' && typeof vb === 'number') {
            cmp = va - vb;
        } else {
            cmp = String(va).localeCompare(String(vb), undefined, {
                numeric: true,
                sensitivity: 'base',
            });
        }

        if (cmp === 0) {
            const ta = getRunSortValue(a, 'started_at') || 0;
            const tb = getRunSortValue(b, 'started_at') || 0;
            cmp = ta - tb;
        }

        return runsSortAsc ? cmp : -cmp;
    });
    return sorted;
}

function defaultSortDirectionForKey(key) {
    return key === 'run_id' || key === 'plan_name';
}

function bindRunsTableSortHandlers(container) {
    const headers = container.querySelectorAll('th.sortable');
    headers.forEach((th) => {
        const key = th.dataset.sort;
        if (!key) return;

        const active = key === runsSortKey;
        th.classList.toggle('sort-active', active);
        th.dataset.order = active ? (runsSortAsc ? 'asc' : 'desc') : '';
        th.setAttribute('aria-sort', active ? (runsSortAsc ? 'ascending' : 'descending') : 'none');

        th.addEventListener('click', () => {
            if (runsSortKey === key) {
                runsSortAsc = !runsSortAsc;
            } else {
                runsSortKey = key;
                runsSortAsc = defaultSortDirectionForKey(key);
            }
            currentPage = 1;
            renderRunsTable();
        });
    });
}

function renderRunsTable() {
    const container = document.getElementById('runs-list');
    if (!container) return;

    // Apply filters
    const statusFilter = document.getElementById('filter-status')?.value || 'all';
    const searchFilter = (document.getElementById('filter-search')?.value || '').toLowerCase();

    let filtered = [...allRuns];
    if (statusFilter !== 'all') {
        filtered = filtered.filter(r => {
            const s = r.active ? 'running' : (r.success ? 'success' : (r.success === false ? 'failed' : 'pending'));
            return s === statusFilter;
        });
    }
    if (searchFilter) {
        filtered = filtered.filter(r =>
            (r.plan_name || '').toLowerCase().includes(searchFilter) ||
            (r.run_id || '').toLowerCase().includes(searchFilter)
        );
    }

    filtered = sortRuns(filtered);

    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">&#9654;</div>
                <h4>${allRuns.length === 0 ? 'No runs yet' : 'No matching runs'}</h4>
                <p>${allRuns.length === 0 ? 'Start a run to see it here' : 'Try adjusting your filters'}</p>
            </div>`;
        renderPagination(0, 0);
        return;
    }

    // Pagination
    const totalPages = Math.ceil(filtered.length / RUNS_PER_PAGE);
    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;
    const startIdx = (currentPage - 1) * RUNS_PER_PAGE;
    const pageItems = filtered.slice(startIdx, startIdx + RUNS_PER_PAGE);

    let html = `<table class="runs-table">
        <thead><tr>
            <th class="sortable" data-sort="run_id">Run</th>
            <th class="sortable" data-sort="plan_name">Plan</th>
            <th>Status</th>
            <th>Mode</th>
            <th class="sortable" data-sort="started_at">Started</th>
            <th>Tasks</th>
            <th class="col-collab">Collab</th>
            <th class="sortable" data-sort="duration_sec">Duration</th>
            <th class="sortable" data-sort="total_cost_usd">Cost</th>
            <th class="col-actions"></th>
        </tr></thead><tbody>`;

    for (const r of pageItems) {
        const status = r.active
            ? 'running'
            : (r.success ? 'success' : (r.success === false ? 'failed' : 'pending'));
        const shortId = r.run_id.length > 30
            ? r.run_id.substring(0, 30) + '\u2026'
            : r.run_id;

        // Duration calc
        let durText = '—';
        const durSec = runDurationSeconds(r);
        if (durSec != null) durText = formatDuration(durSec);

        const costText = r.total_cost_usd != null ? `$${r.total_cost_usd.toFixed(2)}` : '—';
        let modeBadge = '<span class="badge badge-pending">unknown</span>';
        if (r.dry_run === true) {
            modeBadge = badgeFor('dry_run');
        } else if (r.dry_run === false) {
            modeBadge = '<span class="badge badge-normal">normal</span>';
        }

        html += `<tr>
            <td><a class="run-link" href="/static/run.html?id=${encodeURIComponent(r.run_id)}">${escapeHtml(shortId)}</a></td>
            <td>${escapeHtml(r.plan_name || '—')}</td>
            <td>${badgeFor(status)}</td>
            <td>${modeBadge}</td>
            <td class="text-2">${formatDate(r.started_at)}</td>
            <td>${r.task_count || r.task_ids?.length || '—'}</td>
            <td>${renderRunCollaborationSummary(r.collaboration_summary)}</td>
            <td class="text-2">${durText}</td>
            <td class="text-2">${costText}</td>
            <td class="col-actions"><button type="button" class="btn btn-danger-ghost btn-sm" onclick="deleteRun('${escapeHtml(r.run_id)}')">&#128465;</button></td>
        </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
    bindRunsTableSortHandlers(container);

    renderPagination(filtered.length, totalPages);
}

function renderPagination(totalItems, totalPages) {
    const container = document.getElementById('runs-pagination');
    if (!container) return;

    if (totalPages <= 1) {
        container.innerHTML = totalItems > 0
            ? `<span class="pagination-info">${totalItems} run${totalItems !== 1 ? 's' : ''}</span>`
            : '';
        return;
    }

    const startItem = (currentPage - 1) * RUNS_PER_PAGE + 1;
    const endItem = Math.min(currentPage * RUNS_PER_PAGE, totalItems);

    let html = `<span class="pagination-info">${startItem}–${endItem} of ${totalItems}</span>`;
    html += '<div class="pagination-buttons">';

    // Previous
    html += `<button class="btn btn-ghost btn-sm" ${currentPage <= 1 ? 'disabled' : ''} onclick="goToPage(${currentPage - 1})">&larr; Prev</button>`;

    // Page numbers (show max 7: first, ..., current-1, current, current+1, ..., last)
    const pages = _buildPageNumbers(currentPage, totalPages);
    for (const p of pages) {
        if (p === '...') {
            html += '<span class="pagination-ellipsis">&hellip;</span>';
        } else {
            const active = p === currentPage ? ' pagination-active' : '';
            html += `<button class="btn btn-ghost btn-sm${active}" onclick="goToPage(${p})">${p}</button>`;
        }
    }

    // Next
    html += `<button class="btn btn-ghost btn-sm" ${currentPage >= totalPages ? 'disabled' : ''} onclick="goToPage(${currentPage + 1})">Next &rarr;</button>`;

    html += '</div>';
    container.innerHTML = html;
}

function _buildPageNumbers(current, total) {
    if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
    const pages = [];
    pages.push(1);
    if (current > 3) pages.push('...');
    for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) {
        pages.push(i);
    }
    if (current < total - 2) pages.push('...');
    pages.push(total);
    return pages;
}

function goToPage(page) {
    currentPage = page;
    renderRunsTable();
}

async function deleteRun(runId) {
    if (!confirm(`Delete run ${runId}?`)) return;
    try {
        await apiFetch(`/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
        showToast('Run deleted', 'success');
        loadRuns();
    } catch (err) {
        showToast('Error deleting run: ' + err.message, 'error');
    }
}

// ---------------------------------------------------------------------------
// Validate Plan
// ---------------------------------------------------------------------------

async function validatePlan() {
    const textarea = document.getElementById('validate-yaml');
    const result = document.getElementById('validate-result');
    if (!textarea || !result) return;

    const yaml = textarea.value.trim();
    if (!yaml) {
        showToast('Paste a plan YAML first', 'info');
        return;
    }

    result.className = 'validate-result';
    result.textContent = 'Validating...';
    result.classList.add('show');

    try {
        const data = await apiFetch('/plans/validate', {
            method: 'POST',
            body: JSON.stringify({ yaml_content: yaml }),
        });
        if (data.valid) {
            result.className = 'validate-result show valid';
            result.textContent = `Valid — ${data.plan.tasks} tasks, max_parallel=${data.plan.max_parallel}`;
        } else {
            result.className = 'validate-result show invalid';
            result.textContent = `Invalid — ${data.error}`;
        }
    } catch (err) {
        result.className = 'validate-result show invalid';
        result.textContent = `Error: ${err.message}`;
    }
}

// ---------------------------------------------------------------------------
// Dashboard init
// ---------------------------------------------------------------------------

function initDashboard() {
    applyChartDefaults();
    loadStats();
    loadRuns();
    loadRunRoots();
    setInterval(loadRuns, 10000);
    setInterval(loadRunRoots, 30000);

    // Filter event listeners (reset page on filter change)
    const statusFilter = document.getElementById('filter-status');
    if (statusFilter) statusFilter.addEventListener('change', () => { currentPage = 1; renderRunsTable(); });

    const searchFilter = document.getElementById('filter-search');
    if (searchFilter) searchFilter.addEventListener('input', () => { currentPage = 1; renderRunsTable(); });
}
