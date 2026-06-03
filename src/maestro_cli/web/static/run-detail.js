/* Maestro UI — Run Detail logic */

let eventSource = null;
let currentRunDetail = null;
const taskStatuses = {};
let runDetailCharts = { duration: null, statusDonut: null, cost: null, tokens: null, tokensStacked: null, engineCost: null, engineTokens: null };

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function initRunDetail() {
    applyChartDefaults();

    const params = new URLSearchParams(window.location.search);
    const runId = params.get('id');
    if (!runId || runId === 'undefined' || runId === 'null') {
        showToast('Missing run id. Start a run from the wizard.', 'error');
        window.location.href = '/static/new-run.html';
        return;
    }

    document.getElementById('run-id-display').textContent = runId;
    loadRunDetail(runId);
    connectSSE(runId);
}

// ---------------------------------------------------------------------------
// Load run detail + render everything
// ---------------------------------------------------------------------------

async function loadRunDetail(runId) {
    try {
        const data = await apiFetch(`/runs/${encodeURIComponent(runId)}`);
        renderRunDetail(data, runId);
    } catch (err) {
        console.error('Error loading run detail:', err);
        showToast('Error loading run details', 'error');
    }
}

function renderRunDetail(data, runId) {
    currentRunDetail = data;

    const nameEl = document.getElementById('run-plan-name');
    const statusEl = document.getElementById('run-status');
    const durationEl = document.getElementById('run-duration');

    if (nameEl) nameEl.textContent = data.plan_name || '';
    if (statusEl) {
        if (data.active) {
            statusEl.innerHTML = badgeFor('running');
        } else if (data.success != null) {
            statusEl.innerHTML = badgeFor(data.success ? 'success' : 'failed');
        }
    }

    const wallDuration = computeWallDuration(data);
    if (durationEl) durationEl.textContent = formatDuration(wallDuration);

    const taskResults = data.task_results || {};
    const taskIds = Array.isArray(data.task_ids) && data.task_ids.length > 0
        ? data.task_ids
        : Object.keys(taskResults);

    renderMetricsCards(data, wallDuration, taskIds, taskResults);
    renderCollaboration(data, taskIds, taskResults);
    renderTaskCards(taskIds, taskResults, runId, data.collaboration?.tasks || {});
    updateProgress(taskIds, taskResults);
    renderTaskCharts(taskResults);
    renderGanttTimeline(data, taskResults);
}

function computeWallDuration(data) {
    if (!data?.started_at) return null;
    const startedAt = new Date(data.started_at);
    if (!Number.isFinite(startedAt.getTime())) return null;

    const end = data.finished_at ? new Date(data.finished_at) : new Date();
    if (!Number.isFinite(end.getTime())) return null;

    return Math.max(0, (end - startedAt) / 1000);
}

// ---------------------------------------------------------------------------
// Metrics Cards
// ---------------------------------------------------------------------------

function renderMetricsCards(data, wallDuration, taskIds, taskResults) {
    const container = document.getElementById('run-metrics');
    if (!container) return;

    // Count statuses
    let okCount = 0, failCount = 0, skipCount = 0;
    for (const tid of taskIds) {
        const tr = taskResults[tid];
        const s = tr?.status || 'pending';
        if (s === 'success' || s === 'dry_run') okCount++;
        else if (s === 'failed' || s === 'soft_failed') failCount++;
        else if (s === 'skipped') skipCount++;
    }

    // Parallelism info
    let parallelismText = '—';
    if (data.sequential_duration_sec && wallDuration) {
        const savings = data.parallelism_savings_pct || 0;
        parallelismText = `${formatDuration(wallDuration)} wall / ${formatDuration(data.sequential_duration_sec)} seq`;
        if (savings > 0) parallelismText += ` (${savings.toFixed(0)}% saved)`;
    } else if (wallDuration) {
        parallelismText = formatDuration(wallDuration);
    }

    // Compute total tokens from manifest or by summing task results
    let totalTokens = null;
    if (data.total_tokens != null) {
        totalTokens = data.total_tokens;
    } else {
        let sum = 0;
        let hasAny = false;
        for (const tid of taskIds) {
            const tu = taskResults[tid]?.token_usage;
            if (tu && tu.total_tokens != null) {
                sum += tu.total_tokens;
                hasAny = true;
            }
        }
        if (hasAny) totalTokens = sum;
    }
    const tokenText = totalTokens != null ? formatTokenCount(totalTokens) : '—';
    const tokenDetail = totalTokens != null ? `across ${taskIds.length} tasks` : 'no token data';

    // Efficiency metrics
    let costPer1KText = null;
    if (data.total_cost_usd != null && totalTokens != null && totalTokens > 0) {
        const costPer1K = (data.total_cost_usd / totalTokens) * 1000;
        if (costPer1K >= 0.01) {
            costPer1KText = `$${costPer1K.toFixed(3)}`;
        } else if (costPer1K >= 0.001) {
            costPer1KText = `$${costPer1K.toFixed(4)}`;
        } else {
            costPer1KText = `$${costPer1K.toFixed(5)}`;
        }
    }
    let cacheHitText = null;
    let totalCached = 0;
    let totalInputPlusCached = 0;
    let hasTokenBreakdown = false;
    for (const tid of taskIds) {
        const tu = taskResults[tid]?.token_usage;
        if (tu) {
            hasTokenBreakdown = true;
            totalCached += tu.cached_tokens || 0;
            totalInputPlusCached += (tu.input_tokens || 0) + (tu.cached_tokens || 0);
        }
    }
    if (hasTokenBreakdown && totalInputPlusCached > 0) {
        cacheHitText = `${((totalCached / totalInputPlusCached) * 100).toFixed(0)}%`;
    }

    const costText = data.total_cost_usd != null ? `$${data.total_cost_usd.toFixed(2)}` : '—';
    const rawProfile = typeof data.execution_profile === 'string'
        ? data.execution_profile.toLowerCase()
        : '';
    const knownProfiles = new Set(['plan', 'safe', 'yolo']);
    const profileText = knownProfiles.has(rawProfile) ? rawProfile : '';
    const profileIcons = { yolo: '\u26A1', safe: '\uD83D\uDEE1\uFE0F', plan: '\uD83D\uDCCB' };
    const profileIcon = profileIcons[profileText] || '';
    const profileLabel = profileText ? `${profileIcon} ${badgeFor(profileText)}` : '<span class="text-2">unknown</span>';
    const profileDetail = profileText === 'yolo'
        ? 'no approvals'
        : profileText === 'safe'
            ? 'sandboxed'
            : profileText === 'plan'
                ? 'as planned'
                : 'profile not reported';

    container.innerHTML = `
        <div class="stat-card">
            <div class="stat-card-label">Duration</div>
            <div class="stat-card-value">${wallDuration != null ? formatDuration(wallDuration) : '—'}</div>
            <div class="stat-card-detail">${parallelismText}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Tasks</div>
            <div class="stat-card-value">${taskIds.length}</div>
            <div class="stat-card-detail">${okCount} ok / ${failCount} failed / ${skipCount} skipped</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Cost</div>
            <div class="stat-card-value">${costText}</div>
            <div class="stat-card-detail">${data.total_cost_usd != null ? taskIds.length + ' tasks' : 'no cost data'}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Profile</div>
            <div class="stat-card-value profile-badge">${profileLabel}</div>
            <div class="stat-card-detail">${profileDetail}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Tokens</div>
            <div class="stat-card-value">${tokenText}</div>
            <div class="stat-card-detail">${tokenDetail}</div>
        </div>
        ${costPer1KText != null ? `
        <div class="stat-card">
            <div class="stat-card-label">Cost / 1K Tokens</div>
            <div class="stat-card-value">${costPer1KText}</div>
            <div class="stat-card-detail">efficiency</div>
        </div>` : ''}
        ${cacheHitText != null ? `
        <div class="stat-card">
            <div class="stat-card-label">Cache Hit Rate</div>
            <div class="stat-card-value">${cacheHitText}</div>
            <div class="stat-card-detail">${totalCached.toLocaleString('en-US')} cached tokens</div>
        </div>` : ''}
    `;
}

// ---------------------------------------------------------------------------
// Collaboration Surface
// ---------------------------------------------------------------------------

function renderCollaboration(data, taskIds, taskResults) {
    const container = document.getElementById('collaboration-surface');
    if (!container) return;

    const collaboration = deriveCollaborationView(data, taskIds, taskResults);
    const owners = collaboration.owners || [];
    const blockedTasks = collaboration.blockedTasks || [];
    const activity = collaboration.activity || [];
    const unassignedTasks = collaboration.unassignedTasks || [];

    const ownersHtml = owners.length > 0
        ? owners.map(owner => `
            <div class="collab-item">
                <div class="collab-item-title">${escapeHtml(owner.label)}</div>
                <div class="collab-item-meta">
                    ${owner.taskCount} tasks · ${owner.activeCount} active · ${owner.blockedCount} blocked
                </div>
            </div>`).join('')
        : `<div class="collab-muted">No explicit task owners in this run.</div>`;

    const blockedHtml = blockedTasks.length > 0
        ? blockedTasks.map(item => `
            <div class="collab-item">
                <div class="collab-item-title">${escapeHtml(item.taskId)}</div>
                <div class="collab-item-meta">
                    waiting on ${escapeHtml(item.blockedBy.join(', '))}
                </div>
            </div>`).join('')
        : `<div class="collab-muted">No tasks are currently blocked.</div>`;

    const unassignedHtml = unassignedTasks.length > 0
        ? `<div class="collab-item-meta">${escapeHtml(unassignedTasks.join(', '))}</div>`
        : `<div class="collab-muted">Every task shown here has an owner or runtime identity.</div>`;

    const activityHtml = activity.length > 0
        ? activity.map(item => `
            <div class="collab-item">
                <div class="collab-item-title">${escapeHtml(item.message)}</div>
                <div class="collab-item-meta">
                    ${item.owner ? `${escapeHtml(item.owner)} · ` : ''}${formatDate(item.timestamp)}
                </div>
            </div>`).join('')
        : `<div class="collab-muted">No recent activity captured for this run yet.</div>`;

    container.innerHTML = `
        <div class="card">
            <div class="card-header">
                <span class="card-title">Collaboration</span>
                <span class="text-2">${owners.length} owners · ${blockedTasks.length} blocked</span>
            </div>
            <div class="collab-summary-grid">
                <div class="collab-card">
                    <h4>Owners</h4>
                    <div class="collab-inline-metric">${owners.length}</div>
                    <div class="collab-list">${ownersHtml}</div>
                </div>
                <div class="collab-card">
                    <h4>Blocked Tasks</h4>
                    <div class="collab-inline-metric">${blockedTasks.length}</div>
                    <div class="collab-list">${blockedHtml}</div>
                </div>
                <div class="collab-card">
                    <h4>Unassigned</h4>
                    <div class="collab-inline-metric">${unassignedTasks.length}</div>
                    ${unassignedHtml}
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <span class="card-title">Recent Activity</span>
                <span class="text-2">${activity.length} updates</span>
            </div>
            <div class="collab-list">${activityHtml}</div>
        </div>
    `;
}

function deriveCollaborationView(data, taskIds, taskResults) {
    const tasks = data?.collaboration?.tasks || {};
    const ownerMap = new Map();
    const blockedTasks = [];
    const unassignedTasks = [];
    const successLike = new Set(['success', 'dry_run']);
    const terminal = new Set(['success', 'failed', 'soft_failed', 'skipped', 'dry_run']);

    for (const tid of taskIds) {
        const taskMeta = tasks[tid] || {};
        const owner = taskMeta.owner || null;
        const dependsOn = Array.isArray(taskMeta.depends_on) ? taskMeta.depends_on : [];
        const status = taskResults[tid]?.status || taskStatuses[tid] || 'pending';
        const blockedBy = dependsOn.filter(depId => !successLike.has(taskResults[depId]?.status || taskStatuses[depId] || 'pending'));

        if (owner) {
            if (!ownerMap.has(owner)) {
                ownerMap.set(owner, { label: owner, taskCount: 0, activeCount: 0, blockedCount: 0 });
            }
            const bucket = ownerMap.get(owner);
            bucket.taskCount += 1;
            if (!terminal.has(status)) bucket.activeCount += 1;
            if (blockedBy.length > 0 && status !== 'running' && !terminal.has(status)) bucket.blockedCount += 1;
        } else {
            unassignedTasks.push(tid);
        }

        if (blockedBy.length > 0 && status !== 'running' && !terminal.has(status)) {
            blockedTasks.push({ taskId: tid, blockedBy });
        }
    }

    const activity = Array.isArray(data?.collaboration?.activity)
        ? data.collaboration.activity.slice(-8)
        : [];

    return {
        owners: Array.from(ownerMap.values()).sort((a, b) => b.taskCount - a.taskCount || a.label.localeCompare(b.label)),
        blockedTasks,
        unassignedTasks,
        activity,
    };
}

// ---------------------------------------------------------------------------
// Task Cards
// ---------------------------------------------------------------------------

function renderTaskCards(taskIds, taskResults, runId, collaborationTasks = {}) {
    const container = document.getElementById('task-list');
    if (!container) return;

    if (taskIds.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">&#9881;</div>
                <h4>No tasks</h4>
            </div>`;
        return;
    }

    let html = '';
    for (const tid of taskIds) {
        const tr = taskResults[tid];
        const taskMeta = collaborationTasks[tid] || {};
        const status = tr?.status || taskStatuses[tid] || 'pending';
        const duration = tr?.duration_sec;
        const costText = (tr?.cost_usd != null) ? `$${tr.cost_usd.toFixed(2)}` : '';
        const tu = tr?.token_usage;
        const taskTokens = tu?.total_tokens;
        const owner = taskMeta.owner || '';
        const runtime = taskMeta.runtime || '';
        const description = taskMeta.description || '';
        const dependsOn = Array.isArray(taskMeta.depends_on) ? taskMeta.depends_on : [];
        const lastProgressPct = tr?.last_progress_pct ?? taskMeta.last_progress_pct;
        const blockedBy = dependsOn.filter(depId => !['success', 'dry_run'].includes(taskResults[depId]?.status || taskStatuses[depId] || 'pending'));
        const tokenLabel = taskTokens != null ? formatTokenCount(taskTokens) : '';
        const tokenTitle = tu && taskTokens != null
            ? `in: ${(tu.input_tokens || 0).toLocaleString('en-US')} | cached: ${(tu.cached_tokens || 0).toLocaleString('en-US')} | out: ${(tu.output_tokens || 0).toLocaleString('en-US')}`
            : '';
        const pills = [
            owner ? `<span class="pill pill-owner">${escapeHtml(owner)}</span>` : '',
            runtime ? `<span class="pill pill-runtime">${escapeHtml(runtime)}</span>` : '',
            blockedBy.length > 0 && status !== 'running' && !['success', 'failed', 'soft_failed', 'skipped', 'dry_run'].includes(status)
                ? `<span class="pill pill-blocked">blocked by ${escapeHtml(blockedBy.join(', '))}</span>`
                : '',
            Number.isFinite(lastProgressPct)
                ? `<span class="pill pill-progress">${Math.round(lastProgressPct)}%</span>`
                : '',
        ].filter(Boolean).join('');
        const subtitleParts = [];
        if (description) subtitleParts.push(escapeHtml(description));
        if (!description && dependsOn.length > 0) subtitleParts.push(`depends on ${escapeHtml(dependsOn.join(', '))}`);
        if (description && dependsOn.length > 0) subtitleParts.push(`deps: ${escapeHtml(dependsOn.join(', '))}`);
        html += `
            <div class="task-card" data-task-id="${escapeHtml(tid)}" data-status="${status}" onclick="toggleLog('${escapeHtml(tid)}')">
                <div class="task-info">
                    <div class="task-title-block">
                        ${badgeFor(status)}
                        <span class="task-id">${escapeHtml(tid)}</span>
                        <div class="task-pills">${pills}</div>
                    </div>
                    ${subtitleParts.length > 0 ? `<div class="task-subtitle">${subtitleParts.join(' · ')}</div>` : ''}
                </div>
                <div class="task-card-right">
                    ${tokenLabel ? `<span class="task-tokens" title="${escapeHtml(tokenTitle)}">${tokenLabel}</span>` : ''}
                    ${costText ? `<span class="task-cost">${costText}</span>` : ''}
                    <span class="task-meta">${formatDuration(duration)}</span>
                    <a class="btn btn-ghost btn-xs" href="/static/logs.html?run_id=${encodeURIComponent(runId || '')}&task_id=${encodeURIComponent(tid)}" onclick="event.stopPropagation()" title="Full Log">&#128196;</a>
                </div>
            </div>
            <div class="log-viewer" id="log-${escapeHtml(tid)}"></div>`;
    }
    container.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Task Charts
// ---------------------------------------------------------------------------

function renderTaskCharts(taskResults) {
    if (runDetailCharts.duration) { runDetailCharts.duration.destroy(); runDetailCharts.duration = null; }
    if (runDetailCharts.statusDonut) { runDetailCharts.statusDonut.destroy(); runDetailCharts.statusDonut = null; }
    if (runDetailCharts.cost) { runDetailCharts.cost.destroy(); runDetailCharts.cost = null; }
    if (runDetailCharts.tokens) { runDetailCharts.tokens.destroy(); runDetailCharts.tokens = null; }
    if (runDetailCharts.tokensStacked) { runDetailCharts.tokensStacked.destroy(); runDetailCharts.tokensStacked = null; }
    if (runDetailCharts.engineCost) { runDetailCharts.engineCost.destroy(); runDetailCharts.engineCost = null; }
    if (runDetailCharts.engineTokens) { runDetailCharts.engineTokens.destroy(); runDetailCharts.engineTokens = null; }

    const chartRow = document.getElementById('task-charts');
    if (!chartRow) return;

    const hasDuration = Object.values(taskResults).some(tr => tr.duration_sec != null);
    const hasCost = Object.values(taskResults).some(tr => tr.cost_usd != null && tr.cost_usd > 0);

    // Build status distribution from task results
    const statusDist = {};
    for (const tr of Object.values(taskResults)) {
        const s = tr.status || 'unknown';
        statusDist[s] = (statusDist[s] || 0) + 1;
    }
    const hasStatus = Object.keys(statusDist).length > 0;

    if (!hasDuration && !hasStatus) {
        chartRow.classList.add('hidden');
    } else {
        chartRow.classList.remove('hidden');

        // Duration bar
        const durationCard = document.getElementById('chart-task-duration');
        if (hasDuration && durationCard) {
            durationCard.classList.remove('hidden');
            const height = Math.max(200, Object.keys(taskResults).length * 36);
            const canvasContainer = durationCard.querySelector('.chart-canvas-container');
            if (canvasContainer) canvasContainer.style.height = `${height}px`;
            runDetailCharts.duration = createDurationBar('canvas-task-duration', taskResults);
        } else if (durationCard) {
            durationCard.classList.add('hidden');
        }

        // Status donut
        const statusCard = document.getElementById('chart-task-status');
        if (hasStatus && statusCard) {
            statusCard.classList.remove('hidden');
            runDetailCharts.statusDonut = createStatusDonut('canvas-task-status', statusDist);
        } else if (statusCard) {
            statusCard.classList.add('hidden');
        }
    }

    // Cost + Engine Cost row (only if cost data exists)
    const costRow = document.getElementById('task-cost-charts');
    if (hasCost && costRow) {
        costRow.classList.remove('hidden');
        // Cost bar (left)
        const costEntries = Object.entries(taskResults).filter(([, tr]) => tr.cost_usd != null && tr.cost_usd > 0);
        const costHeight = Math.max(200, costEntries.length * 36);
        const costCard = document.getElementById('chart-task-cost');
        if (costCard) {
            const cc = costCard.querySelector('.chart-canvas-container');
            if (cc) cc.style.height = `${costHeight}px`;
        }
        runDetailCharts.cost = createCostBar('canvas-task-cost', taskResults);
        // Engine cost donut (right)
        const engineCostCard = document.getElementById('chart-engine-cost');
        if (engineCostCard) {
            engineCostCard.classList.remove('hidden');
            runDetailCharts.engineCost = createEngineCostDonut('canvas-engine-cost', taskResults);
        }
    } else if (costRow) {
        costRow.classList.add('hidden');
    }

    // Token Usage + Engine Tokens row
    const tokenRow = document.getElementById('task-token-charts');
    const hasTokens = Object.values(taskResults).some(tr => tr.token_usage?.total_tokens != null);
    if (hasTokens && tokenRow) {
        tokenRow.classList.remove('hidden');
        // Token bar (left)
        const tokenEntries = Object.values(taskResults).filter(tr => tr.token_usage?.total_tokens != null);
        const tokenHeight = Math.max(200, tokenEntries.length * 36);
        const tokenCard = document.getElementById('chart-task-tokens');
        if (tokenCard) {
            const cc = tokenCard.querySelector('.chart-canvas-container');
            if (cc) cc.style.height = `${tokenHeight}px`;
        }
        runDetailCharts.tokens = createTokenBar('canvas-task-tokens', taskResults);
        // Engine tokens donut (right)
        const engineTokenCard = document.getElementById('chart-engine-tokens');
        if (engineTokenCard) {
            engineTokenCard.classList.remove('hidden');
            runDetailCharts.engineTokens = createEngineTokenDonut('canvas-engine-tokens', taskResults);
        }
    } else if (tokenRow) {
        tokenRow.classList.add('hidden');
    }

    // Token Breakdown stacked (full width, separate row)
    const tokenBreakdownRow = document.getElementById('task-token-breakdown');
    if (hasTokens && tokenBreakdownRow) {
        tokenBreakdownRow.classList.remove('hidden');
        const stackedEntries = Object.values(taskResults).filter(tr => tr.token_usage?.total_tokens != null);
        const stackedHeight = Math.max(200, stackedEntries.length * 36);
        const stackedCard = document.getElementById('chart-task-tokens-stacked');
        if (stackedCard) {
            const cc = stackedCard.querySelector('.chart-canvas-container');
            if (cc) cc.style.height = `${stackedHeight}px`;
        }
        runDetailCharts.tokensStacked = createTokenStackedBar('canvas-task-tokens-stacked', taskResults);
    } else if (tokenBreakdownRow) {
        tokenBreakdownRow.classList.add('hidden');
    }
}

// ---------------------------------------------------------------------------
// Gantt Timeline
// ---------------------------------------------------------------------------

function renderGanttTimeline(runData, taskResults) {
    const container = document.getElementById('gantt-timeline');
    if (!container) return;

    const entries = Object.entries(taskResults).filter(([, tr]) => tr.started_at && tr.finished_at);
    if (entries.length === 0) {
        container.classList.add('hidden');
        return;
    }
    container.classList.remove('hidden');

    // Find global min/max timestamps
    let globalStart = Infinity;
    let globalEnd = -Infinity;
    for (const [, tr] of entries) {
        const s = new Date(tr.started_at).getTime();
        const e = new Date(tr.finished_at).getTime();
        if (s < globalStart) globalStart = s;
        if (e > globalEnd) globalEnd = e;
    }
    const totalMs = globalEnd - globalStart;
    if (totalMs <= 0) { container.classList.add('hidden'); return; }

    // Sort by start time
    entries.sort(([, a], [, b]) => new Date(a.started_at).getTime() - new Date(b.started_at).getTime());

    // Time axis labels
    const axisTicks = 5;
    let axisHtml = '<div class="gantt-axis">';
    for (let i = 0; i <= axisTicks; i++) {
        const sec = (totalMs / 1000) * (i / axisTicks);
        axisHtml += `<span class="gantt-tick" style="left:${(i / axisTicks * 100).toFixed(1)}%">${formatDuration(sec)}</span>`;
    }
    axisHtml += '</div>';

    // Bars
    let barsHtml = '';
    for (const [tid, tr] of entries) {
        const startMs = new Date(tr.started_at).getTime() - globalStart;
        const durationMs = new Date(tr.finished_at).getTime() - new Date(tr.started_at).getTime();
        const leftPct = (startMs / totalMs * 100).toFixed(2);
        const widthPct = Math.max(0.5, durationMs / totalMs * 100).toFixed(2);
        const status = tr.status || 'pending';
        const costLabel = tr.cost_usd != null ? ` · $${tr.cost_usd.toFixed(2)}` : '';

        barsHtml += `
            <div class="gantt-row">
                <div class="gantt-label">${escapeHtml(tid)}</div>
                <div class="gantt-track">
                    <div class="gantt-bar gantt-bar-${status}"
                         style="left:${leftPct}%;width:${widthPct}%"
                         title="${escapeHtml(tid)}: ${formatDuration(durationMs / 1000)} (${status})${costLabel}">
                    </div>
                </div>
            </div>`;
    }

    container.innerHTML = `
        <div class="card">
            <div class="card-header">
                <span class="card-title">Execution Timeline</span>
                <span class="text-2" style="font-size:0.78rem">${formatDuration(totalMs / 1000)} total</span>
            </div>
            <div class="gantt-container">
                ${axisHtml}
                ${barsHtml}
            </div>
        </div>`;
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

function updateProgress(taskIds, taskResults) {
    const total = taskIds.length;
    let completed = 0;
    let hasFailures = false;

    for (const tid of taskIds) {
        const tr = taskResults[tid];
        const status = tr?.status || taskStatuses[tid] || 'pending';
        if (['success', 'failed', 'soft_failed', 'skipped', 'dry_run'].includes(status)) {
            completed++;
        }
        if (['failed', 'soft_failed'].includes(status)) {
            hasFailures = true;
        }
    }

    const pct = total > 0 ? (completed / total * 100) : 0;
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    if (fill) {
        fill.style.width = `${pct}%`;
        if (hasFailures) fill.classList.add('has-failures');
        else fill.classList.remove('has-failures');
    }
    if (text) text.textContent = `${completed} / ${total} tasks`;
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

function connectSSE(runId) {
    if (eventSource) eventSource.close();

    eventSource = new EventSource(`${API}/runs/${encodeURIComponent(runId)}/events`);

    eventSource.addEventListener('task_complete', (e) => {
        const data = JSON.parse(e.data);
        taskStatuses[data.task_id] = data.status;
        if (!currentRunDetail) {
            loadRunDetail(runId);
            return;
        }

        if (!currentRunDetail.task_results) currentRunDetail.task_results = {};
        currentRunDetail.task_results[data.task_id] = {
            ...(currentRunDetail.task_results[data.task_id] || {}),
            ...data,
        };
        if (!Array.isArray(currentRunDetail.task_ids)) {
            currentRunDetail.task_ids = [];
        }
        if (!currentRunDetail.task_ids.includes(data.task_id)) {
            currentRunDetail.task_ids.push(data.task_id);
        }
        appendActivity(currentRunDetail, data);
        renderRunDetail(currentRunDetail, runId);
    });

    eventSource.addEventListener('run_complete', (e) => {
        const data = JSON.parse(e.data);
        const statusEl = document.getElementById('run-status');
        if (statusEl) {
            statusEl.innerHTML = badgeFor(data.success ? 'success' : 'failed');
        }
        const durationEl = document.getElementById('run-duration');
        if (durationEl && data.finished_at && data.started_at) {
            const dur = (new Date(data.finished_at) - new Date(data.started_at)) / 1000;
            durationEl.textContent = formatDuration(dur);
        }

        showToast(
            data.success ? 'Run completed successfully' : 'Run finished with failures',
            data.success ? 'success' : 'error'
        );

        // Reload to get full data for charts
        const params = new URLSearchParams(window.location.search);
        const runId = params.get('id');
        if (runId) loadRunDetail(runId);

        eventSource.close();
    });

    eventSource.onerror = () => {
        eventSource.close();
    };
}

function appendActivity(runDetail, taskResult) {
    if (!runDetail.collaboration) runDetail.collaboration = {};
    if (!Array.isArray(runDetail.collaboration.activity)) {
        runDetail.collaboration.activity = [];
    }

    const taskMeta = runDetail.collaboration?.tasks?.[taskResult.task_id] || {};
    const status = taskResult.status || 'completed';
    runDetail.collaboration.activity.push({
        timestamp: taskResult.finished_at || new Date().toISOString(),
        event: 'task_complete',
        task_id: taskResult.task_id,
        owner: taskMeta.owner || null,
        message: `${taskResult.task_id} finished as ${status}`,
    });
    runDetail.collaboration.activity = runDetail.collaboration.activity.slice(-8);
}

// ---------------------------------------------------------------------------
// Log toggle (inline)
// ---------------------------------------------------------------------------

async function toggleLog(taskId) {
    const viewer = document.getElementById(`log-${taskId}`);
    if (!viewer) return;

    if (viewer.classList.contains('open')) {
        viewer.classList.remove('open');
        return;
    }

    viewer.textContent = 'Loading...';
    viewer.classList.add('open');

    const params = new URLSearchParams(window.location.search);
    const runId = params.get('id');

    try {
        const data = await apiFetch(`/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(taskId)}/log`);
        viewer.innerHTML = `<pre class="log-pre">${simpleMarkdown(data.content || '(empty log)')}</pre>`;
    } catch {
        viewer.textContent = '(log not available yet)';
    }
}
