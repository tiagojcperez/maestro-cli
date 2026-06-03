/* Maestro UI — Chart.js wrappers (dark theme) */

// ---------------------------------------------------------------------------
// Engine inference helper
// ---------------------------------------------------------------------------

function _inferEngine(command) {
    if (!command) return 'shell';
    const cmd = Array.isArray(command) ? command.join(' ') : String(command);
    if (/\bcodex\b/i.test(cmd)) return 'codex';
    if (/\bclaude\b/i.test(cmd)) return 'claude';
    if (/\bgemini\b/i.test(cmd)) return 'gemini';
    return 'shell';
}

const ENGINE_COLOR_KEYS = { claude: 'accent', codex: 'running', gemini: 'success', shell: 'skipped' };

// ---------------------------------------------------------------------------
// Theme colors — read from CSS custom properties
// ---------------------------------------------------------------------------

function getChartColors() {
    const cs = getComputedStyle(document.documentElement);
    const get = (v) => cs.getPropertyValue(v).trim();
    return {
        success: get('--success') || '#3ddc84',
        failed: get('--failed') || '#f55',
        warning: get('--warning') || '#f5a623',
        running: get('--running') || '#5b9bf5',
        skipped: get('--skipped') || '#6b7280',
        accent: get('--accent') || '#c0785a',
        accentHover: get('--accent-hover') || '#d4926e',
        text1: get('--text-1') || '#c4c9d4',
        text2: get('--text-2') || '#8891a4',
        text3: get('--text-3') || '#545d72',
        bg2: get('--bg-2') || '#151921',
        grid: 'rgba(255,255,255,0.06)',
    };
}

const STATUS_COLOR_MAP = {
    success: 'success',
    failed: 'failed',
    soft_failed: 'warning',
    skipped: 'skipped',
    dry_run: 'warning',
    running: 'running',
    pending: 'skipped',
};

const STATUS_LABELS = {
    success: 'Success',
    failed: 'Failed',
    soft_failed: 'Soft Failed',
    skipped: 'Skipped',
    dry_run: 'Dry Run',
    running: 'Running',
    pending: 'Pending',
};

function statusColor(status, colors) {
    const key = STATUS_COLOR_MAP[status] || 'skipped';
    return colors[key];
}

// ---------------------------------------------------------------------------
// Chart.js global defaults
// ---------------------------------------------------------------------------

function applyChartDefaults() {
    if (typeof Chart === 'undefined') return;
    const c = getChartColors();
    Chart.defaults.color = c.text2;
    Chart.defaults.borderColor = c.grid;
    Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(14,17,23,0.95)';
    Chart.defaults.plugins.tooltip.titleColor = '#f0f2f5';
    Chart.defaults.plugins.tooltip.bodyColor = '#c4c9d4';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.padding = 10;
}

// ---------------------------------------------------------------------------
// Status Donut
// ---------------------------------------------------------------------------

function createStatusDonut(canvasId, distribution) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const labels = [];
    const data = [];
    const bgColors = [];

    const order = ['success', 'failed', 'soft_failed', 'skipped', 'dry_run'];
    for (const s of order) {
        const count = distribution[s];
        if (count && count > 0) {
            labels.push(STATUS_LABELS[s] || s);
            data.push(count);
            bgColors.push(statusColor(s, c));
        }
    }
    // Any remaining statuses
    for (const [s, count] of Object.entries(distribution)) {
        if (!order.includes(s) && count > 0) {
            labels.push(STATUS_LABELS[s] || s);
            data.push(count);
            bgColors.push(c.skipped);
        }
    }

    if (data.length === 0) return null;

    const total = data.reduce((a, b) => a + b, 0);

    return new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data,
                backgroundColor: bgColors,
                borderWidth: 0,
                hoverOffset: 4,
            }],
        },
        options: {
            cutout: '65%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 14, usePointStyle: true, pointStyle: 'circle' },
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.raw} (${(ctx.raw / total * 100).toFixed(0)}%)`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Cost Trend (Line)
// ---------------------------------------------------------------------------

function createCostTrend(canvasId, costByRun) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined' || !costByRun.length) return null;

    const c = getChartColors();
    const sorted = [...costByRun].reverse(); // oldest first

    const labels = sorted.map(r => {
        if (!r.started_at) return '';
        const d = new Date(r.started_at);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    });

    const data = sorted.map(r => r.total_cost_usd || 0);

    return new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Cost (USD)',
                data,
                borderColor: c.accent,
                backgroundColor: c.accent + '18',
                fill: true,
                tension: 0.35,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: c.accent,
                pointBorderColor: c.bg2,
                pointBorderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    ticks: { callback: (v) => `$${v.toFixed(2)}` },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const idx = items[0].dataIndex;
                            return sorted[idx].plan_name || '';
                        },
                        label: (ctx) => ` $${ctx.raw.toFixed(2)}`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Cost by Model (horizontal bars)
// ---------------------------------------------------------------------------

function createCostByModel(canvasId, costByModel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined' || !costByModel.length) return null;

    const c = getChartColors();
    const sorted = [...costByModel]
        .filter(row => row && typeof row.total_cost_usd === 'number' && row.total_cost_usd > 0)
        .sort((a, b) => b.total_cost_usd - a.total_cost_usd)
        .slice(0, 10);

    if (!sorted.length) return null;

    const labels = sorted.map(row => row.model || 'unknown');
    const data = sorted.map(row => row.total_cost_usd);

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Total Cost (USD)',
                data,
                backgroundColor: c.accent + 'cc',
                hoverBackgroundColor: c.accentHover,
                borderRadius: 4,
                barPercentage: 0.72,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: (v) => `$${Number(v).toFixed(2)}`,
                    },
                },
                y: {
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => sorted[items[0].dataIndex].model || 'unknown',
                        label: (ctx) => ` Total: $${ctx.raw.toFixed(2)}`,
                        afterLabel: (ctx) => {
                            const row = sorted[ctx.dataIndex];
                            const tasks = Number(row.task_count || 0);
                            const avg = Number(row.avg_cost_usd || 0);
                            return ` Tasks: ${tasks} · Avg/task: $${avg.toFixed(2)}`;
                        },
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Duration Trend (vertical bars — recent runs)
// ---------------------------------------------------------------------------

function createDurationTrend(canvasId, recentRuns) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined' || !recentRuns.length) return null;

    const c = getChartColors();
    const sorted = [...recentRuns].reverse(); // oldest first

    const labels = sorted.map(r => {
        if (!r.started_at) return '';
        const d = new Date(r.started_at);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    });

    const data = sorted.map(r => r.duration_sec || 0);
    const bgColors = sorted.map(r => {
        if (r.success === true) return c.success + 'cc';
        if (r.success === false) return c.failed + 'cc';
        return c.skipped + 'cc';
    });
    const borderColors = sorted.map(r => {
        if (r.success === true) return c.success;
        if (r.success === false) return c.failed;
        return c.skipped;
    });

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Duration',
                data,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 3,
                barPercentage: 0.75,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    ticks: { callback: (v) => formatDuration(v) },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const idx = items[0].dataIndex;
                            return sorted[idx].plan_name || '';
                        },
                        label: (ctx) => ` ${formatDuration(ctx.raw)}`,
                        afterLabel: (ctx) => {
                            const r = sorted[ctx.dataIndex];
                            return r.success === true ? '  Success' : r.success === false ? '  Failed' : '';
                        },
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Duration Bar (horizontal)
// ---------------------------------------------------------------------------

function createDurationBar(canvasId, taskResults) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const entries = Object.entries(taskResults)
        .filter(([, tr]) => tr.duration_sec != null)
        .sort(([, a], [, b]) => (b.duration_sec || 0) - (a.duration_sec || 0));

    if (entries.length === 0) return null;

    const labels = entries.map(([tid]) => tid);
    const data = entries.map(([, tr]) => tr.duration_sec);
    const bgColors = entries.map(([, tr]) => statusColor(tr.status, c));

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Duration (s)',
                data,
                backgroundColor: bgColors,
                borderRadius: 4,
                barPercentage: 0.7,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: { callback: (v) => formatDuration(v) },
                },
                y: {
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${formatDuration(ctx.raw)}`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Cost Bar (horizontal)
// ---------------------------------------------------------------------------

function createCostBar(canvasId, taskResults) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const entries = Object.entries(taskResults)
        .filter(([, tr]) => tr.cost_usd != null && tr.cost_usd > 0)
        .sort(([, a], [, b]) => (b.cost_usd || 0) - (a.cost_usd || 0));

    if (entries.length === 0) return null;

    const labels = entries.map(([tid]) => tid);
    const data = entries.map(([, tr]) => tr.cost_usd);

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Cost (USD)',
                data,
                backgroundColor: c.accent + 'cc',
                hoverBackgroundColor: c.accentHover,
                borderRadius: 4,
                barPercentage: 0.7,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: { callback: (v) => `$${v.toFixed(2)}` },
                },
                y: {
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` $${ctx.raw.toFixed(2)}`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Token Bar (horizontal, per task, colored by status)
// ---------------------------------------------------------------------------

function createTokenBar(canvasId, taskResults) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const entries = Object.entries(taskResults)
        .filter(([, tr]) => tr.token_usage?.total_tokens != null)
        .sort(([, a], [, b]) => (b.token_usage?.total_tokens || 0) - (a.token_usage?.total_tokens || 0));

    if (entries.length === 0) return null;

    const labels = entries.map(([tid]) => tid);
    const data = entries.map(([, tr]) => tr.token_usage.total_tokens);
    const bgColors = entries.map(([, tr]) => statusColor(tr.status, c));

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Total Tokens',
                data,
                backgroundColor: bgColors,
                borderRadius: 4,
                barPercentage: 0.7,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: (v) => {
                            if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
                            if (v >= 1e3) return `${Math.round(v / 1e3)}K`;
                            return String(Math.round(v));
                        },
                    },
                },
                y: {
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.raw.toLocaleString('en-US')} tokens`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Token Stacked Bar (input / cached / output per task)
// ---------------------------------------------------------------------------

function createTokenStackedBar(canvasId, taskResults) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const entries = Object.entries(taskResults)
        .filter(([, tr]) => tr.token_usage?.total_tokens != null)
        .sort(([, a], [, b]) => (b.token_usage?.total_tokens || 0) - (a.token_usage?.total_tokens || 0));

    if (entries.length === 0) return null;

    const labels = entries.map(([tid]) => tid);
    const inputData = entries.map(([, tr]) => tr.token_usage?.input_tokens || 0);
    const cachedData = entries.map(([, tr]) => tr.token_usage?.cached_tokens || 0);
    const outputData = entries.map(([, tr]) => tr.token_usage?.output_tokens || 0);

    const fmtAxis = (v) => {
        if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
        if (v >= 1e3) return `${Math.round(v / 1e3)}K`;
        return String(Math.round(v));
    };

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Input',
                    data: inputData,
                    backgroundColor: c.running + 'cc',
                    borderRadius: 0,
                    barPercentage: 0.7,
                },
                {
                    label: 'Cached',
                    data: cachedData,
                    backgroundColor: c.success + 'cc',
                    borderRadius: 0,
                    barPercentage: 0.7,
                },
                {
                    label: 'Output',
                    data: outputData,
                    backgroundColor: c.accent + 'cc',
                    borderRadius: 4,
                    barPercentage: 0.7,
                },
            ],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, beginAtZero: true, ticks: { callback: fmtAxis } },
                y: { stacked: true, ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } } },
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 14, usePointStyle: true, pointStyle: 'circle' },
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toLocaleString('en-US')}`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Engine Cost Donut
// ---------------------------------------------------------------------------

function createEngineCostDonut(canvasId, taskResults) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const engineCost = {};
    for (const tr of Object.values(taskResults)) {
        if (tr.cost_usd == null || tr.cost_usd <= 0) continue;
        const engine = _inferEngine(tr.command);
        engineCost[engine] = (engineCost[engine] || 0) + tr.cost_usd;
    }

    const entries = Object.entries(engineCost).filter(([, v]) => v > 0);
    if (entries.length === 0) return null;

    const labels = entries.map(([k]) => k);
    const data = entries.map(([, v]) => v);
    const bgColors = entries.map(([k]) => c[ENGINE_COLOR_KEYS[k] || 'skipped']);
    const total = data.reduce((a, b) => a + b, 0);

    return new Chart(canvas, {
        type: 'doughnut',
        data: { labels, datasets: [{ data, backgroundColor: bgColors, borderWidth: 0, hoverOffset: 4 }] },
        options: {
            cutout: '65%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, usePointStyle: true, pointStyle: 'circle' } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: $${ctx.raw.toFixed(3)} (${(ctx.raw / total * 100).toFixed(0)}%)`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Engine Token Donut
// ---------------------------------------------------------------------------

function createEngineTokenDonut(canvasId, taskResults) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return null;

    const c = getChartColors();
    const engineTokens = {};
    for (const tr of Object.values(taskResults)) {
        const tok = tr.token_usage?.total_tokens;
        if (tok == null || tok <= 0) continue;
        const engine = _inferEngine(tr.command);
        engineTokens[engine] = (engineTokens[engine] || 0) + tok;
    }

    const entries = Object.entries(engineTokens).filter(([, v]) => v > 0);
    if (entries.length === 0) return null;

    const labels = entries.map(([k]) => k);
    const data = entries.map(([, v]) => v);
    const bgColors = entries.map(([k]) => c[ENGINE_COLOR_KEYS[k] || 'skipped']);
    const total = data.reduce((a, b) => a + b, 0);

    return new Chart(canvas, {
        type: 'doughnut',
        data: { labels, datasets: [{ data, backgroundColor: bgColors, borderWidth: 0, hoverOffset: 4 }] },
        options: {
            cutout: '65%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 14, usePointStyle: true, pointStyle: 'circle' } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString('en-US')} (${(ctx.raw / total * 100).toFixed(0)}%)`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Token Trend (line — dashboard)
// ---------------------------------------------------------------------------

function createTokenTrend(canvasId, recentRuns) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined' || !recentRuns.length) return null;

    const c = getChartColors();
    const sorted = [...recentRuns].reverse(); // oldest first
    const filtered = sorted.filter(r => r.total_tokens != null);
    if (filtered.length === 0) return null;

    const labels = filtered.map(r => {
        if (!r.started_at) return '';
        const d = new Date(r.started_at);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    });
    const data = filtered.map(r => r.total_tokens || 0);

    return new Chart(canvas, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Tokens',
                data,
                borderColor: c.running,
                backgroundColor: c.running + '18',
                fill: true,
                tension: 0.35,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: c.running,
                pointBorderColor: c.bg2,
                pointBorderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: (v) => {
                            if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
                            if (v >= 1e3) return `${Math.round(v / 1e3)}K`;
                            return String(Math.round(v));
                        },
                    },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => filtered[items[0].dataIndex].plan_name || '',
                        label: (ctx) => ` ${ctx.raw.toLocaleString('en-US')} tokens`,
                    },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Tokens by Model (horizontal bars — dashboard)
// ---------------------------------------------------------------------------

function createTokenByModel(canvasId, tokensByModel) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined' || !tokensByModel.length) return null;

    const c = getChartColors();
    const sorted = [...tokensByModel]
        .filter(row => row && typeof row.total_tokens === 'number' && row.total_tokens > 0)
        .sort((a, b) => b.total_tokens - a.total_tokens)
        .slice(0, 10);

    if (!sorted.length) return null;

    const labels = sorted.map(row => row.model || 'unknown');
    const data = sorted.map(row => row.total_tokens);

    return new Chart(canvas, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Total Tokens',
                data,
                backgroundColor: c.running + 'cc',
                hoverBackgroundColor: c.running,
                borderRadius: 4,
                barPercentage: 0.72,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        callback: (v) => {
                            if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
                            if (v >= 1e3) return `${Math.round(v / 1e3)}K`;
                            return String(Math.round(v));
                        },
                    },
                },
                y: {
                    ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => sorted[items[0].dataIndex].model || 'unknown',
                        label: (ctx) => ` Total: ${ctx.raw.toLocaleString('en-US')} tokens`,
                        afterLabel: (ctx) => {
                            const row = sorted[ctx.dataIndex];
                            return ` Tasks: ${row.task_count || 0} · Avg/task: ${(row.avg_tokens || 0).toLocaleString('en-US')}`;
                        },
                    },
                },
            },
        },
    });
}
