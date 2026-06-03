/* Maestro UI — Log Viewer (dedicated page) */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let logLines = [];       // All lines (raw strings)
let filteredIndices = []; // Indices into logLines that pass the current filter
let matchIndices = [];    // Indices into filteredIndices that match search
let currentMatch = -1;    // Current match position
let currentFilter = 'all';

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function initLogViewer() {
    const params = new URLSearchParams(window.location.search);
    const runId = params.get('run_id');
    const taskId = params.get('task_id');

    if (!runId || !taskId) {
        document.getElementById('log-content').innerHTML =
            '<div class="empty-state"><h4>Missing run_id or task_id</h4></div>';
        return;
    }

    // Set header info
    const taskLabel = document.getElementById('log-task-id');
    if (taskLabel) taskLabel.textContent = taskId;

    const backLink = document.getElementById('log-back-link');
    if (backLink) backLink.href = `/static/run.html?id=${encodeURIComponent(runId)}`;

    loadFullLog(runId, taskId);
    setupKeyboardShortcuts();
}

// ---------------------------------------------------------------------------
// Load log
// ---------------------------------------------------------------------------

async function loadFullLog(runId, taskId) {
    const content = document.getElementById('log-content');
    if (!content) return;

    content.innerHTML = '<div class="log-loading">Loading log...</div>';

    try {
        const data = await apiFetch(
            `/runs/${encodeURIComponent(runId)}/tasks/${encodeURIComponent(taskId)}/log`
        );
        const raw = data.content || '';
        logLines = raw.split('\n');
        filteredIndices = logLines.map((_, i) => i);
        renderLog();
        updateStatusBar();
    } catch (err) {
        content.innerHTML = `<div class="empty-state"><h4>Error loading log</h4><p>${escapeHtml(err.message)}</p></div>`;
    }
}

// ---------------------------------------------------------------------------
// Render log lines
// ---------------------------------------------------------------------------

function renderLog() {
    const content = document.getElementById('log-content');
    if (!content) return;

    if (logLines.length === 0) {
        content.innerHTML = '<div class="empty-state"><h4>Empty log</h4></div>';
        return;
    }

    const searchTerm = (document.getElementById('log-search')?.value || '').toLowerCase();
    matchIndices = [];
    currentMatch = -1;

    let html = '';
    for (let fi = 0; fi < filteredIndices.length; fi++) {
        const lineIdx = filteredIndices[fi];
        const line = logLines[lineIdx];
        const lineNum = lineIdx + 1;
        const levelClass = getLineClass(line);

        let displayLine = escapeHtml(line);
        // Apply inline markdown (bold, code, headings)
        displayLine = displayLine.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        displayLine = displayLine.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');
        const hm = displayLine.match(/^(#{2,4})\s+(.+)$/);
        if (hm) displayLine = `<strong class="md-h${hm[1].length}">${hm[2]}</strong>`;

        // Highlight search matches
        if (searchTerm && line.toLowerCase().includes(searchTerm)) {
            matchIndices.push(fi);
            displayLine = highlightMatches(displayLine, searchTerm);
        }

        html += `<div class="log-line ${levelClass}" data-line="${lineNum}" id="log-line-${lineNum}">` +
            `<span class="log-line-num">${lineNum}</span>` +
            `<span class="log-line-content">${displayLine}</span>` +
            `</div>`;
    }

    content.innerHTML = html;
    updateMatchCounter();
}

function getLineClass(line) {
    const lower = line.toLowerCase();
    if (/\[error\]|error:|error !|traceback|exception/i.test(line)) return 'log-line-error';
    if (/\[warn\]|warning:|warn:/i.test(line)) return 'log-line-warn';
    if (/\[info\]|info:/i.test(line)) return 'log-line-info';
    if (/^\d{4}-\d{2}-\d{2}|^\[\d{4}/.test(line)) return 'log-line-timestamp';
    return '';
}

function highlightMatches(escapedLine, searchTerm) {
    // Case-insensitive highlight on already-escaped HTML
    const regex = new RegExp(`(${escapeRegex(escapeHtml(searchTerm))})`, 'gi');
    return escapedLine.replace(regex, '<mark class="log-match">$1</mark>');
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

function filterByLevel(level) {
    currentFilter = level;
    const filterBtn = document.getElementById('filter-level');
    if (filterBtn) filterBtn.value = level;

    if (level === 'all') {
        filteredIndices = logLines.map((_, i) => i);
    } else {
        filteredIndices = [];
        for (let i = 0; i < logLines.length; i++) {
            const cls = getLineClass(logLines[i]);
            if (level === 'error' && cls === 'log-line-error') filteredIndices.push(i);
            else if (level === 'warn' && cls === 'log-line-warn') filteredIndices.push(i);
            else if (level === 'info' && cls === 'log-line-info') filteredIndices.push(i);
        }
    }

    renderLog();
    updateStatusBar();
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

function searchLog() {
    renderLog();
    if (matchIndices.length > 0) {
        currentMatch = 0;
        scrollToMatch(currentMatch);
    }
}

function nextMatch() {
    if (matchIndices.length === 0) return;
    currentMatch = (currentMatch + 1) % matchIndices.length;
    scrollToMatch(currentMatch);
}

function prevMatch() {
    if (matchIndices.length === 0) return;
    currentMatch = (currentMatch - 1 + matchIndices.length) % matchIndices.length;
    scrollToMatch(currentMatch);
}

function scrollToMatch(matchIdx) {
    const fi = matchIndices[matchIdx];
    const lineIdx = filteredIndices[fi];
    const lineNum = lineIdx + 1;
    const el = document.getElementById(`log-line-${lineNum}`);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Flash highlight
        el.classList.add('log-line-active');
        setTimeout(() => el.classList.remove('log-line-active'), 1500);
    }
    updateMatchCounter();
}

function updateMatchCounter() {
    const counter = document.getElementById('match-counter');
    if (!counter) return;
    if (matchIndices.length === 0) {
        const searchVal = document.getElementById('log-search')?.value || '';
        counter.textContent = searchVal ? 'No matches' : '';
    } else {
        counter.textContent = `${currentMatch + 1} / ${matchIndices.length}`;
    }
}

// ---------------------------------------------------------------------------
// Jump to line
// ---------------------------------------------------------------------------

function jumpToLine() {
    const input = document.getElementById('jump-line');
    if (!input) return;
    const lineNum = parseInt(input.value, 10);
    if (isNaN(lineNum) || lineNum < 1) return;

    const el = document.getElementById(`log-line-${lineNum}`);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('log-line-active');
        setTimeout(() => el.classList.remove('log-line-active'), 1500);
    }
    input.value = '';
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function updateStatusBar() {
    const totalEl = document.getElementById('log-total-lines');
    const errorEl = document.getElementById('log-error-count');
    const warnEl = document.getElementById('log-warn-count');
    const shownEl = document.getElementById('log-shown-lines');

    if (totalEl) totalEl.textContent = `${logLines.length} lines`;

    let errors = 0, warns = 0;
    for (const line of logLines) {
        const cls = getLineClass(line);
        if (cls === 'log-line-error') errors++;
        else if (cls === 'log-line-warn') warns++;
    }
    if (errorEl) errorEl.textContent = `${errors} errors`;
    if (warnEl) warnEl.textContent = `${warns} warnings`;
    if (shownEl) shownEl.textContent = currentFilter !== 'all'
        ? `showing ${filteredIndices.length} of ${logLines.length}`
        : '';
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+F / Cmd+F → focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
            e.preventDefault();
            const search = document.getElementById('log-search');
            if (search) search.focus();
            return;
        }

        // Escape → clear search
        if (e.key === 'Escape') {
            const search = document.getElementById('log-search');
            if (search && document.activeElement === search) {
                search.value = '';
                searchLog();
                search.blur();
                return;
            }
        }

        // Enter / Shift+Enter in search → next/prev match
        if (e.key === 'Enter' && document.activeElement?.id === 'log-search') {
            e.preventDefault();
            if (e.shiftKey) prevMatch();
            else nextMatch();
            return;
        }

        // Enter in jump-line → jump
        if (e.key === 'Enter' && document.activeElement?.id === 'jump-line') {
            e.preventDefault();
            jumpToLine();
            return;
        }

        // Don't trigger shortcuts when typing in inputs
        if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') return;

        // G → focus jump to line
        if (e.key === 'g' || e.key === 'G') {
            e.preventDefault();
            const jumpInput = document.getElementById('jump-line');
            if (jumpInput) jumpInput.focus();
            return;
        }

        // E → filter errors only
        if (e.key === 'e' || e.key === 'E') {
            e.preventDefault();
            filterByLevel(currentFilter === 'error' ? 'all' : 'error');
            return;
        }

        // W → filter warnings only
        if (e.key === 'w' || e.key === 'W') {
            e.preventDefault();
            filterByLevel(currentFilter === 'warn' ? 'all' : 'warn');
            return;
        }

        // Home → scroll to top
        if (e.key === 'Home') {
            document.getElementById('log-content')?.scrollTo({ top: 0, behavior: 'smooth' });
            return;
        }

        // End → scroll to bottom
        if (e.key === 'End') {
            const content = document.getElementById('log-content');
            if (content) content.scrollTo({ top: content.scrollHeight, behavior: 'smooth' });
            return;
        }
    });
}
