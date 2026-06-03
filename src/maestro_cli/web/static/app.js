/* Maestro UI — Shared utilities (v3) */

const API = '/api';

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

async function apiFetch(endpoint, options = {}) {
    const url = `${API}${endpoint}`;
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const opts = { ...defaults, ...options };
    const resp = await fetch(url, opts);
    if (!resp.ok && resp.status !== 400) {
        throw new Error(`API error: ${resp.status} ${resp.statusText}`);
    }
    return resp.json();
}

function badgeFor(status) {
    const s = (status || 'pending').toLowerCase();
    return `<span class="badge badge-${s}">${s}</span>`;
}

function formatDuration(sec) {
    if (sec == null) return '—';
    if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
    if (sec < 60) return `${sec.toFixed(1)}s`;
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(0);
    return `${m}m ${s}s`;
}

function formatTokenCount(n) {
    if (n == null || !Number.isFinite(n)) return '—';
    return Math.round(n).toLocaleString('en-US');
}

function formatDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString();
}

function escapeHtml(text) {
    const el = document.createElement('div');
    el.textContent = text;
    return el.innerHTML;
}

/**
 * Lightweight markdown-to-HTML for log/report rendering.
 * Escapes HTML first (safe), then applies inline formatting.
 * Supports: **bold**, `code`, headings (##), and blank-line paragraphs.
 */
function simpleMarkdown(text) {
    if (!text) return '';
    const lines = escapeHtml(text).split('\n');
    const out = [];
    for (const line of lines) {
        let l = line;
        // Headings (## ... ####)
        const hm = l.match(/^(#{2,4})\s+(.+)$/);
        if (hm) {
            const lvl = hm[1].length;
            out.push(`<strong class="md-h${lvl}">${hm[2]}</strong>`);
            continue;
        }
        // Bold **text**
        l = l.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // Inline code `text`
        l = l.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');
        out.push(l);
    }
    return out.join('\n');
}

// ---------------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------------

function showToast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('hiding');
        toast.addEventListener('animationend', () => toast.remove());
    }, duration);
}

// ---------------------------------------------------------------------------
// Collapsible / Toggle helpers
// ---------------------------------------------------------------------------

function toggleCollapsible(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('hidden');
}

function hideValidateSection() {
    const el = document.getElementById('validate-section');
    if (el) el.classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Modal handlers (global)
// ---------------------------------------------------------------------------

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
        e.target.classList.remove('open');
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.open').forEach(el => el.classList.remove('open'));
    }
});
