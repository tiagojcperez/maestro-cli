/* Maestro UI — New Run Wizard */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentStep = 1;
let validatedPlan = null;   // full /plans/validate response (.plan)
let taskSelections = {};     // { taskId: boolean }
let validateTimeout = null;  // debounce timer
let droppedYamlContent = null; // content from drag & drop (no filesystem path)
const WIZARD_STATE_KEY = 'maestro.newRunWizard.v1';
const WIZARD_STEP_MIN = 1;
const WIZARD_STEP_MAX = 3;

const PROFILE_DESCRIPTIONS = {
    plan: 'Use YAML args exactly as written',
    safe: 'Strip dangerous flags, add sandbox gates',
    yolo: 'Bypass approvals and sandboxing',
};

function normalizeValidatedPlan(plan) {
    const input = plan || {};
    const ids = Array.isArray(input.task_ids) ? input.task_ids : [];
    const rawDetails = Array.isArray(input.task_details) ? input.task_details : [];

    const detailsById = new Map();
    for (const d of rawDetails) {
        if (!d || !d.id) continue;
        detailsById.set(d.id, {
            id: d.id,
            description: d.description || '',
            engine: d.engine ?? null,
            model: d.model ?? null,
            has_command: Boolean(d.has_command),
            depends_on: Array.isArray(d.depends_on) ? d.depends_on : [],
            allow_failure: Boolean(d.allow_failure),
        });
    }

    const normalizedDetails = [];
    const seen = new Set();

    for (const id of ids) {
        seen.add(id);
        normalizedDetails.push(
            detailsById.get(id) || {
                id,
                description: '',
                engine: null,
                model: null,
                has_command: false,
                depends_on: [],
                allow_failure: false,
            }
        );
    }

    for (const [id, detail] of detailsById.entries()) {
        if (seen.has(id)) continue;
        normalizedDetails.push(detail);
    }

    const normalizedIds = ids.length > 0
        ? ids
        : normalizedDetails.map(d => d.id);

    return {
        ...input,
        task_ids: normalizedIds,
        task_details: normalizedDetails,
    };
}

function compactText(value) {
    return String(value ?? '')
        .replace(/\s+/g, ' ')
        .trim();
}

function truncateText(value, maxLen = 48) {
    const text = compactText(value);
    if (text.length <= maxLen) return text;
    return `${text.slice(0, maxLen - 1)}…`;
}

function clampWizardStep(step) {
    const parsed = Number.parseInt(step, 10);
    if (!Number.isFinite(parsed)) return WIZARD_STEP_MIN;
    return Math.min(WIZARD_STEP_MAX, Math.max(WIZARD_STEP_MIN, parsed));
}

function parseStepParam(rawStep) {
    if (rawStep == null || rawStep === '') return null;
    return clampWizardStep(rawStep);
}

function readPersistedWizardState() {
    try {
        const raw = sessionStorage.getItem(WIZARD_STATE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return null;
        return parsed;
    } catch {
        return null;
    }
}

function syncWizardUrl(step = currentStep, explicitPlanPath = null, hasDroppedContent = Boolean(droppedYamlContent)) {
    const url = new URL(window.location.href);
    const normalizedStep = clampWizardStep(step);

    if (normalizedStep > 1) {
        url.searchParams.set('step', String(normalizedStep));
    } else {
        url.searchParams.delete('step');
    }

    const pathInput = document.getElementById('plan-path-input');
    const planPath = explicitPlanPath != null
        ? compactText(explicitPlanPath)
        : compactText(pathInput?.value || '');

    if (planPath && !hasDroppedContent) {
        url.searchParams.set('plan', planPath);
    } else {
        url.searchParams.delete('plan');
    }

    const nextUrl = `${url.pathname}${url.search}${url.hash}`;
    const currentUrl = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextUrl !== currentUrl) {
        window.history.replaceState({}, '', nextUrl);
    }
}

function persistWizardState() {
    const pathInput = document.getElementById('plan-path-input');
    const profileInput = document.getElementById('cfg-profile');
    const parallelInput = document.getElementById('cfg-parallel');
    const dryRunInput = document.getElementById('cfg-dry-run');

    const snapshot = {
        step: clampWizardStep(currentStep),
        planPath: compactText(pathInput?.value || ''),
        droppedYamlContent: droppedYamlContent || null,
        validatedPlan: validatedPlan || null,
        taskSelections: { ...taskSelections },
        profile: profileInput?.value || 'plan',
        maxParallel: parallelInput?.value || '',
        dryRun: Boolean(dryRunInput?.checked),
        savedAt: Date.now(),
    };

    try {
        sessionStorage.setItem(WIZARD_STATE_KEY, JSON.stringify(snapshot));
    } catch {
        // ignore storage failures (quota/privacy mode)
    }

    syncWizardUrl(snapshot.step, snapshot.planPath, Boolean(snapshot.droppedYamlContent));
}

function clearPersistedWizardState() {
    try {
        sessionStorage.removeItem(WIZARD_STATE_KEY);
    } catch {
        // ignore
    }
}

function applyPersistedExecutionOptions(state) {
    if (!state || typeof state !== 'object') return;

    const profileInput = document.getElementById('cfg-profile');
    const parallelInput = document.getElementById('cfg-parallel');
    const dryRunInput = document.getElementById('cfg-dry-run');

    if (profileInput && typeof state.profile === 'string') {
        const allowedProfiles = ['plan', 'safe', 'yolo'];
        if (allowedProfiles.includes(state.profile)) {
            profileInput.value = state.profile;
        }
    }

    if (parallelInput && state.maxParallel != null && state.maxParallel !== '') {
        parallelInput.value = String(state.maxParallel);
    }

    if (dryRunInput && typeof state.dryRun === 'boolean') {
        dryRunInput.checked = state.dryRun;
    }
}

function applyPersistedTaskSelections(savedSelections) {
    if (!validatedPlan || !savedSelections || typeof savedSelections !== 'object') return;

    const merged = {};
    for (const tid of validatedPlan.task_ids || []) {
        merged[tid] = savedSelections[tid] !== false;
    }
    taskSelections = merged;
}

// ---------------------------------------------------------------------------
// Stepper navigation
// ---------------------------------------------------------------------------

function goToStep(step) {
    const targetStep = clampWizardStep(step);

    // Prevent forward navigation without valid plan
    if (targetStep >= 2 && !validatedPlan) return;

    // Render step content on entry
    if (targetStep === 2) renderTaskSelector();
    if (targetStep === 3) renderReviewSummary();

    // Hide current, show target
    const panels = document.querySelectorAll('.wizard-panel');
    panels.forEach(p => p.classList.remove('active'));

    const targetPanel = document.querySelectorAll('.wizard-panel')[targetStep - 1];
    if (targetPanel) targetPanel.classList.add('active');

    currentStep = targetStep;
    updateStepperUI();
    persistWizardState();
}

function updateStepperUI() {
    const steps = document.querySelectorAll('.wizard-step');
    steps.forEach(el => {
        const s = parseInt(el.dataset.step);
        el.classList.remove('active', 'completed');
        if (s === currentStep) {
            el.classList.add('active');
        } else if (s < currentStep) {
            el.classList.add('completed');
            el.querySelector('.wizard-step-circle').innerHTML = '&#10003;';
        } else {
            el.querySelector('.wizard-step-circle').textContent = s;
        }
    });

    // Connectors
    const c12 = document.getElementById('connector-1-2');
    const c23 = document.getElementById('connector-2-3');
    if (c12) c12.classList.toggle('completed', currentStep > 1);
    if (c23) c23.classList.toggle('completed', currentStep > 2);
}

// ---------------------------------------------------------------------------
// Step 1: Plan Selection
// ---------------------------------------------------------------------------

function onPlanPathInput() {
    const input = document.getElementById('plan-path-input');
    const path = input.value.trim();

    // Manual typing clears any dropped content
    droppedYamlContent = null;

    // Clear previous state
    clearValidation();

    if (!path) return;

    // Debounce
    if (validateTimeout) clearTimeout(validateTimeout);
    validateTimeout = setTimeout(() => validatePlanPath(path), 600);
}

function clearValidation(shouldPersist = true) {
    validatedPlan = null;
    taskSelections = {};
    droppedYamlContent = null;
    document.getElementById('btn-next-1').disabled = true;
    document.getElementById('plan-summary').classList.add('hidden');
    document.getElementById('plan-summary').innerHTML = '';

    const errEl = document.getElementById('validation-error');
    errEl.classList.remove('show');
    errEl.textContent = '';

    setValidationIndicator('idle');
    if (shouldPersist) persistWizardState();
}

function setValidationIndicator(state) {
    const el = document.getElementById('validation-indicator');
    if (!el) return;
    el.className = 'validation-indicator';
    if (state === 'loading') {
        el.className = 'validation-indicator loading';
        el.innerHTML = '<div class="validation-spinner"></div>';
    } else if (state === 'valid') {
        el.className = 'validation-indicator valid';
        el.innerHTML = '&#10003;';
    } else if (state === 'invalid') {
        el.className = 'validation-indicator invalid';
        el.innerHTML = '&#10007;';
    } else {
        el.innerHTML = '';
    }
}

async function validatePlanPath(path) {
    setValidationIndicator('loading');

    try {
        const data = await apiFetch('/plans/validate', {
            method: 'POST',
            body: JSON.stringify({ path }),
        });

        if (data.valid) {
            validatedPlan = normalizeValidatedPlan(data.plan);
            // Init all tasks as selected
            taskSelections = {};
            for (const tid of validatedPlan.task_ids) {
                taskSelections[tid] = true;
            }
            renderPlanSummary(validatedPlan);
            setValidationIndicator('valid');
            document.getElementById('btn-next-1').disabled = false;

            // Set max_parallel from plan
            const parallelInput = document.getElementById('cfg-parallel');
            if (parallelInput) parallelInput.value = validatedPlan.max_parallel || 1;
            persistWizardState();
            return true;
        } else {
            setValidationIndicator('invalid');
            const errEl = document.getElementById('validation-error');
            errEl.textContent = data.error || 'Invalid plan';
            errEl.classList.add('show');
            persistWizardState();
            return false;
        }
    } catch (err) {
        setValidationIndicator('invalid');
        const errEl = document.getElementById('validation-error');
        errEl.textContent = err.message;
        errEl.classList.add('show');
        persistWizardState();
        return false;
    }
}

function renderPlanSummary(plan) {
    const container = document.getElementById('plan-summary');
    if (!container) return;

    const taskChips = plan.task_ids.map(tid =>
        `<span class="example-chip">${escapeHtml(tid)}</span>`
    ).join('');

    container.innerHTML = `
        <div class="card">
            <div class="plan-summary-header">
                <span class="plan-summary-name">${escapeHtml(plan.name)}</span>
                <span class="text-2">${plan.tasks} tasks</span>
            </div>
            <div class="plan-summary-meta">
                <span>max_parallel: <strong>${plan.max_parallel}</strong></span>
                <span>fail_fast: <strong>${plan.fail_fast}</strong></span>
            </div>
            <div class="plan-summary-tasks">${taskChips}</div>
        </div>`;
    container.classList.remove('hidden');
}

function selectExamplePlan(path) {
    const input = document.getElementById('plan-path-input');
    if (input) {
        input.value = path;
        // Clear debounce, validate immediately
        if (validateTimeout) clearTimeout(validateTimeout);
        clearValidation();
        setValidationIndicator('loading');
        validatePlanPath(path);
    }
}

async function loadExamplePlans() {
    const container = document.getElementById('example-chips');
    if (!container) return;

    try {
        // Probe available API routes first to avoid noisy 404s in browsers
        // when connected to older backend versions.
        let hasBrowse = false;
        let hasExamples = false;
        try {
            const resp = await fetch('/openapi.json');
            if (resp.ok) {
                const spec = await resp.json();
                const paths = spec?.paths || {};
                hasBrowse = Boolean(paths['/api/files/browse']);
                hasExamples = Boolean(paths['/api/plans/examples']);
            }
        } catch {
            // ignore and fall through
        }

        let examples = [];
        if (hasBrowse) {
            const files = await apiFetch('/files/browse');
            const yamlFiles = (files || []).filter(f =>
                f && typeof f.path === 'string' && (f.path.endsWith('.yaml') || f.path.endsWith('.yml'))
            );
            examples = yamlFiles.filter(f =>
                f.path.startsWith('examples/') || f.path.startsWith('plans/')
            );
            if (examples.length === 0) {
                examples = yamlFiles;
            }
            examples = examples.slice(0, 24).map(f => ({
                name: (f.name || '').replace(/\.(yaml|yml)$/i, '') || f.path,
                path: f.path,
            }));
        } else if (hasExamples) {
            examples = await apiFetch('/plans/examples');
        } else {
            container.innerHTML = '';
            return;
        }

        if (examples.length === 0) {
            container.innerHTML = '';
            return;
        }
        container.innerHTML = examples.map(ex =>
            `<button class="example-chip" onclick="selectExamplePlan('${escapeHtml(ex.path)}')">${escapeHtml(ex.name)}</button>`
        ).join('');
    } catch {
        // Silent fail — examples are optional
        container.innerHTML = '';
    }
}

// ---------------------------------------------------------------------------
// Step 2: Configure
// ---------------------------------------------------------------------------

function renderTaskSelector() {
    const container = document.getElementById('task-select-list');
    if (!container || !validatedPlan) return;

    const details = validatedPlan.task_details || [];
    let html = '';

    for (const t of details) {
        const taskId = compactText(t.id);
        if (!taskId) continue;

        const checked = taskSelections[taskId] !== false ? 'checked' : '';
        const excluded = taskSelections[taskId] === false ? ' excluded' : '';
        const description = compactText(t.description);
        const deps = Array.isArray(t.depends_on) ? t.depends_on : [];

        // Engine badge
        let engineHtml = '';
        if (t.engine) {
            const modelLabelFull = t.model ? `${t.engine}/${t.model}` : String(t.engine);
            const modelLabel = truncateText(modelLabelFull, 42);
            engineHtml = `<span class="task-engine-badge" title="${escapeHtml(modelLabelFull)}">${escapeHtml(modelLabel)}</span>`;
        } else if (t.has_command) {
            engineHtml = '<span class="task-engine-badge task-engine-shell">shell</span>';
        }

        // Dependency pills
        let depsHtml = '';
        if (deps.length > 0) {
            depsHtml = deps.map(dep => {
                const depText = compactText(dep);
                return `<span class="task-dep-pill" title="${escapeHtml(depText)}">&rarr; ${escapeHtml(truncateText(depText, 26))}</span>`;
            }
            ).join('');
        }

        // Description
        const descHtml = description
            ? `<span class="task-select-desc" title="${escapeHtml(description)}">${escapeHtml(truncateText(description, 120))}</span>`
            : '';

        // Allow failure indicator
        const failureHtml = t.allow_failure
            ? '<span class="task-dep-pill task-dep-pill-warn">allow_failure</span>'
            : '';

        html += `
            <label class="task-select-item${excluded}" data-task-id="${escapeHtml(taskId)}">
                <input type="checkbox" ${checked} onchange="toggleTask('${escapeHtml(taskId)}', this.checked)">
                <div class="task-select-main">
                    <span class="task-select-id" title="${escapeHtml(taskId)}">${escapeHtml(truncateText(taskId, 42))}</span>
                    ${descHtml}
                    <div class="task-select-right">
                        ${engineHtml}
                        ${failureHtml}
                        ${depsHtml}
                    </div>
                </div>
            </label>`;
    }

    container.innerHTML = html;
    updateTaskCountLabel();
    checkDependencyWarnings();
}

function toggleTask(taskId, checked) {
    taskSelections[taskId] = checked;

    // Update visual
    const item = document.querySelector(`[data-task-id="${taskId}"]`);
    if (item) item.classList.toggle('excluded', !checked);

    updateTaskCountLabel();
    checkDependencyWarnings();
    persistWizardState();
}

function toggleAllTasks(checked) {
    for (const tid of Object.keys(taskSelections)) {
        taskSelections[tid] = checked;
    }
    // Update all checkboxes
    const checkboxes = document.querySelectorAll('.task-select-item input[type="checkbox"]');
    checkboxes.forEach(cb => { cb.checked = checked; });

    // Update visuals
    const items = document.querySelectorAll('.task-select-item');
    items.forEach(el => el.classList.toggle('excluded', !checked));

    updateTaskCountLabel();
    checkDependencyWarnings();
    persistWizardState();
}

function updateTaskCountLabel() {
    const label = document.getElementById('task-count-label');
    if (!label || !validatedPlan) return;
    const total = validatedPlan.task_ids.length;
    const selected = Object.values(taskSelections).filter(Boolean).length;
    label.textContent = `(${selected} of ${total} selected)`;
}

function checkDependencyWarnings() {
    const container = document.getElementById('dep-warnings');
    if (!container || !validatedPlan) return;

    const details = validatedPlan.task_details || [];
    const warnings = [];

    for (const t of details) {
        if (!taskSelections[t.id]) continue; // skip excluded tasks
        for (const dep of (t.depends_on || [])) {
            if (taskSelections[dep] === false) {
                warnings.push(`<strong>${escapeHtml(t.id)}</strong> depends on <strong>${escapeHtml(dep)}</strong> which is excluded`);
            }
        }
    }

    if (warnings.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = warnings.map(w =>
        `<div class="dep-warning">&#9888; ${w}</div>`
    ).join('');
}

function onProfileChange() {
    const select = document.getElementById('cfg-profile');
    const desc = document.getElementById('profile-desc');
    if (select && desc) {
        desc.textContent = PROFILE_DESCRIPTIONS[select.value] || '';
    }
    persistWizardState();
}

// ---------------------------------------------------------------------------
// Step 3: Review & Launch
// ---------------------------------------------------------------------------

function renderReviewSummary() {
    const container = document.getElementById('review-content');
    if (!container || !validatedPlan) return;

    const planPath = document.getElementById('plan-path-input').value.trim();
    const profile = document.getElementById('cfg-profile').value;
    const maxParallel = document.getElementById('cfg-parallel').value;
    const dryRun = document.getElementById('cfg-dry-run').checked;

    const total = validatedPlan.task_ids.length;
    const selectedTasks = validatedPlan.task_ids.filter(tid => taskSelections[tid] !== false);
    const skippedTasks = validatedPlan.task_ids.filter(tid => taskSelections[tid] === false);

    const selectedChips = selectedTasks.map(tid =>
        `<span class="review-task-chip">${escapeHtml(tid)}</span>`
    ).join('');

    const skippedChips = skippedTasks.length > 0
        ? skippedTasks.map(tid =>
            `<span class="review-task-chip skipped">${escapeHtml(tid)}</span>`
        ).join('')
        : '<span class="text-2">None</span>';

    container.innerHTML = `
        <div class="review-label">Plan</div>
        <div class="review-value">${escapeHtml(validatedPlan.name)}</div>

        <div class="review-label">Path</div>
        <div class="review-value" style="font-family:var(--mono);font-size:0.82rem">${escapeHtml(planPath)}</div>

        <div class="review-label">Tasks</div>
        <div class="review-value">
            <div style="margin-bottom:0.4rem">${selectedTasks.length} of ${total} selected</div>
            <div class="review-task-list">${selectedChips}</div>
        </div>

        <div class="review-label">Skipped</div>
        <div class="review-value">
            <div class="review-task-list">${skippedChips}</div>
        </div>

        <div class="review-label">Profile</div>
        <div class="review-value">${badgeFor(profile)} <span class="text-2">${PROFILE_DESCRIPTIONS[profile] || ''}</span></div>

        <div class="review-label">Parallel</div>
        <div class="review-value">${escapeHtml(maxParallel)}</div>

        <div class="review-label">Dry Run</div>
        <div class="review-value">${dryRun ? '<span class="badge badge-warning">Yes</span>' : 'No'}</div>
    `;

    // Clear previous launch error
    const errEl = document.getElementById('launch-error');
    if (errEl) { errEl.classList.remove('show'); errEl.textContent = ''; }
}

async function launchRun() {
    const btn = document.getElementById('btn-launch');
    if (btn) { btn.disabled = true; btn.textContent = 'Launching...'; }

    const planPath = document.getElementById('plan-path-input').value.trim();
    const profile = document.getElementById('cfg-profile').value;
    const maxParallel = parseInt(document.getElementById('cfg-parallel').value) || null;
    const dryRun = document.getElementById('cfg-dry-run').checked;

    // Build only/skip lists
    const total = validatedPlan.task_ids.length;
    const selectedTasks = validatedPlan.task_ids.filter(tid => taskSelections[tid] !== false);
    const skippedTasks = validatedPlan.task_ids.filter(tid => taskSelections[tid] === false);

    const body = {
        dry_run: dryRun,
        execution_profile: profile,
    };

    // If plan came from drag & drop, always send yaml_content.
    // The input field may contain only the filename (not a real path).
    if (droppedYamlContent) {
        body.yaml_content = droppedYamlContent;
    } else {
        body.plan_path = planPath;
    }

    if (maxParallel) body.max_parallel = maxParallel;

    // Use skip if fewer tasks are excluded, only if fewer are included
    if (skippedTasks.length > 0 && skippedTasks.length <= selectedTasks.length) {
        body.skip = skippedTasks;
    } else if (skippedTasks.length > 0) {
        body.only = selectedTasks;
    }

    try {
        const data = await apiFetch('/runs', {
            method: 'POST',
            body: JSON.stringify(body),
        });
        const runId = data?.run_id;
        if (!runId) {
            const reason = data?.detail || data?.error || 'Run did not return run_id';
            throw new Error(reason);
        }
        clearPersistedWizardState();
        window.location.href = `/static/run.html?id=${encodeURIComponent(runId)}`;
    } catch (err) {
        const errEl = document.getElementById('launch-error');
        if (errEl) {
            errEl.textContent = 'Error: ' + err.message;
            errEl.classList.add('show');
        }
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Launch Run'; }
    }
}

// ---------------------------------------------------------------------------
// File Browser
// ---------------------------------------------------------------------------

function openFileBrowser() {
    const overlay = document.getElementById('file-browser-modal');
    if (overlay) overlay.classList.add('open');
    loadFileBrowser();
}

function closeFileBrowser() {
    const overlay = document.getElementById('file-browser-modal');
    if (overlay) overlay.classList.remove('open');
}

async function loadFileBrowser() {
    const container = document.getElementById('file-browser-list');
    if (!container) return;

    container.innerHTML = '<div class="empty-state"><h4>Loading...</h4></div>';

    try {
        const files = await apiFetch('/files/browse');
        if (files.length === 0) {
            container.innerHTML = '<div class="empty-state"><h4>No YAML files found</h4><p>Add .yaml files to your project</p></div>';
            return;
        }

        // Group by directory
        const groups = {};
        for (const f of files) {
            const dir = f.dir || '(root)';
            if (!groups[dir]) groups[dir] = [];
            groups[dir].push(f);
        }

        let html = '';
        for (const [dir, items] of Object.entries(groups)) {
            html += `<div class="file-browser-group">`;
            html += `<div class="file-browser-dir">${escapeHtml(dir)}</div>`;
            for (const item of items) {
                html += `<div class="file-browser-item" onclick="selectBrowsedFile('${escapeHtml(item.path)}')">
                    <span class="file-browser-item-icon">&#128196;</span>
                    <span>${escapeHtml(item.name)}</span>
                </div>`;
            }
            html += '</div>';
        }
        container.innerHTML = html;
    } catch {
        container.innerHTML = '<div class="empty-state"><h4>Error loading files</h4></div>';
    }
}

function selectBrowsedFile(path) {
    const input = document.getElementById('plan-path-input');
    if (input) {
        input.value = path;
        droppedYamlContent = null; // clear any dropped content
        if (validateTimeout) clearTimeout(validateTimeout);
        clearValidation();
        setValidationIndicator('loading');
        validatePlanPath(path);
    }
    closeFileBrowser();
}

// ---------------------------------------------------------------------------
// Drag & Drop
// ---------------------------------------------------------------------------

function setupDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const card = document.getElementById('plan-file-card');
    if (!dropZone) return;

    // Prevent default drag behaviors on the whole card
    for (const el of [dropZone, card]) {
        if (!el) continue;
        el.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });

        el.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    }

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer?.files;
        if (!files || files.length === 0) return;

        const file = files[0];
        if (!file.name.endsWith('.yaml') && !file.name.endsWith('.yml')) {
            showToast('Please drop a .yaml or .yml file', 'info');
            return;
        }

        const reader = new FileReader();
        reader.onload = (evt) => {
            const content = evt.target.result;
            droppedYamlContent = content;

            // Show filename in input (informational — not a real path)
            const input = document.getElementById('plan-path-input');
            if (input) input.value = file.name;

            // Validate using yaml_content
            clearValidation();
            droppedYamlContent = content; // re-set after clearValidation clears it
            setValidationIndicator('loading');
            validateDroppedContent(content);
        };
        reader.readAsText(file);
    });

    // Also allow clicking the drop zone to trigger file input
    const hiddenInput = document.createElement('input');
    hiddenInput.type = 'file';
    hiddenInput.accept = '.yaml,.yml';
    hiddenInput.style.display = 'none';
    document.body.appendChild(hiddenInput);

    dropZone.addEventListener('click', () => hiddenInput.click());

    hiddenInput.addEventListener('change', () => {
        const file = hiddenInput.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (evt) => {
            const content = evt.target.result;
            droppedYamlContent = content;

            const input = document.getElementById('plan-path-input');
            if (input) input.value = file.name;

            clearValidation();
            droppedYamlContent = content;
            setValidationIndicator('loading');
            validateDroppedContent(content);
        };
        reader.readAsText(file);
        hiddenInput.value = ''; // reset for re-selection
    });
}

async function validateDroppedContent(content) {
    try {
        const data = await apiFetch('/plans/validate', {
            method: 'POST',
            body: JSON.stringify({ yaml_content: content }),
        });

        if (data.valid) {
            validatedPlan = normalizeValidatedPlan(data.plan);
            taskSelections = {};
            for (const tid of validatedPlan.task_ids) {
                taskSelections[tid] = true;
            }
            renderPlanSummary(validatedPlan);
            setValidationIndicator('valid');
            document.getElementById('btn-next-1').disabled = false;

            const parallelInput = document.getElementById('cfg-parallel');
            if (parallelInput) parallelInput.value = validatedPlan.max_parallel || 1;
            persistWizardState();
            return true;
        } else {
            setValidationIndicator('invalid');
            const errEl = document.getElementById('validation-error');
            errEl.textContent = data.error || 'Invalid plan';
            errEl.classList.add('show');
            persistWizardState();
            return false;
        }
    } catch (err) {
        setValidationIndicator('invalid');
        const errEl = document.getElementById('validation-error');
        errEl.textContent = err.message;
        errEl.classList.add('show');
        persistWizardState();
        return false;
    }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function initNewRun() {
    // Plan path input listener
    const pathInput = document.getElementById('plan-path-input');
    if (pathInput) {
        pathInput.addEventListener('input', () => {
            onPlanPathInput();
            persistWizardState();
        });
        pathInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (validateTimeout) clearTimeout(validateTimeout);
                const path = pathInput.value.trim();
                if (path) {
                    clearValidation();
                    setValidationIndicator('loading');
                    validatePlanPath(path);
                }
            }
        });
    }

    const parallelInput = document.getElementById('cfg-parallel');
    if (parallelInput) {
        parallelInput.addEventListener('input', persistWizardState);
    }

    const dryRunInput = document.getElementById('cfg-dry-run');
    if (dryRunInput) {
        dryRunInput.addEventListener('change', persistWizardState);
    }

    // Setup drag & drop zone
    setupDropZone();

    // Load example plan chips
    loadExamplePlans();

    // Restore execution options before loading any plan.
    const persisted = readPersistedWizardState();
    applyPersistedExecutionOptions(persisted);
    onProfileChange();

    // Deep link support: ?plan=path/to/plan.yaml&step=2
    const params = new URLSearchParams(window.location.search);
    const planParam = compactText(params.get('plan') || '');
    const stepParam = parseStepParam(params.get('step'));

    const persistedPlanPath = compactText(persisted?.planPath || '');
    const persistedHasDropped = Boolean(persisted?.droppedYamlContent);
    const targetStep = stepParam || parseStepParam(persisted?.step) || 1;

    let restoreSource = null;
    if (planParam) {
        restoreSource = { type: 'path', value: planParam, label: planParam };
    } else if (persistedHasDropped) {
        restoreSource = {
            type: 'yaml',
            value: persisted.droppedYamlContent,
            label: persistedPlanPath || 'dropped-plan.yaml',
        };
    } else if (persistedPlanPath) {
        restoreSource = { type: 'path', value: persistedPlanPath, label: persistedPlanPath };
    }

    let restoredValidPlan = false;
    if (restoreSource && pathInput) {
        pathInput.value = restoreSource.label;
        clearValidation(false);

        if (restoreSource.type === 'yaml') {
            droppedYamlContent = restoreSource.value;
            setValidationIndicator('loading');
            restoredValidPlan = await validateDroppedContent(restoreSource.value);
        } else {
            setValidationIndicator('loading');
            restoredValidPlan = await validatePlanPath(restoreSource.value);
        }
    }

    if (restoredValidPlan && targetStep >= 2 && persisted?.taskSelections) {
        applyPersistedTaskSelections(persisted.taskSelections);
    }

    if (restoredValidPlan && targetStep >= 2) {
        goToStep(targetStep);
    } else {
        currentStep = 1;
        updateStepperUI();
        persistWizardState();
    }
}
