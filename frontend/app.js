/* =========================================================
   COLOSSEUM FRONTEND ENGINE
   ========================================================= */

// =========================================================
// GLOBAL STATE
// =========================================================

const state = {
    session: null,
    task: null,
    fileType: 'tabular',
    socket: null
};
let comparisonChartInstance = null;
// =========================================================
// DOM REFERENCES
// =========================================================

const dom = {

    // Steps
    steps: {
        upload: document.getElementById('step-upload'),
        task: document.getElementById('step-task'),
        config: document.getElementById('step-config'),
        stream: document.getElementById('step-stream')
    },

    // Upload
    dropZone: document.getElementById('dropZone'),
    fileInput: document.getElementById('fileInput'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    uploadBtn: document.getElementById('uploadBtn'),

    // Status
    statusText: document.getElementById('statusText'),
    statusTime: document.getElementById('statusTime'),
    sysDot: document.getElementById('sysDot'),
    sysLabel: document.getElementById('sysLabel'),
    wsDot: document.getElementById('wsDot'),
    wsLabel: document.getElementById('wsLabel'),

    // Config
    targetCol: document.getElementById('targetCol'),
    textCol: document.getElementById('textCol'),
    textColGroup: document.getElementById('textColGroup'),
    splitSlider: document.getElementById('splitSlider'),
    splitLabel: document.getElementById('splitLabel'),
    seedInput: document.getElementById('seedInput'),
    finishBtn: document.getElementById('finishBtn'),

    // Config Preview
    configPreview: document.getElementById('configPreview'),
    configPreviewJson: document.getElementById('configPreviewJson'),

    // Stream
    startEvalBtn: document.getElementById('startEvalBtn'),
    liveUiContainer: document.getElementById('liveUiContainer'),
    logFeed: document.getElementById('log-feed'),

    // Winner
    winnerReveal: document.getElementById('winnerReveal'),
    winnerName: document.getElementById('winnerName')
};

// =========================================================
// API ROUTES
// =========================================================

const API = {
    upload: '/api/upload',
    task: '/api/task',
    config: '/api/config',
    evaluate: '/api/evaluate',
    socket: 'ws://localhost:8000/api/filter-stream'
};

// =========================================================
// STATUS HELPERS
// =========================================================

function updateStatus(message) {
    dom.statusText.innerText = message;
}

function updateSystemStatus(type, label) {

    dom.sysDot.className = 'sys-dot';
    dom.sysDot.classList.add(type);

    dom.sysLabel.innerText = label;
}

function updateSocketStatus(connected) {

    dom.wsDot.className = 'ws-dot';

    if (connected) {
        dom.wsDot.classList.add('connected');
        dom.wsLabel.innerText = 'CONNECTED';
    }
    else {
        dom.wsDot.classList.add('disconnected');
        dom.wsLabel.innerText = 'DISCONNECTED';
    }
}

// =========================================================
// STEP NAVIGATION
// =========================================================

function showStep(stepName) {

    Object.values(dom.steps).forEach(step => {
        step.classList.remove('active');
    });

    dom.steps[stepName].classList.add('active');
}

// =========================================================
// RAIL PROGRESS
// =========================================================

function markRailStep(stepNumber, stateType) {

    const railStep = document.getElementById(`rail-${stepNumber}`);

    if (!railStep) return;

    railStep.classList.remove('active');

    if (stateType === 'done') {
        railStep.classList.add('done');
    }

    if (stateType === 'active') {
        railStep.classList.add('active');
    }
}

// =========================================================
// LIVE CLOCK
// =========================================================

function startClock() {

    function updateClock() {
        const now = new Date();
        dom.statusTime.innerText = now.toLocaleTimeString();
    }

    updateClock();
    setInterval(updateClock, 1000);
}

// =========================================================
// SPLIT SLIDER
// =========================================================

function initializeSplitSlider() {

    dom.splitSlider.addEventListener('input', () => {

        const train = dom.splitSlider.value;
        const test = 100 - train;

        dom.splitLabel.innerText = `${train} / ${test}`;
    });
}

// =========================================================
// FILE UPLOAD UI
// =========================================================

function initializeDropZone() {

    dom.dropZone.onclick = () => dom.fileInput.click();

    dom.dropZone.ondragover = (event) => {
        event.preventDefault();
        dom.dropZone.classList.add('dragover');
    };

    dom.dropZone.ondragleave = () => {
        dom.dropZone.classList.remove('dragover');
    };

    dom.dropZone.ondrop = (event) => {

        event.preventDefault();
        dom.dropZone.classList.remove('dragover');

        const files = event.dataTransfer.files;

        if (files.length) {
            dom.fileInput.files = files;
            dom.fileInput.dispatchEvent(new Event('change'));
        }
    };
}

function initializeFileSelection() {

    dom.fileInput.onchange = (event) => {

        const file = event.target.files[0];

        if (!file) return;

        dom.fileName.innerText = file.name;

        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        dom.fileSize.innerText = `${sizeMB} MB`;

        dom.fileInfo.classList.remove('hidden');
    };
}

// =========================================================
// API UTIL
// =========================================================

async function postJSON(url, payload) {

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
    }

    return data;
}

// =========================================================
// STEP 1 — FILE UPLOAD
// =========================================================

async function uploadDataset() {

    const file = dom.fileInput.files[0];

    if (!file) return;

    updateStatus('Uploading dataset...');
    updateSystemStatus('active', 'UPLOADING');

    dom.uploadBtn.disabled = true;

    try {

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(API.upload, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        state.session = data.session;

        updateStatus('Upload complete');
        updateSystemStatus('done', 'UPLOAD COMPLETE');

        markRailStep(1, 'done');
        markRailStep(2, 'active');

        showStep('task');
    }

    catch (error) {

        console.error(error);

        updateStatus(error.message);
        updateSystemStatus('error', 'UPLOAD ERROR');
    }

    finally {
        dom.uploadBtn.disabled = false;
    }
}

// =========================================================
// STEP 2 — TASK SELECTION
// =========================================================

window.selectTask = async function(task, fileType) {

    state.task = task;
    state.fileType = fileType;

    updateStatus(`Configuring ${task} pipeline...`);
    updateSystemStatus('active', 'TASK ROUTING');

    try {

        const payload = {
            task: capitalize(task),
            problem_type: task,
            file_type: fileType
        };

        await postJSON(API.task, payload);

        Object.assign(state.session, payload);

        if (fileType === 'text') {
            dom.textColGroup.classList.remove('hidden');
        }
        else {
            dom.textColGroup.classList.add('hidden');
        }

        updateStatus('Task configured');
        updateSystemStatus('done', 'TASK READY');

        markRailStep(2, 'done');
        markRailStep(3, 'active');

        showStep('config');
    }

    catch (error) {

        console.error(error);

        updateStatus(error.message);
        updateSystemStatus('error', 'TASK ERROR');
    }
};

// =========================================================
// STEP 3 — CONFIGURATION
// =========================================================

async function saveConfiguration() {

    const targetColumn = dom.targetCol.value.trim();
    const textColumn = dom.textCol.value.trim();

    if (!targetColumn) {
        alert('Please enter a target column');
        return;
    }

    const config = {
        target_column: targetColumn,
        text_column: textColumn || null,
        split_ratio: parseInt(dom.splitSlider.value) / 100,
        random_seed: parseInt(dom.seedInput.value)
    };

    previewConfiguration(config);

    updateStatus('Saving configuration...');
    updateSystemStatus('active', 'CONFIGURING');

    try {

        await postJSON(API.config, config);

        Object.assign(state.session, config);

        updateStatus('Configuration saved');
        updateSystemStatus('done', 'ENGINE ARMED');

        markRailStep(3, 'done');
        markRailStep(4, 'active');

        showStep('stream');
    }

    catch (error) {

        console.error(error);

        updateStatus(error.message);
        updateSystemStatus('error', 'CONFIG ERROR');
    }
}

function previewConfiguration(config) {

    dom.configPreview.classList.remove('hidden');

    dom.configPreviewJson.innerText = JSON.stringify(
        config,
        null,
        2
    );
}

// =========================================================
// STEP 4 — WEBSOCKET
// =========================================================

function connectWebSocket() {

    state.socket = new WebSocket(API.socket);

    state.socket.onopen = () => {

        updateSocketStatus(true);
        updateStatus('WebSocket connected');

        startEvaluation();
    };

    state.socket.onmessage = (event) => {

        const data = JSON.parse(event.data);

        // If it's a stage update, animate the tracker
        if (data.type === 'stage') {
            updatePipelineStage(data.stage, data.status, data.msg);
        }

        // Otherwise, it's a log/model update, send it to the feed
        else {

            addLogEntry(data);
            updateModelVisualState(data);

        }
    };

    state.socket.onerror = () => {

        updateSocketStatus(false);
        updateStatus('WebSocket connection failed');
        updateSystemStatus('error', 'SOCKET ERROR');
    };

    state.socket.onclose = () => {
        updateSocketStatus(false);
    };
}

// =========================================================
// PIPELINE LOGS
// =========================================================

function addLogEntry(data) {

    let badgeClass = 'badge-info';

    if (data.action === 'exclude') {
        badgeClass = 'badge-exclude';
    }

    if (data.action === 'warn') {
        badgeClass = 'badge-warn';
    }

    if (data.action === 'restrict') {
        badgeClass = 'badge-restrict';
    }

    const logHTML = `
        <div class="log-entry">

            <span class="log-badge ${badgeClass}">
                ${data.action.toUpperCase()}
            </span>

            <div class="log-entry-body">
                <div class="log-entry-target">
                    ${data.target}
                </div>

                <div class="log-entry-reason">
                    ${data.reason}
                </div>
            </div>

        </div>
    `;

    dom.logFeed.insertAdjacentHTML('beforeend', logHTML);

    dom.logFeed.scrollTop = dom.logFeed.scrollHeight;
}

// =========================================================
// MODEL VISUAL STATE
// =========================================================

function updateModelVisualState(data) {

    if (data.target === 'All') return;

    const modelElement = document.getElementById(
        `model-${data.target}`
    );

    if (!modelElement) return;

    if (data.action === 'exclude') {
        modelElement.classList.add('excluded');
    }

    if (data.action === 'warn') {
        modelElement.classList.add('warned');
    }
}

// =========================================================
// EVALUATION
// =========================================================

async function startEvaluation() {

    updateStatus('Running evaluation pipeline...');
    updateSystemStatus('active', 'GAUNTLET RUNNING');

    try {

        const result = await postJSON(API.evaluate, {
            session: state.session
        });

        updateStatus('Evaluation complete');
        updateSystemStatus('done', 'PIPELINE COMPLETE');

        revealWinner(result);

        if (state.socket) {
            state.socket.close();
        }
    }

    catch (error) {

        console.error(error);

        updateStatus(error.message);
        updateSystemStatus('error', 'EVALUATION FAILED');

        if (state.socket) {
            state.socket.close();
        }
    }
}

function revealWinner(data) {
    // 1. Hide the live stream UI
    dom.steps['stream'].classList.remove('active');
    
    // 2. Show the new Results UI
    const resultsStep = document.getElementById('step-results');
    resultsStep.classList.add('active');

    const res = data.results; 
    const winnerData = res.models[res.winner.name];

    // 3. Populate the Hero Card
    document.getElementById('finalWinnerName').innerText = res.winner.name;
    document.getElementById('finalExplanation').innerText = `"${res.explanation}"`;
    
    const confLabel = document.getElementById('finalConfidence');
    confLabel.innerText = `${res.confidence.level.toUpperCase()} CONFIDENCE (${res.confidence.value}%)`;
    
    if (res.confidence.level === 'high') confLabel.style.color = 'var(--green)';
    if (res.confidence.level === 'medium') confLabel.style.color = 'var(--amber)';
    if (res.confidence.level === 'low') confLabel.style.color = 'var(--red)';

   // 4. Populate Metrics (Defaults to the winner on load)
    updateMetricsPanel(res.winner.name, winnerData, true);
    
    
    ['accuracy', 'f1', 'roc_auc'].forEach(metric => {
        if (winnerData.metrics && winnerData.metrics[metric] !== undefined) {
            metricsGrid.innerHTML += `
                <div class="info-tile">
                    <div class="info-tile-label">${metric.replace('_', ' ').toUpperCase()}</div>
                    <div class="info-tile-val" style="font-size: 20px;">${winnerData.metrics[metric].toFixed(4)}</div>
                </div>`;
        }
    });

    // 5. Update Rail and Draw Chart
    markRailStep(4, 'done');
    renderChart(res.models);
}

// =========================================================
// INTERACTIVE METRICS UPDATER
// =========================================================
function updateMetricsPanel(modelName, modelData, isWinner = false) {
    const metricsGrid = document.getElementById('metricsGrid');
    metricsGrid.innerHTML = ''; // Clear previous tiles

    // Change the panel header dynamically
    const headerSpan = document.querySelector('#step-results .live-log-panel:first-child .log-header span');
    if (headerSpan) {
        headerSpan.innerText = isWinner ? 'WINNER TELEMETRY' : `${modelName.toUpperCase()} TELEMETRY`;
    }

    // Add Composite Score Tile
    metricsGrid.innerHTML += `
        <div class="info-tile" style="border-color: ${isWinner ? 'var(--accent)' : 'var(--border-hi)'}; transition: all 0.3s ease;">
            <div class="info-tile-label" style="color: ${isWinner ? 'var(--accent)' : 'var(--text-muted)'};">COMPOSITE SCORE</div>
            <div class="info-tile-val" style="font-size: 24px;">${modelData.composite_score.toFixed(4)}</div>
        </div>`;
    
    // Add the other 3 tiles
    ['accuracy', 'f1', 'roc_auc'].forEach(metric => {
        if (modelData.metrics && modelData.metrics[metric] !== undefined) {
            metricsGrid.innerHTML += `
                <div class="info-tile" style="transition: all 0.3s ease;">
                    <div class="info-tile-label">${metric.replace('_', ' ').toUpperCase()}</div>
                    <div class="info-tile-val" style="font-size: 20px;">${modelData.metrics[metric].toFixed(4)}</div>
                </div>`;
        }
    });
}

function renderChart(modelsData) {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (comparisonChartInstance) comparisonChartInstance.destroy();
    
    const modelNames = Object.keys(modelsData);
    const scores = modelNames.map(name => modelsData[name].composite_score);
    
    comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Composite Score', data: scores,
                backgroundColor: 'rgba(0, 212, 255, 0.4)', borderColor: 'rgba(0, 212, 255, 1)',
                borderWidth: 1, borderRadius: 4
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, indexAxis: 'y',
            plugins: { legend: { display: false } },
            
            // --- ADD THIS ONCLICK HANDLER ---
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const barIndex = elements[0].index;
                    const clickedModelName = modelNames[barIndex];
                    const clickedModelData = modelsData[clickedModelName];
                    
                    // Update the tiles when a bar is clicked!
                    updateMetricsPanel(clickedModelName, clickedModelData, false);
                }
            },
            // --------------------------------

            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#4a6478' } },
                y: { grid: { display: false }, ticks: { color: '#c9d8e8', font: { family: 'JetBrains Mono' } } }
            }
        }
    });
}
// =========================================================
// START BUTTON
// =========================================================

function initializeEvaluationButton() {

    dom.startEvalBtn.onclick = () => {

        dom.startEvalBtn.classList.add('hidden');
        dom.liveUiContainer.classList.remove('hidden');

        connectWebSocket();
    };
}

// =========================================================
// UTILITIES
// =========================================================

function capitalize(text) {
    return text.charAt(0).toUpperCase() + text.slice(1);
}

// =========================================================
// INITIALIZATION
// =========================================================

function initializeApp() {

    updateSystemStatus('idle', 'IDLE');
    updateSocketStatus(false);
    updateStatus('READY');

    initializeDropZone();
    initializeFileSelection();
    initializeSplitSlider();
    initializeEvaluationButton();

    dom.uploadBtn.onclick = uploadDataset;
    dom.finishBtn.onclick = saveConfiguration;

    startClock();
}

initializeApp();

// =========================================================
// PIPELINE STAGE ANIMATOR
// =========================================================
function updatePipelineStage(stageId, status, message) {
    // stageId will be 'sample', 'profile', 'preprocess', 'filter', or 'train'
    const stageDiv = document.getElementById(`pipe-${stageId}`);
    const statusText = document.getElementById(`ps-${stageId}`);
    const badge = document.getElementById(`pb-${stageId}`);

    if (!stageDiv || !statusText || !badge) return;

    // Reset classes
    stageDiv.className = 'pipe-stage';
    badge.className = 'pipe-badge';

    // Apply new classes
    stageDiv.classList.add(status); // adds 'running' or 'done'
    badge.classList.add(status);
    
    // Update text
    badge.innerText = status.toUpperCase();
    if (message) {
        statusText.innerText = message;
    }
}