// --- STATE MANAGEMENT ---
let currentSession = null;
let selectedTask = null;
let selectedFileType = 'tabular';

// --- DOM ELEMENTS ---
const steps = {
    upload: document.getElementById('step-upload'),
    task: document.getElementById('step-task'),
    config: document.getElementById('step-config'),
    stream: document.getElementById('step-stream') // Added Step 4
};

const statusText = document.getElementById('statusText');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileNameSpan = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');

// --- STEP 1: UPLOAD LOGIC ---
// This is the line that was missing! It links the visual box to the hidden input
dropZone.onclick = () => fileInput.click();

// Added: Cool drag-and-drop visual effects
dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#38bdf8';
    dropZone.style.background = '#1e293b';
};
dropZone.ondragleave = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#334155';
    dropZone.style.background = 'transparent';
};
dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#334155';
    dropZone.style.background = 'transparent';
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event('change')); // Trigger the change event
    }
};

// When a file is selected (either by click or drop)
fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameSpan.innerText = file.name;
        fileInfo.classList.remove('hidden');
    }
};

// Upload button logic
uploadBtn.onclick = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    updateStatus("Uploading...");
    uploadBtn.disabled = true;
    uploadBtn.innerText = "Uploading...";

    try {
        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();
        
        if (resp.ok) {
            currentSession = data.session; // Store the session metadata!
            updateStatus("Upload complete!");
            showStep('task');
        } else {
            updateStatus("Upload failed: " + data.detail);
        }
    } catch (err) {
        updateStatus("Error: Could not reach backend.");
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerText = "Upload & Continue";
    }
};

// --- STEP 2: TASK SELECTION ---
window.selectTask = async (task, fileType) => {
    selectedTask = task;
    selectedFileType = fileType;

    updateStatus(`Setting task to ${task}...`);

    try {
        const resp = await fetch('/api/task', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                task: task.charAt(0).toUpperCase() + task.slice(1), 
                problem_type: task 
            })
        });

        if (resp.ok) {
            // Update our local state
            currentSession.task = task.charAt(0).toUpperCase() + task.slice(1);
            currentSession.problem_type = task;
            currentSession.file_type = fileType;
            
            updateStatus("Task set.");
            const textGroup = document.getElementById('textColGroup');
            fileType === 'text' ? textGroup.classList.remove('hidden') : textGroup.classList.add('hidden');
            showStep('config');
        }
    } catch (err) {
        updateStatus("Error setting task.");
    }
};

// --- STEP 3: CONFIGURATION ---
document.getElementById('finishBtn').onclick = async () => {
    const targetCol = document.getElementById('targetCol').value;
    const textCol = document.getElementById('textCol').value;

    if (!targetCol) {
        alert("Please specify a target column.");
        return;
    }

    updateStatus("Saving configuration...");

    try {
        const resp = await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_column: targetCol,
                text_column: textCol || null,
                split_ratio: 0.8,
                random_seed: 42
            })
        });

        if (resp.ok) {
            // Update our local state so the Gauntlet has the right data
            currentSession.target_column = targetCol;
            if (textCol) currentSession.text_column = textCol;
            
            updateStatus("Config saved. Ready for gauntlet.");
            showStep('stream');
        }
    } catch (err) {
        updateStatus("Error saving config.");
    }
};

// --- STEP 4: LIVE EVALUATION STREAM ---
let filterSocket;

document.getElementById('startEvalBtn').onclick = async () => {
    document.getElementById('startEvalBtn').classList.add('hidden');
    document.getElementById('liveUiContainer').classList.remove('hidden');
    
    // 1. Open the WebSocket
    filterSocket = new WebSocket("ws://localhost:8000/api/filter-stream");
    
    filterSocket.onopen = () => {
        document.querySelector('.status-dot').style.backgroundColor = '#22c55e'; // Green
        updateStatus("WebSocket connected. Starting engine...");
        triggerEvaluationAPI(); // 2. Trigger the API once socket is listening
    };

    filterSocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const feed = document.getElementById('log-feed');
        
        let badgeClass = 'badge-warn';
        if (data.action === 'exclude') badgeClass = 'badge-exclude';
        if (data.action === 'restrict') badgeClass = 'badge-restrict';

        const logHTML = `
            <div class="log-entry">
                <div><span class="log-badge ${badgeClass}">${data.action.toUpperCase()}</span><strong>${data.target}</strong></div>
                <div style="margin-top: 4px; color: #94a3b8;">${data.reason}</div>
            </div>`;
        feed.insertAdjacentHTML('beforeend', logHTML);
        feed.scrollTop = feed.scrollHeight;

        // Cross out the model
        if (data.action === 'exclude' && data.target !== 'All') {
            const modelElement = document.getElementById(`model-${data.target}`);
            if (modelElement) modelElement.classList.add('model-excluded');
        }
    };
    
    filterSocket.onclose = () => { 
        document.querySelector('.status-dot').style.backgroundColor = '#ef4444'; // Red
    };
};

// Fire the actual API payload
async function triggerEvaluationAPI() {
    try {
        // We pass the fully updated currentSession to Layer 3
        const resp = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session: currentSession }) 
        });
        
        if (resp.ok) {
            updateStatus("Evaluation Complete! Results ready.");
            filterSocket.close();
            document.getElementById('goToDashboardBtn').classList.remove('hidden');
        } else {
            updateStatus("Evaluation failed to complete.");
            filterSocket.close();
        }
    } catch (err) { 
        updateStatus("Error running evaluation."); 
        if(filterSocket) filterSocket.close();
    }
}

// --- UTILS ---
function showStep(stepName) {
    Object.values(steps).forEach(s => {
        if(s) s.classList.remove('active');
    });
    const activeStep = document.getElementById(`step-${stepName}`);
    if (activeStep) activeStep.classList.add('active');
}

function updateStatus(text) { 
    if(statusText) statusText.innerText = text; 
}