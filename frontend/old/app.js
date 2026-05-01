// --- STATE MANAGEMENT ---
let currentSession = null;
let selectedTask = null;
let selectedFileType = 'tabular';

// --- DOM ELEMENTS ---
const steps = {
    upload: document.getElementById('step-upload'),
    task: document.getElementById('step-task'),
    config: document.getElementById('step-config')
};

const statusText = document.getElementById('statusText');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileNameSpan = document.getElementById('fileName');

// --- STEP 1: UPLOAD LOGIC ---
dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
        fileNameSpan.innerText = file.name;
        fileInfo.classList.remove('hidden');
    }
};

document.getElementById('uploadBtn').onclick = async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    updateStatus("Uploading...");

    try {
        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();
        
        if (resp.ok) {
            currentSession = data.session;
            updateStatus("Upload complete!");
            showStep('task');
        } else {
            updateStatus("Upload failed: " + data.detail);
        }
    } catch (err) {
        updateStatus("Error: Could not reach backend.");
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
            updateStatus("Task set.");
            // Show/hide text column input based on selection
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
    const targetCol = document.getElementById('targetCol').value.trim();
    const textCol = document.getElementById('textCol').value.trim();

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
            updateStatus("Ready! You can now switch to the Streamlit Dashboard.");
            alert("Configuration Saved! Open the Streamlit app at localhost:8501 to run the evaluation.");
        }
    } catch (err) {
        updateStatus("Error saving config.");
    }
};

// --- UTILS ---
function showStep(stepName) {
    Object.values(steps).forEach(s => s.classList.remove('active'));
    steps[stepName].classList.add('active');
}

function updateStatus(text) {
    statusText.innerText = text;
}