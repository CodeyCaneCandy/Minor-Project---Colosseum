<div align="center">

```
 ██████╗ ██████╗ ██╗      ██████╗ ███████╗███████╗███████╗██╗   ██╗███╗   ███╗
██╔════╝██╔═══██╗██║     ██╔═══██╗██╔════╝██╔════╝██╔════╝██║   ██║████╗ ████║
██║     ██║   ██║██║     ██║   ██║███████╗███████╗█████╗  ██║   ██║██╔████╔██║
██║     ██║   ██║██║     ██║   ██║╚════██║╚════██║██╔══╝  ██║   ██║██║╚██╔╝██║
╚██████╗╚██████╔╝███████╗╚██████╔╝███████║███████║███████╗╚██████╔╝██║ ╚═╝ ██║
 ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝
```

### *Upload your dataset. Tell us the task. We tell you which model wins — and why.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Claude](https://img.shields.io/badge/Claude-Sonnet_3.5-CC785C?style=flat-square)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

</div>

---

## What is Colosseum?

Most ML beginners spend **3–5 days** guessing which model suits their data — reading tutorials, copy-pasting code, tweaking parameters, and never knowing if they made the right call. AutoML tools automate this but treat the process as a black box. Manual comparison teaches you something, but takes forever.

**Colosseum sits in between.** It runs the right models in parallel, compares them on task-specific metrics, and generates a plain-English explanation of the result — so you finish in minutes and actually understand what happened.

> *"Random Forest is the best fit because your 12 categorical features benefit from tree-based splitting without one-hot explosion. Your 1:7 class imbalance also benefits from RF's bootstrap sampling. SVM was excluded before training because your 85,000 rows would make training prohibitively slow."*

---

## ✦ Three Things No Other Tool Does Together

<table>
<tr>
<td width="33%" align="center">

### 🔬 Dataset-Aware Filtering
Models are eliminated **before training** based on theoretical fit — not after wasting compute. Each exclusion is logged with a plain-English reason.

</td>
<td width="33%" align="center">

### 🧠 LLM Explanation
After ranking, dataset profile + metrics are sent to **Claude** to generate a human-readable verdict tied to your specific data's properties.

</td>
<td width="33%" align="center">

### 📊 Confidence Score
Quantifies how decisive the winning model's lead is. When results are too close to call, Colosseum tells you to choose by deployment constraints instead.

</td>
</tr>
</table>

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Vanilla HTML · CSS · JS | Static UI, no framework overhead |
| Backend | FastAPI · Uvicorn | Three modular REST routers |
| Tabular ML | scikit-learn · XGBoost | LR · RF · SVM · KNN · MLP · XGB |
| Image ML | PyTorch · torchvision | MobileNetV2 · ResNet-50 · EfficientNet-B0 |
| Text ML | scikit-learn · sentence-transformers | TF-IDF · all-MiniLM-L6-v2 embeddings |
| LLM | Anthropic Claude Sonnet 3.5 | Plain-English result explanation |
| Parallelism | joblib | Simultaneous model training |
| Validation | Pydantic | Request + session schema enforcement |
| State | Local filesystem (`session.json`) | No database — directory as DBMS |

---

## 🏛 Architecture — 7 Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  BROWSER  →  index.html (upload · task · config)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 1 · Input & Routing                                      │
│  upload_router.py  ·  task_router.py  ·  config_router.py      │
│  Writes ──▶  data/datasets/<file>  +  data/sessions/session.json│
└──────────────────────────┬──────────────────────────────────────┘
                           │ status: "ready"
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 2 · Preprocessing                                        │
│  Profile · Sample · Split · TF-IDF / MobileNetV2 / sklearn     │
│  Writes ──▶  data/features/<vectors>.npy                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ profile JSON
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 3 · Dataset-Aware Filter          ★ Novel               │
│  YAML rule engine · prunes model pool · logs each exclusion     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ eligible model list
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 4 · Parallel Runner                                      │
│  joblib.Parallel(n_jobs=-1) · ModelWrapper · CV scoring        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ predictions + timing
┌──────────────────────────▼──────────────────────────────────────┐
│  LAYER 5 · Metrics Engine & Ranker                              │
│  F1 · AUC · RMSE · R² · composite score · confidence score     │
└──────────┬───────────────────────────────┬──────────────────────┘
           │ top-3 scores + filter log     │ ranked results
┌──────────▼──────────────────┐  ┌────────▼────────────────────── ┐
│  LAYER 6 · LLM Explainer    │  │  LAYER 7 · Dashboard           │
│  Claude Sonnet 3.5          │  │  Radar · Bar · Heatmap         │
│  Plain-English verdict      │  │  Hero card · Export CSV/PDF    │
└─────────────────────────────┘  └────────────────────────────────┘
```

### Layer breakdown

<details>
<summary><strong>Layer 1 — Input & Routing</strong></summary>
<br/>

Three independent FastAPI routers, each owning one concern:

- `upload_router.py` — validates file format, writes dataset to `data/datasets/`, creates `session.json` with `status: uploaded`
- `task_router.py` — validates task type against detected file type, updates session to `status: configured`
- `config_router.py` — stores split ratio, random seed, and CV fold count, advances session to `status: ready`

The frontend is a single `index.html` with three cards that unlock sequentially. No card 2 without card 1 completing successfully.

</details>

<details>
<summary><strong>Layer 2 — Preprocessing</strong></summary>
<br/>

Reads `session.json` when `status == "ready"`. Three parallel paths:

| Data type | Pipeline |
|---|---|
| **Tabular** | `SimpleImputer` → `StandardScaler` → `ColumnTransformer` (OneHot / Ordinal) → serialised `.pkl` |
| **Text** | TF-IDF sparse matrix (fast, < 10s) + optional `all-MiniLM-L6-v2` dense embeddings (~1 min) |
| **Image** | Resize 224×224 → frozen MobileNetV2 → 1280-dim avg-pool vectors → `.npy` cache (extracted once) |

Applies a stratified sampling gate before extraction: > 50k rows or > 5k images per class triggers sampling. Sample size shown transparently in the UI.

</details>

<details>
<summary><strong>Layer 3 — Dataset-Aware Filter ★</strong></summary>
<br/>

A YAML rule table checked against the dataset profile. Every failing rule produces a log entry that later feeds the LLM explainer.

```
Condition              →  Action                 →  User-facing reason
──────────────────────────────────────────────────────────────────────
n_rows > 100,000       →  Exclude SVM            →  O(n²) training complexity
dtype = image          →  Exclude LR, NB, KNN    →  Cannot process pixel spaces
n_classes > 20         →  Warn Logistic Reg       →  OvR overhead with many classes
imbalance_ratio > 10:1 →  Switch metric to F1    →  Accuracy is misleading here
```

</details>

<details>
<summary><strong>Layer 4 — Parallel Runner</strong></summary>
<br/>

All eligible models are wrapped in a common `ModelWrapper` interface and dispatched with `joblib.Parallel(n_jobs=-1)` — every CPU core is used simultaneously. Training wall-clock time is recorded per model.

**Model pool:**

| Modality | Models |
|---|---|
| Tabular | Logistic Regression · Random Forest · XGBoost · SVM · Naive Bayes · KNN · MLP |
| Text | Same as tabular (operating on TF-IDF or embedding vectors) |
| Image | MobileNetV2 · ResNet-50 · EfficientNet-B0 (frozen backbone + linear head) |

Optional 5-fold stratified CV per model. High standard deviation flags an unstable model in the output.

</details>

<details>
<summary><strong>Layer 5 — Metrics Engine & Ranker</strong></summary>
<br/>

Computes task-appropriate metrics, then applies a weighted composite score:

```
S = w₁·F1 + w₂·AUC + w₃·accuracy − w₄·train_time_norm − w₅·model_size_norm
```

Weights are configurable via the config panel.

**Confidence score (novel):**
```
C = (S₁ − S₂) / S₁ × 100

C > 15%   →  High confidence   (clear winner)
C 5–15%   →  Medium confidence
C < 5%    →  Low confidence    (check deployment constraints)
```

</details>

<details>
<summary><strong>Layer 6 — LLM Explanation Engine ★</strong></summary>
<br/>

Assembles a structured prompt from:
- Dataset profile summary (size, feature types, balance ratio)
- Filter exclusion log (which models were dropped and why)
- Top-3 model scores with full metric breakdown
- Task type and confidence level

Sends to **Claude Sonnet 3.5** via the Anthropic Python SDK. Handles rate limits and API failures with a template-based fallback. Includes a "Show raw prompt" toggle in the UI for educational transparency.

</details>

<details>
<summary><strong>Layer 7 — Dashboard</strong></summary>
<br/>

- **Hero card** — winning model, composite score, confidence badge (High / Medium / Low)
- **Radar chart** — all models across 5 normalised metrics
- **Bar chart** — composite scores ranked
- **Confusion matrix heatmap** (classification) or **residual scatter** (regression)
- **LLM explanation card** — the plain-English verdict
- **Live progress log** — streams which model is currently training
- **Export** — full CSV results + PDF report (charts + explanation)

</details>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/colosseum.git
cd colosseum

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > backend/.env

# 5. Run the backend
cd backend
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` in your browser (or use VS Code Live Server).

The application runs entirely on `http://localhost:8000`.

### Directory structure

```
colosseum/
├── backend/
│   ├── main.py               ← FastAPI entry point
│   ├── upload_router.py      ← Layer 1a
│   ├── task_router.py        ← Layer 1b
│   ├── config_router.py      ← Layer 1c
│   ├── session_model.py      ← Pydantic session schema
│   ├── preprocessor.py       ← Layer 2
│   ├── filter_engine.py      ← Layer 3
│   ├── model_runner.py       ← Layer 4
│   ├── metrics_engine.py     ← Layer 5
│   ├── llm_explainer.py      ← Layer 6
│   └── rules.yaml            ← Filter rule definitions
├── frontend/
│   └── index.html            ← Layer 1 UI + Layer 7 dashboard
└── data/
    ├── datasets/             ← Uploaded files land here
    ├── sessions/             ← session.json lives here
    └── features/             ← Cached .npy feature vectors
```

---

## 👥 The Team

| Member | Ownership |
|---|---|
| **Person 1** | Layer 2 — Preprocessing & Feature Extraction |
| **Person 2** | Layers 3 & 4 — Rule-based Filter & Parallel Model Runner |
| **Person 3** | Layer 5 — Metrics Engine, Weighted Ranker & Confidence Score |
| **Person 4** | Layers 6 & 7 — LLM Explainer & UI Dashboard |

---

## ⚠️ Scope & Limitations

This is a **locally-hosted prototype** built for a minor project presentation. By design it does not include:

- Multi-user or concurrent upload support
- Hyperparameter tuning (models run with sensible defaults)
- Cloud deployment or authentication
- A production database (directory + `session.json` is the intentional DBMS substitute)

---

<div align="center">

*Built as a minor project · Computer Science · 2025*

</div>
