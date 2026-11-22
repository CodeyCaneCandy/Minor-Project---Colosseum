# Colosseum

Colosseum is a locally hosted machine learning model comparison tool. Upload a dataset, configure a task, and compare candidate models with evaluation metrics and an optional AI explanation.

The project uses a FastAPI backend and a static HTML, CSS, and JavaScript frontend. The backend serves the frontend, so you can use the app at one address without starting a separate frontend server.

## Features

- Upload tabular, text, and image datasets through the web interface.
- Profile and preprocess data before training.
- Filter candidate models using dataset rules.
- Compare model metrics and rank the results.
- Optionally ask the built-in assistant questions about the results using Groq.
- View a separate Streamlit results dashboard (optional).

## Requirements

- Python 3.11 recommended.
- `pip` and `venv` (included with standard Python installations).
- macOS, Linux, or Windows.
- Optional: a Groq API key for the AI assistant.

Python 3.11 is recommended for compatibility with the project's scientific and machine learning packages. XGBoost is optional at runtime: if its native library cannot load, Colosseum starts without XGBoost and keeps the other models available.

## Setup and run

Run these commands from a terminal in the repository directory.

### macOS and Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example backend/.env
cd backend
python -m uvicorn main:app --reload --port 8000
```

### Windows PowerShell

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example backend\.env
Set-Location backend
python -m uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** in your browser. The API documentation is at **http://127.0.0.1:8000/docs**.

If the `python3.11` or `py -3.11` command is unavailable, install Python 3.11 first, then reopen the terminal and repeat the setup steps.

## Optional Groq assistant

The app runs without an API key; only the AI chat assistant needs one. Get a key from [Groq Console](https://console.groq.com/keys), then put it in `backend/.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Restart the backend after changing the key. **Do not commit `backend/.env` or share your API key.** The root `.env.example` is a safe template and contains no real key.

## Optional XGBoost on macOS

If the server logs that XGBoost is unavailable because `libomp.dylib` is missing, install the OpenMP runtime with Homebrew:

```bash
brew install libomp
```

Then restart the backend. Without this runtime, the app can still use its other models.

## Optional Streamlit dashboard

The repository also includes a separate dashboard at `frontend/app.py`. Start it from the repository root in a second terminal, with the same virtual environment activated:

```bash
streamlit run frontend/app.py
```

The dashboard expects a completed session from the main web app.

## Project layout

```text
backend/
  main.py                 FastAPI application and static frontend server
  routers/                Upload, configuration, evaluation, and chat endpoints
  services/               Profiling, preprocessing, models, and explanations
  layer3_filter/          Dataset-aware model filtering
  layer5_training/        Model training, metrics, and ranking
  custom_models/          Custom model implementations
frontend/
  index.html              Main web application
  styles.css              Frontend styles
  app.js                  Frontend behavior
  app.py                  Optional Streamlit dashboard
configs/                  Model filter configuration
requirements.txt          Python dependencies
```


