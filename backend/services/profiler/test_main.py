import json
import sys
from pathlib import Path

# ── PATH SETUP ──────────────────────────────────────────────────────────────
# This file lives at: backend/services/profiler/test_main.py
# Project root is:    three levels up  (profiler → services → backend → root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))   # lets Python find "backend.*" imports

# ── IMPORTS ─────────────────────────────────────────────────────────────────
from backend.services.profiler.tabular_profiler import profile_tabular
# from backend.services.profiler.text_profiler import profile_text


# ── DISPATCHER ──────────────────────────────────────────────────────────────
def run_profiler(session):
    task_type = session["task_type"]

    if task_type == "tabular":
        return profile_tabular(
            file_path=session["dataset_path"],
            target_column=session["target_column"],
            problem_type=session["problem_type"],
        )

    elif task_type == "text":
        return profile_text(
            file_path=session["dataset_path"],
            text_column=session["text_column"],
            target_column=session.get("target_column"),
        )

    else:
        raise ValueError(f"Unsupported task_type: '{task_type}'")


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # All paths resolved from project root — works regardless of cwd
    DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)   # create if missing

    # ── Mock session ─────────────────────────────────────────────────────────
    session = {
        "id":             "test_001",
        "task_type":      "tabular",          # switch to "text" to test text profiler
        "dataset_path":   str(DATASETS_DIR / "wine" / "winequality-red.csv"),
        "target_column":  "quality",
        "problem_type":   "classification",   # or "regression"
    }

    print(f"\n[test_main] Project root : {PROJECT_ROOT}")
    print(f"[test_main] Dataset path : {session['dataset_path']}\n")

    # ── Run profiler ─────────────────────────────────────────────────────────
    profile = run_profiler(session)

    # ── Print result ─────────────────────────────────────────────────────────
    print("=== DATASET PROFILE ===\n")
    print(json.dumps(profile, indent=2))

    # ── Save output (Layer 2 handoff) ─────────────────────────────────────────
    out_path = FEATURES_DIR / f"{session['id']}_profile.json"
    out_path.write_text(json.dumps(profile, indent=2))
    print(f"\nProfile saved → {out_path}\n")