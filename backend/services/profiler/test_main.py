import json
import sys
from pathlib import Path

# ── PATH SETUP ───────────────────────────────────────────────────────────────
# This file lives at: backend/services/profiler/test_main.py
# Project root is four levels up: profiler → services → backend → Colosseum
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── IMPORTS ──────────────────────────────────────────────────────────────────
from services.profiler.tabular_profiler import profile_tabular
from services.profiler.text_profiler    import profile_text
# from backend.services.profiler.image_profiler import profile_image   # Layer 2 todo


# ── DISPATCHER ───────────────────────────────────────────────────────────────
def run_profiler(session: dict) -> dict:
    """
    Route the session to the correct profiler based on task_type.
    Mirrors how Layer 2 will call profilers in production.
    """
    task_type = session["task_type"]

    if task_type == "tabular":
        return profile_tabular(
            file_path      = session["dataset_path"],
            target_column  = session["target_column"],
            problem_type   = session["problem_type"],
        )

    elif task_type == "text":
        return profile_text(
            file_path      = session["dataset_path"],
            text_column    = session["text_column"],
            target_column  = session.get("target_column"),   # optional for unsupervised
        )

    # elif task_type == "image":
    #     return profile_image(
    #         folder_path   = session["dataset_path"],
    #         target_column = session.get("target_column"),
    #     )

    else:
        raise ValueError(
            f"Unsupported task_type: '{task_type}'. "
            f"Expected one of: 'tabular', 'text', 'image'."
        )


# ── SESSIONS ─────────────────────────────────────────────────────────────────
# Switch which session dict is active by commenting/uncommenting.
# Each one simulates a different upload scenario.

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

SESSIONS = {

    # ── Test 1: tabular classification (wine quality) ────────────────────────
    "tabular_classification": {
        "id":             "test_tabular_cls",
        "task_type":      "tabular",
        "dataset_path":   str(DATASETS_DIR / "wine" / "winequality-red.csv"),
        "target_column":  "quality",
        "problem_type":   "classification",
    },

    # ── Test 2: tabular regression (same dataset, different framing) ─────────
    "tabular_regression": {
        "id":             "test_tabular_reg",
        "task_type":      "tabular",
        "dataset_path":   str(DATASETS_DIR / "wine" / "winequality-white.csv"),
        "target_column":  "quality",
        "problem_type":   "regression",
    },

    # ── Test 3: text classification ──────────────────────────────────────────
    # Point dataset_path at your own text CSV.
    # The file must have a column of raw text and a column of labels.
    "text_classification": {
        "id":             "test_text_cls",
        "task_type":      "text",
        "dataset_path":   r"C:\Users\amesh\OneDrive\Desktop\Colosseum\data\datasets\imdb\IMDB Dataset.csv",
        "text_column":    "review",       # change to match your column name
        "target_column":  "sentiment",   # change to match your label column
    },

    # ── Test 4: text without labels (unsupervised / topic modelling) ─────────
    "text_unsupervised": {
        "id":             "test_text_unsup",
        "task_type":      "text",
        "dataset_path":   str(DATASETS_DIR / "reviews" / "sample_reviews.csv"),
        "text_column":    "review",
        "target_column":  None,          # no label column
    },

}


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ┌─────────────────────────────────────────────────────────────┐
    # │  CHANGE THIS LINE to switch which test you are running      │
    # └─────────────────────────────────────────────────────────────┘
    # ACTIVE_SESSION = "tabular_classification"
    # ACTIVE_SESSION = "tabular_regression"
    ACTIVE_SESSION = "text_classification"
    # ACTIVE_SESSION = "text_unsupervised"

    session = SESSIONS[ACTIVE_SESSION]

    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Running profiler : {ACTIVE_SESSION}")
    print(f"  Project root     : {PROJECT_ROOT}")
    print(f"  Dataset path     : {session['dataset_path']}")
    print(f"{'='*55}\n")

    # ── Run ──────────────────────────────────────────────────────────────────
    profile = run_profiler(session)

    # ── Print ────────────────────────────────────────────────────────────────
    print("=== DATASET PROFILE ===\n")
    print(json.dumps(profile, indent=2))

    # ── Save (Layer 2 to Layer 3 handoff) ────────────────────────────────────
    out_path = FEATURES_DIR / f"{session['id']}_profile.json"
    out_path.write_text(json.dumps(profile, indent=2))

    print(f"\n  Profile saved -> {out_path}")
    print(f"  Summary: {profile['summary']}\n")
