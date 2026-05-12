from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback
import os 
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Import Layer 2 & 4 modules
from services.profiler.tabular_profiler import profile_tabular
from services.profiler.text_profiler import profile_text
from services.preprocessing.tabular_preprocess import DynamicPreprocessor
from services.preprocessing.text_preprocess import TextPreprocessor
from services.preprocessing.image_preprocess import ImagePreprocessor
# Import Layer 3 modules
from core.ws_manager import manager
from core.schemas import DatasetProfile
from layer3_filter.engine import DatasetModelFilter

# Import Layer 5 module
from layer5_training.runner import ModelRunner

router = APIRouter()

class EvaluateRequest(BaseModel):
    session: dict

# ── LAYER 1.5: THE SMART SAMPLER ──────────────────────────────────────────────
def smart_sample(df: pd.DataFrame, target_col: str, max_rows: int = 15000) -> tuple[pd.DataFrame, dict]:
    """Trims massive datasets down using stratified sampling for speed."""
    total_rows = len(df)
    
    if total_rows <= max_rows:
        return df, {
            "sampling_applied": False,
            "original_rows": total_rows,
            "sampled_rows": total_rows,
            "sample_fraction": 1.0,
            "reason": f"Dataset is under the {max_rows}-row threshold."
        }

    print(f"\n[LAYER 1.5] Dataset too large ({total_rows} rows). Trimming to {max_rows} rows...")
    try:
        sampled_df, _ = train_test_split(df, train_size=max_rows, stratify=df[target_col], random_state=42)
    except Exception:
        sampled_df = df.sample(n=max_rows, random_state=42)

    report = {
        "sampling_applied": True,
        "original_rows": total_rows,
        "sampled_rows": max_rows,
        "sample_fraction": round(max_rows / total_rows, 3),
        "reason": f"Capped at {max_rows:,} rows to ensure rapid model evaluation."
    }
    return sampled_df, report

# ── API ENDPOINT ──────────────────────────────────────────────────────────────
@router.post("/evaluate")
async def run_evaluation(request: EvaluateRequest):
    session = dict(request.session)
    temp_path = None  

    try:
        task_type = session.get("file_type", "tabular")
        dataset_path = session["dataset_path"]
        
        # Safely get target_column (Images don't have one in the UI)
        target_column = session.get("target_column", "folder_names")
        
        # ── 0. LOAD AND SAMPLE DATA (Layer 1.5) ───────────────────────────
        await manager.broadcast({"type": "stage", "stage": "sample", "status": "running", "msg": "Checking dataset size..."})
        
        if task_type == "image":
            # SKIP PANDAS: Images are handled by the ImagePreprocessor
            sampling_report = {"reason": "Image dataset. Sampling deferred to image preprocessor."}
            await manager.broadcast({"type": "stage", "stage": "sample", "status": "done", "msg": "Image directory validated."})
        else:
            # ORIGINAL TABULAR/TEXT LOGIC
            df = pd.read_csv(dataset_path, sep=None, engine='python', encoding='latin-1')
            
            if len(df.columns) == 6 and 'target' not in df.columns:
                df = pd.read_csv(dataset_path, encoding='latin-1', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])
            
            df, sampling_report = smart_sample(df, target_column, max_rows=15000)
            
            p = Path(dataset_path)
            temp_path = str(p.with_stem(p.stem + "_trimmed"))
            df.to_csv(temp_path, index=False)
            session["dataset_path"] = temp_path
            
            await manager.broadcast({"type": "stage", "stage": "sample", "status": "done", "msg": "Sampling complete."})
        
        # ── 1. TRIGGER PROFILER ────────────────────────────
        await manager.broadcast({"type": "stage", "stage": "profile", "status": "running", "msg": "Analyzing dataset shape & cardinality..."})
        
        if task_type == "tabular":
            profile = profile_tabular(session["dataset_path"], session["target_column"], session["problem_type"])
        elif task_type == "text":
            profile = profile_text(session["dataset_path"], session["text_column"], session["target_column"])
        elif task_type == "image":
            # --- UPDATED DUMMY PROFILE ---
            # Giving Pydantic the required keys so DatasetProfile validation passes
            profile = {
                "dataset_type": "image",
                "n_rows": 100,               # Dummy integer
                "n_columns": 12288,          # Dummy integer (e.g., 64x64x3)
                "dtype": "numeric",          # Dummy string
                "missing_cells": 0,          # Assumed 0 for images
                "target_type": "categorical",
                "problem_type": "classification",
                "flags": {
                    
                    "has_missing": False,
                    "high_missing": False,
                    "imbalanced": False,
                    "large_dataset": False
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported task type.")
            
        await manager.broadcast({"type": "stage", "stage": "profile", "status": "done", "msg": "Profiling complete."})

        # ── 2. TRIGGER PREPROCESSOR ────────────────────────────
        await manager.broadcast({"type": "stage", "stage": "preprocess", "status": "running", "msg": "Imputing, Scaling & Encoding..."})
        
        if task_type == "tabular":
            preprocessor = DynamicPreprocessor(session["dataset_path"], session["target_column"])
        elif task_type == "text":
            preprocessor = TextPreprocessor(session["dataset_path"], session["text_column"], session["target_column"])
        elif task_type == "image":
            # Import and trigger the new Image Preprocessor
            from services.preprocessing.image_preprocess import ImagePreprocessor
            preprocessor = ImagePreprocessor(target_size=(64, 64))
            
        processed_data = preprocessor.run(session)
        await manager.broadcast({"type": "stage", "stage": "preprocess", "status": "done", "msg": "Data transformed to matrices."})

        # ── 3. TRIGGER LAYER 3 (The Model Filter) ────────────────────────
        await manager.broadcast({"type": "stage", "stage": "filter", "status": "running", "msg": "Applying mathematical constraints..."})
        
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'layer3_filter', 'rules', 'master_rules.yaml')
        dataset_profile = DatasetProfile(**profile)
        
        filter_engine = DatasetModelFilter(yaml_path)
        filter_engine.logger.broadcast_callback = manager.broadcast
        filter_results = await filter_engine.apply_rules(dataset_profile)
        
        await manager.broadcast({"type": "stage", "stage": "filter", "status": "done", "msg": "Unsuitable models excluded."})

        # ── 5. TRIGGER LAYER 5 (The Parallel Training Engine) ────────────
        await manager.broadcast({"type": "stage", "stage": "train", "status": "running", "msg": "Distributing models to CPU/GPU cores..."})
        
        ready_models = filter_results["ready_models"]
        
        X_train = processed_data["X_train"]
        X_test = processed_data["X_test"]
        y_train = processed_data["y_train"]
        y_test = processed_data["y_test"]
        
        runner = ModelRunner()
        final_results = runner.train_and_evaluate(ready_models, X_train, X_test, y_train, y_test)
        
        await manager.broadcast({"type": "stage", "stage": "train", "status": "done", "msg": "Training and cross-validation complete."})
        
        final_results["sampling_report"] = sampling_report
        
        project_root = Path(__file__).resolve().parent.parent.parent
        results_file = project_root / "data" / "features" / "results.json"
        
        runner.save_results(final_results, results_file)

        winner_obj = final_results.get("winner", {"name": "Unknown", "score": 0})
        if isinstance(winner_obj, str): 
            winner_obj = {"name": winner_obj, "score": 0}

        return {
            "status": "success", 
            "message": "All Layers complete! Results saved to disk.",
            "winner": winner_obj,
            "confidence": final_results.get("confidence", {"level": "high", "value": 100.0, "gap": 0}),
            "explanation": final_results.get("explanation", "Evaluation completed successfully."),
            "results": final_results 
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)