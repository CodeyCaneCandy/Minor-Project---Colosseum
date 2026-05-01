from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback
import os # <-- Added for file paths

# Import Layer 2 modules
from services.profiler.tabular_profiler import profile_tabular
from services.profiler.text_profiler import profile_text
from services.preprocessing.tabular_preprocess import DynamicPreprocessor
from services.preprocessing.text_preprocess import TextPreprocessor

# --- NEW LAYER 3 IMPORTS ---
from core.ws_manager import manager
from core.schemas import DatasetProfile
from layer3_filter.engine import DatasetModelFilter

router = APIRouter()

class EvaluateRequest(BaseModel):
    session: dict

@router.post("/evaluate")
async def run_evaluation(request: EvaluateRequest):
    session = request.session
    
    try:
        task_type = session.get("file_type", "tabular")
        
        # ── 1. TRIGGER PROFILER ───────────────────────────────────────────
        if task_type == "tabular":
            profile = profile_tabular(
                file_path=session["dataset_path"],
                target_column=session["target_column"],
                problem_type=session["problem_type"]
            )
            
            # ── 2. TRIGGER PREPROCESSOR ───────────────────────────────────
            preprocessor = DynamicPreprocessor(
                file_path=session["dataset_path"], 
                target_col=session["target_column"]
            )
            processed_data = preprocessor.run(session)
            
        elif task_type == "text":
            profile = profile_text(
                file_path=session["dataset_path"],
                text_column=session["text_column"],
                target_column=session["target_column"]
            )
            
            # ── 2. TRIGGER PREPROCESSOR ───────────────────────────────────
            preprocessor = TextPreprocessor(
                file_path=session["dataset_path"],
                text_col=session["text_column"],
                target_col=session["target_column"]
            )
            processed_data = preprocessor.run(session)
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported task type.")

        # ── 3. TRIGGER LAYER 3 (The Model Filter) ────────────────────────
        
        # Find the YAML file dynamically so it doesn't break
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'layer3_filter', 'rules', 'master_rules.yaml')
        
        # A. Convert the raw dictionary into our strict Pydantic contract
        dataset_profile = DatasetProfile(**profile)
        
        # B. Initialize the engine
        filter_engine = DatasetModelFilter(yaml_path)
        
        # C. Hook the logger to the WebSocket manager!
        filter_engine.logger.broadcast_callback = manager.broadcast
        
        # D. Run the gauntlet (This awaits the live stream)
        filter_results = await filter_engine.apply_rules(dataset_profile)

        # ── 4. RETURN FINAL RESPONSE ─────────────────────────────────────
        return {
            "status": "success", 
            "message": "Layer 2 Profiling, Preprocessing, and Layer 3 Filtering complete!",
            "profile_summary": profile["summary"],
            "layer3_results": filter_results, # <-- Added the final model list here!
            "data_shapes": {
                "X_train": list(processed_data["X_train"].shape),
                "X_test": list(processed_data["X_test"].shape)
            }
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))