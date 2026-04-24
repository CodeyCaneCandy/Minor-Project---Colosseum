from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback

# Import Layer 2 modules
from services.profiler.tabular_profiler import profile_tabular
from services.profiler.text_profiler import profile_text
from services.preprocessing.tabular_preprocess import DynamicPreprocessor
from services.preprocessing.text_preprocess import TextPreprocessor

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

        # Note: In the future, Layer 3 (Filter) and Layer 4 (Model Runner) 
        # will be called right here using `profile` and `processed_data`.

        return {
            "status": "success", 
            "message": "Layer 2 Profiling and Preprocessing complete!",
            "profile_summary": profile["summary"],
            "data_shapes": {
                "X_train": list(processed_data["X_train"].shape),
                "X_test": list(processed_data["X_test"].shape)
            }
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))