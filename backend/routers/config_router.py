from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_FILE = PROJECT_ROOT / "data" / "sessions" / "session.json"

class ConfigRequest(BaseModel):
    target_column: Optional[str] = None
    text_column: Optional[str] = None
    split_ratio: float = 0.8
    random_seed: int = 42

@router.post("/config")
async def set_config(request: ConfigRequest):
    if not SESSION_FILE.exists():
        raise HTTPException(status_code=400, detail="No active session found.")

    try:
        session_data = json.loads(SESSION_FILE.read_text())

        # Update session with configurations
        if request.target_column:
            session_data["target_column"] = request.target_column
        if request.text_column:
            session_data["text_column"] = request.text_column
            
        session_data["split_ratio"] = request.split_ratio
        session_data["random_seed"] = request.random_seed
        
        # Crucial state change: This signals the Streamlit app that it can proceed
        session_data["status"] = "ready"

        # Save back to disk
        SESSION_FILE.write_text(json.dumps(session_data, indent=2))
        
        return {"message": "Configuration saved. Ready for evaluation.", "session": session_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")