from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
from pathlib import Path

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SESSION_FILE = PROJECT_ROOT / "data" / "sessions" / "session.json"

class TaskRequest(BaseModel):
    task: str           # e.g., 'Classification', 'Regression'
    problem_type: str   # Usually maps to the same concept for the backend logic
    file_type: Optional[str] = "tabular"
    
@router.post("/task")
async def set_task(request: TaskRequest):
    if not SESSION_FILE.exists():
        raise HTTPException(status_code=400, detail="No active session. Please upload a file first.")

    try:
        session_data = json.loads(SESSION_FILE.read_text())
        
        # Update session
        session_data["task"] = request.task
        session_data["problem_type"] = request.problem_type
        session_data["status"] = "configured"
        session_data["file_type"] = request.file_type 

        # Save back to disk
        SESSION_FILE.write_text(json.dumps(session_data, indent=2))
        
        return {"message": "Task successfully set", "session": session_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set task: {str(e)}")