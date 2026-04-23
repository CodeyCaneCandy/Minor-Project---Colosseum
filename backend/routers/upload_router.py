from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from pathlib import Path
import uuid
import json

router = APIRouter()

# Dynamically find the project root (backend/routers -> backend -> Colosseum)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Ensure directories exist
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        file_path = DATASETS_DIR / file.filename
        
        # Stream the upload to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Basic heuristic for file type
        ext = file_path.suffix.lower()
        file_type = "tabular" if ext in [".csv", ".tsv", ".xlsx"] else "text" if ext in [".txt"] else "unknown"

        # Initialize the session state
        session_data = {
            "id": f"session_{uuid.uuid4().hex[:8]}",
            "status": "uploaded",
            "filename": file.filename,
            "dataset_path": str(file_path),
            "file_type": file_type
        }

        # Save to session.json
        session_file = SESSIONS_DIR / "session.json"
        session_file.write_text(json.dumps(session_data, indent=2))

        return {"message": "File uploaded successfully", "session": session_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")