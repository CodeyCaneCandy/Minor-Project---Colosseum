from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import zipfile # <--- 1. ADD THIS IMPORT
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

        # ==========================================
        # 2. START OF NEW ZIP HANDLING LOGIC
        # ==========================================
        ext = file_path.suffix.lower()
        dataset_target_path = str(file_path) # Default to the file itself
        
        if ext == ".zip":
            file_type = "image"
            # Create a folder with the same name as the zip (minus the .zip)
            extract_dir = DATASETS_DIR / file.filename.replace('.zip', '')
            extract_dir.mkdir(exist_ok=True)
            
            # Extract the zip contents into that new folder
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # Point the Colosseum engine to the FOLDER, not the zip file
            dataset_target_path = str(extract_dir)
            
            # Delete the original zip file to save space
            # --- NEW SOFT-FAIL DELETION ---
            try:
                file_path.unlink() 
            except PermissionError:
                print(f"Warning: Could not delete {file_path}. Likely locked by OneDrive.")
            except Exception as e:
                print(f"Warning: Failed to clean up zip file: {e}")
            # ------------------------------
            
        else:
            # original heuristic for tabular/text
            file_type = "tabular" if ext in [".csv", ".tsv", ".xlsx"] else "text" if ext in [".txt"] else "unknown"

        # Initialize the session state
        session_data = {
            "id": f"session_{uuid.uuid4().hex[:8]}",
            "status": "uploaded",
            "filename": file.filename,
            "dataset_path": dataset_target_path, # <--- 3. Make sure this uses the new variable!
            "file_type": file_type
        }
        # ==========================================
        # END OF NEW LOGIC
        # ==========================================

        # Save to session.json
        session_file = SESSIONS_DIR / "session.json"
        print(f"DEBUG: Writing session to {session_file}")
        session_file.write_text(json.dumps(session_data, indent=2))

        return {"message": "File uploaded successfully", "session": session_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")