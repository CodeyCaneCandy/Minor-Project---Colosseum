from dotenv import load_dotenv
load_dotenv()  # loads .env from the project root before anything else imports os.getenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from routers.upload_router import router as upload_router
from routers.task_router   import router as task_router
from routers.config_router import router as config_router
from routers.evaluate_router import router as evaluate_router
from routers.chat_router   import router as chat_router
from core.ws_manager import manager

app = FastAPI(title="Colosseum API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_origins=["*"] is fine for a localhost demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LAYER 1 ROUTERS ───────────────────────────────────────────────────────────
app.include_router(upload_router, prefix="/api")
app.include_router(task_router,   prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(evaluate_router, prefix="/api")
app.include_router(chat_router,   prefix="/api")

# ── OPTIONAL: health check ────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ── LAYER 3: WEBSOCKET STREAM ─────────────────────────────────────────────────
@app.websocket("/api/filter-stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the pipe open. We only send data, we don't expect to receive.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── SERVE FRONTEND ────────────────────────────────────────────────────────────
# Mounts frontend/index.html at /
# Must come LAST — it catches everything not matched by the API routes above
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

# ── RUN ───────────────────────────────────────────────────────────────────────
# From the backend/ directory:
#   uvicorn main:app --reload --port 8000