"""
chat_router.py
──────────────
POST /api/chat   — single-turn chatbot endpoint backed by llm_explainer.py
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from services.explainer.llm_explainer import chat

router = APIRouter()


class ChatMessage(BaseModel):
    role: str    # "user" or "model"
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    session: Optional[dict] = None
    results: Optional[dict] = None


@router.post("/chat")
def handle_chat(req: ChatRequest):
    # Convert our simple history format to Gemini's format
    gemini_history = [
        {"role": m.role, "parts": [m.text]}
        for m in req.history
    ]

    reply = chat(
        user_message=req.message,
        history=gemini_history,
        session=req.session,
        results=req.results,
    )

    return {"reply": reply}
