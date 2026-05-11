"""
llm_explainer.py
────────────────
Groq-powered chat assistant for the Colosseum ML platform.
Uses Groq's free API (llama-3.3-70b) — no billing required.

Get a free key at: https://console.groq.com  (takes 30 seconds)
Add to .env:  GROQ_API_KEY=your_key_here
"""

import os
import textwrap
from typing import Optional

from groq import Groq

# ── Configure Groq ────────────────────────────────────────────────────────────
_api_key = os.getenv("GROQ_API_KEY", "")
_client  = Groq(api_key=_api_key) if _api_key else None

_MODEL_NAME = "llama-3.3-70b-versatile"   # free, fast, excellent reasoning

_SYSTEM_PROMPT = textwrap.dedent("""
    You are Colosseum's built-in AI assistant — a friendly ML explainer embedded
    inside the Colosseum automated model-selection platform.

    Your job:
    • Answer questions about the dataset that was uploaded.
    • Explain the training results: why a model won, what the metrics mean, what
      the confidence score tells the user, why other models were filtered out.
    • Give plain-English explanations suitable for someone learning ML.
    • Keep answers concise (3–5 sentences unless asked to elaborate).
    • Never make up numbers — only use the context you are given.
    • Use a slightly technical but friendly tone — like a senior data scientist
      explaining to a junior colleague.

    Context you have access to (injected below):
    {context_block}

    If the user asks something outside this context, say so honestly and offer
    to explain a general ML concept instead.
""").strip()


# ── Build context block ───────────────────────────────────────────────────────
def _build_context(session: Optional[dict], results: Optional[dict]) -> str:
    parts = []

    if session:
        parts.append("=== SESSION INFO ===")
        parts.append(f"Task type  : {session.get('problem_type', 'unknown')} / {session.get('file_type', 'unknown')}")
        parts.append(f"Target col : {session.get('target_column', '—')}")
        if session.get("text_column"):
            parts.append(f"Text col   : {session['text_column']}")
        parts.append(f"Train split: {session.get('train_split', 0.8) * 100:.0f}%")

    if results:
        parts.append("\n=== EVALUATION RESULTS ===")
        winner = results.get("winner", {})
        if isinstance(winner, str):
            winner = {"name": winner}
        parts.append(f"Winner     : {winner.get('name', '—')}")
        parts.append(f"Score      : {winner.get('score', '—')}")

        conf = results.get("confidence", {})
        parts.append(f"Confidence : {conf.get('level', '—')} ({conf.get('value', '—')}%)")

        if results.get("explanation"):
            parts.append(f"Auto-explanation: {results['explanation']}")

        models = results.get("models", {})
        if models:
            parts.append("\nAll model scores:")
            for name, info in models.items():
                cs  = info.get("composite_score", 0)
                m   = info.get("metrics", {})
                acc = m.get("accuracy", "—")
                f1  = m.get("f1", "—")
                auc = m.get("roc_auc", "—")
                parts.append(
                    f"  {name:22s}  composite={cs:.4f}"
                    f"  acc={acc if acc=='—' else f'{acc:.4f}'}"
                    f"  f1={f1 if f1=='—' else f'{f1:.4f}'}"
                    f"  auc={auc if auc=='—' else f'{auc:.4f}'}"
                )

        sampling = results.get("sampling_report", {})
        if sampling.get("sampling_applied"):
            parts.append(
                f"\nSampling: {sampling['original_rows']:,} → {sampling['sampled_rows']:,} rows kept"
            )

    if not parts:
        return "No session or results context available yet — answer general ML questions."

    return "\n".join(parts)


def _build_system_prompt(session: Optional[dict], results: Optional[dict]) -> str:
    ctx = _build_context(session, results)
    return _SYSTEM_PROMPT.format(context_block=ctx)


# ── Public API ────────────────────────────────────────────────────────────────
def chat(
    user_message: str,
    history: list,          # [{"role": "user"|"model", "parts": [str]}, ...]
    session: Optional[dict] = None,
    results: Optional[dict] = None,
) -> str:
    if not _api_key or not _client:
        return (
            "⚠️  GROQ_API_KEY is not set in your .env file. "
            "Get a free key at https://console.groq.com (30 seconds) "
            "then add  GROQ_API_KEY=your_key  to your .env and restart."
        )

    try:
        system_prompt = _build_system_prompt(session, results)

        # Groq uses OpenAI-compatible format
        messages = [{"role": "system", "content": system_prompt}]

        # Convert history (Gemini format "model" → Groq format "assistant")
        for turn in history:
            role = "assistant" if turn["role"] == "model" else turn["role"]
            messages.append({"role": role, "content": turn["parts"][0]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        response = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        return f"❌ Groq error: {exc}"


def explain_results(session: dict, results: dict) -> str:
    """One-shot post-Gauntlet explanation."""
    prompt = (
        "The Gauntlet just finished. In 3-4 sentences explain which model won, "
        "why it beat the others based on the scores, and what the confidence level means. "
        "Be friendly and beginner-friendly."
    )
    return chat(prompt, history=[], session=session, results=results)