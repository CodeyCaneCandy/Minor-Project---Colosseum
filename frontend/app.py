"""
app.py — Colosseum Layer 7 Dashboard (Streamlit)
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Colosseum — Model Evaluator",
    page_icon="🏛",
    layout="wide",
)

# ── PATHS ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SESSION_FILE  = PROJECT_ROOT / "data" / "sessions" / "session.json"
RESULTS_FILE  = PROJECT_ROOT / "data" / "features" / "results.json"

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #e2e8f0; }
h1 { color: #38bdf8; font-size: 36px; font-weight: 800; }
h2, h3 { color: #94a3b8; font-weight: 600; }
section[data-testid="stSidebar"] { background-color: #020617; }
.card {
    background-color: #1e293b;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 12px;
    border: 1px solid #334155;
}
.metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 28px; font-weight: 800; color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h1>🏛 Colosseum — Model Evaluator</h1>", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_session() -> dict | None:
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return None

def load_results() -> dict | None:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return None

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Training Log", "Results"],
    help="Navigate between session details, engine logs, and final metrics."
)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if menu == "Overview":
    st.header("⚙️ Session Overview")
    
    with st.expander("ℹ️ What is this dashboard for?"):
        st.write("""
        This dashboard is the final stop in the Colosseum pipeline. 
        While the web app handles uploading and live data filtering, this dashboard 
        digs into the heavy analytics: comparing model metrics, viewing confusion matrices, 
        and reading the AI's explanation for the winner.
        """)

    session = load_session()

    if session is None or session.get("status") != "ready":
        st.warning("No completed session found. Please run the gauntlet at localhost:8000 first.")
        st.stop()

    # Session summary cards
    st.subheader("Current Active Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card"><div class="metric-label">File</div><div class="metric-value" style="font-size:16px;">{session["filename"]}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="metric-label">Data type</div><div class="metric-value">{session["file_type"]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="metric-label">Task</div><div class="metric-value">{session.get("task","—")}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="metric-label">Target</div><div class="metric-value">{session.get("target_column","—")}</div></div>', unsafe_allow_html=True)

    st.success("✅ Engine run complete. Navigate to the Results tab to see the winner.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING LOG
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Training Log":
    st.header("📋 Training Log")

    results = load_results()
    if results is None:
        st.info("Evaluation has not run yet. Complete the gauntlet first.")
        st.stop()

    if "sampling_report" in results:
        sr = results["sampling_report"]
        if sr.get("sampling_applied"):
            st.markdown(f"""
            <div class="card">
            <b>Sampling applied</b><br>
            {sr['original_rows']:,} rows → {sr['sampled_rows']:,} rows &nbsp;
            ({sr['sample_fraction']:.1%} kept)<br>
            <span style="color:#94a3b8;font-size:13px;">{sr['reason']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card" style="border-color:#22c55e;">
            <b>No sampling needed</b> — full dataset used ({sr['original_rows']:,} rows)
            </div>
            """, unsafe_allow_html=True)

    if "filter_log" in results:
        with st.expander("🔬 Model filter log (Layer 3)"):
            for entry in results["filter_log"]:
                icon = "🚫" if entry.get("action") == "exclude" else "⚠️"
                st.markdown(f"{icon} **{entry.get('model')}** — {entry.get('reason')}", unsafe_allow_html=True)

    if "model_times" in results:
        st.subheader("⏱ Training times")
        times_df = pd.DataFrame(list(results["model_times"].items()), columns=["Model", "Time (s)"]).sort_values("Time (s)")
        fig = px.bar(times_df, x="Model", y="Time (s)", color_discrete_sequence=["#38bdf8"], template="plotly_dark")
        fig.update_layout(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Results":
    st.header("🏆 Evaluation Results")

    results = load_results()
    if results is None or not results.get("models"):
        st.info("No results available yet.")
        st.stop()

    models_data = results["models"]
    winner = results.get("winner", {})
    confidence = results.get("confidence", {})

    conf_colour = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(confidence.get("level", "low"), "#64748b")

    st.markdown(f"""
    <div class="card" style="border:2px solid {conf_colour}; padding:20px 24px;">
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.1em;">Recommended model</div>
    <div style="font-size:32px;font-weight:900;color:{conf_colour};margin:6px 0;">{winner.get('name','—')}</div>
    <div style="font-size:13px;color:#94a3b8;">
    Composite score: <b>{winner.get('score','—')}</b> &nbsp;|&nbsp;
    Confidence: <b style="color:{conf_colour};">{confidence.get('level','—').upper()}</b> ({confidence.get('value','—')}%)
    </div></div>
    """, unsafe_allow_html=True)

    if "explanation" in results:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #f59e0b;">
        <div style="font-size:11px;color:#f59e0b;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;">Why this model won</div>
        <div style="font-size:14px;color:#e2e8f0;line-height:1.7;font-style:italic;">"{results['explanation']}"</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Model comparison")
    df_results = pd.DataFrame([{"Model": k, **v} for k, v in models_data.items()])
    st.dataframe(df_results.style.highlight_max(subset=[c for c in df_results.columns if c != "Model"], color="#166534", axis=0), use_container_width=True)

    if "confusion_matrix" in results:
        st.subheader("🔥 Confusion matrix — " + winner.get("name", ""))
        cm = np.array(results["confusion_matrix"])
        classes = results.get("label_classes") or [str(i) for i in range(cm.shape[0])]
        fig = go.Figure(go.Heatmap(z=cm, x=[f"Predicted: {c}" for c in classes], y=[f"Actual: {c}" for c in classes], colorscale="Blues", text=cm, texttemplate="%{text}"))
        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)