"""
app.py — Colosseum Layer 7 Dashboard (Streamlit)

Run from the project root:
    streamlit run frontend/app.py

This app does NOT upload files or call /split.
It reads data/sessions/session.json (written by Layer 1)
and data/features/results.json (written by Layer 5).

The HTML frontend (index.html) handles upload → task → config.
This app handles the evaluation trigger and results display.
"""

import json
import time
import requests
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
# app.py lives at frontend/app.py
# project root is one level up
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SESSION_FILE  = PROJECT_ROOT / "data" / "sessions" / "session.json"
RESULTS_FILE  = PROJECT_ROOT / "data" / "features" / "results.json"
BACKEND_URL   = "http://127.0.0.1:8000"

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: #e2e8f0; }
h1 { color: #38bdf8; font-size: 36px; font-weight: 800; }
h2, h3 { color: #94a3b8; font-weight: 600; }
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9, #22c55e);
    color: white; font-weight: 700;
    border-radius: 8px; border: none;
    padding: 10px 24px; font-size: 15px;
}
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
st.caption("Upload your dataset at localhost:8000, then evaluate and compare models here.")

# ── SIDEBAR NAV ───────────────────────────────────────────────────────────────
menu = st.sidebar.selectbox(
    "Navigation",
    ["Setup & Evaluate", "Training Log", "Results"],
)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_session() -> dict | None:
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return None


def load_results() -> dict | None:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return None


def status_badge(status: str) -> str:
    colours = {
        "uploaded":   "#f59e0b",
        "configured": "#3b82f6",
        "ready":      "#22c55e",
        "running":    "#8b5cf6",
        "done":       "#22c55e",
    }
    c = colours.get(status, "#64748b")
    return (
        f'<span style="background:{c};color:white;padding:2px 10px;'
        f'border-radius:4px;font-size:11px;font-weight:700;">'
        f'{status.upper()}</span>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SETUP & EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
if menu == "Setup & Evaluate":
    st.header("⚙️ Session Setup")

    session = load_session()

    # ── No session yet ────────────────────────────────────────────────────────
    if session is None:
        st.markdown("""
        <div class="card">
        <b>No session found.</b><br><br>
        Go to <a href="http://localhost:8000" target="_blank"
        style="color:#38bdf8;">localhost:8000</a>
        to upload your dataset, select a task, and save your config.
        Once that's done, come back here to run the evaluation.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Session loaded ────────────────────────────────────────────────────────
    st.markdown(
        f"**Current session** &nbsp; {status_badge(session['status'])}",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Session summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">File</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="font-size:16px;">{session["filename"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Data type</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{session["file_type"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Task</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{session.get("task","—")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Split</div>', unsafe_allow_html=True)
        split_pct = int(session.get("split_ratio", 0.8) * 100)
        st.markdown(f'<div class="metric-value">{split_pct}/{100-split_pct}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Full session JSON expander
    with st.expander("View full session.json"):
        st.json(session)

    # ── Not ready yet ─────────────────────────────────────────────────────────
    if session["status"] != "ready":
        st.warning(
            f"Session status is **{session['status']}** — "
            "complete all three steps at localhost:8000 before evaluating."
        )
        st.stop()

    # ── Ready — show evaluate button ──────────────────────────────────────────
    st.success("Session is ready. Click below to run the evaluation pipeline.")

    if st.button("🚀 Run Evaluation"):
        try:
            with st.spinner("Calling backend — running Layers 2 → 5..."):
                resp = requests.post(
                    f"{BACKEND_URL}/api/evaluate",
                    json={"session": session},
                    timeout=300,
                )
            if resp.status_code == 200:
                st.success("Evaluation complete! Switch to the Results tab.")
                st.session_state["eval_done"] = True
            else:
                st.error(f"Backend returned {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the backend at localhost:8000. "
                "Make sure uvicorn is running."
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING LOG
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Training Log":
    st.header("📋 Training Log")

    session = load_session()
    if session is None:
        st.warning("No session found. Upload a dataset at localhost:8000 first.")
        st.stop()

    results = load_results()
    if results is None:
        st.info("Evaluation has not run yet. Go to Setup & Evaluate to start it.")
        st.stop()

    # Show the sampling report if present
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

    # Show filter exclusion log
    if "filter_log" in results:
        with st.expander("🔬 Model filter log (Layer 3)"):
            for entry in results["filter_log"]:
                icon = "🚫" if entry.get("action") == "exclude" else "⚠️"
                st.markdown(
                    f"{icon} **{entry.get('model')}** — {entry.get('reason')}",
                    unsafe_allow_html=True,
                )

    # Show per-model training times if available
    if "model_times" in results:
        st.subheader("⏱ Training times")
        times_df = pd.DataFrame(
            list(results["model_times"].items()),
            columns=["Model", "Time (s)"],
        ).sort_values("Time (s)")
        fig = px.bar(
            times_df, x="Model", y="Time (s)",
            color_discrete_sequence=["#38bdf8"],
            template="plotly_dark",
        )
        fig.update_layout(
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "Results":
    st.header("🏆 Evaluation Results")

    results = load_results()

    if results is None:
        st.info("No results yet. Run the evaluation from the Setup & Evaluate tab.")
        st.stop()

    models_data = results.get("models", {})
    if not models_data:
        st.warning("Results file found but no model data inside.")
        st.stop()

    # ── Winner card ───────────────────────────────────────────────────────────
    winner     = results.get("winner", {})
    confidence = results.get("confidence", {})

    conf_colour = {
        "high":   "#22c55e",
        "medium": "#f59e0b",
        "low":    "#ef4444",
    }.get(confidence.get("level", "low"), "#64748b")

    st.markdown(f"""
    <div class="card" style="border:2px solid {conf_colour}; padding:20px 24px;">
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.1em;">
    Recommended model</div>
    <div style="font-size:32px;font-weight:900;color:{conf_colour};margin:6px 0;">
    {winner.get('name','—')}</div>
    <div style="font-size:13px;color:#94a3b8;">
    Composite score: <b>{winner.get('score','—')}</b> &nbsp;|&nbsp;
    Confidence: <b style="color:{conf_colour};">
    {confidence.get('level','—').upper()}</b>
    ({confidence.get('value','—')}%)
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── LLM Explanation ───────────────────────────────────────────────────────
    if "explanation" in results:
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #f59e0b;">
        <div style="font-size:11px;color:#f59e0b;text-transform:uppercase;
        letter-spacing:.1em;margin-bottom:8px;">Why this model won</div>
        <div style="font-size:14px;color:#e2e8f0;line-height:1.7;font-style:italic;">
        "{results['explanation']}"
        </div>
        </div>
        """, unsafe_allow_html=True)

        if "raw_prompt" in results:
            with st.expander("Show raw LLM prompt"):
                st.code(results["raw_prompt"], language="text")

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.subheader("📊 Model comparison")

    rows = []
    for model_name, metrics in models_data.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df_results = pd.DataFrame(rows)
    st.dataframe(
        df_results.style.highlight_max(
            subset=[c for c in df_results.columns if c != "Model"],
            color="#166534",
            axis=0,
        ),
        use_container_width=True,
    )

    # ── Radar chart ───────────────────────────────────────────────────────────
    radar_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    available     = [m for m in radar_metrics if m in df_results.columns]

    if len(available) >= 3:
        st.subheader("🕸 Radar — normalised metrics")
        fig = go.Figure()
        for _, row in df_results.iterrows():
            values = [row[m] for m in available]
            values += [values[0]]   # close the polygon
            fig.add_trace(go.Scatterpolar(
                r    = values,
                theta = available + [available[0]],
                fill  = "toself",
                name  = row["Model"],
                opacity = 0.7,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="#1e293b",
            ),
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Composite score bar chart ─────────────────────────────────────────────
    if "composite_score" in df_results.columns:
        st.subheader("📊 Composite scores")
        fig = px.bar(
            df_results.sort_values("composite_score", ascending=True),
            x="composite_score", y="Model",
            orientation="h",
            color="composite_score",
            color_continuous_scale="teal",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    if "confusion_matrix" in results:
        st.subheader("🔥 Confusion matrix — " + winner.get("name", ""))
        cm      = np.array(results["confusion_matrix"])
        classes = results.get("label_classes") or [str(i) for i in range(cm.shape[0])]

        fig = go.Figure(go.Heatmap(
            z          = cm,
            x          = [f"Predicted: {c}" for c in classes],
            y          = [f"Actual: {c}"    for c in classes],
            colorscale = "Blues",
            text       = cm,
            texttemplate = "%{text}",
        ))
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("💾 Export")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_results.to_csv(index=False)
        st.download_button(
            "Download results CSV",
            data=csv,
            file_name="colosseum_results.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download full results JSON",
            data=json.dumps(results, indent=2),
            file_name="colosseum_results.json",
            mime="application/json",
        )
