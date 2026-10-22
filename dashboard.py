"""
Self-Improving Agent Dashboard
===============================
Streamlit dashboard showing improvement metrics, experience memory, and agent pipeline.

Run: uv run streamlit run dashboard.py
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd

MEMORY_FILE = Path(__file__).parent / "experience_memory.json"


def load_experiences() -> list[dict]:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return []


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Improving Agent",
    page_icon="🧠",
    layout="wide",
)

st.title("Self-Improving IT Incident Agent")
st.caption("Reflexion loop + experience memory = measurable improvement")

# ── Load data ────────────────────────────────────────────────
experiences = load_experiences()

# ── Sidebar: Controls ───────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("Refresh Data"):
        st.rerun()

    st.divider()
    st.header("Architecture")
    st.markdown("""
    ```
    Incident
       ↓
    [Triage] → classify
       ↓
    [Resolution] → propose fix
       ↓
    ┌─[Critic] → score (5 dims)
    │    ↓
    │  score >= 0.85? → Store + Exit
    │    ↓ no
    │  [Refiner] → improve
    └────┘ (max 5 loops)
       ↓
    [Narrator] → TTS summary
    ```
    """)

    st.divider()
    st.header("Sponsor Tools")
    st.markdown("""
    - **Google ADK** — Agent framework
    - **ElevenLabs** — Voice narration
    - **Datadog** — LLM Observability
    - **Braintrust** — Eval scoring
    """)

# ── Top metrics row ──────────────────────────────────────────
if experiences:
    scores = [e["score"] for e in experiences]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resolutions", len(scores))
    col2.metric("Average Score", f"{sum(scores)/len(scores):.3f}")
    col3.metric("Best Score", f"{max(scores):.3f}")
    improvement = scores[-1] - scores[0] if len(scores) > 1 else 0
    col4.metric(
        "Improvement",
        f"{scores[-1]:.3f}",
        delta=f"{improvement:+.3f}" if len(scores) > 1 else None,
    )
else:
    st.info("No experiences stored yet. Run the agent to see improvement metrics.")

# ── Charts ───────────────────────────────────────────────────
if experiences:
    st.header("Improvement Over Time")

    df = pd.DataFrame(experiences)
    df["index"] = range(1, len(df) + 1)
    df["cumulative_avg"] = df["score"].expanding().mean().round(3)

    tab1, tab2 = st.tabs(["Score Trajectory", "By Category"])

    with tab1:
        chart_df = df[["index", "score", "cumulative_avg"]].set_index("index")
        st.line_chart(chart_df, height=350)
        st.caption("Blue = individual score, Red = cumulative average")

    with tab2:
        categories = df["category"].unique()
        cat_data = {}
        for cat in categories:
            cat_scores = df[df["category"] == cat]["score"].tolist()
            cat_data[cat] = {
                "count": len(cat_scores),
                "avg": round(sum(cat_scores) / len(cat_scores), 3),
                "best": max(cat_scores),
            }
        cat_df = pd.DataFrame(cat_data).T
        cat_df.index.name = "Category"
        st.bar_chart(cat_df["avg"])
        st.dataframe(cat_df, use_container_width=True)

# ── Experience Memory Table ──────────────────────────────────
if experiences:
    st.header("Experience Memory")

    display_df = pd.DataFrame(experiences)
    display_df = display_df[["id", "timestamp", "category", "score", "resolution"]]
    display_df["resolution"] = display_df["resolution"].str[:200] + "..."
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
        "%H:%M:%S"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=1, format="%.3f"
            ),
            "category": st.column_config.TextColumn("Category"),
            "timestamp": st.column_config.TextColumn("Time"),
            "resolution": st.column_config.TextColumn("Resolution (preview)"),
        },
    )

# ── Demo quick-start section ────────────────────────────────
st.divider()
st.header("Demo Quick Start")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sample Incidents")
    st.code(
        """# CPU spike
High CPU utilization at 98% on web-prod-03.
Deployment v2.4.1 rolled out 2 hours ago.
Response times degraded from 200ms to 3s.

# Memory leak
Memory leak detected in api-gateway-01.
Heap at 94% and growing steadily.
Connection pool exhaustion warnings in logs.

# Disk space
Disk space critical at 95% on db-replica-02.
Log files consuming 40GB. Database temp files growing.""",
        language=None,
    )

with col2:
    st.subheader("Commands")
    st.code(
        """# Start agent (ADK web UI, from the repo root)
uv run adk web .

# Start with Datadog tracing
bash scripts/run.sh

# Run evals (before/after)
uv run python -m eval.braintrust_eval v1
uv run python -m eval.braintrust_eval v2

# Clear memory for demo reset
python -c "from self_improving_agent.tools import clear_experience_memory; clear_experience_memory()"
""",
        language="bash",
    )
