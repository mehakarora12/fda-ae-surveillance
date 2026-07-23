# Phase 6 — Streamlit Dashboard
# Status: not started.
"""
Phase 6 — Streamlit Dashboard
===============================
Ties every previous phase's output into one interactive view:
  - Overview: project-wide KPIs and categorization insights (Phase 1b)
  - Anomaly Timeline: actual vs. expected counts per drug, anomalies marked (Phase 3)
  - Anomaly Explanations: the RAG-generated root-cause notes (Phase 4)
  - Evaluation Metrics: the quantified accuracy/precision/faithfulness numbers (Phase 5)

Each section checks for its required data file and shows a clear "run this
command first" message instead of crashing if an earlier phase hasn't been
run yet — the dashboard should degrade gracefully, not require the whole
pipeline to be complete just to look at what IS ready.

Usage
-----
    streamlit run app/dashboard.py
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

st.set_page_config(page_title="FDA AE Surveillance Dashboard", layout="wide")

DATA = config.PROCESSED_DIR
REPORTS_DIR = config.ROOT_DIR / "outputs" / "reports"

COUNTS_PATH = DATA / "daily_counts.csv"
CATEGORIZED_PATH = DATA / "categorized_reports.parquet"
ANOMALIES_PATH = DATA / "anomalies.csv"
EXPLANATIONS_PATH = DATA / "anomaly_explanations.csv"
FAITHFULNESS_PATH = DATA / "faithfulness_scores.csv"
CATEGORIZATION_SCORE_PATH = REPORTS_DIR / "categorization_accuracy.txt"
ANOMALY_SCORE_PATH = REPORTS_DIR / "anomaly_precision.txt"


# --------------------------------------------------------------------------
# Cached data loaders — each returns None if the file doesn't exist yet,
# so callers can show a helpful message instead of crashing.
# --------------------------------------------------------------------------
@st.cache_data
def load_counts():
    if not COUNTS_PATH.exists():
        return None
    return pd.read_csv(COUNTS_PATH, parse_dates=["date"])


@st.cache_data
def load_categorized():
    if not CATEGORIZED_PATH.exists():
        return None
    return pd.read_parquet(CATEGORIZED_PATH)


@st.cache_data
def load_anomalies():
    if not ANOMALIES_PATH.exists():
        return None
    return pd.read_csv(ANOMALIES_PATH, parse_dates=["date"])


@st.cache_data
def load_explanations():
    if not EXPLANATIONS_PATH.exists():
        return None
    df = pd.read_csv(EXPLANATIONS_PATH)

    def _safe_parse_list(v):
        try:
            return ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [v]
        except (ValueError, SyntaxError):
            return [v]

    df["dominant_organ_systems"] = df["dominant_organ_systems"].apply(_safe_parse_list)
    return df


@st.cache_data
def load_faithfulness():
    if not FAITHFULNESS_PATH.exists():
        return None
    return pd.read_csv(FAITHFULNESS_PATH)


def missing_data_notice(command: str) -> None:
    st.info(f"No data yet for this section. Run:\n\n```\n{command}\n```")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("💊 FDA Adverse Event Surveillance & RAG Diagnostics")
st.caption("End-to-end pipeline: openFDA ingestion → LLM categorization → vector search → forecasting & anomaly detection → RAG diagnostics → evaluation")

tab_overview, tab_timeline, tab_explanations, tab_eval = st.tabs(
    ["📊 Overview", "📈 Anomaly Timeline", "🔍 Anomaly Explanations", "✅ Evaluation Metrics"]
)


# --------------------------------------------------------------------------
# Tab 1: Overview
# --------------------------------------------------------------------------
with tab_overview:
    categorized = load_categorized()
    counts = load_counts()
    anomalies = load_anomalies()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Drugs tracked", counts["drug"].nunique() if counts is not None else "—")
    col2.metric("Total categorized reports", f"{len(categorized):,}" if categorized is not None else "—")
    col3.metric(
        "High-confidence anomalies",
        int(anomalies["is_anomaly_high_confidence"].sum()) if anomalies is not None else "—",
    )
    col4.metric(
        "Date range",
        f"{counts['date'].min().date()} – {counts['date'].max().date()}" if counts is not None else "—",
    )

    st.divider()

    if categorized is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Reports by organ system")
            organ_counts = categorized["primary_organ_system"].value_counts()
            fig = go.Figure(go.Bar(x=organ_counts.values, y=organ_counts.index, orientation="h"))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.subheader("Reports by severity")
            severity_counts = categorized["severity"].value_counts()
            fig = go.Figure(go.Pie(labels=severity_counts.index, values=severity_counts.values, hole=0.4))
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width='stretch')
    else:
        missing_data_notice("python src/extraction.py merge")


# --------------------------------------------------------------------------
# Tab 2: Anomaly Timeline
# --------------------------------------------------------------------------
with tab_timeline:
    counts = load_counts()
    anomalies = load_anomalies()

    if counts is None:
        missing_data_notice("python src/data_ingestion.py counts")
    elif anomalies is None:
        missing_data_notice("python src/forecasting.py run")
    else:
        drug = st.selectbox("Select a drug", sorted(anomalies["drug"].unique()))
        drug_data = anomalies[anomalies["drug"] == drug].sort_values("date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drug_data["date"], y=drug_data["count"], mode="lines", name="Actual",
            line=dict(color="#4C72B0", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=drug_data["date"], y=drug_data["expected_count"], mode="lines", name="Expected (Holt-Winters)",
            line=dict(color="#55A868", width=1.5, dash="dash"),
        ))
        flagged = drug_data[drug_data["is_anomaly_high_confidence"]]
        fig.add_trace(go.Scatter(
            x=flagged["date"], y=flagged["count"], mode="markers", name="High-confidence anomaly",
            marker=dict(color="#C44E52", size=10, symbol="circle"),
        ))
        fig.update_layout(
            title=f"Daily serious AE reports — {drug}", height=500,
            xaxis_title="Date", yaxis_title="Report count", hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

        st.subheader(f"Flagged anomalies for {drug}")
        st.dataframe(
            flagged[["date", "count", "expected_count", "zscore"]].reset_index(drop=True),
            width='stretch',
        )


# --------------------------------------------------------------------------
# Tab 3: Anomaly Explanations
# --------------------------------------------------------------------------
with tab_explanations:
    explanations = load_explanations()

    if explanations is None:
        missing_data_notice("python src/rag_diagnostics.py run")
    else:
        col1, col2 = st.columns(2)
        drug_filter = col1.multiselect("Filter by drug", sorted(explanations["drug"].unique()))
        severity_filter = col2.multiselect("Filter by severity", sorted(explanations["dominant_severity"].unique()))

        filtered = explanations.copy()
        if drug_filter:
            filtered = filtered[filtered["drug"].isin(drug_filter)]
        if severity_filter:
            filtered = filtered[filtered["dominant_severity"].isin(severity_filter)]

        st.caption(f"Showing {len(filtered)} of {len(explanations)} anomaly explanations")

        for _, row in filtered.sort_values("date").iterrows():
            with st.expander(f"**{row['drug']}** — {row['date']}  ·  {row['dominant_severity']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Actual count", int(row["count"]))
                col2.metric("Expected count", f"{row['expected_count']:.1f}")
                col3.metric("Z-score", f"{row['zscore']:.2f}")

                st.markdown(f"**Summary:** {row['summary']}")
                st.markdown(f"**Organ systems:** {', '.join(row['dominant_organ_systems'])}")
                st.markdown(f"**Notable pattern:** {row['notable_pattern']}")
                st.caption(row["confidence_caveat"])


# --------------------------------------------------------------------------
# Tab 4: Evaluation Metrics
# --------------------------------------------------------------------------
with tab_eval:
    st.subheader("Categorization accuracy (Phase 1b vs. independent labeling)")
    if CATEGORIZATION_SCORE_PATH.exists():
        st.code(CATEGORIZATION_SCORE_PATH.read_text())
    else:
        missing_data_notice("python src/evaluation.py sample-for-labeling --n 100  (then label it, then score-categorization)")

    st.subheader("Anomaly precision (Phase 3 vs. documented real-world signals)")
    if ANOMALY_SCORE_PATH.exists():
        st.code(ANOMALY_SCORE_PATH.read_text())
    else:
        missing_data_notice("python src/evaluation.py sample-anomalies  (then label it, then score-anomalies)")

    st.subheader("RAG faithfulness (Phase 4 explanations vs. retrieved sources)")
    faithfulness = load_faithfulness()
    if faithfulness is not None:
        st.metric("Mean faithfulness score", f"{faithfulness['score'].mean():.2f}", help=f"n={len(faithfulness)}")
        st.dataframe(faithfulness.sort_values("score"), width='stretch')
    else:
        missing_data_notice("python src/evaluation.py score-faithfulness")
