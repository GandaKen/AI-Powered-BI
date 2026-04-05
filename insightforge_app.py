"""InsightForge BI Assistant — Streamlit dashboard with RAG-powered AI chat.

A multi-view analytics dashboard backed by a local Ollama LLM and a
FAISS vector store.  Views include Sales Overview, Product Analysis,
Regional Analysis, Customer Demographics, Advanced Analytics, and an
AI Assistant that answers natural-language questions about the dataset.

Requirements:
    - Ollama running locally with ``llama3.2:3b`` and ``nomic-embed-text``
    - ``sales_data.csv`` in the working directory
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from insightforge.agent import create_agent
from insightforge.config import settings as agent_settings
from insightforge.observability.tracing import make_trace_collector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — centralised knobs so nothing is buried in function bodies
# ---------------------------------------------------------------------------

DATA_PATH: Path = Path("sales_data.csv")

LLM_MODEL: str = "llama3.2:3b"
EMBEDDING_MODEL: str = "nomic-embed-text"
LLM_TEMPERATURE: float = 0
RAG_TOP_K: int = 5
HEAVY_MODEL: str = agent_settings.llm_model_heavy
CIRCUIT_BREAKER_THRESHOLD: int = 3

CHART_TEMPLATE: str = "plotly_dark"

COLORS: list[str] = [
    "#6366f1", "#22d3ee", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6",
]
PRODUCT_COLORS: dict[str, str] = {
    "Widget A": "#6366f1",
    "Widget B": "#22d3ee",
    "Widget C": "#f59e0b",
    "Widget D": "#ef4444",
}
REGION_COLORS: dict[str, str] = {
    "North": "#6366f1",
    "South": "#22d3ee",
    "East": "#f59e0b",
    "West": "#ef4444",
}

APP_VERSION: str = "2.0"

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InsightForge BI Assistant", page_icon="📊", layout="wide",
)

st.markdown(
    """
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); }
    [data-testid="stSidebar"] * { color: #e0e7ff !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        padding: 1.2rem; border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.3);
        text-align: center;
    }
    .metric-card h3 { color: #a5b4fc; font-size: 0.85rem; margin: 0; font-weight: 500; }
    .metric-card h1 { color: #e0e7ff; font-size: 1.8rem; margin: 0.3rem 0 0 0; font-weight: 700; }
    .metric-card .delta-pos { color: #34d399; font-size: 0.8rem; }
    .metric-card .delta-neg { color: #f87171; font-size: 0.8rem; }
    .section-header { border-left: 4px solid #6366f1; padding-left: 12px; margin: 1.5rem 0 1rem 0; }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .trace-card {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;
        display: flex; align-items: center; gap: 0.8rem;
    }
    .trace-card .step-name {
        font-weight: 600; font-size: 0.82rem; color: #e0e7ff;
        min-width: 150px;
    }
    .trace-bar-bg {
        flex: 1; height: 8px; background: rgba(99, 102, 241, 0.15);
        border-radius: 4px; overflow: hidden;
    }
    .trace-bar-fill { height: 100%; border-radius: 4px; }
    .trace-stat {
        font-size: 0.75rem; color: #a5b4fc; min-width: 60px; text-align: right;
    }
    .trace-summary-row {
        display: flex; gap: 1.2rem; margin-bottom: 0.6rem;
        flex-wrap: wrap;
    }
    .trace-summary-item {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px; padding: 0.5rem 0.9rem; text-align: center;
    }
    .trace-summary-item .label { font-size: 0.7rem; color: #a5b4fc; }
    .trace-summary-item .value { font-size: 1rem; font-weight: 700; color: #e0e7ff; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    """Read *sales_data.csv* and enrich with derived time / demographic columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns plus ``Month``, ``Quarter``, ``Year``,
        ``DayOfWeek``, ``Age_Group``, and ``Satisfaction_Level``.

    Raises
    ------
    FileNotFoundError
        If ``DATA_PATH`` does not point to an existing file.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at '{DATA_PATH.resolve()}'. "
            "Ensure sales_data.csv is in the working directory."
        )

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.day_name()
    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56-69"],
    )
    df["Satisfaction_Level"] = pd.cut(
        df["Customer_Satisfaction"],
        bins=[0, 2, 3, 4, 5.01],
        labels=["Low (1-2)", "Medium (2-3)", "High (3-4)", "Very High (4-5)"],
    )
    return df


# ---------------------------------------------------------------------------
# RAG system
# ---------------------------------------------------------------------------


@st.cache_resource
def initialize_rag_system(_df: pd.DataFrame):
    """Build a FAISS-backed RAG pipeline over aggregated sales statistics.

    Creates LangChain ``Document`` objects covering dataset overview, per-product,
    per-region, cross-segment, gender, and monthly trend summaries, then indexes
    them in a FAISS vector store using Ollama embeddings.

    Parameters
    ----------
    _df : pd.DataFrame
        The full (unfiltered) sales DataFrame.  The leading underscore tells
        Streamlit not to hash this argument.

    Returns
    -------
    tuple[ChatOllama, FAISS]
        The LLM instance and the populated vector store.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    from insightforge.llm.provider import get_embeddings, get_llm

    llm = get_llm(agent_settings, tier="light")
    embeddings = get_embeddings(agent_settings)

    documents: list[Document] = []
    total_sales = _df["Sales"].sum()

    # -- Dataset-wide overview
    documents.append(Document(
        page_content=(
            f"Dataset Overview: {len(_df)} transactions from "
            f"{_df['Date'].min().date()} to {_df['Date'].max().date()}. "
            f"Total Sales: ${total_sales:,.2f}. "
            f"Average Transaction: ${_df['Sales'].mean():.2f}. "
            f"Median Transaction: ${_df['Sales'].median():.2f}. "
            f"Products: {', '.join(_df['Product'].unique())}. "
            f"Regions: {', '.join(_df['Region'].unique())}. "
            f"Customer Age Range: {_df['Customer_Age'].min()}"
            f"-{_df['Customer_Age'].max()} "
            f"(avg {_df['Customer_Age'].mean():.1f}). "
            f"Gender Split: {dict(_df['Customer_Gender'].value_counts())}. "
            f"Overall Satisfaction: "
            f"{_df['Customer_Satisfaction'].mean():.2f}/5."
        ),
        metadata={"type": "overview"},
    ))

    # -- Per-product summaries
    for product in _df["Product"].unique():
        pdf = _df[_df["Product"] == product]
        share = pdf["Sales"].sum() / total_sales * 100
        top_region = pdf.groupby("Region")["Sales"].sum().idxmax()
        documents.append(Document(
            page_content=(
                f"Product {product}: Total ${pdf['Sales'].sum():,.2f} "
                f"({share:.1f}% share), Avg ${pdf['Sales'].mean():.2f}, "
                f"Median ${pdf['Sales'].median():.2f}, {len(pdf)} transactions. "
                f"Satisfaction {pdf['Customer_Satisfaction'].mean():.2f}/5. "
                f"Top region: {top_region}. "
                f"Avg customer age: {pdf['Customer_Age'].mean():.1f}."
            ),
            metadata={"type": "product"},
        ))

    # -- Per-region summaries
    for region in _df["Region"].unique():
        rdf = _df[_df["Region"] == region]
        share = rdf["Sales"].sum() / total_sales * 100
        top_product = rdf.groupby("Product")["Sales"].sum().idxmax()
        documents.append(Document(
            page_content=(
                f"Region {region}: Total ${rdf['Sales'].sum():,.2f} "
                f"({share:.1f}% share), Avg ${rdf['Sales'].mean():.2f}, "
                f"{len(rdf)} transactions. "
                f"Satisfaction {rdf['Customer_Satisfaction'].mean():.2f}/5. "
                f"Top product: {top_product}. "
                f"Gender split: "
                f"{dict(rdf['Customer_Gender'].value_counts())}."
            ),
            metadata={"type": "region"},
        ))

    # -- Product × Region cross-segments
    for product in _df["Product"].unique():
        for region in _df["Region"].unique():
            subset = _df[
                (_df["Product"] == product) & (_df["Region"] == region)
            ]
            if len(subset) > 0:
                documents.append(Document(
                    page_content=(
                        f"{product} in {region}: "
                        f"${subset['Sales'].sum():,.2f} total, "
                        f"Avg ${subset['Sales'].mean():.2f}, "
                        f"{len(subset)} transactions, "
                        f"Satisfaction "
                        f"{subset['Customer_Satisfaction'].mean():.2f}/5."
                    ),
                    metadata={"type": "cross"},
                ))

    # -- Gender summaries
    for gender in _df["Customer_Gender"].unique():
        gdf = _df[_df["Customer_Gender"] == gender]
        documents.append(Document(
            page_content=(
                f"{gender} customers: {len(gdf)} transactions, "
                f"Total ${gdf['Sales'].sum():,.2f}, "
                f"Avg ${gdf['Sales'].mean():.2f}, "
                f"Satisfaction "
                f"{gdf['Customer_Satisfaction'].mean():.2f}/5, "
                f"Avg age {gdf['Customer_Age'].mean():.1f}."
            ),
            metadata={"type": "gender"},
        ))

    # -- Monthly trend highlights
    monthly = (
        _df.groupby("Month")
        .agg({"Sales": ["sum", "mean", "count"]})
        .reset_index()
    )
    monthly.columns = ["Month", "Total", "Avg", "Count"]
    best_month = monthly.loc[monthly["Total"].idxmax()]
    worst_month = monthly.loc[monthly["Total"].idxmin()]
    documents.append(Document(
        page_content=(
            f"Monthly Trends: Best month {best_month['Month']} "
            f"(${best_month['Total']:,.0f}), "
            f"Worst month {worst_month['Month']} "
            f"(${worst_month['Total']:,.0f}). "
            f"Average monthly sales: ${monthly['Total'].mean():,.0f}."
        ),
        metadata={"type": "trend"},
    ))

    vectorstore = FAISS.from_documents(documents, embeddings)
    return llm, vectorstore


@st.cache_resource
def initialize_agent_system(_df: pd.DataFrame):
    """Build the LangGraph-based agentic RAG system."""
    return create_agent(_df, agent_settings)


def run_basic_rag(prompt: str, llm, vectorstore) -> str:
    """Run the legacy chain-based RAG as fallback path."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    retriever = vectorstore.as_retriever(search_kwargs={"k": RAG_TOP_K})
    docs = retriever.invoke(prompt)
    context = "\n".join(d.page_content for d in docs)

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are InsightForge, an expert BI analyst. "
            "Answer questions using ONLY the data context provided. "
            "Be precise with numbers. Use bullet points for comparisons. "
            "If the data doesn't contain the answer, say so clearly.",
        ),
        (
            "human",
            "Data Context:\n{context}\n\nQuestion: {question}",
        ),
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": prompt})


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------


def render_metric(label: str, value: str) -> None:
    """Render a styled KPI metric card via raw HTML."""
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )


def render_section_header(title: str) -> None:
    """Render a view section header with the accent left-border style."""
    st.markdown(
        f'<div class="section-header"><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Present sidebar filter widgets and return the filtered DataFrame.

    Filters include date range, product, region, gender, and customer age.
    A summary count is displayed below the filter expander.
    """
    filtered = df.copy()
    with st.sidebar.expander("🔍 Filters", expanded=True):
        date_range = st.date_input(
            "Date Range",
            value=(df["Date"].min().date(), df["Date"].max().date()),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
        )
        if len(date_range) == 2:
            filtered = filtered[
                (filtered["Date"].dt.date >= date_range[0])
                & (filtered["Date"].dt.date <= date_range[1])
            ]

        products = st.multiselect(
            "Products",
            df["Product"].unique(),
            default=list(df["Product"].unique()),
        )
        if products:
            filtered = filtered[filtered["Product"].isin(products)]

        regions = st.multiselect(
            "Regions",
            df["Region"].unique(),
            default=list(df["Region"].unique()),
        )
        if regions:
            filtered = filtered[filtered["Region"].isin(regions)]

        genders = st.multiselect(
            "Gender",
            df["Customer_Gender"].unique(),
            default=list(df["Customer_Gender"].unique()),
        )
        if genders:
            filtered = filtered[filtered["Customer_Gender"].isin(genders)]

        age_range = st.slider(
            "Age Range",
            int(df["Customer_Age"].min()),
            int(df["Customer_Age"].max()),
            (int(df["Customer_Age"].min()), int(df["Customer_Age"].max())),
        )
        filtered = filtered[
            (filtered["Customer_Age"] >= age_range[0])
            & (filtered["Customer_Age"] <= age_range[1])
        ]

    st.sidebar.markdown(
        f"**Showing {len(filtered):,} of {len(df):,} records**"
    )
    return filtered


def render_trace_steps(trace_data: dict) -> None:
    """Render vertical step cards for a pipeline trace."""
    if not trace_data:
        return

    steps = trace_data.get("steps", [])
    total_ms = trace_data.get("total_latency_ms", 0)
    total_in = trace_data.get("total_tokens_input", 0)
    total_out = trace_data.get("total_tokens_output", 0)
    langfuse_url = trace_data.get("langfuse_url", "")

    summary_html = '<div class="trace-summary-row">'
    summary_html += (
        f'<div class="trace-summary-item">'
        f'<div class="label">Total Latency</div>'
        f'<div class="value">{total_ms:,} ms</div></div>'
    )
    summary_html += (
        f'<div class="trace-summary-item">'
        f'<div class="label">Tokens In</div>'
        f'<div class="value">{total_in:,}</div></div>'
    )
    summary_html += (
        f'<div class="trace-summary-item">'
        f'<div class="label">Tokens Out</div>'
        f'<div class="value">{total_out:,}</div></div>'
    )
    summary_html += (
        f'<div class="trace-summary-item">'
        f'<div class="label">Steps</div>'
        f'<div class="value">{len(steps)}</div></div>'
    )
    summary_html += "</div>"
    st.markdown(summary_html, unsafe_allow_html=True)

    max_ms = max((s.get("latency_ms", 0) for s in steps), default=1) or 1
    step_colors = {
        "query_planner": "#6366f1",
        "retrieval_planner": "#8b5cf6",
        "information_retriever": "#22d3ee",
        "context_assembler": "#10b981",
        "generator": "#f59e0b",
        "response_qa": "#ef4444",
    }

    for step in steps:
        name = step.get("step_name", "unknown")
        ms = step.get("latency_ms", 0)
        tok_in = step.get("tokens_input", 0)
        tok_out = step.get("tokens_output", 0)
        bar_pct = min(int(ms / max_ms * 100), 100) if max_ms else 0
        color = step_colors.get(name, "#6366f1")

        display_name = name.replace("_", " ").title()
        tokens_str = ""
        if tok_in or tok_out:
            tokens_str = f" · {tok_in}/{tok_out} tok"

        st.markdown(
            f'<div class="trace-card">'
            f'<span class="step-name" style="color:{color}">{display_name}</span>'
            f'<div class="trace-bar-bg">'
            f'<div class="trace-bar-fill" style="width:{bar_pct}%;background:{color};"></div>'
            f'</div>'
            f'<span class="trace-stat">{ms:,} ms{tokens_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if langfuse_url:
        st.markdown(
            f'<a href="{langfuse_url}" target="_blank" '
            f'style="font-size:0.8rem; color:#a5b4fc;">View full trace in Langfuse</a>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------

try:
    df_raw = load_data()
except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
    st.error(f"Failed to load data: {exc}")
    st.stop()

st.sidebar.image(
    "https://img.icons8.com/fluency/48/combo-chart.png", width=40,
)
st.sidebar.title("InsightForge")
st.sidebar.markdown("---")

views = [
    "📊 Sales Overview",
    "📦 Product Analysis",
    "🗺️ Regional Analysis",
    "👥 Customer Demographics",
    "🔬 Advanced Analytics",
    "🤖 AI Assistant",
    "📡 Observability",
]
view = st.sidebar.radio("Navigation", views, label_visibility="collapsed")

df = apply_filters(df_raw)

# ──────────────────────── SALES OVERVIEW ────────────────────────

if view == "📊 Sales Overview":
    render_section_header("Sales Overview Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    with c2:
        render_metric("Transactions", f"{len(df):,}")
    with c3:
        render_metric("Avg Transaction", f"${df['Sales'].mean():,.0f}")
    with c4:
        render_metric("Median Sale", f"${df['Sales'].median():,.0f}")
    with c5:
        render_metric(
            "Avg Satisfaction",
            f"{df['Customer_Satisfaction'].mean():.2f}/5",
        )

    st.markdown("")

    monthly = (
        df.groupby("Month")
        .agg({"Sales": ["sum", "mean", "count"]})
        .reset_index()
    )
    monthly.columns = ["Month", "Total_Sales", "Avg_Sale", "Transactions"]
    monthly["Cumulative"] = monthly["Total_Sales"].cumsum()
    monthly["MA_3"] = monthly["Total_Sales"].rolling(3, min_periods=1).mean()

    tab1, tab2 = st.tabs(["📈 Monthly Trend", "📊 Cumulative Sales"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Total_Sales"],
            mode="lines+markers", name="Monthly Sales",
            line={"color": "#6366f1", "width": 2}, marker={"size": 5},
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["MA_3"],
            mode="lines", name="3-Month Moving Avg",
            line={"color": "#f59e0b", "width": 2, "dash": "dash"},
        ))
        fig.update_layout(
            template=CHART_TEMPLATE, height=400,
            xaxis_title="Month", yaxis_title="Sales ($)",
            hovermode="x unified", xaxis={"tickangle": -45, "dtick": 3},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.area(
            monthly, x="Month", y="Cumulative",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400,
            xaxis_title="Month", yaxis_title="Cumulative Sales ($)",
            xaxis={"tickangle": -45, "dtick": 3},
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        quarterly = df.groupby("Quarter")["Sales"].sum().reset_index()
        fig = px.bar(
            quarterly, x="Quarter", y="Sales",
            color_discrete_sequence=["#6366f1"], text_auto=",.0f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=350,
            title="Quarterly Revenue", xaxis={"tickangle": -45},
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dow_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        dow = (
            df.groupby("DayOfWeek")["Sales"]
            .mean()
            .reindex(dow_order)
            .reset_index()
        )
        fig = px.bar(
            dow, x="DayOfWeek", y="Sales",
            color_discrete_sequence=["#22d3ee"], text_auto=",.0f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=350,
            title="Avg Sales by Day of Week",
            xaxis_title="", yaxis_title="Avg Sales ($)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── PRODUCT ANALYSIS ────────────────────────

elif view == "📦 Product Analysis":
    render_section_header("Product Performance Analysis")

    prod_agg = (
        df.groupby("Product")
        .agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sale=("Sales", "mean"),
            Transactions=("Sales", "count"),
            Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .round(2)
        .reset_index()
    )

    cols = st.columns(len(prod_agg))
    for i, row in prod_agg.iterrows():
        with cols[i % len(cols)]:
            color = PRODUCT_COLORS.get(row["Product"], COLORS[i % len(COLORS)])
            st.markdown(
                f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{row['Product']}</h3>
                <h1>${row['Total_Sales']:,.0f}</h1>
                <p style="color:#a5b4fc; font-size:0.8rem; margin:0;">
                    {row['Transactions']:.0f} txns · Avg ${row['Avg_Sale']:,.0f} · ⭐ {row['Satisfaction']:.2f}
                </p>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            prod_agg, x="Product", y="Total_Sales",
            color="Product", color_discrete_map=PRODUCT_COLORS,
            text_auto=",.0f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400,
            title="Total Sales by Product", showlegend=False,
            yaxis_title="Sales ($)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            prod_agg, names="Product", values="Total_Sales",
            color="Product", color_discrete_map=PRODUCT_COLORS, hole=0.45,
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400,
            title="Revenue Share by Product",
        )
        fig.update_traces(textinfo="percent+label", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product × Region Heatmap")
    cross = df.pivot_table(
        values="Sales", index="Product", columns="Region", aggfunc="sum",
    )
    fig = px.imshow(
        cross, text_auto=",.0f", color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig.update_layout(template=CHART_TEMPLATE, height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monthly Sales by Product")
    prod_monthly = (
        df.groupby(["Month", "Product"])["Sales"].sum().reset_index()
    )
    fig = px.line(
        prod_monthly, x="Month", y="Sales", color="Product",
        color_discrete_map=PRODUCT_COLORS, markers=True,
    )
    fig.update_layout(
        template=CHART_TEMPLATE, height=400,
        xaxis={"tickangle": -45, "dtick": 3}, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product Comparison Radar")
    radar_data = prod_agg.copy()
    for col in ["Total_Sales", "Avg_Sale", "Transactions", "Satisfaction"]:
        col_range = radar_data[col].max() - radar_data[col].min()
        radar_data[col + "_norm"] = (
            (radar_data[col] - radar_data[col].min())
            / (col_range if col_range > 0 else 1)
        )

    fig = go.Figure()
    categories = ["Total Sales", "Avg Sale", "Transactions", "Satisfaction"]
    for _, row in radar_data.iterrows():
        values = [
            row["Total_Sales_norm"], row["Avg_Sale_norm"],
            row["Transactions_norm"], row["Satisfaction_norm"],
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Product"],
            line={"color": PRODUCT_COLORS.get(row["Product"])},
        ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=450,
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
    )
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── REGIONAL ANALYSIS ────────────────────────

elif view == "🗺️ Regional Analysis":
    render_section_header("Regional Performance Analysis")

    reg_agg = (
        df.groupby("Region")
        .agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sale=("Sales", "mean"),
            Transactions=("Sales", "count"),
            Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .round(2)
        .reset_index()
    )

    cols = st.columns(len(reg_agg))
    for i, row in reg_agg.iterrows():
        with cols[i % len(cols)]:
            color = REGION_COLORS.get(row["Region"], COLORS[i % len(COLORS)])
            st.markdown(
                f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{row['Region']}</h3>
                <h1>${row['Total_Sales']:,.0f}</h1>
                <p style="color:#a5b4fc; font-size:0.8rem; margin:0;">
                    {row['Transactions']:.0f} txns · Avg ${row['Avg_Sale']:,.0f} · ⭐ {row['Satisfaction']:.2f}
                </p>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            reg_agg.sort_values("Total_Sales", ascending=True),
            x="Total_Sales", y="Region", orientation="h",
            color="Region", color_discrete_map=REGION_COLORS,
            text_auto=",.0f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=350,
            title="Total Sales by Region", showlegend=False,
            xaxis_title="Sales ($)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.treemap(
            df, path=["Region", "Product"], values="Sales",
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=350,
            title="Sales Treemap: Region → Product",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Regional Satisfaction Comparison")
    fig = px.box(
        df, x="Region", y="Customer_Satisfaction", color="Region",
        color_discrete_map=REGION_COLORS, points="outliers",
    )
    fig.update_layout(
        template=CHART_TEMPLATE, height=400, showlegend=False,
        yaxis_title="Satisfaction Score",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monthly Trend by Region")
    reg_monthly = (
        df.groupby(["Month", "Region"])["Sales"].sum().reset_index()
    )
    fig = px.line(
        reg_monthly, x="Month", y="Sales", color="Region",
        color_discrete_map=REGION_COLORS, markers=True,
    )
    fig.update_layout(
        template=CHART_TEMPLATE, height=400,
        xaxis={"tickangle": -45, "dtick": 3}, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Region × Product Performance")
    reg_prod = (
        df.groupby(["Region", "Product"])
        .agg({"Sales": "sum", "Customer_Satisfaction": "mean"})
        .reset_index()
    )
    fig = px.bar(
        reg_prod, x="Region", y="Sales", color="Product",
        color_discrete_map=PRODUCT_COLORS, barmode="group",
        text_auto=",.0f",
    )
    fig.update_layout(
        template=CHART_TEMPLATE, height=400, yaxis_title="Sales ($)",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── CUSTOMER DEMOGRAPHICS ────────────────────────

elif view == "👥 Customer Demographics":
    render_section_header("Customer Demographics & Behavior")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric(
            "Avg Customer Age", f"{df['Customer_Age'].mean():.1f} yrs",
        )
    with col2:
        render_metric(
            "Female Customers",
            f"{(df['Customer_Gender'] == 'Female').sum():,}",
        )
    with col3:
        render_metric(
            "Male Customers",
            f"{(df['Customer_Gender'] == 'Male').sum():,}",
        )

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="Customer_Age", nbins=25,
            color_discrete_sequence=["#6366f1"], marginal="box",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400, title="Age Distribution",
            xaxis_title="Age", yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_agg = (
            df.groupby("Customer_Gender")
            .agg(
                Total_Sales=("Sales", "sum"),
                Avg_Sale=("Sales", "mean"),
                Count=("Sales", "count"),
                Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .reset_index()
        )
        fig = px.bar(
            gender_agg, x="Customer_Gender",
            y=["Total_Sales", "Avg_Sale"], barmode="group",
            color_discrete_sequence=["#6366f1", "#22d3ee"],
            text_auto=",.0f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400, title="Sales by Gender",
            xaxis_title="", yaxis_title="Amount ($)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sales by Age Group")
    age_agg = (
        df.groupby("Age_Group", observed=True)
        .agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sale=("Sales", "mean"),
            Count=("Sales", "count"),
            Satisfaction=("Customer_Satisfaction", "mean"),
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            age_agg, x="Age_Group", y="Total_Sales", color="Age_Group",
            color_discrete_sequence=COLORS, text_auto=",.0f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400,
            title="Revenue by Age Group", showlegend=False,
            xaxis_title="Age Group", yaxis_title="Total Sales ($)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            age_agg, x="Age_Group", y="Satisfaction", color="Age_Group",
            color_discrete_sequence=COLORS, text_auto=".2f",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=400,
            title="Satisfaction by Age Group", showlegend=False,
            xaxis_title="Age Group", yaxis_title="Avg Satisfaction",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Customer Scatter: Age vs Sales vs Satisfaction")
    fig = px.scatter(
        df, x="Customer_Age", y="Sales",
        color="Customer_Satisfaction",
        color_continuous_scale="Viridis", size="Sales", size_max=10,
        opacity=0.6,
        hover_data=["Product", "Region", "Customer_Gender"],
    )
    fig.update_layout(
        template=CHART_TEMPLATE, height=450,
        xaxis_title="Customer Age", yaxis_title="Sales ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Gender × Product Breakdown")
    gp = (
        df.groupby(["Customer_Gender", "Product"])["Sales"]
        .sum()
        .reset_index()
    )
    fig = px.bar(
        gp, x="Product", y="Sales", color="Customer_Gender",
        barmode="group", color_discrete_sequence=["#6366f1", "#ef4444"],
        text_auto=",.0f",
    )
    fig.update_layout(
        template=CHART_TEMPLATE, height=400, yaxis_title="Sales ($)",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── ADVANCED ANALYTICS ────────────────────────

elif view == "🔬 Advanced Analytics":
    render_section_header("Advanced Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Correlations", "Distributions", "Time Decomposition", "Segmentation"],
    )

    with tab1:
        st.markdown("#### Correlation Matrix")
        corr_cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]
        corr_matrix = df[corr_cols].corr().round(3)
        fig = px.imshow(
            corr_matrix, text_auto=".3f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto",
        )
        fig.update_layout(template=CHART_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pairwise Scatter Matrix")
        fig = px.scatter_matrix(
            df, dimensions=corr_cols, color="Product",
            color_discrete_map=PRODUCT_COLORS, opacity=0.4, height=600,
        )
        fig.update_layout(template=CHART_TEMPLATE)
        fig.update_traces(diagonal_visible=False, marker={"size": 3})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Sales Distribution by Product")
        fig = px.violin(
            df, x="Product", y="Sales", color="Product",
            color_discrete_map=PRODUCT_COLORS, box=True,
            points="outliers",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=450, showlegend=False,
            yaxis_title="Sales ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Satisfaction Distribution by Region")
        fig = px.violin(
            df, x="Region", y="Customer_Satisfaction", color="Region",
            color_discrete_map=REGION_COLORS, box=True, points="outliers",
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=450, showlegend=False,
            yaxis_title="Satisfaction Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Monthly Sales Decomposition")
        monthly_ts = (
            df.groupby("Month")["Sales"]
            .agg(["sum", "mean", "count", "std"])
            .reset_index()
        )
        monthly_ts.columns = ["Month", "Total", "Mean", "Count", "StdDev"]
        monthly_ts["StdDev"] = monthly_ts["StdDev"].fillna(0)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=(
                "Total Sales", "Transaction Count",
                "Sales Volatility (Std Dev)",
            ),
            vertical_spacing=0.08,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_ts["Month"], y=monthly_ts["Total"],
                mode="lines+markers", line={"color": "#6366f1"},
                name="Total Sales",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=monthly_ts["Month"], y=monthly_ts["Count"],
                marker_color="#22d3ee", name="Transactions",
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=monthly_ts["Month"], y=monthly_ts["StdDev"],
                mode="lines", fill="tozeroy",
                line={"color": "#f59e0b"}, name="Std Dev",
            ),
            row=3, col=1,
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=700, showlegend=False,
            xaxis3={"tickangle": -45, "dtick": 6},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown(
            "#### Customer Segmentation: Age Group × Satisfaction Level"
        )
        seg = (
            df.groupby(["Age_Group", "Satisfaction_Level"], observed=True)
            .agg(Count=("Sales", "count"), Avg_Sales=("Sales", "mean"))
            .reset_index()
        )
        fig = px.scatter(
            seg, x="Age_Group", y="Satisfaction_Level",
            size="Count", color="Avg_Sales",
            color_continuous_scale="Viridis", size_max=40,
            hover_data=["Count", "Avg_Sales"],
        )
        fig.update_layout(
            template=CHART_TEMPLATE, height=450,
            xaxis_title="Age Group", yaxis_title="Satisfaction Level",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top / Bottom Performing Segments")
        seg_detail = (
            df.groupby(["Product", "Region", "Customer_Gender"])
            .agg(
                Revenue=("Sales", "sum"),
                Transactions=("Sales", "count"),
                Avg_Sale=("Sales", "mean"),
                Satisfaction=("Customer_Satisfaction", "mean"),
            )
            .round(2)
            .reset_index()
            .sort_values("Revenue", ascending=False)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏆 Top 10 Segments by Revenue**")
            st.dataframe(
                seg_detail.head(10),
                use_container_width=True, hide_index=True,
            )
        with col2:
            st.markdown("**⚠️ Bottom 10 Segments by Revenue**")
            st.dataframe(
                seg_detail.tail(10).sort_values("Revenue"),
                use_container_width=True, hide_index=True,
            )

# ──────────────────────── AI ASSISTANT ────────────────────────

elif view == "🤖 AI Assistant":
    render_section_header("Chat with InsightForge")

    agent_graph = None
    fallback_rag = None

    try:
        with st.spinner("Loading agentic RAG..."):
            agent_graph = initialize_agent_system(df_raw)
    except Exception:
        logger.exception("Agentic RAG initialization failed; trying fallback chain")

    try:
        with st.spinner("Loading fallback RAG..."):
            fallback_rag = initialize_rag_system(df_raw)
    except Exception:
        logger.exception("Fallback RAG initialisation failed")
        if agent_graph is None:
            st.error(
                "Could not initialise the AI assistant. "
                "Make sure Ollama is running with the required models "
                f"(`{LLM_MODEL}`, `{HEAVY_MODEL}`, and `{EMBEDDING_MODEL}`)."
            )
            st.stop()

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_consecutive_failures = 0
            st.rerun()
        st.markdown("**💡 Try asking:**")
        sample_qs = [
            "Which product has the highest sales?",
            "Compare regional performance",
            "What's the average customer satisfaction?",
            "How does Widget A perform in the South?",
            "What month had the best sales?",
        ]
        for q in sample_qs:
            st.caption(f"• {q}")

    with col1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your sales data..."):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                trace_data = None
                try:
                    agent_result = None
                    consecutive_failures = st.session_state.get(
                        "agent_consecutive_failures", 0
                    )
                    use_agent = (
                        agent_graph is not None
                        and consecutive_failures < CIRCUIT_BREAKER_THRESHOLD
                    )
                    if use_agent:
                        with st.status("Running agent workflow...", expanded=False) as status:
                            status.write("Analyzing query...")
                            status.write("Planning retrieval...")
                            status.write("Searching data sources...")
                            status.write("Generating and quality-checking response...")
                            start = time.perf_counter()
                            collector = make_trace_collector(agent_settings)
                            invoke_config = {}
                            if collector is not None:
                                invoke_config = {"callbacks": [collector]}
                            try:
                                agent_result = agent_graph.invoke(
                                    {
                                        "query": prompt,
                                        "messages": st.session_state.messages,
                                        "retry_count": 0,
                                        "trace_steps": [],
                                    },
                                    config=invoke_config,
                                )
                                st.session_state.agent_consecutive_failures = 0
                                logger.info(
                                    json.dumps({
                                        "event": "agent_success",
                                        "latency_ms": (time.perf_counter() - start) * 1000,
                                        "query_len": len(prompt),
                                        "trace": agent_result.get("trace_steps", []),
                                    })
                                )
                            except Exception as agent_err:
                                st.session_state.agent_consecutive_failures = (
                                    consecutive_failures + 1
                                )
                                logger.warning(
                                    json.dumps({
                                        "event": "agent_failure",
                                        "consecutive_failures": st.session_state.agent_consecutive_failures,
                                        "error": str(agent_err),
                                    })
                                )
                                if collector is not None:
                                    try:
                                        collector.finalize(
                                            query=prompt,
                                            response=str(agent_err),
                                            status="error",
                                            session_id=st.session_state.get("_session_id", ""),
                                        )
                                    except Exception:
                                        pass
                                raise
                            status.update(label="Agent workflow complete", state="complete")

                    if agent_result is not None:
                        response = agent_result.get("final_response") or agent_result.get(
                            "generated_response", "I could not generate a response."
                        )
                        score = agent_result.get("evaluation", {}).get("score")

                        if collector is not None:
                            try:
                                trace_data = collector.finalize(
                                    query=prompt,
                                    response=response,
                                    status="success",
                                    quality_score=float(score) if score is not None else None,
                                    session_id=st.session_state.get("_session_id", ""),
                                )
                            except Exception:
                                logger.debug("Trace persistence failed", exc_info=True)

                        if score is not None:
                            st.caption(f"Quality score: **{score}/10**")

                        st.markdown(response)

                        with st.expander("Pipeline Trace", expanded=False):
                            if trace_data and trace_data.get("steps"):
                                render_trace_steps(trace_data)
                            else:
                                st.write(
                                    "Nodes:",
                                    " -> ".join(agent_result.get("trace_steps", [])),
                                )
                            st.write("Safety:", agent_result.get("safety_check", {}))
                            st.write(
                                "Retrieval strategy:",
                                agent_result.get("retrieval_strategy", {}),
                            )
                            st.write("Evaluation:", agent_result.get("evaluation", {}))

                    elif fallback_rag is not None:
                        llm, vectorstore = fallback_rag
                        response = run_basic_rag(prompt, llm, vectorstore)
                        st.markdown(response)
                    else:
                        response = "AI assistant is temporarily unavailable."
                        st.markdown(response)

                except Exception:
                    logger.exception("Agent pipeline failed; trying fallback chain")
                    if fallback_rag is not None:
                        try:
                            llm, vectorstore = fallback_rag
                            response = run_basic_rag(prompt, llm, vectorstore)
                            st.info("Used fallback RAG path due to agent runtime error.")
                        except Exception:
                            logger.exception("Fallback chain also failed")
                            response = (
                                "Sorry, I encountered an error while processing your "
                                "question. Please check that Ollama is running and "
                                "try again."
                            )
                    else:
                        response = (
                            "Sorry, I encountered an error while processing your "
                            "question. Please check that Ollama is running and "
                            "try again."
                        )
                    st.error(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response},
            )

# ──────────────────────── OBSERVABILITY ────────────────────────

elif view == "📡 Observability":
    render_section_header("Observability Dashboard")

    db_url = agent_settings.database_url
    _obs_available = True
    try:
        from insightforge.observability.repository import (
            get_latency_metrics,
            get_quality_metrics,
            get_steps_dataframe,
            get_traces_dataframe,
            get_usage_metrics,
        )
    except ImportError:
        _obs_available = False

    if not _obs_available or not db_url:
        st.warning(
            "Observability requires a DATABASE_URL and the observability "
            "dependencies (sqlalchemy, psycopg2-binary). Run `alembic upgrade head` "
            "after starting Postgres to create the trace tables."
        )
    else:
        days = st.selectbox("Time window", [7, 14, 30, 90], index=2)

        traces_df = get_traces_dataframe(db_url, days=days)
        steps_df = get_steps_dataframe(db_url, days=days)

        if traces_df.empty:
            st.info(
                "No traces recorded yet. Ask a question in the AI Assistant to "
                "start collecting data."
            )
        else:
            # ── Latency KPIs ──
            latency = get_latency_metrics(db_url, days=days)
            quality = get_quality_metrics(db_url, days=days)
            usage = get_usage_metrics(db_url, days=days)

            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            with kc1:
                render_metric("Total Queries", f"{latency['count']:,}")
            with kc2:
                render_metric("Avg Latency", f"{latency['avg']:,.0f} ms")
            with kc3:
                render_metric("P95 Latency", f"{latency['p95']:,} ms")
            with kc4:
                q_score = quality.get("avg_quality")
                render_metric(
                    "Avg Quality",
                    f"{q_score:.1f}/10" if q_score else "N/A",
                )
            with kc5:
                render_metric("Error Rate", f"{quality['error_rate']:.1f}%")

            st.markdown("")

            tab_lat, tab_qual, tab_usage, tab_traces = st.tabs(
                ["Latency", "Quality", "Usage", "Trace Log"],
            )

            # ── Latency tab ──
            with tab_lat:
                st.markdown("#### Latency by Pipeline Stage")
                breakdown = latency.get("step_breakdown", [])
                if breakdown:
                    bd_df = pd.DataFrame(breakdown)
                    fig = px.bar(
                        bd_df, x="step", y="avg_ms",
                        color_discrete_sequence=COLORS,
                        text_auto=",.0f",
                    )
                    fig.update_layout(
                        template=CHART_TEMPLATE, height=350,
                        xaxis_title="Pipeline Stage",
                        yaxis_title="Avg Latency (ms)",
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Latency Over Time")
                if not traces_df.empty and "created_at" in traces_df.columns:
                    fig = px.scatter(
                        traces_df, x="created_at", y="total_latency_ms",
                        color="status",
                        color_discrete_sequence=COLORS,
                        hover_data=["query"],
                    )
                    fig.update_layout(
                        template=CHART_TEMPLATE, height=350,
                        xaxis_title="Time", yaxis_title="Latency (ms)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if not steps_df.empty:
                    st.markdown("#### Per-Step Latency Distribution")
                    fig = px.box(
                        steps_df, x="step_name", y="latency_ms",
                        color="step_name", color_discrete_sequence=COLORS,
                    )
                    fig.update_layout(
                        template=CHART_TEMPLATE, height=400,
                        showlegend=False,
                        xaxis_title="Step", yaxis_title="Latency (ms)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Quality tab ──
            with tab_qual:
                qc1, qc2, qc3 = st.columns(3)
                with qc1:
                    render_metric(
                        "Avg Quality Score",
                        f"{quality['avg_quality']:.1f}/10" if quality["avg_quality"] else "N/A",
                    )
                with qc2:
                    render_metric("Fallback Rate", f"{quality['fallback_rate']:.1f}%")
                with qc3:
                    render_metric("Error Rate", f"{quality['error_rate']:.1f}%")

                st.markdown("#### Quality Score Over Time")
                scored = traces_df.dropna(subset=["quality_score"])
                if not scored.empty:
                    fig = px.scatter(
                        scored, x="created_at", y="quality_score",
                        color_discrete_sequence=["#10b981"],
                        hover_data=["query"],
                    )
                    fig.add_hline(
                        y=scored["quality_score"].mean(),
                        line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"Avg: {scored['quality_score'].mean():.1f}",
                    )
                    fig.update_layout(
                        template=CHART_TEMPLATE, height=350,
                        xaxis_title="Time", yaxis_title="Quality Score",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No quality scores recorded yet.")

                st.markdown("#### Status Distribution")
                status_counts = traces_df["status"].value_counts().reset_index()
                status_counts.columns = ["status", "count"]
                fig = px.pie(
                    status_counts, names="status", values="count",
                    color_discrete_sequence=COLORS, hole=0.4,
                )
                fig.update_layout(template=CHART_TEMPLATE, height=350)
                st.plotly_chart(fig, use_container_width=True)

            # ── Usage tab ──
            with tab_usage:
                uc1, uc2, uc3 = st.columns(3)
                with uc1:
                    render_metric(
                        "Total Tokens In",
                        f"{usage['total_tokens_input']:,}",
                    )
                with uc2:
                    render_metric(
                        "Total Tokens Out",
                        f"{usage['total_tokens_output']:,}",
                    )
                with uc3:
                    total_tok = usage["total_tokens_input"] + usage["total_tokens_output"]
                    render_metric("Total Tokens", f"{total_tok:,}")

                st.markdown("#### Token Usage Over Time")
                if not traces_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=traces_df["created_at"],
                        y=traces_df["total_tokens_input"],
                        mode="lines+markers", name="Input",
                        line={"color": "#6366f1"},
                    ))
                    fig.add_trace(go.Scatter(
                        x=traces_df["created_at"],
                        y=traces_df["total_tokens_output"],
                        mode="lines+markers", name="Output",
                        line={"color": "#22d3ee"},
                    ))
                    fig.update_layout(
                        template=CHART_TEMPLATE, height=350,
                        xaxis_title="Time", yaxis_title="Tokens",
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                tool_usage = usage.get("tool_usage", {})
                if tool_usage:
                    st.markdown("#### Pipeline Stage Frequency")
                    tu_df = pd.DataFrame(
                        list(tool_usage.items()),
                        columns=["stage", "count"],
                    )
                    fig = px.pie(
                        tu_df, names="stage", values="count",
                        color_discrete_sequence=COLORS, hole=0.4,
                    )
                    fig.update_layout(template=CHART_TEMPLATE, height=350)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Trace Log tab ──
            with tab_traces:
                st.markdown("#### Recent Traces")
                display_df = traces_df[
                    ["created_at", "query", "status", "total_latency_ms",
                     "quality_score", "total_tokens_input", "total_tokens_output"]
                ].copy()
                display_df.columns = [
                    "Time", "Query", "Status", "Latency (ms)",
                    "Quality", "Tokens In", "Tokens Out",
                ]
                st.dataframe(
                    display_df.sort_values("Time", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                )

    if agent_settings.langfuse_host:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"[Open Langfuse Dashboard]({agent_settings.langfuse_host})"
        )

# ──────────────────────── FOOTER ────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption(
    f"InsightForge v{APP_VERSION} · Powered by Ollama + LangChain"
)
