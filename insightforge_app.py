import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

COLORS = ["#6366f1", "#22d3ee", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]
PRODUCT_COLORS = {"Widget A": "#6366f1", "Widget B": "#22d3ee", "Widget C": "#f59e0b", "Widget D": "#ef4444"}
REGION_COLORS = {"North": "#6366f1", "South": "#22d3ee", "East": "#f59e0b", "West": "#ef4444"}

st.set_page_config(page_title="InsightForge BI Assistant", page_icon="📊", layout="wide")

st.markdown("""
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
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
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


@st.cache_resource
def initialize_rag_system(_df):
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    documents = []
    total_sales = _df["Sales"].sum()
    documents.append(Document(
        page_content=(
            f"Dataset Overview: {len(_df)} transactions from {_df['Date'].min().date()} to {_df['Date'].max().date()}. "
            f"Total Sales: ${total_sales:,.2f}. Average Transaction: ${_df['Sales'].mean():.2f}. "
            f"Median Transaction: ${_df['Sales'].median():.2f}. "
            f"Products: {', '.join(_df['Product'].unique())}. Regions: {', '.join(_df['Region'].unique())}. "
            f"Customer Age Range: {_df['Customer_Age'].min()}-{_df['Customer_Age'].max()} (avg {_df['Customer_Age'].mean():.1f}). "
            f"Gender Split: {dict(_df['Customer_Gender'].value_counts())}. "
            f"Overall Satisfaction: {_df['Customer_Satisfaction'].mean():.2f}/5."
        ),
        metadata={"type": "overview"},
    ))

    for product in _df["Product"].unique():
        pdf = _df[_df["Product"] == product]
        share = pdf["Sales"].sum() / total_sales * 100
        top_region = pdf.groupby("Region")["Sales"].sum().idxmax()
        documents.append(Document(
            page_content=(
                f"Product {product}: Total ${pdf['Sales'].sum():,.2f} ({share:.1f}% share), "
                f"Avg ${pdf['Sales'].mean():.2f}, Median ${pdf['Sales'].median():.2f}, "
                f"{len(pdf)} transactions. Satisfaction {pdf['Customer_Satisfaction'].mean():.2f}/5. "
                f"Top region: {top_region}. "
                f"Avg customer age: {pdf['Customer_Age'].mean():.1f}."
            ),
            metadata={"type": "product"},
        ))

    for region in _df["Region"].unique():
        rdf = _df[_df["Region"] == region]
        share = rdf["Sales"].sum() / total_sales * 100
        top_product = rdf.groupby("Product")["Sales"].sum().idxmax()
        documents.append(Document(
            page_content=(
                f"Region {region}: Total ${rdf['Sales'].sum():,.2f} ({share:.1f}% share), "
                f"Avg ${rdf['Sales'].mean():.2f}, {len(rdf)} transactions. "
                f"Satisfaction {rdf['Customer_Satisfaction'].mean():.2f}/5. "
                f"Top product: {top_product}. Gender split: {dict(rdf['Customer_Gender'].value_counts())}."
            ),
            metadata={"type": "region"},
        ))

    for product in _df["Product"].unique():
        for region in _df["Region"].unique():
            subset = _df[(_df["Product"] == product) & (_df["Region"] == region)]
            if len(subset) > 0:
                documents.append(Document(
                    page_content=(
                        f"{product} in {region}: ${subset['Sales'].sum():,.2f} total, "
                        f"Avg ${subset['Sales'].mean():.2f}, {len(subset)} transactions, "
                        f"Satisfaction {subset['Customer_Satisfaction'].mean():.2f}/5."
                    ),
                    metadata={"type": "cross"},
                ))

    for gender in _df["Customer_Gender"].unique():
        gdf = _df[_df["Customer_Gender"] == gender]
        documents.append(Document(
            page_content=(
                f"{gender} customers: {len(gdf)} transactions, Total ${gdf['Sales'].sum():,.2f}, "
                f"Avg ${gdf['Sales'].mean():.2f}, Satisfaction {gdf['Customer_Satisfaction'].mean():.2f}/5, "
                f"Avg age {gdf['Customer_Age'].mean():.1f}."
            ),
            metadata={"type": "gender"},
        ))

    monthly = _df.groupby("Month").agg({"Sales": ["sum", "mean", "count"]}).reset_index()
    monthly.columns = ["Month", "Total", "Avg", "Count"]
    best_month = monthly.loc[monthly["Total"].idxmax()]
    worst_month = monthly.loc[monthly["Total"].idxmin()]
    documents.append(Document(
        page_content=(
            f"Monthly Trends: Best month {best_month['Month']} (${best_month['Total']:,.0f}), "
            f"Worst month {worst_month['Month']} (${worst_month['Total']:,.0f}). "
            f"Average monthly sales: ${monthly['Total'].mean():,.0f}."
        ),
        metadata={"type": "trend"},
    ))

    vectorstore = FAISS.from_documents(documents, embeddings)
    return llm, vectorstore


def render_metric(label, value):
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )


def apply_filters(df):
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

        products = st.multiselect("Products", df["Product"].unique(), default=list(df["Product"].unique()))
        if products:
            filtered = filtered[filtered["Product"].isin(products)]

        regions = st.multiselect("Regions", df["Region"].unique(), default=list(df["Region"].unique()))
        if regions:
            filtered = filtered[filtered["Region"].isin(regions)]

        genders = st.multiselect("Gender", df["Customer_Gender"].unique(), default=list(df["Customer_Gender"].unique()))
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

    st.sidebar.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
    return filtered


# ──────────────────────── MAIN ────────────────────────

df_raw = load_data()

st.sidebar.image("https://img.icons8.com/fluency/48/combo-chart.png", width=40)
st.sidebar.title("InsightForge")
st.sidebar.markdown("---")

views = [
    "📊 Sales Overview",
    "📦 Product Analysis",
    "🗺️ Regional Analysis",
    "👥 Customer Demographics",
    "🔬 Advanced Analytics",
    "🤖 AI Assistant",
]
view = st.sidebar.radio("Navigation", views, label_visibility="collapsed")

df = apply_filters(df_raw)

# ──────────────────────── SALES OVERVIEW ────────────────────────

if view == "📊 Sales Overview":
    st.markdown(
        '<div class="section-header"><h2>Sales Overview Dashboard</h2></div>',
        unsafe_allow_html=True,
    )

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
        render_metric("Avg Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f}/5")

    st.markdown("")

    monthly = df.groupby("Month").agg({"Sales": ["sum", "mean", "count"]}).reset_index()
    monthly.columns = ["Month", "Total_Sales", "Avg_Sale", "Transactions"]
    monthly["Cumulative"] = monthly["Total_Sales"].cumsum()
    monthly["MA_3"] = monthly["Total_Sales"].rolling(3, min_periods=1).mean()

    tab1, tab2 = st.tabs(["📈 Monthly Trend", "📊 Cumulative Sales"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Total_Sales"],
            mode="lines+markers", name="Monthly Sales",
            line=dict(color="#6366f1", width=2), marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["MA_3"],
            mode="lines", name="3-Month Moving Avg",
            line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Month", yaxis_title="Sales ($)",
            hovermode="x unified", xaxis=dict(tickangle=-45, dtick=3),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.area(monthly, x="Month", y="Cumulative", color_discrete_sequence=["#6366f1"])
        fig.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Month", yaxis_title="Cumulative Sales ($)",
            xaxis=dict(tickangle=-45, dtick=3),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        quarterly = df.groupby("Quarter")["Sales"].sum().reset_index()
        fig = px.bar(quarterly, x="Quarter", y="Sales", color_discrete_sequence=["#6366f1"], text_auto=",.0f")
        fig.update_layout(template="plotly_dark", height=350, title="Quarterly Revenue", xaxis=dict(tickangle=-45))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df.groupby("DayOfWeek")["Sales"].mean().reindex(dow_order).reset_index()
        fig = px.bar(dow, x="DayOfWeek", y="Sales", color_discrete_sequence=["#22d3ee"], text_auto=",.0f")
        fig.update_layout(
            template="plotly_dark", height=350,
            title="Avg Sales by Day of Week", xaxis_title="", yaxis_title="Avg Sales ($)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── PRODUCT ANALYSIS ────────────────────────

elif view == "📦 Product Analysis":
    st.markdown(
        '<div class="section-header"><h2>Product Performance Analysis</h2></div>',
        unsafe_allow_html=True,
    )

    prod_agg = df.groupby("Product").agg(
        Total_Sales=("Sales", "sum"),
        Avg_Sale=("Sales", "mean"),
        Transactions=("Sales", "count"),
        Satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2).reset_index()

    cols = st.columns(len(prod_agg))
    for i, row in prod_agg.iterrows():
        with cols[i % len(cols)]:
            color = PRODUCT_COLORS.get(row["Product"], COLORS[i % len(COLORS)])
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{row['Product']}</h3>
                <h1>${row['Total_Sales']:,.0f}</h1>
                <p style="color:#a5b4fc; font-size:0.8rem; margin:0;">
                    {row['Transactions']:.0f} txns · Avg ${row['Avg_Sale']:,.0f} · ⭐ {row['Satisfaction']:.2f}
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            prod_agg, x="Product", y="Total_Sales",
            color="Product", color_discrete_map=PRODUCT_COLORS, text_auto=",.0f",
        )
        fig.update_layout(template="plotly_dark", height=400, title="Total Sales by Product", showlegend=False, yaxis_title="Sales ($)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            prod_agg, names="Product", values="Total_Sales",
            color="Product", color_discrete_map=PRODUCT_COLORS, hole=0.45,
        )
        fig.update_layout(template="plotly_dark", height=400, title="Revenue Share by Product")
        fig.update_traces(textinfo="percent+label", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product × Region Heatmap")
    cross = df.pivot_table(values="Sales", index="Product", columns="Region", aggfunc="sum")
    fig = px.imshow(cross, text_auto=",.0f", color_continuous_scale="Viridis", aspect="auto")
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monthly Sales by Product")
    prod_monthly = df.groupby(["Month", "Product"])["Sales"].sum().reset_index()
    fig = px.line(prod_monthly, x="Month", y="Sales", color="Product", color_discrete_map=PRODUCT_COLORS, markers=True)
    fig.update_layout(template="plotly_dark", height=400, xaxis=dict(tickangle=-45, dtick=3), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Product Comparison Radar")
    radar_data = prod_agg.copy()
    for col in ["Total_Sales", "Avg_Sale", "Transactions", "Satisfaction"]:
        col_range = radar_data[col].max() - radar_data[col].min()
        radar_data[col + "_norm"] = (radar_data[col] - radar_data[col].min()) / (col_range if col_range > 0 else 1)

    fig = go.Figure()
    categories = ["Total Sales", "Avg Sale", "Transactions", "Satisfaction"]
    for _, row in radar_data.iterrows():
        values = [row["Total_Sales_norm"], row["Avg_Sale_norm"], row["Transactions_norm"], row["Satisfaction_norm"]]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["Product"],
            line=dict(color=PRODUCT_COLORS.get(row["Product"])),
        ))
    fig.update_layout(template="plotly_dark", height=450, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── REGIONAL ANALYSIS ────────────────────────

elif view == "🗺️ Regional Analysis":
    st.markdown(
        '<div class="section-header"><h2>Regional Performance Analysis</h2></div>',
        unsafe_allow_html=True,
    )

    reg_agg = df.groupby("Region").agg(
        Total_Sales=("Sales", "sum"),
        Avg_Sale=("Sales", "mean"),
        Transactions=("Sales", "count"),
        Satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2).reset_index()

    cols = st.columns(len(reg_agg))
    for i, row in reg_agg.iterrows():
        with cols[i % len(cols)]:
            color = REGION_COLORS.get(row["Region"], COLORS[i % len(COLORS)])
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {color};">
                <h3>{row['Region']}</h3>
                <h1>${row['Total_Sales']:,.0f}</h1>
                <p style="color:#a5b4fc; font-size:0.8rem; margin:0;">
                    {row['Transactions']:.0f} txns · Avg ${row['Avg_Sale']:,.0f} · ⭐ {row['Satisfaction']:.2f}
                </p>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            reg_agg.sort_values("Total_Sales", ascending=True),
            x="Total_Sales", y="Region", orientation="h",
            color="Region", color_discrete_map=REGION_COLORS, text_auto=",.0f",
        )
        fig.update_layout(template="plotly_dark", height=350, title="Total Sales by Region", showlegend=False, xaxis_title="Sales ($)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.treemap(df, path=["Region", "Product"], values="Sales", color_discrete_sequence=COLORS)
        fig.update_layout(template="plotly_dark", height=350, title="Sales Treemap: Region → Product")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Regional Satisfaction Comparison")
    fig = px.box(df, x="Region", y="Customer_Satisfaction", color="Region", color_discrete_map=REGION_COLORS, points="outliers")
    fig.update_layout(template="plotly_dark", height=400, showlegend=False, yaxis_title="Satisfaction Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Monthly Trend by Region")
    reg_monthly = df.groupby(["Month", "Region"])["Sales"].sum().reset_index()
    fig = px.line(reg_monthly, x="Month", y="Sales", color="Region", color_discrete_map=REGION_COLORS, markers=True)
    fig.update_layout(template="plotly_dark", height=400, xaxis=dict(tickangle=-45, dtick=3), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Region × Product Performance")
    reg_prod = df.groupby(["Region", "Product"]).agg({"Sales": "sum", "Customer_Satisfaction": "mean"}).reset_index()
    fig = px.bar(reg_prod, x="Region", y="Sales", color="Product", color_discrete_map=PRODUCT_COLORS, barmode="group", text_auto=",.0f")
    fig.update_layout(template="plotly_dark", height=400, yaxis_title="Sales ($)")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── CUSTOMER DEMOGRAPHICS ────────────────────────

elif view == "👥 Customer Demographics":
    st.markdown(
        '<div class="section-header"><h2>Customer Demographics & Behavior</h2></div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric("Avg Customer Age", f"{df['Customer_Age'].mean():.1f} yrs")
    with col2:
        render_metric("Female Customers", f"{(df['Customer_Gender'] == 'Female').sum():,}")
    with col3:
        render_metric("Male Customers", f"{(df['Customer_Gender'] == 'Male').sum():,}")

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="Customer_Age", nbins=25, color_discrete_sequence=["#6366f1"], marginal="box")
        fig.update_layout(template="plotly_dark", height=400, title="Age Distribution", xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_agg = df.groupby("Customer_Gender").agg(
            Total_Sales=("Sales", "sum"),
            Avg_Sale=("Sales", "mean"),
            Count=("Sales", "count"),
            Satisfaction=("Customer_Satisfaction", "mean"),
        ).reset_index()
        fig = px.bar(
            gender_agg, x="Customer_Gender", y=["Total_Sales", "Avg_Sale"],
            barmode="group", color_discrete_sequence=["#6366f1", "#22d3ee"], text_auto=",.0f",
        )
        fig.update_layout(template="plotly_dark", height=400, title="Sales by Gender", xaxis_title="", yaxis_title="Amount ($)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sales by Age Group")
    age_agg = df.groupby("Age_Group", observed=True).agg(
        Total_Sales=("Sales", "sum"),
        Avg_Sale=("Sales", "mean"),
        Count=("Sales", "count"),
        Satisfaction=("Customer_Satisfaction", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(age_agg, x="Age_Group", y="Total_Sales", color="Age_Group", color_discrete_sequence=COLORS, text_auto=",.0f")
        fig.update_layout(template="plotly_dark", height=400, title="Revenue by Age Group", showlegend=False, xaxis_title="Age Group", yaxis_title="Total Sales ($)")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(age_agg, x="Age_Group", y="Satisfaction", color="Age_Group", color_discrete_sequence=COLORS, text_auto=".2f")
        fig.update_layout(template="plotly_dark", height=400, title="Satisfaction by Age Group", showlegend=False, xaxis_title="Age Group", yaxis_title="Avg Satisfaction")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Customer Scatter: Age vs Sales vs Satisfaction")
    fig = px.scatter(
        df, x="Customer_Age", y="Sales", color="Customer_Satisfaction",
        color_continuous_scale="Viridis", size="Sales", size_max=10, opacity=0.6,
        hover_data=["Product", "Region", "Customer_Gender"],
    )
    fig.update_layout(template="plotly_dark", height=450, xaxis_title="Customer Age", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Gender × Product Breakdown")
    gp = df.groupby(["Customer_Gender", "Product"])["Sales"].sum().reset_index()
    fig = px.bar(gp, x="Product", y="Sales", color="Customer_Gender", barmode="group", color_discrete_sequence=["#6366f1", "#ef4444"], text_auto=",.0f")
    fig.update_layout(template="plotly_dark", height=400, yaxis_title="Sales ($)")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────── ADVANCED ANALYTICS ────────────────────────

elif view == "🔬 Advanced Analytics":
    st.markdown(
        '<div class="section-header"><h2>Advanced Analytics</h2></div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Distributions", "Time Decomposition", "Segmentation"])

    with tab1:
        st.markdown("#### Correlation Matrix")
        corr_cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]
        corr_matrix = df[corr_cols].corr().round(3)
        fig = px.imshow(corr_matrix, text_auto=".3f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pairwise Scatter Matrix")
        fig = px.scatter_matrix(df, dimensions=corr_cols, color="Product", color_discrete_map=PRODUCT_COLORS, opacity=0.4, height=600)
        fig.update_layout(template="plotly_dark")
        fig.update_traces(diagonal_visible=False, marker=dict(size=3))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Sales Distribution by Product")
        fig = px.violin(df, x="Product", y="Sales", color="Product", color_discrete_map=PRODUCT_COLORS, box=True, points="outliers")
        fig.update_layout(template="plotly_dark", height=450, showlegend=False, yaxis_title="Sales ($)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Satisfaction Distribution by Region")
        fig = px.violin(df, x="Region", y="Customer_Satisfaction", color="Region", color_discrete_map=REGION_COLORS, box=True, points="outliers")
        fig.update_layout(template="plotly_dark", height=450, showlegend=False, yaxis_title="Satisfaction Score")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Monthly Sales Decomposition")
        monthly_ts = df.groupby("Month")["Sales"].agg(["sum", "mean", "count", "std"]).reset_index()
        monthly_ts.columns = ["Month", "Total", "Mean", "Count", "StdDev"]
        monthly_ts["StdDev"] = monthly_ts["StdDev"].fillna(0)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Total Sales", "Transaction Count", "Sales Volatility (Std Dev)"),
            vertical_spacing=0.08,
        )
        fig.add_trace(go.Scatter(x=monthly_ts["Month"], y=monthly_ts["Total"], mode="lines+markers", line=dict(color="#6366f1"), name="Total Sales"), row=1, col=1)
        fig.add_trace(go.Bar(x=monthly_ts["Month"], y=monthly_ts["Count"], marker_color="#22d3ee", name="Transactions"), row=2, col=1)
        fig.add_trace(go.Scatter(x=monthly_ts["Month"], y=monthly_ts["StdDev"], mode="lines", fill="tozeroy", line=dict(color="#f59e0b"), name="Std Dev"), row=3, col=1)
        fig.update_layout(template="plotly_dark", height=700, showlegend=False, xaxis3=dict(tickangle=-45, dtick=6))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### Customer Segmentation: Age Group × Satisfaction Level")
        seg = df.groupby(["Age_Group", "Satisfaction_Level"], observed=True).agg(
            Count=("Sales", "count"), Avg_Sales=("Sales", "mean"),
        ).reset_index()
        fig = px.scatter(
            seg, x="Age_Group", y="Satisfaction_Level",
            size="Count", color="Avg_Sales", color_continuous_scale="Viridis",
            size_max=40, hover_data=["Count", "Avg_Sales"],
        )
        fig.update_layout(template="plotly_dark", height=450, xaxis_title="Age Group", yaxis_title="Satisfaction Level")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top / Bottom Performing Segments")
        seg_detail = df.groupby(["Product", "Region", "Customer_Gender"]).agg(
            Revenue=("Sales", "sum"),
            Transactions=("Sales", "count"),
            Avg_Sale=("Sales", "mean"),
            Satisfaction=("Customer_Satisfaction", "mean"),
        ).round(2).reset_index().sort_values("Revenue", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏆 Top 10 Segments by Revenue**")
            st.dataframe(seg_detail.head(10), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**⚠️ Bottom 10 Segments by Revenue**")
            st.dataframe(seg_detail.tail(10).sort_values("Revenue"), use_container_width=True, hide_index=True)

# ──────────────────────── AI ASSISTANT ────────────────────────

elif view == "🤖 AI Assistant":
    st.markdown(
        '<div class="section-header"><h2>Chat with InsightForge</h2></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Loading AI models..."):
        llm, vectorstore = initialize_rag_system(df_raw)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
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
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(prompt)
                    context = "\n".join([d.page_content for d in docs])

                    template = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are InsightForge, an expert BI analyst. Answer questions using ONLY the data context provided. "
                         "Be precise with numbers. Use bullet points for comparisons. "
                         "If the data doesn't contain the answer, say so clearly."),
                        ("human", "Data Context:\n{context}\n\nQuestion: {question}"),
                    ])

                    chain = template | llm | StrOutputParser()
                    response = chain.invoke({"context": context, "question": prompt})

                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ──────────────────────── FOOTER ────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption("InsightForge v2.0 · Powered by Ollama + LangChain")
