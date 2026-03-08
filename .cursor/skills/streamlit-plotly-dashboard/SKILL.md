---
name: streamlit-plotly-dashboard
description: Build and enhance Streamlit dashboards with Plotly charts, custom CSS, sidebar navigation, and dark themes. Use when creating Streamlit pages, adding Plotly visualizations, styling with custom CSS, or building interactive dashboard layouts.
---

# Streamlit + Plotly Dashboard Patterns

## App Configuration

```python
st.set_page_config(page_title="Title", page_icon="icon", layout="wide")
```

Always set `layout="wide"` for dashboards.

## Custom CSS Injection

Inject global styles for branded look:

```python
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b, #312e81); }
    .metric-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        padding: 1.2rem; border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.3);
        text-align: center;
    }
    .metric-card h3 { color: #a5b4fc; font-size: 0.85rem; }
    .metric-card h1 { color: #e0e7ff; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)
```

## Metric Cards

Use HTML-based cards instead of `st.metric` for more control:

```python
def render_metric(label, value):
    st.markdown(
        f'<div class="metric-card"><h3>{label}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )
```

## Sidebar Navigation

```python
views = ["Overview", "Analysis", "AI Chat"]
view = st.sidebar.radio("Navigation", views, label_visibility="collapsed")
```

Use `if/elif` blocks for each view. Filters go in `st.sidebar.expander`.

## Plotly Chart Conventions

All charts should use consistent settings:

```python
fig.update_layout(
    template="plotly_dark",
    height=400,
    hovermode="x unified",
    xaxis=dict(tickangle=-45, dtick=3),
)
st.plotly_chart(fig, use_container_width=True)
```

### Common Chart Types

| Chart | Use Case | Plotly Function |
|-------|----------|-----------------|
| Line + markers | Time series trends | `go.Scatter(mode="lines+markers")` |
| Bar | Category comparison | `px.bar(text_auto=",.0f")` |
| Pie / Donut | Share breakdown | `px.pie(hole=0.45)` |
| Heatmap | Cross-dimensional | `px.imshow(text_auto=",.0f")` |
| Treemap | Hierarchical | `px.treemap(path=[...])` |
| Violin | Distribution | `px.violin(box=True, points="outliers")` |
| Scatter | Relationship | `px.scatter(size=..., color=...)` |
| Radar | Multi-metric compare | `go.Scatterpolar(fill="toself")` |
| Subplots | Multiple related | `make_subplots(rows=n, cols=1)` |

### Color Palettes

Define named color maps for consistency:

```python
PRODUCT_COLORS = {"Widget A": "#6366f1", "Widget B": "#22d3ee", ...}
fig = px.bar(..., color="Product", color_discrete_map=PRODUCT_COLORS)
```

## Caching

```python
@st.cache_data       # For data loading / transformations (serializable)
@st.cache_resource   # For ML models, DB connections (non-serializable)
```

Prefix non-hashable args with `_` (e.g., `def init(_df)`).

## Filter Pattern

```python
def apply_filters(df):
    filtered = df.copy()
    with st.sidebar.expander("Filters", expanded=True):
        products = st.multiselect("Products", df["Product"].unique(), default=list(df["Product"].unique()))
        if products:
            filtered = filtered[filtered["Product"].isin(products)]
    return filtered
```

## Layout Patterns

```python
col1, col2 = st.columns(2)         # Side-by-side charts
tab1, tab2 = st.tabs(["A", "B"])   # Tabbed views
with st.expander("Details"):       # Collapsible sections
```

## Chat Interface

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```
