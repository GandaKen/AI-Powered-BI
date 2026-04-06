---
name: insightforge-bi-project
description: Project-specific knowledge for the InsightForge AI-Powered BI Assistant capstone project. Use when working on any file in this repo, modifying the Streamlit app, editing the Jupyter notebook, or asking questions about the project architecture.
---

# InsightForge BI Project Guide

## Architecture

```
insightforge_app.py          # Streamlit dashboard (main app)
InsightForge_Assistant.ipynb  # Jupyter notebook (development & evaluation)
sales_data.csv               # Source data (2,500 transactions)
sales_data_quiz.md           # Test questions for RAG evaluation
```

## Data Schema (`sales_data.csv`)

| Column | Type | Description |
|--------|------|-------------|
| Date | date | 2022-01-01 to 2028-10-27 |
| Product | str | Widget A, Widget B, Widget C, Widget D |
| Region | str | North, South, East, West |
| Sales | int | Transaction amount (USD) |
| Customer_Age | int | 18–69 |
| Customer_Gender | str | Male, Female |
| Customer_Satisfaction | float | 1.0–5.0 |

Derived columns added at load time: `Month`, `Quarter`, `Year`, `DayOfWeek`, `Age_Group`, `Satisfaction_Level`.

## Streamlit App (`insightforge_app.py`)

### Views (sidebar radio nav)
1. **Sales Overview** — KPI cards, monthly trend with 3-month MA, cumulative sales, quarterly bars, day-of-week bars
2. **Product Analysis** — Per-product metrics, bar/pie charts, product×region heatmap, monthly line, radar chart
3. **Regional Analysis** — Per-region metrics, horizontal bars, treemap, satisfaction box plots, monthly line, grouped bars
4. **Customer Demographics** — Age histogram, gender comparison, age group bars, scatter (age vs sales vs satisfaction), gender×product
5. **Advanced Analytics** — Correlation matrix, scatter matrix, violin plots, time decomposition (subplots), segmentation bubble chart, top/bottom segments
6. **AI Assistant** — RAG-powered chat using Ollama + LangChain
7. **Model Evaluation** — QAEvalChain with data-derived ground-truth Q&A pairs

### Key Conventions
- Color palette: `COLORS`, `PRODUCT_COLORS`, `REGION_COLORS` constants at top
- All charts: `template="plotly_dark"`, consistent height values
- Custom CSS in `st.markdown` for metric cards and sidebar gradient
- Filters: sidebar expander with date range, product, region, gender, age slider
- Caching: `@st.cache_data` for data load, `@st.cache_resource` for RAG system

### Tech Stack
| Component | Streamlit App | Notebook |
|-----------|--------------|----------|
| LLM | Ollama `llama3.2:3b` | OpenAI `gpt-3.5-turbo` |
| Embeddings | Ollama `nomic-embed-text` | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS | FAISS |
| Charts | Plotly | Matplotlib + Seaborn |
| Framework | Streamlit | Jupyter |

## Notebook Structure (`InsightForge_Assistant.ipynb`)

### Part 1: AI-Powered BI Assistant
- Step 1: Data Preparation — load CSV, add derived columns, descriptive stats
- Step 2: Knowledge Base Creation — generate Document objects from aggregated stats
- Step 3: LLM Application — ChatOpenAI, prompt templates, custom data retriever
- Step 4: Chain Prompts — multi-step analysis chains with StructuredOutputParser
- Step 5: RAG System — FAISS vector store, RetrievalQA chain, hybrid retriever
- Step 6: Memory — ConversationBufferMemory, ConversationSummaryMemory

### Part 2: LLMOps
- Step 7a: Model Evaluation — QAEvalChain with ground-truth Q&A pairs
- Step 7b: Data Visualization — matplotlib/seaborn charts
- Step 7c: Streamlit UI — documents the Streamlit app design

## Running the App

```bash
source venv/bin/activate
streamlit run insightforge_app.py
```

Requires Ollama running locally with `llama3.2:3b` and `nomic-embed-text` models pulled.
