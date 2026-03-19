# InsightForge — AI-Powered Business Intelligence Assistant

> An end-to-end BI solution combining interactive Plotly dashboards with a
> RAG-powered AI analyst that answers natural-language questions about your
> sales data — all running locally with Ollama.

---

## Table of Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data Schema](#data-schema)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Notebook](#notebook)
- [Configuration](#configuration)
- [Code Quality](#code-quality)

---

## Overview

InsightForge demonstrates how large language models can be integrated into a traditional BI workflow to:

- Explore 2,500 sales transactions across six interactive dashboard views
- Answer free-form business questions via a RAG pipeline (FAISS + Ollama)
- Evaluate LLM response quality with `QAEvalChain`

---

## Screenshots

| Sales Overview | AI Assistant |
|---|---|
| *Monthly trend, cumulative sales, quarterly & day-of-week breakdowns* | *Conversational RAG chat powered by Ollama `llama3.2:3b`* |

---

## Architecture

The AI Assistant uses a **6-node LangGraph agentic RAG pipeline**:

```
QueryPlanner → RetrievalPlanner → InformationRetriever → ContextAssembler → Generator → ResponseQA
       ↑                                                                                    │
       └────────────────────────── retry (max 1) ──────────────────────────────────────────┘
```

- **QueryPlanner**: Intent classification, task breakdown, LLM guardrail
- **RetrievalPlanner**: Maps tasks to tools (vector_search, data_analysis, statistical)
- **InformationRetriever**: Executes tool calls
- **ContextAssembler**: Merges chunks, dynamic token budget
- **Generator**: BI analyst response (llama3.1:8b)
- **ResponseQA**: Evaluate + refine, optional retry

**Infrastructure** (Docker Compose): Streamlit App → Bifrost Gateway → Ollama. Optional Langfuse for observability.

---

## Tech Stack

| Layer | Streamlit App | Jupyter Notebook |
|---|---|---|
| **LLM** | Ollama `llama3.2:3b` | OpenAI `gpt-3.5-turbo` |
| **Embeddings** | Ollama `nomic-embed-text` | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS | FAISS |
| **Orchestration** | LangChain | LangChain |
| **Charts** | Plotly | Matplotlib + Seaborn |
| **UI** | Streamlit | Jupyter |

---

## Project Structure

```
AI-Powered-BI/
├── insightforge_app.py          # Streamlit dashboard (main entry point)
├── insightforge/                 # Agentic RAG package
│   ├── config.py                # Pydantic Settings (env-driven)
│   ├── llm/                     # Provider, guardrail
│   ├── retrieval/               # Documents, FAISS vectorstore
│   ├── agent/                   # LangGraph nodes, tools, graph
│   ├── observability/           # Langfuse tracing
│   └── prompts/
├── helm/                        # Kubernetes Helm chart
├── bifrost/config.json          # Bifrost → Ollama routing
├── docker-compose.yml           # Ollama + Bifrost + Langfuse + App
├── Dockerfile
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Data Schema

`sales_data.csv` — 2,500 rows, date range **2022-01-01 → 2028-10-27**

| Column | Type | Description |
|---|---|---|
| `Date` | date | Transaction date |
| `Product` | str | Widget A / B / C / D |
| `Region` | str | North / South / East / West |
| `Sales` | int | Transaction amount (USD) |
| `Customer_Age` | int | 18 – 69 |
| `Customer_Gender` | str | Male / Female |
| `Customer_Satisfaction` | float | 1.0 – 5.0 |

**Derived columns** added at load time: `Month`, `Quarter`, `Year`, `DayOfWeek`, `Age_Group`, `Satisfaction_Level`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.12 |
| [Ollama](https://ollama.com) | Running locally on default port 11434 |
| `llama3.2:3b` model | `ollama pull llama3.2:3b` |
| `llama3.1:8b` model | `ollama pull llama3.1:8b` (for agent quality nodes) |
| `nomic-embed-text` model | `ollama pull nomic-embed-text` |
| OpenAI API key *(notebook only)* | Set in `.env` as `OPENAI_API_KEY` |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/AI-Powered-BI.git
cd AI-Powered-BI
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 5. (Notebook only) Configure OpenAI API key

```bash
cp .env.example .env            # then edit .env
# OPENAI_API_KEY=sk-...
```

---

## Running the App

### Local (Ollama)

```bash
# Ensure Ollama is running in a separate terminal
ollama serve

# Launch the Streamlit dashboard
streamlit run insightforge_app.py
```

### Docker Compose (Ollama + Bifrost + App + Langfuse)

```bash
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501). The first run pulls Ollama models via an init container.

### Kubernetes

```bash
helm install insightforge ./helm
```

### Dashboard Views

| View | Description |
|---|---|
| 📊 **Sales Overview** | KPI cards, monthly trend + 3-month MA, cumulative sales, quarterly bars, day-of-week bars |
| 📦 **Product Analysis** | Per-product metrics, bar/pie charts, product × region heatmap, monthly line, radar chart |
| 🗺️ **Regional Analysis** | Per-region metrics, horizontal bars, treemap, satisfaction box plots, monthly line, grouped bars |
| 👥 **Customer Demographics** | Age histogram, gender comparison, age-group bars, scatter (age vs sales vs satisfaction) |
| 🔬 **Advanced Analytics** | Correlation matrix, scatter matrix, violin plots, time decomposition, segmentation bubble chart |
| 🤖 **AI Assistant** | RAG-powered chat — ask natural-language questions about the data |

### Sidebar Filters

All views respect global filters applied from the sidebar: **date range**, **product**, **region**, **gender**, and **customer age range**.

---

## Notebook

`InsightForge_Assistant.ipynb` walks through the full development workflow:

### Part 1 — AI-Powered BI Assistant

| Step | Description |
|---|---|
| 1 | **Data Preparation** — load CSV, derive columns, descriptive stats |
| 2 | **Knowledge Base Creation** — generate `Document` objects from aggregated stats |
| 3 | **LLM Application** — `ChatOpenAI`, prompt templates, `SalesDataRetriever` |
| 4 | **Chain Prompts** — `SALES_ANALYSIS`, `TREND_ANALYSIS`, `CUSTOMER_INSIGHT`, `RECOMMENDATION` templates |
| 5 | **RAG System** — FAISS vector store, `RetrievalQA` chain, hybrid retriever |
| 6 | **Memory** — `ConversationBufferMemory`, `InsightForgeAssistant` with sliding-window history |

### Part 2 — LLMOps

| Step | Description |
|---|---|
| 7a | **Model Evaluation** — `QAEvalChain` with ground-truth Q&A pairs |
| 7b | **Data Visualization** — matplotlib / seaborn charts |
| 7c | **Streamlit UI** — documents the production app design |

---

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `BIFROST_BASE_URL` | `http://localhost:8080` | Bifrost gateway (or direct Ollama fallback) |
| `LLM_MODEL_LIGHT` | `llama3.2:3b` | Planning / routing nodes |
| `LLM_MODEL_HEAVY` | `llama3.1:8b` | Generator / ResponseQA |
| `EMBEDDING_MODEL` | `nomic-embed-text` | FAISS embeddings |
| `RAG_TOP_K` | `5` | Retrieved chunks per query |
| `MAX_EVAL_RETRIES` | `1` | Retry loop cap |
| `LANGFUSE_*` | (optional) | Observability |

---

## Code Quality

```bash
# Lint
ruff check .

# Tests
pytest tests/ -v --cov=insightforge

# Type check
mypy insightforge/ --ignore-missing-imports
```

The project uses `ruff.toml` with rules `E`, `F`, `W`, `I`, `C4`, `UP`, `B` at line-length 100.

