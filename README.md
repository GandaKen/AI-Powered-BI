# InsightForge — AI-Powered Business Intelligence Assistant

[![Tests](https://github.com/GandaKen/AI-Powered-BI/actions/workflows/tests.yml/badge.svg)](https://github.com/GandaKen/AI-Powered-BI/actions/workflows/tests.yml)
[![Pylint](https://github.com/GandaKen/AI-Powered-BI/actions/workflows/pylint.yml/badge.svg)](https://github.com/GandaKen/AI-Powered-BI/actions/workflows/pylint.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-green.svg)](https://github.com/langchain-ai/langgraph)

> An end-to-end BI solution that combines interactive Plotly dashboards with a
> **6-node agentic RAG pipeline** (LangGraph + Ollama) to answer natural-language
> questions about sales data — including built-in observability, guardrails, and
> Kubernetes-ready deployment.

---

## Table of Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data Schema](#data-schema)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Dashboard Views](#dashboard-views)
- [Agentic RAG Pipeline](#agentic-rag-pipeline)
- [Observability](#observability)
- [Notebook](#notebook)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
  - [Docker Compose](#docker-compose)
  - [Kubernetes (Helm)](#kubernetes-helm)
- [CI/CD](#cicd)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

InsightForge demonstrates how large language models can be integrated into a
traditional BI workflow. The project provides:

- **Seven interactive dashboard views** built with Streamlit and Plotly, exploring
  2,500 sales transactions across products, regions, and customer demographics
- **An agentic RAG pipeline** (LangGraph) with query planning, multi-tool retrieval,
  response generation, and self-evaluation with retry
- **Built-in guardrails** that block prompt-injection attempts and off-topic queries
  using both heuristic rules and LLM classification
- **Dual-write observability** — local Postgres traces plus optional Langfuse
  integration — surfaced in a dedicated Observability dashboard view
- **Production-ready packaging** via Docker Compose (Ollama + Bifrost gateway +
  Postgres + Redis) and a Kubernetes Helm chart

---

## Screenshots

| Sales Overview | AI Assistant |
|---|---|
| *Monthly trend, cumulative sales, quarterly & day-of-week breakdowns* | *Conversational agentic RAG chat with pipeline trace and quality score* |

---

## Architecture

### Agentic RAG Graph (LangGraph)

```
START → QueryPlanner → RetrievalPlanner → InformationRetriever → ContextAssembler → Generator → ResponseQA → END
               ↑                                                                                      │
               └──────────────────────────────── retry (max 1) ──────────────────────────────────────┘
```

| Node | Responsibility |
|---|---|
| **QueryPlanner** | Guardrail check, intent classification, task breakdown (llama3.2:3b) |
| **RetrievalPlanner** | Maps tasks to tools: `vector_search`, `data_analysis`, `statistical` |
| **InformationRetriever** | Executes selected tools against the FAISS store and DataFrame |
| **ContextAssembler** | Deduplicates chunks, applies dynamic token budget (4k simple / 12k complex) |
| **Generator** | Produces a BI-analyst response (llama3.1:8b) |
| **ResponseQA** | Heuristic + LLM evaluation; may refine the response or trigger a retry |

### Infrastructure

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   Bifrost    │────▶│    Ollama    │
│  App (:8501) │     │  GW (:8080)  │     │   (:11434)   │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       ├──▶ Postgres (:5432)  — trace storage
       ├──▶ Redis (:6379)     — caching (planned)
       └──▶ Langfuse (opt.)   — cloud observability
```

---

## Tech Stack

| Layer | Streamlit App | Jupyter Notebook |
|---|---|---|
| **LLM** | Ollama `llama3.2:3b` (light) / `llama3.1:8b` (heavy) | OpenAI `gpt-3.5-turbo` |
| **Embeddings** | Ollama `nomic-embed-text` | HuggingFace `all-MiniLM-L6-v2` |
| **Orchestration** | LangGraph (agentic) | LangChain (chain-based) |
| **Vector Store** | FAISS | FAISS |
| **Charts** | Plotly (dark theme) | Matplotlib + Seaborn |
| **Observability** | Langfuse + Postgres | — |
| **UI** | Streamlit | Jupyter |
| **Gateway** | Bifrost (OpenAI-compatible proxy to Ollama) | — |

---

## Project Structure

```
AI-Powered-BI/
├── insightforge_app.py              # Streamlit dashboard (main entry point)
├── insightforge/                     # Core Python package
│   ├── config.py                    # Pydantic Settings (env-driven)
│   ├── llm/
│   │   ├── provider.py              # LLM & embeddings (Bifrost → Ollama fallback)
│   │   └── guardrail.py             # Prompt-injection & off-topic blocking
│   ├── retrieval/
│   │   ├── documents.py             # Build knowledge-base docs from DataFrame
│   │   └── vectorstore.py           # FAISS build / save / load with hash invalidation
│   ├── agent/
│   │   ├── state.py                 # AgentState TypedDict
│   │   ├── graph.py                 # LangGraph workflow definition
│   │   ├── nodes/                   # query_planner, retrieval_planner, etc.
│   │   └── tools/                   # vector_search, data_analysis, statistical
│   ├── observability/
│   │   ├── collector.py             # TraceCollector (LangChain callbacks → Postgres)
│   │   ├── models.py                # SQLAlchemy models (Trace, TraceStep)
│   │   ├── repository.py            # CRUD + metrics queries
│   │   └── tracing.py              # Langfuse + TraceCollector setup
│   ├── prompts/
│   │   └── templates.py             # QUERY_PLANNER, GENERATOR, RESPONSE_QA prompts
│   └── db/
│       └── connection.py            # SQLAlchemy session factory
├── alembic/                          # Database migrations
│   └── versions/
│       └── 001_create_trace_tables.py
├── bifrost/
│   └── config.json                  # Bifrost → Ollama provider routing
├── helm/                             # Kubernetes Helm chart
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
├── tests/
│   ├── unit/                        # config, guardrail, provider, tools, vectorstore, documents
│   ├── integration/                 # agent graph end-to-end
│   └── e2e/                         # full chat flow
├── .github/workflows/
│   ├── tests.yml                    # pytest + mypy on push/PR
│   └── pylint.yml                   # pylint on push
├── docker-compose.yml               # Ollama + Bifrost + Postgres + Redis + App
├── Dockerfile                       # Python 3.11-slim container
├── Makefile                         # Dev shortcuts (see below)
├── pyproject.toml                   # Package metadata + dev deps
├── requirements.txt                 # Pinned runtime dependencies
├── ruff.toml                        # Linter config (E, F, W, I, C4, UP, B)
├── alembic.ini                      # Alembic config
├── .env.example                     # Environment template
├── .env.cloud                       # Langfuse Cloud preset
├── .env.selfhosted                  # Self-hosted Langfuse preset
├── InsightForge_Assistant.ipynb      # Development & evaluation notebook
├── sales_data.csv                   # Source dataset (2,500 transactions)
└── sales_data_quiz.md               # Ground-truth Q&A pairs for evaluation
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

**Derived columns** added at load time: `Month`, `Quarter`, `Year`, `DayOfWeek`,
`Age_Group`, `Satisfaction_Level`.

---

## Getting Started

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on 3.12 |
| [Ollama](https://ollama.com) | Running locally on default port 11434 |
| `llama3.2:3b` model | `ollama pull llama3.2:3b` |
| `llama3.1:8b` model | `ollama pull llama3.1:8b` (used by Generator and ResponseQA nodes) |
| `nomic-embed-text` model | `ollama pull nomic-embed-text` |
| Docker & Docker Compose | Required only for containerized deployment |
| OpenAI API key | Required only for the Jupyter notebook |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/GandaKen/AI-Powered-BI.git
cd AI-Powered-BI

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .                # install insightforge package in editable mode

# 4. Pull Ollama models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 5. Set up environment
cp .env.example .env            # then edit .env as needed
```

### Running the App

**Local (direct Ollama):**

```bash
# Start Ollama in a separate terminal
ollama serve

# Launch the Streamlit dashboard
streamlit run insightforge_app.py
```

**Using the Makefile:**

```bash
make dev              # Start Streamlit in dev mode
make docker-up        # Start the full Docker Compose stack
make docker-down      # Stop the stack
make db-migrate       # Run Alembic migrations
make install          # Install all dependencies
make help             # Show all available targets
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Dashboard Views

The sidebar provides radio navigation across seven views, with global filters
(date range, product, region, gender, customer age) applied from a collapsible
expander.

| View | Highlights |
|---|---|
| **Sales Overview** | KPI cards (revenue, transactions, avg/median sale, satisfaction), monthly trend + 3-month moving average, cumulative sales, quarterly revenue bars, day-of-week analysis |
| **Product Analysis** | Per-product metrics, total sales bar/pie charts, product × region heatmap, monthly trend by product, product comparison radar chart |
| **Regional Analysis** | Per-region metrics, horizontal bars, sales treemap (region → product), satisfaction box plots, monthly trend by region, region × product grouped bars |
| **Customer Demographics** | Age histogram + box plot, sales by gender, revenue and satisfaction by age group, age vs sales vs satisfaction scatter, gender × product breakdown |
| **Advanced Analytics** | Tabs — Correlations (matrix + scatter matrix), Distributions (violin by product/region), Time Decomposition (3 subplots), Segmentation (bubble chart + top/bottom segments) |
| **AI Assistant** | Agentic RAG chat interface with pipeline trace expander (step timing, token counts), quality score, and safety-check indicator |
| **Observability** | Latency, quality, and usage KPIs; tabbed views for latency trends, quality distribution, token usage, and full trace log; Langfuse dashboard link |

---

## Agentic RAG Pipeline

### Tools

The pipeline dynamically selects from three retrieval tools:

| Tool | Operations |
|---|---|
| **vector_search** | FAISS similarity search over the pre-built knowledge base |
| **data_analysis** | `groupby_agg`, `pivot`, `describe`, `filter_agg` — allowlisted DataFrame operations |
| **statistical** | `correlation`, `percentile`, `yoy_growth` |

### Resilience

- **Guardrail**: Heuristic blocklist + LLM-based classification (`safe` / `injection_attempt` / `off_topic`)
- **Self-evaluation retry**: ResponseQA can send the pipeline back to RetrievalPlanner (max 1 retry)
- **Circuit breaker**: After 3 consecutive agent failures, the app falls back to a basic RAG chain

---

## Observability

InsightForge uses a **dual-write** observability strategy:

| Sink | Purpose |
|---|---|
| **Postgres** (local) | Every request is traced via `TraceCollector` (LangChain callbacks). Step timing, token counts, and quality scores are stored in `traces` / `trace_steps` tables and surfaced in the Observability dashboard view. |
| **Langfuse** (optional) | When `LANGFUSE_*` keys are set, traces are also sent to Langfuse Cloud or a self-hosted instance for deeper analysis. |

**Database migrations** are managed by Alembic:

```bash
make db-migrate       # alembic upgrade head
make db-reset         # downgrade + upgrade (destructive)
```

**Environment presets** simplify switching between Langfuse configurations:

```bash
make env-cloud        # copies .env.cloud → .env
make env-selfhosted   # copies .env.selfhosted → .env
```

---

## Notebook

`InsightForge_Assistant.ipynb` walks through the full development and evaluation
workflow using OpenAI models.

### Part 1 — AI-Powered BI Assistant

| Step | Description |
|---|---|
| 1 | **Data Preparation** — load CSV, derive columns, descriptive stats |
| 2 | **Knowledge Base Creation** — generate `Document` objects from aggregated stats |
| 3 | **LLM Application** — `ChatOpenAI`, prompt templates, custom `SalesDataRetriever` |
| 4 | **Chain Prompts** — `SALES_ANALYSIS`, `TREND_ANALYSIS`, `CUSTOMER_INSIGHT`, `RECOMMENDATION` templates |
| 5 | **RAG System** — FAISS vector store, `RetrievalQA` chain, hybrid retriever |
| 6 | **Memory** — `ConversationBufferMemory`, `InsightForgeAssistant` with sliding-window history |

### Part 2 — LLMOps

| Step | Description |
|---|---|
| 7a | **Model Evaluation** — `QAEvalChain` with ground-truth Q&A pairs from `sales_data_quiz.md` |
| 7b | **Data Visualization** — matplotlib / seaborn charts |
| 7c | **Streamlit UI** — documents the production app design |

---

## Configuration

All settings are managed through environment variables (see `.env.example`).
The `insightforge.config.Settings` class uses Pydantic Settings with validation.

| Variable | Default | Description |
|---|---|---|
| `BIFROST_BASE_URL` | `http://localhost:8080` | Bifrost gateway URL (falls back to direct Ollama) |
| `LLM_MODEL_LIGHT` | `llama3.2:3b` | Model for planning / routing nodes |
| `LLM_MODEL_HEAVY` | `llama3.1:8b` | Model for Generator / ResponseQA |
| `EMBEDDING_MODEL` | `nomic-embed-text` | FAISS embedding model |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature |
| `RAG_TOP_K` | `5` | Number of retrieved chunks per query |
| `MAX_EVAL_RETRIES` | `1` | ResponseQA retry loop cap |
| `TOKEN_BUDGET_SIMPLE` | `4000` | Context token budget for simple queries |
| `TOKEN_BUDGET_COMPLEX` | `12000` | Context token budget for complex queries |
| `DATABASE_URL` | `postgresql://...@localhost/insightforge` | Postgres connection for trace storage |
| `LANGFUSE_PUBLIC_KEY` | *(empty)* | Langfuse public key (optional) |
| `LANGFUSE_SECRET_KEY` | *(empty)* | Langfuse secret key (optional) |
| `LANGFUSE_HOST` | *(empty)* | Langfuse host URL (optional) |

---

## Testing

The test suite is organized into three tiers:

| Tier | Directory | Scope |
|---|---|---|
| **Unit** | `tests/unit/` | Config validation, guardrail, LLM provider, tools, vectorstore, documents |
| **Integration** | `tests/integration/` | Agent graph creation and invocation with mocked LLM/tools |
| **E2E** | `tests/e2e/` | Full chat flow with mocked agent |

```bash
# Run all tests with coverage
pytest tests/ -v --cov=insightforge --cov-report=term-missing

# Type checking
mypy insightforge/ --ignore-missing-imports

# Linting
ruff check .
```

---

## Deployment

### Docker Compose

The `docker-compose.yml` orchestrates five services:

| Service | Image | Port | Purpose |
|---|---|---|---|
| **ollama** | `ollama/ollama:latest` | 11434 | LLM inference server |
| **ollama-init** | `ollama/ollama:latest` | — | Init container that pulls required models |
| **bifrost** | `maximhq/bifrost:latest` | 8080 | OpenAI-compatible gateway routing to Ollama |
| **redis** | `redis:7-alpine` | 6379 | Redis with AOF persistence |
| **postgres** | `postgres:15-alpine` | 5432 | Trace storage database |
| **app** | Built from `Dockerfile` | 8501 | Streamlit dashboard |

```bash
docker compose up --build
```

The app is available at [http://localhost:8501](http://localhost:8501) once all services are healthy.

### Kubernetes (Helm)

A Helm chart is provided in `helm/` for Kubernetes deployment:

```bash
helm install insightforge ./helm
```

The chart deploys the Streamlit app as a `Deployment` with a `ClusterIP` Service.
See `helm/values.yaml` for replica count, resource limits, image configuration,
and environment variable overrides.

---

## CI/CD

Two GitHub Actions workflows run automatically:

| Workflow | Trigger | Steps |
|---|---|---|
| **Tests** (`tests.yml`) | Push to `main`, pull requests | Install deps → pytest with coverage → mypy type check |
| **Pylint** (`pylint.yml`) | Every push | Lint all Python files with pylint |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests
4. Ensure all checks pass (`pytest`, `ruff check .`, `mypy`)
5. Commit with a descriptive message
6. Open a pull request against `main`

---

## License

This project is part of the Simplilearn AI/ML Engineering Capstone. No license
file is currently included — contact the repository owner for usage terms.
