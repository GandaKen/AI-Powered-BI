.DEFAULT_GOAL := up

.PHONY: up down build logs status restart dev install db-migrate db-reset env-cloud env-selfhosted help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Default (Docker Compose) ─────────────────────────────────

up: ## Start the full stack (default target)
	docker compose up --build -d
	@echo ""
	@echo "InsightForge is running at http://localhost:8501"
	@echo "Bifrost gateway at http://localhost:8080"
	@echo ""

down: ## Stop the full stack
	docker compose down

build: ## Rebuild the app image without starting
	docker compose build

logs: ## Tail logs from all services
	docker compose logs -f

status: ## Show status of all services
	docker compose ps

restart: ## Restart the app container (keeps Ollama/Postgres/Redis running)
	docker compose restart app
	@echo "App restarted at http://localhost:8501"

# ── Local development (no Docker) ────────────────────────────

dev: ## Start Streamlit locally (requires Ollama running separately)
	streamlit run insightforge_app.py

install: ## Install Python dependencies
	pip install -r requirements.txt && pip install -e .

# ── Database ───────────────────────────────────────────────────

db-migrate: ## Run Alembic migrations (creates trace tables)
	alembic upgrade head

db-reset: ## Drop and recreate trace tables
	alembic downgrade base
	alembic upgrade head

# ── Environment profiles ───────────────────────────────────────

env-cloud: ## Switch to Langfuse Cloud config
	@cp .env.cloud .env
	@echo "Switched to Langfuse Cloud environment."

env-selfhosted: ## Switch to self-hosted Langfuse config
	@cp .env.selfhosted .env
	@echo "Switched to self-hosted Langfuse environment."
