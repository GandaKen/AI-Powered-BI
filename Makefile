.PHONY: env-cloud env-selfhosted db-migrate db-reset dev docker-up docker-down help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment profiles ───────────────────────────────────────

env-cloud: ## Switch to Langfuse Cloud config
	@cp .env.cloud .env
	@echo "Switched to Langfuse Cloud environment."

env-selfhosted: ## Switch to self-hosted Langfuse config
	@cp .env.selfhosted .env
	@echo "Switched to self-hosted Langfuse environment."

# ── Database ───────────────────────────────────────────────────

db-migrate: ## Run Alembic migrations (creates trace tables)
	alembic upgrade head

db-reset: ## Drop and recreate trace tables
	alembic downgrade base
	alembic upgrade head

# ── Development ────────────────────────────────────────────────

dev: ## Start Streamlit in development mode
	streamlit run insightforge_app.py

docker-up: ## Start Docker Compose stack
	docker compose up

docker-down: ## Stop Docker Compose stack
	docker compose down

install: ## Install Python dependencies
	pip install -r requirements.txt && pip install -e .
