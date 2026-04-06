"""Configuration for InsightForge agentic RAG."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bifrost_base_url: str = Field(default="http://localhost:8080", alias="BIFROST_BASE_URL")
    llm_model_light: str = Field(default="llama3.2:3b", alias="LLM_MODEL_LIGHT")
    llm_model_heavy: str = Field(default="llama3.1:8b", alias="LLM_MODEL_HEAVY")
    embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    max_eval_retries: int = Field(default=1, alias="MAX_EVAL_RETRIES")
    token_budget_simple: int = Field(default=4000, alias="TOKEN_BUDGET_SIMPLE")
    token_budget_complex: int = Field(default=12000, alias="TOKEN_BUDGET_COMPLEX")

    database_url: str = Field(
        default="postgresql://insightforge:insightforge@localhost:5432/insightforge",
        alias="DATABASE_URL",
    )

    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="", alias="LANGFUSE_HOST")

    @field_validator("llm_model_light", "llm_model_heavy", "embedding_model")
    @classmethod
    def reject_empty_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("bifrost_base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("Bifrost URL must start with http:// or https://")
        return v.rstrip("/")


settings = Settings()
