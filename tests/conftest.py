from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    data = {
        "Date": pd.date_range("2024-01-01", periods=20, freq="D"),
        "Product": ["Widget A", "Widget B", "Widget C", "Widget D"] * 5,
        "Region": ["North", "South", "East", "West"] * 5,
        "Sales": [100 + i * 10 for i in range(20)],
        "Customer_Age": [25 + (i % 10) for i in range(20)],
        "Customer_Gender": ["Male", "Female"] * 10,
        "Customer_Satisfaction": [3.5 + (i % 3) * 0.3 for i in range(20)],
    }
    df = pd.DataFrame(data)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Date"].dt.year
    return df


@pytest.fixture
def mock_llm():
    """Mock LLM returning canned responses."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='{"intent":"factual","complexity":"simple","entities":[],"tasks":["q"]}'
    )
    return llm


@pytest.fixture
def mock_embeddings():
    """Fake embeddings for tests (768 dims matches nomic-embed-text)."""
    from langchain_community.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=768)

