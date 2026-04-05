"""E2E test for chat flow via agent graph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from insightforge.agent import create_agent
    from insightforge.config import Settings
except ImportError as e:
    pytest.skip(f"Agent imports unavailable: {e}", allow_module_level=True)


@pytest.fixture
def mock_settings():
    s = Settings()
    s.bifrost_base_url = "http://localhost:8080"
    s.max_eval_retries = 1
    return s


@pytest.fixture
def sample_df():
    data = {
        "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "Product": ["Widget A", "Widget B"] * 15,
        "Region": ["North", "South"] * 15,
        "Sales": [100 + i * 5 for i in range(30)],
        "Customer_Age": [30] * 30,
        "Customer_Gender": ["Male", "Female"] * 15,
        "Customer_Satisfaction": [4.0] * 30,
    }
    df = pd.DataFrame(data)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Date"].dt.year
    return df


def test_chat_flow_query_to_response(mock_settings, sample_df):
    """Simulate chat: query -> agent invoke -> response in state."""
    from langchain_community.embeddings import FakeEmbeddings
    from langchain_core.tools import tool

    def _mock_search(query: str) -> str:
        """Mock vector search."""
        return "Widget A: $2000. Widget B: $1500."

    mock_vector_tool = tool(_mock_search)

    with patch("insightforge.agent.get_llm") as mock_get_llm:
        with patch("insightforge.agent.get_embeddings") as mock_get_emb:
            with patch(
                "insightforge.agent.build_vector_search_tool",
                return_value=mock_vector_tool,
            ):
                mock_llm = MagicMock()
                mock_llm.invoke.side_effect = [
                    MagicMock(content='{"label":"safe","reason":"ok"}'),
                    MagicMock(
                        content='{"intent":"factual","complexity":"simple",'
                        '"entities":[],"tasks":["Top product?"]}'
                    ),
                    MagicMock(content="Widget A leads with $2000."),
                    MagicMock(
                        content='{"score":9,"issues":[],'
                        '"refined_response":"Widget A leads with $2000."}'
                    ),
                ]
                mock_get_llm.return_value = mock_llm
                mock_get_emb.return_value = FakeEmbeddings(size=768)

                agent = create_agent(sample_df, mock_settings)
                result = agent.invoke({
                    "query": "Which product has highest sales?",
                    "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
                    "retry_count": 0,
                    "trace_steps": [],
                })

                assert "final_response" in result
                assert "Widget" in result["final_response"] or "product" in result["final_response"].lower()
