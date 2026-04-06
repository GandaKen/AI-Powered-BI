"""Integration test for the agent graph with mocked LLM."""

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
    s.llm_model_light = "llama3.2:3b"
    s.llm_model_heavy = "llama3.1:8b"
    s.max_eval_retries = 1
    return s


@pytest.fixture
def sample_df():
    data = {
        "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "Product": ["Widget A", "Widget B", "Widget C", "Widget D"] * 7 + ["Widget A", "Widget B"],
        "Region": ["North", "South", "East", "West"] * 7 + ["North", "South"],
        "Sales": [100 + i * 5 for i in range(30)],
        "Customer_Age": [25 + (i % 15) for i in range(30)],
        "Customer_Gender": ["Male", "Female"] * 15,
        "Customer_Satisfaction": [3.5 + (i % 3) * 0.2 for i in range(30)],
    }
    df = pd.DataFrame(data)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Year"] = df["Date"].dt.year
    return df


def test_create_agent_returns_compiled_graph(mock_settings, sample_df):
    """Agent factory returns a runnable graph."""
    with patch("insightforge.agent.get_llm") as mock_get_llm:
        with patch("insightforge.agent.get_embeddings") as mock_get_emb:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content='{"intent":"factual","complexity":"simple","entities":[],"tasks":["q"]}'
            )
            mock_get_llm.return_value = mock_llm

            from langchain_community.embeddings import FakeEmbeddings
            mock_get_emb.return_value = FakeEmbeddings(size=768)

            agent = create_agent(sample_df, mock_settings)
            assert agent is not None
            assert hasattr(agent, "invoke")


def test_agent_invoke_produces_final_response(mock_settings, sample_df):
    """Full graph invocation produces final_response in state."""
    from langchain_core.tools import tool

    def _mock_search(query: str) -> str:
        """Mock vector search."""
        return "Product Widget A: $5000. Region North: $2000."

    mock_vector_tool = tool(_mock_search)

    def mock_build_vector_search(*args, **kwargs):
        return mock_vector_tool

    with patch("insightforge.agent.get_llm") as mock_get_llm:
        with patch("insightforge.agent.get_embeddings") as mock_get_emb:
            with patch(
                "insightforge.agent.build_vector_search_tool",
                side_effect=mock_build_vector_search,
            ):
                mock_llm = MagicMock()
                mock_llm.invoke.side_effect = [
                    MagicMock(content='{"label":"safe","reason":"ok"}'),  # guardrail
                    MagicMock(
                        content='{"intent":"factual","complexity":"simple","entities":[],'
                        '"tasks":["Which product has highest sales?"]}'
                    ),
                    MagicMock(content="Widget A has the highest sales at $X."),
                    MagicMock(
                        content='{"score":8,"issues":[],'
                        '"refined_response":"Widget A has the highest sales."}'
                    ),
                ]
                mock_get_llm.return_value = mock_llm

                from langchain_community.embeddings import FakeEmbeddings
                mock_get_emb.return_value = FakeEmbeddings(size=768)

                agent = create_agent(sample_df, mock_settings)
                result = agent.invoke({
                    "query": "Which product has highest sales?",
                    "messages": [],
                    "retry_count": 0,
                    "trace_steps": [],
                })

                assert "final_response" in result or "generated_response" in result
                assert "trace_steps" in result
                assert "QueryPlanner" in result.get("trace_steps", [])
