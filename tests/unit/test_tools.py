"""Unit tests for agent tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from insightforge.agent.tools.data_analysis import build_data_analysis_tool
from insightforge.agent.tools.statistical import build_statistical_tool
from insightforge.agent.tools.vector_search import build_vector_search_tool


def test_vector_search_tool() -> None:
    """Vector search returns concatenated doc contents."""
    mock_manager = MagicMock()
    mock_manager.search.return_value = [
        MagicMock(page_content="Product Widget A: $1000 total."),
        MagicMock(page_content="Region North: $500 total."),
    ]
    tool = build_vector_search_tool(vector_manager=mock_manager, top_k=2)
    result = tool.invoke({"query": "Widget A"})
    assert isinstance(result, str)
    assert "Widget A" in result
    mock_manager.search.assert_called_once_with(query="Widget A", k=2)


def test_data_analysis_groupby_agg(sample_df) -> None:
    """groupby_agg returns aggregated results."""
    tool = build_data_analysis_tool(sample_df)
    req = json.dumps({
        "operation": "groupby_agg",
        "groupby": ["Product"],
        "metric": "Sales",
        "agg": "sum",
    })
    result = tool.invoke({"request_json": req})
    assert "Widget" in result


def test_data_analysis_pivot(sample_df) -> None:
    """pivot returns pivot table."""
    tool = build_data_analysis_tool(sample_df)
    req = json.dumps({
        "operation": "pivot",
        "index": "Product",
        "columns": "Region",
        "values": "Sales",
        "agg": "sum",
    })
    result = tool.invoke({"request_json": req})
    assert isinstance(result, str)


def test_data_analysis_describe(sample_df) -> None:
    """describe returns stats."""
    tool = build_data_analysis_tool(sample_df)
    req = json.dumps({"operation": "describe", "metric": "Sales"})
    result = tool.invoke({"request_json": req})
    assert "mean" in result.lower() or "count" in result.lower()


def test_data_analysis_filter_agg(sample_df) -> None:
    """filter_agg returns filtered aggregation."""
    tool = build_data_analysis_tool(sample_df)
    req = json.dumps({
        "operation": "filter_agg",
        "filter_col": "Product",
        "filter_val": "Widget A",
        "metric": "Sales",
        "agg": "sum",
    })
    result = tool.invoke({"request_json": req})
    assert isinstance(result, str)
    assert len(result) > 0


def test_data_analysis_rejects_unknown_column(sample_df) -> None:
    """Unknown column raises ValueError."""
    tool = build_data_analysis_tool(sample_df)
    req = json.dumps({
        "operation": "groupby_agg",
        "groupby": ["InvalidCol"],
        "metric": "Sales",
        "agg": "sum",
    })
    with pytest.raises(ValueError, match="Unknown column"):
        tool.invoke({"request_json": req})


def test_statistical_correlation(sample_df) -> None:
    """Correlation returns matrix."""
    tool = build_statistical_tool(sample_df)
    req = json.dumps({"operation": "correlation"})
    result = tool.invoke({"request_json": req})
    assert isinstance(result, str)


def test_statistical_percentile(sample_df) -> None:
    """Percentile returns value."""
    tool = build_statistical_tool(sample_df)
    req = json.dumps({"operation": "percentile", "metric": "Sales", "p": 0.9})
    result = tool.invoke({"request_json": req})
    assert "P90" in result or "90" in result
