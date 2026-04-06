"""Safe, structured pandas data analysis tool."""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool

SAFE_AGGS = {"sum", "mean", "count", "median", "min", "max"}


def _validate_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")


def build_data_analysis_tool(df: pd.DataFrame):
    """Create a structured data analysis tool bound to a DataFrame."""

    @tool("data_analysis")
    def data_analysis(request_json: str) -> str:
        """Run allowlisted operations using JSON request."""
        request = json.loads(request_json)
        operation = request.get("operation", "groupby_agg")

        if operation == "groupby_agg":
            groupby_cols = request.get("groupby", [])
            metric = request.get("metric", "Sales")
            agg = request.get("agg", "sum")
            if agg not in SAFE_AGGS:
                raise ValueError(f"Unsupported aggregation: {agg}")
            for col in groupby_cols + [metric]:
                _validate_column(df, col)
            result = (
                df.groupby(groupby_cols, observed=True)[metric]
                .agg(agg)
                .sort_values(ascending=False)
                .head(20)
            )
            return result.to_string()

        if operation == "pivot":
            index_col = request["index"]
            columns_col = request["columns"]
            values_col = request.get("values", "Sales")
            agg = request.get("agg", "sum")
            for col in [index_col, columns_col, values_col]:
                _validate_column(df, col)
            if agg not in SAFE_AGGS:
                raise ValueError(f"Unsupported aggregation: {agg}")
            result = pd.pivot_table(
                df,
                index=index_col,
                columns=columns_col,
                values=values_col,
                aggfunc=agg,
                fill_value=0,
            )
            return result.to_string()

        if operation == "describe":
            metric = request.get("metric", "Sales")
            _validate_column(df, metric)
            return df[metric].describe().to_string()

        if operation == "filter_agg":
            filter_col = request.get("filter_col")
            filter_val = request.get("filter_val")
            metric = request.get("metric", "Sales")
            agg = request.get("agg", "sum")
            if agg not in SAFE_AGGS:
                raise ValueError(f"Unsupported aggregation: {agg}")
            for col in [filter_col, metric]:
                _validate_column(df, col)
            subset = df[df[filter_col] == filter_val]
            return str(subset[metric].agg(agg))

        raise ValueError(f"Unsupported operation: {operation}")

    return data_analysis
