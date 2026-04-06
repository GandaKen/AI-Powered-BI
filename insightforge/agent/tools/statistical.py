"""Statistical analysis tool."""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.tools import tool


def build_statistical_tool(df: pd.DataFrame):
    """Create a statistical computation tool bound to DataFrame."""

    @tool("statistical")
    def statistical(request_json: str) -> str:
        """Compute statistical summaries from safe operation requests."""
        request = json.loads(request_json)
        operation = request.get("operation")

        if operation == "correlation":
            cols = request.get(
                "columns",
                ["Sales", "Customer_Age", "Customer_Satisfaction"],
            )
            numeric_cols = [col for col in cols if col in df.select_dtypes("number").columns]
            if not numeric_cols:
                return "No valid numeric columns provided."
            return df[numeric_cols].corr().round(3).to_string()

        if operation == "percentile":
            metric = request.get("metric", "Sales")
            percentile = float(request.get("p", 0.9))
            if metric not in df.columns:
                return f"Unknown metric: {metric}"
            return f"{metric} P{int(percentile * 100)}: {df[metric].quantile(percentile):.2f}"

        if operation == "yoy_growth":
            yearly = df.groupby("Year")["Sales"].sum().sort_index()
            if len(yearly) < 2:
                return "Not enough yearly data for YoY calculation."
            rows = []
            for i in range(1, len(yearly)):
                prev = yearly.iloc[i - 1]
                curr = yearly.iloc[i]
                growth = ((curr - prev) / prev) * 100 if prev else 0
                rows.append(f"{yearly.index[i - 1]}->{yearly.index[i]}: {growth:.2f}%")
            return "\n".join(rows)

        return "Unsupported statistical operation."

    return statistical
