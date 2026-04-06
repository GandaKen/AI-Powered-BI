"""Build retrieval documents from sales DataFrame."""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Create rich BI summary documents from the source DataFrame."""
    docs: list[Document] = []
    total_sales = df["Sales"].sum()

    docs.append(
        Document(
            page_content=(
                f"Dataset Overview: {len(df)} transactions from "
                f"{df['Date'].min().date()} to {df['Date'].max().date()}. "
                f"Total Sales: ${total_sales:,.2f}. "
                f"Average Transaction: ${df['Sales'].mean():.2f}. "
                f"Median Transaction: ${df['Sales'].median():.2f}. "
                f"Products: {', '.join(df['Product'].unique())}. "
                f"Regions: {', '.join(df['Region'].unique())}. "
                f"Customer Age Range: {df['Customer_Age'].min()}-{df['Customer_Age'].max()} "
                f"(avg {df['Customer_Age'].mean():.1f}). "
                f"Gender Split: {dict(df['Customer_Gender'].value_counts())}. "
                f"Overall Satisfaction: {df['Customer_Satisfaction'].mean():.2f}/5."
            ),
            metadata={"type": "overview"},
        )
    )

    for product in df["Product"].unique():
        product_df = df[df["Product"] == product]
        share = product_df["Sales"].sum() / total_sales * 100
        top_region = product_df.groupby("Region")["Sales"].sum().idxmax()
        docs.append(
            Document(
                page_content=(
                    f"Product {product}: Total ${product_df['Sales'].sum():,.2f} "
                    f"({share:.1f}% share), Avg ${product_df['Sales'].mean():.2f}, "
                    f"Median ${product_df['Sales'].median():.2f}, {len(product_df)} transactions. "
                    f"Satisfaction {product_df['Customer_Satisfaction'].mean():.2f}/5. "
                    f"Top region: {top_region}."
                ),
                metadata={"type": "product", "product": product},
            )
        )

    for region in df["Region"].unique():
        region_df = df[df["Region"] == region]
        share = region_df["Sales"].sum() / total_sales * 100
        top_product = region_df.groupby("Product")["Sales"].sum().idxmax()
        docs.append(
            Document(
                page_content=(
                    f"Region {region}: Total ${region_df['Sales'].sum():,.2f} "
                    f"({share:.1f}% share), Avg ${region_df['Sales'].mean():.2f}, "
                    f"{len(region_df)} transactions. Top product: {top_product}. "
                    f"Satisfaction {region_df['Customer_Satisfaction'].mean():.2f}/5."
                ),
                metadata={"type": "region", "region": region},
            )
        )

    monthly = df.groupby("Month").agg({"Sales": ["sum", "mean", "count"]}).reset_index()
    monthly.columns = ["Month", "Total", "Avg", "Count"]
    best_month = monthly.loc[monthly["Total"].idxmax()]
    worst_month = monthly.loc[monthly["Total"].idxmin()]
    docs.append(
        Document(
            page_content=(
                f"Monthly Trend: Best month {best_month['Month']} (${best_month['Total']:,.0f}), "
                f"Worst month {worst_month['Month']} (${worst_month['Total']:,.0f}), "
                f"Average monthly sales ${monthly['Total'].mean():,.0f}."
            ),
            metadata={"type": "monthly_trend"},
        )
    )

    if "Quarter" in df.columns:
        quarterly = df.groupby("Quarter")["Sales"].agg(["sum", "mean", "count"]).reset_index()
        quarterly.columns = ["Quarter", "Total", "Avg", "Count"]
        best_q = quarterly.loc[quarterly["Total"].idxmax()]
        worst_q = quarterly.loc[quarterly["Total"].idxmin()]
        docs.append(
            Document(
                page_content=(
                    f"Quarterly Trend: Best quarter {best_q['Quarter']} "
                    f"(${best_q['Total']:,.0f}), Worst quarter {worst_q['Quarter']} "
                    f"(${worst_q['Total']:,.0f}), Avg quarterly ${quarterly['Total'].mean():,.0f}."
                ),
                metadata={"type": "quarterly_trend"},
            )
        )

    yearly = df.groupby("Year")["Sales"].sum().sort_index()
    if len(yearly) >= 2:
        yoy_lines = []
        for i in range(1, len(yearly)):
            prev_year = yearly.index[i - 1]
            cur_year = yearly.index[i]
            prev_sales = yearly.iloc[i - 1]
            cur_sales = yearly.iloc[i]
            growth = ((cur_sales - prev_sales) / prev_sales) * 100 if prev_sales else 0
            yoy_lines.append(
                f"{prev_year}->{cur_year}: ${prev_sales:,.0f} to ${cur_sales:,.0f} ({growth:.1f}%)"
            )
        docs.append(
            Document(
                page_content="Year-over-year sales comparison: " + "; ".join(yoy_lines),
                metadata={"type": "yoy"},
            )
        )

    return docs

