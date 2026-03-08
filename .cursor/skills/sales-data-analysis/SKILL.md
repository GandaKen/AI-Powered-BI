---
name: sales-data-analysis
description: Perform exploratory data analysis and business analytics on sales datasets using pandas, numpy, and visualization libraries. Use when loading CSV data, computing business metrics, creating aggregations, building pivot tables, or generating sales reports.
---

# Sales Data Analysis Patterns

## Data Loading & Enrichment

```python
import pandas as pd

df = pd.read_csv("sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.to_period("M").astype(str)
df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
df["Year"] = df["Date"].dt.year
df["DayOfWeek"] = df["Date"].dt.day_name()
```

### Binning Continuous Variables

```python
df["Age_Group"] = pd.cut(
    df["Customer_Age"],
    bins=[17, 25, 35, 45, 55, 70],
    labels=["18-25", "26-35", "36-45", "46-55", "56-69"],
)
df["Satisfaction_Level"] = pd.cut(
    df["Customer_Satisfaction"],
    bins=[0, 2, 3, 4, 5.01],
    labels=["Low (1-2)", "Medium (2-3)", "High (3-4)", "Very High (4-5)"],
)
```

## Standard Aggregations

### Per-category summary
```python
agg = df.groupby("Product").agg(
    Total_Sales=("Sales", "sum"),
    Avg_Sale=("Sales", "mean"),
    Transactions=("Sales", "count"),
    Satisfaction=("Customer_Satisfaction", "mean"),
).round(2).reset_index()
```

### Cross-dimensional pivot
```python
pivot = df.pivot_table(values="Sales", index="Product", columns="Region", aggfunc="sum")
```

### Time series with rolling stats
```python
monthly = df.groupby("Month").agg({"Sales": ["sum", "mean", "count"]}).reset_index()
monthly.columns = ["Month", "Total", "Avg", "Count"]
monthly["Cumulative"] = monthly["Total"].cumsum()
monthly["MA_3"] = monthly["Total"].rolling(3, min_periods=1).mean()
```

## Business Metrics

| Metric | Formula |
|--------|---------|
| Revenue share | `product_total / grand_total * 100` |
| Top performer | `grouped.idxmax()` |
| Growth rate | `monthly.pct_change()` |
| Volatility | `monthly.std()` |

## Segmentation Analysis

```python
segments = df.groupby(["Product", "Region", "Customer_Gender"]).agg(
    Revenue=("Sales", "sum"),
    Transactions=("Sales", "count"),
    Avg_Sale=("Sales", "mean"),
    Satisfaction=("Customer_Satisfaction", "mean"),
).round(2).reset_index().sort_values("Revenue", ascending=False)

top_10 = segments.head(10)
bottom_10 = segments.tail(10).sort_values("Revenue")
```

## Correlation Analysis

```python
corr_cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]
corr_matrix = df[corr_cols].corr().round(3)
```

## Visualization Quick Reference

### Matplotlib/Seaborn (for notebooks)
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=agg, x="Product", y="Total_Sales", ax=ax)
plt.tight_layout()
plt.show()
```

### Plotly (for Streamlit dashboards)
```python
import plotly.express as px

fig = px.bar(agg, x="Product", y="Total_Sales", color="Product",
             color_discrete_map=PRODUCT_COLORS, text_auto=",.0f")
fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)
```

## Filtering Pattern

```python
def apply_filters(df, products=None, regions=None, date_range=None, age_range=None):
    filtered = df.copy()
    if products:
        filtered = filtered[filtered["Product"].isin(products)]
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if date_range and len(date_range) == 2:
        filtered = filtered[
            (filtered["Date"].dt.date >= date_range[0])
            & (filtered["Date"].dt.date <= date_range[1])
        ]
    if age_range:
        filtered = filtered[
            (filtered["Customer_Age"] >= age_range[0])
            & (filtered["Customer_Age"] <= age_range[1])
        ]
    return filtered
```

## Radar Chart Normalization

Normalize metrics to 0–1 for radar/spider charts:

```python
for col in ["Total_Sales", "Avg_Sale", "Transactions", "Satisfaction"]:
    col_range = data[col].max() - data[col].min()
    data[col + "_norm"] = (data[col] - data[col].min()) / (col_range if col_range > 0 else 1)
```
