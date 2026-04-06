"""QAEvalChain-based evaluation against ground-truth Q&A pairs.

Generates data-derived ground-truth answers from the DataFrame, runs the
agentic pipeline (or a fallback callable) against each question, then uses
LangChain's ``QAEvalChain`` to grade predictions vs. expected answers.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd
from langchain.evaluation.qa import QAEvalChain

logger = logging.getLogger(__name__)


def build_eval_qa_pairs(df: pd.DataFrame) -> list[dict[str, str]]:
    """Build ground-truth QA pairs dynamically from the DataFrame.

    Every expected answer is computed directly from ``df`` so that the
    evaluation stays in sync with the actual data — no hard-coded numbers.
    """
    total_sales = df["Sales"].sum()
    avg_sale = df["Sales"].mean()
    product_sales = df.groupby("Product")["Sales"].sum()
    best_product = product_sales.idxmax()
    worst_product = product_sales.idxmin()
    region_sales = df.groupby("Region")["Sales"].sum()
    best_region = region_sales.idxmax()
    avg_satisfaction = df["Customer_Satisfaction"].mean()
    products = ", ".join(sorted(df["Product"].unique()))
    regions = ", ".join(sorted(df["Region"].unique()))
    avg_age = df["Customer_Age"].mean()
    n_transactions = len(df)

    product_satisfaction = df.groupby("Product")["Customer_Satisfaction"].mean()
    best_satisfaction_product = product_satisfaction.idxmax()

    region_satisfaction = df.groupby("Region")["Customer_Satisfaction"].mean()
    best_satisfaction_region = region_satisfaction.idxmax()

    gender_avg_sales = df.groupby("Customer_Gender")["Sales"].mean()
    higher_spending_gender = gender_avg_sales.idxmax()

    return [
        {
            "question": "How many transactions are in the dataset?",
            "answer": f"There are {n_transactions:,} transactions in the dataset.",
        },
        {
            "question": "What is the total sales revenue in the dataset?",
            "answer": f"The total sales revenue is approximately ${total_sales:,.0f}.",
        },
        {
            "question": "What is the average transaction amount?",
            "answer": f"The average transaction amount is approximately ${avg_sale:,.2f}.",
        },
        {
            "question": "What products are available in the dataset?",
            "answer": f"The products available are: {products}.",
        },
        {
            "question": "What regions does the data cover?",
            "answer": f"The data covers the following regions: {regions}.",
        },
        {
            "question": "Which product has the highest total sales?",
            "answer": (
                f"{best_product} has the highest total sales "
                f"with ${product_sales[best_product]:,.0f}."
            ),
        },
        {
            "question": "Which product has the lowest total sales?",
            "answer": (
                f"{worst_product} has the lowest total sales "
                f"with ${product_sales[worst_product]:,.0f}."
            ),
        },
        {
            "question": "Which region generates the most revenue?",
            "answer": (
                f"The {best_region} region generates the most revenue "
                f"with total sales of ${region_sales[best_region]:,.0f}."
            ),
        },
        {
            "question": "What is the average customer satisfaction rating?",
            "answer": (
                f"The average customer satisfaction rating is "
                f"{avg_satisfaction:.2f} out of 5."
            ),
        },
        {
            "question": "Which product has the highest customer satisfaction rating?",
            "answer": (
                f"{best_satisfaction_product} has the highest average customer "
                f"satisfaction rating at {product_satisfaction[best_satisfaction_product]:.2f}/5."
            ),
        },
        {
            "question": "Which region has the highest customer satisfaction?",
            "answer": (
                f"The {best_satisfaction_region} region has the highest average customer "
                f"satisfaction at {region_satisfaction[best_satisfaction_region]:.2f}/5."
            ),
        },
        {
            "question": "What is the average age of customers?",
            "answer": f"The average customer age is approximately {avg_age:.1f} years.",
        },
        {
            "question": "Do male or female customers spend more on average?",
            "answer": (
                f"{higher_spending_gender} customers spend more on average "
                f"at ${gender_avg_sales[higher_spending_gender]:,.2f} per transaction."
            ),
        },
    ]


def run_qa_evaluation(
    qa_pairs: list[dict[str, str]],
    predict_fn: Callable[[str], str],
    eval_llm,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run predictions and grade them with ``QAEvalChain``.

    Parameters
    ----------
    qa_pairs:
        List of ``{"question": ..., "answer": ...}`` ground-truth dicts.
    predict_fn:
        Callable that takes a question string and returns the model's answer.
    eval_llm:
        LangChain LLM to power the ``QAEvalChain`` grading.
    progress_callback:
        Optional ``(current_index, total, question)`` callback for UI updates.

    Returns
    -------
    dict with keys:
        ``predictions`` — list of per-question result dicts,
        ``graded_outputs`` — raw ``QAEvalChain`` grades,
        ``correct`` — count of CORRECT grades,
        ``total`` — number of QA pairs,
        ``accuracy`` — percentage score.
    """
    predictions: list[dict[str, str]] = []
    for idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        if progress_callback:
            progress_callback(idx, len(qa_pairs), question)
        try:
            result = predict_fn(question)
        except Exception as exc:
            logger.warning("Prediction failed for %r: %s", question, exc)
            result = f"Error: {exc}"
        predictions.append({
            "question": question,
            "answer": qa["answer"],
            "result": result,
        })

    eval_chain = QAEvalChain.from_llm(eval_llm)
    graded_outputs = eval_chain.evaluate(
        qa_pairs,
        predictions,
        question_key="question",
        answer_key="answer",
        prediction_key="result",
    )

    correct = 0
    enriched: list[dict[str, Any]] = []
    for qa, pred, grade in zip(qa_pairs, predictions, graded_outputs):
        grade_text = str(grade.get("results", grade.get("text", "N/A")))
        is_correct = "CORRECT" in grade_text.upper()
        correct += int(is_correct)
        enriched.append({
            "question": qa["question"],
            "expected": qa["answer"],
            "predicted": pred["result"],
            "grade": grade_text.strip(),
            "is_correct": is_correct,
        })

    total = len(qa_pairs)
    return {
        "predictions": enriched,
        "graded_outputs": graded_outputs,
        "correct": correct,
        "total": total,
        "accuracy": (correct / total * 100) if total else 0.0,
    }
