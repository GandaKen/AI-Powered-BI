from __future__ import annotations

from insightforge.retrieval.documents import build_documents


def test_build_documents_returns_multiple_types(sample_df) -> None:
    docs = build_documents(sample_df)
    assert len(docs) >= 5
    types = {doc.metadata.get("type") for doc in docs}
    assert "overview" in types
    assert "product" in types
    assert "region" in types
