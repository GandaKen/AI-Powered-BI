"""Unit tests for VectorStoreManager."""

from __future__ import annotations

from insightforge.retrieval.documents import build_documents
from insightforge.retrieval.vectorstore import (
    VectorStoreManager,
    _documents_hash,
    dataframe_hash,
)


def test_vectorstore_build(sample_df, mock_embeddings) -> None:
    """Build from docs, verify manager is initialized."""
    docs = build_documents(sample_df)
    manager = VectorStoreManager(embeddings=mock_embeddings).build(docs)
    assert manager.vectorstore is not None
    assert manager._data_hash is not None


def test_vectorstore_save_load_roundtrip(sample_df, mock_embeddings, tmp_path) -> None:
    """Save and load preserves index."""
    docs = build_documents(sample_df)
    manager = VectorStoreManager(embeddings=mock_embeddings).build(docs)
    manager.save(tmp_path)
    assert (tmp_path / "index.faiss").exists() or (tmp_path / "faiss.index").exists()
    loaded = VectorStoreManager(embeddings=mock_embeddings).load(tmp_path)
    assert loaded.vectorstore is not None


def test_documents_hash_deterministic(sample_df) -> None:
    """Hash is deterministic for same documents."""
    docs = build_documents(sample_df)
    h1 = _documents_hash(docs)
    h2 = _documents_hash(docs)
    assert h1 == h2


def test_dataframe_hash_deterministic(sample_df) -> None:
    """DataFrame hash is deterministic."""
    h1 = dataframe_hash(sample_df)
    h2 = dataframe_hash(sample_df)
    assert h1 == h2


def test_vectorstore_needs_rebuild(sample_df, mock_embeddings, tmp_path) -> None:
    """needs_rebuild returns True when hash mismatches."""
    docs = build_documents(sample_df)
    manager = VectorStoreManager(embeddings=mock_embeddings).build(docs)
    manager.save(tmp_path)
    current_hash = _documents_hash(docs)
    assert manager.needs_rebuild(tmp_path, current_hash) is False
    assert manager.needs_rebuild(tmp_path, "different_hash") is True
