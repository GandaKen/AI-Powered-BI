"""FAISS vector store lifecycle management."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def _documents_hash(documents: list[Document]) -> str:
    """Compute deterministic hash of document contents for cache invalidation."""
    contents = sorted(d.page_content for d in documents)
    blob = "\n".join(contents).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class VectorStoreManager:
    """Thin manager around FAISS operations with hash-based persistence."""

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings
        self.vectorstore: FAISS | None = None
        self._data_hash: str | None = None

    def build(self, documents: list[Document]) -> VectorStoreManager:
        self._data_hash = _documents_hash(documents)
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self

    def save(self, path: str | Path) -> None:
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(path))
        if self._data_hash:
            (path / "data_hash.json").write_text(
                json.dumps({"hash": self._data_hash}), encoding="utf-8"
            )

    def load(self, path: str | Path, expected_hash: str | None = None) -> VectorStoreManager:
        path = Path(path)
        self.vectorstore = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        hash_file = path / "data_hash.json"
        if hash_file.exists():
            data = json.loads(hash_file.read_text(encoding="utf-8"))
            self._data_hash = data.get("hash")
            if expected_hash and self._data_hash != expected_hash:
                raise ValueError(
                    f"Index hash mismatch: expected {expected_hash}, got {self._data_hash}"
                )
        return self

    def needs_rebuild(self, path: str | Path, current_hash: str) -> bool:
        """Return True if stored index is stale (hash mismatch)."""
        path = Path(path)
        hash_file = path / "data_hash.json"
        if not hash_file.exists():
            return True
        try:
            data = json.loads(hash_file.read_text(encoding="utf-8"))
            return data.get("hash") != current_hash
        except (json.JSONDecodeError, OSError):
            return True

    def search(self, query: str, k: int = 5):
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized.")
        return self.vectorstore.similarity_search(query, k=k)

    def add_documents(self, docs: list[Document]) -> None:
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized.")
        self.vectorstore.add_documents(docs)


def dataframe_hash(df: pd.DataFrame) -> str:
    """Compute deterministic hash of DataFrame content for cache invalidation."""
    csv_bytes = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()

