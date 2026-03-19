"""Vector search tool."""

from __future__ import annotations

from langchain_core.tools import tool


def build_vector_search_tool(vector_manager, top_k: int):
    """Create a vector search tool bound to a vector manager instance."""

    @tool("vector_search")
    def vector_search(query: str) -> str:
        """Search the FAISS index and return top matching snippets."""
        docs = vector_manager.search(query=query, k=top_k)
        return "\n\n".join(doc.page_content for doc in docs)

    return vector_search

