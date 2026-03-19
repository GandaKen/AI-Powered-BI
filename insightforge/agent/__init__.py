"""Agent factory entry points."""

from __future__ import annotations

from insightforge.agent.graph import build_agent_graph
from insightforge.agent.tools.data_analysis import build_data_analysis_tool
from insightforge.agent.tools.statistical import build_statistical_tool
from insightforge.agent.tools.vector_search import build_vector_search_tool
from insightforge.config import Settings
from insightforge.llm.guardrail import QueryGuardrail
from insightforge.llm.provider import get_embeddings, get_llm
from insightforge.observability.tracing import get_langfuse_callbacks
from insightforge.retrieval.documents import build_documents
from insightforge.retrieval.vectorstore import VectorStoreManager


def create_agent(df, settings: Settings):
    """Construct full agentic RAG graph bound to a dataset."""
    embeddings = get_embeddings(settings)
    docs = build_documents(df)
    vector_manager = VectorStoreManager(embeddings=embeddings).build(docs)

    tool_registry = {
        "vector_search": build_vector_search_tool(
            vector_manager=vector_manager,
            top_k=settings.rag_top_k,
        ),
        "data_analysis": build_data_analysis_tool(df),
        "statistical": build_statistical_tool(df),
    }

    llm_light = get_llm(settings=settings, tier="light")
    llm_heavy = get_llm(settings=settings, tier="heavy")
    guardrail = QueryGuardrail(llm=llm_light)

    graph = build_agent_graph(
        settings=settings,
        llm_light=llm_light,
        llm_heavy=llm_heavy,
        guardrail=guardrail,
        tool_registry=tool_registry,
    )

    callbacks = get_langfuse_callbacks(settings)
    if callbacks:
        return graph.with_config({"callbacks": callbacks})
    return graph

