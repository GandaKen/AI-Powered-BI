"""LangGraph wiring for the 6-node agentic pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from insightforge.agent.nodes.context_assembler import make_context_assembler_node
from insightforge.agent.nodes.generator import make_generator_node
from insightforge.agent.nodes.information_retriever import make_information_retriever_node
from insightforge.agent.nodes.query_planner import make_query_planner_node
from insightforge.agent.nodes.response_qa import make_response_qa_node
from insightforge.agent.nodes.retrieval_planner import make_retrieval_planner_node
from insightforge.agent.state import AgentState


def _route_after_response_qa(state: AgentState) -> str:
    evaluation = state.get("evaluation", {})
    return "retry" if evaluation.get("needs_retry", False) else "accept"


def build_agent_graph(
    settings,
    llm_light,
    llm_heavy,
    guardrail,
    tool_registry: dict,
):
    """Build and compile the InsightForge agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("query_planner", make_query_planner_node(llm_light, guardrail))
    workflow.add_node("retrieval_planner", make_retrieval_planner_node())
    workflow.add_node("information_retriever", make_information_retriever_node(tool_registry))
    workflow.add_node("context_assembler", make_context_assembler_node(settings))
    workflow.add_node("generator", make_generator_node(llm_heavy))
    workflow.add_node("response_qa", make_response_qa_node(llm_heavy, settings))

    workflow.add_edge(START, "query_planner")
    workflow.add_edge("query_planner", "retrieval_planner")
    workflow.add_edge("retrieval_planner", "information_retriever")
    workflow.add_edge("information_retriever", "context_assembler")
    workflow.add_edge("context_assembler", "generator")
    workflow.add_edge("generator", "response_qa")
    workflow.add_conditional_edges(
        "response_qa",
        _route_after_response_qa,
        {
            "retry": "retrieval_planner",
            "accept": END,
        },
    )

    return workflow.compile()

