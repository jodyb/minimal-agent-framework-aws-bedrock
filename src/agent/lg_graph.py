from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.lg_state import LGState
from agent.lg_nodes import reason_node, think_node, retrieve_node, tool_node, answer_node, memory_node
import agent.tools  # noqa: F401

def build_graph():
    g = StateGraph(LGState)
    g.add_node("REASON", reason_node)
    g.add_node("THINK", think_node)
    g.add_node("RETRIEVE", retrieve_node)
    g.add_node("TOOL", tool_node)
    g.add_node("ANSWER", answer_node)
    g.add_node("MEMORY", memory_node)

    g.set_entry_point("REASON")

    g.add_conditional_edges(
        "REASON",
        lambda s: s["next"],
        {"THINK": "THINK", "RETRIEVE": "RETRIEVE", "TOOL": "TOOL", "ANSWER": "ANSWER", "STOP": END},
    )

    g.add_edge("THINK", "MEMORY")
    g.add_edge("MEMORY", "REASON")
    g.add_edge("RETRIEVE", "REASON")
    g.add_edge("TOOL", "REASON")
    g.add_edge("ANSWER", END)

    return g.compile()
