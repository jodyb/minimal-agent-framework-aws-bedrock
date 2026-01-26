"""
lg_graph.py - Builds the LangGraph state machine for the agent.

This file defines the graph topology: which nodes exist, how they connect,
and what determines the transitions between them.

GRAPH STRUCTURE OVERVIEW:
=========================

                         ┌─────────────────────────────────────────┐
                         │                                         │
                         ▼                                         │
    ┌──────────┐    ┌────────┐    ┌────────┐    ┌────────┐        │
    │  START   │───▶│ REASON │───▶│ THINK  │───▶│ MEMORY │────────┘
    └──────────┘    └────────┘    └────────┘    └────────┘
                         │
                         │  (conditional routing based on state["next"])
                         │
                         ├───▶ RETRIEVE ───┐
                         │                 │
                         ├───▶ TOOL ───────┤
                         │                 │
                         ├───▶ ANSWER ───▶ END
                         │
                         └───▶ STOP ─────▶ END

KEY CONCEPTS:
- REASON is the "control plane" - it decides what to do next
- All action nodes (THINK, RETRIEVE, TOOL) loop back to REASON
- REASON sets state["next"] to route to the appropriate node
- ANSWER and STOP both terminate the graph (reach END)
"""

from langgraph.graph import END, StateGraph

from agent.lg_state import LGState
from agent.lg_nodes import (
    reason_node,    # Control plane: decides next action
    think_node,     # Cognitive processing: chain-of-thought reasoning, expands cognitive context
    retrieve_node,  # Knowledge fetching: RAG-style context retrieval
    tool_node,      # Tool execution: runs external tools/functions
    answer_node,    # Final output: formats and returns the answer
    memory_node,    # Memory management: summarizes history to prevent overflow
)
# Import tools module to register all available tools via decorators
# The noqa comment suppresses "unused import" warnings since tools self-register
import agent.tools  # noqa: F401


def build_graph():
    """
    Constructs and compiles the LangGraph state machine.

    Returns:
        CompiledGraph: A runnable graph that accepts LGState and returns final LGState.
                       Call graph.invoke(initial_state) to run the agent.
    """
    # Create a new StateGraph using our typed state definition
    # LGState defines all the fields that will be passed between nodes
    g = StateGraph(LGState)

    # -------------------------------------------------------------------------
    # REGISTER NODES
    # -------------------------------------------------------------------------
    # Each node is a function: (state: LGState) -> dict of state updates
    # The returned dict is merged into the current state

    g.add_node("REASON", reason_node)    # Control plane - sets state["next"]
    g.add_node("THINK", think_node)      # Chain-of-thought reasoning
    g.add_node("RETRIEVE", retrieve_node)  # Fetch knowledge/context
    g.add_node("TOOL", tool_node)        # Execute tool calls
    g.add_node("ANSWER", answer_node)    # Generate final answer
    g.add_node("MEMORY", memory_node)    # Summarize history periodically

    # -------------------------------------------------------------------------
    # SET ENTRY POINT
    # -------------------------------------------------------------------------
    # The graph always starts at REASON, which evaluates the initial state
    # and decides whether to THINK, RETRIEVE, use a TOOL, or ANSWER
    g.set_entry_point("REASON")

    # -------------------------------------------------------------------------
    # CONDITIONAL EDGES (from REASON)
    # -------------------------------------------------------------------------
    # REASON is the router - it sets state["next"] to one of these values,
    # and the graph follows the corresponding edge.
    #
    # The lambda reads state["next"] and returns the routing key.
    # The dict maps routing keys to destination node names.
    g.add_conditional_edges(
        "REASON",                          # Source node
        lambda s: s["next"],               # Routing function: reads state["next"]
        {
            "THINK": "THINK",              # Go think/reason about the problem
            "RETRIEVE": "RETRIEVE",        # Go fetch knowledge
            "TOOL": "TOOL",                # Go execute a tool
            "ANSWER": "ANSWER",            # Go generate final answer
            "STOP": END,                   # Emergency stop (max steps, errors, etc.)
        },
    )

    # -------------------------------------------------------------------------
    # FIXED EDGES (unconditional transitions)
    # -------------------------------------------------------------------------
    # After THINK completes, always check if memory needs summarization
    g.add_edge("THINK", "MEMORY")

    # After MEMORY, return to REASON for next decision
    g.add_edge("MEMORY", "REASON")

    # After RETRIEVE, return to REASON to process the new knowledge
    g.add_edge("RETRIEVE", "REASON")

    # After TOOL execution, return to REASON to evaluate results
    g.add_edge("TOOL", "REASON")

    # ANSWER is terminal - once we have an answer, we're done
    g.add_edge("ANSWER", END)

    # -------------------------------------------------------------------------
    # COMPILE AND RETURN
    # -------------------------------------------------------------------------
    # compile() validates the graph and returns a runnable object
    # The compiled graph can be invoked with: graph.invoke(initial_state)
    return g.compile()
