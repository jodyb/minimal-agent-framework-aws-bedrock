import sys
import time
import uuid


# Add "src" directory to the module search path so we can import from src/agent/
sys.path.insert(0, "src")

# Import the graph builder, output formatter, trace export, and state type definition
from agent.lg_graph import build_graph
from agent.pretty_print import pretty_print_run
from agent.trace import export_trace
from agent.lg_state import LGState

def main() -> None:
    # The question/task for the agent to solve (modify this to test different inputs)
    question = "What is an agent?"  # change me

    # Initialize the agent's state dictionary with all required fields
    state: LGState = {
        # --- Core task ---
        "question": question,           # The user's input question or task
        "next": "THINK",                # Which node to execute next (starts with THINK)

        # --- Step limits (prevents infinite loops) ---
        "step_count": 0,                # Current number of steps taken
        "max_steps": 12,                # Maximum allowed steps before forced termination

        # --- Reasoning trace (cognitive artifacts) ---
        "reasoning_steps": [],          # Log of reasoning steps taken by the agent

        # --- Knowledge retrieval ---
        "knowledge": [],                # Retrieved knowledge/context chunks
        "retrieve_count": 0,            # Number of retrieval operations performed
        "retrieve_cap": 1,              # Maximum retrieval operations allowed

        # --- Tool execution ---
        "tool_request": None,           # Current tool call request from the LLM
        "repaired_tool_request": None,  # Tool request after repair (if validation failed)
        "tool_results": [],             # Results from executed tools
        "tool_fail_count": 0,           # Number of consecutive tool failures
        "tool_fail_cap": 4,             # Max failures before giving up on a tool

        # --- Tool budget ---
        "tool_calls": 0,
        "tool_call_cap": 5,

        "tool_latency_ms": 0,
        "tool_latency_cap_ms": 10,

        # --- Error handling ---
        "last_error": None,             # Most recent error message (for debugging/repair)

        # --- THINK node tracking ---
        "think_count": 0,               # Number of times THINK node has been invoked

        # --- Memory/summarization (for long-running tasks) ---
        "memory_summary": "",           # Compressed summary of prior conversation
        "memory_every": 4,              # Summarize memory every N steps
        "last_memory_at": 0,            # Step count when last memory summary was made

        # --- Guardrails ---
        "max_tool_risk": "medium",      # Maximum risk level for tools (low/medium/high)

        # --- Planning ---
        "plan": [],                      # List of planned steps (if using planning mode)

        # --- Observability: control-plane decision events ---
        "events": [],                    # Log of key decision events for auditing/debugging

        # --- Trace metadata ---
        "run_id": str(uuid.uuid4()),    # Unique ID for this agent run/session
        "started_at": time.time(),      # Timestamp when the run started
    }

    # Build the LangGraph state machine (nodes + edges)
    graph = build_graph()
    # Run the graph starting from initial state; returns final state after completion
    final_state = graph.invoke(state)
    # Display a formatted summary of the agent's execution
    pretty_print_run(final_state)

    # Export the trace to JSON for later analysis
    trace_path = export_trace(final_state)
    print(f"\n[Trace exported to {trace_path}]")

# Standard Python idiom: only run main() if this file is executed directly
if __name__ == "__main__":
    main()
