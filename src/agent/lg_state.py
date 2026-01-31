from typing import Any, Dict, List, Optional, Literal
# TypedDict lets us define a dictionary with specific keys and value types
# This gives us type checking while still using a PLAIN DICT AT RUNTIME
from typing_extensions import TypedDict


class LGState(TypedDict):
    """
    The central state object passed between all nodes in the LangGraph.

    LangGraph works like a state machine: each node reads from this state,
    performs some action, and returns updates to merge back into the state.
    The 'next' field determines which node runs next (routing).
    """

    # -------------------------------------------------------------------------
    # CORE TASK
    # -------------------------------------------------------------------------
    question: str  # The user's input question or task to solve

    # -------------------------------------------------------------------------
    # CONTROL FLOW (state machine routing)
    # -------------------------------------------------------------------------
    # Which node should execute next; this is how the graph routes between nodes
    next: Literal["THINK", "RETRIEVE", "TOOL", "ANSWER", "STOP"]
    step_count: int   # How many steps have been taken so far
    max_steps: int    # Hard limit to prevent infinite loops

    # -------------------------------------------------------------------------
    # REASONING TRACE (cognitive artifacts from THINK node)
    # -------------------------------------------------------------------------
    reasoning_steps: List[str]  # Log of the agent's chain-of-thought reasoning

    # -------------------------------------------------------------------------
    # KNOWLEDGE RETRIEVAL (RAG-style context fetching)
    # -------------------------------------------------------------------------
    knowledge: List[Dict[str, Any]]  # Retrieved chunks/documents for context
    retrieve_count: int              # How many times RETRIEVE has been called
    retrieve_cap: int                # Max retrieval calls allowed (prevents over-fetching)

    # -------------------------------------------------------------------------
    # TOOL EXECUTION
    # -------------------------------------------------------------------------
    tool_request: Optional[Dict[str, Any]]          # Tool call requested by LLM
    repaired_tool_request: Optional[Dict[str, Any]] # Fixed version if original failed validation
    tool_results: List[Dict[str, Any]]              # Outputs from executed tools

    # -------------------------------------------------------------------------
    # ERROR HANDLING & RETRY LOGIC
    # -------------------------------------------------------------------------
    tool_fail_count: int        # Consecutive failures for current tool attempt
    tool_fail_cap: int          # Max retries before abandoning the tool call
    last_error: Optional[str]   # Most recent error message (used for repair logic)

    # -------------------------------------------------------------------------
    # THINK NODE TRACKING
    # -------------------------------------------------------------------------
    think_count: int  # Number of times THINK node has run (for debugging/limits)

    # -------------------------------------------------------------------------
    # MEMORY MANAGEMENT (prevents context overflow on long tasks)
    # -------------------------------------------------------------------------
    memory_summary: str   # Compressed summary of older conversation history
    memory_every: int     # Trigger summarization every N steps
    last_memory_at: int   # Step count when last summary was generated

    # -------------------------------------------------------------------------
    # GUARDRAILS (safety constraints)
    # -------------------------------------------------------------------------
    # Maximum allowed risk level for tool execution
    # Tools self-declare their risk; agent won't run tools above this threshold
    max_tool_risk: Literal["low", "medium", "high"]

    # -------------------------------------------------------------------------
    # PLANNING (short-horizon lookahead)
    # -------------------------------------------------------------------------
    # Short execution plan (2-3 steps ahead) to guide the agent's next moves
    plan: List[str]  # e.g. ["RETRIEVE", "THINK", "ANSWER"]

    # -------------------------------------------------------------------------
    # TOOL CALL TRACKING & LIMITS
    # -------------------------------------------------------------------------
    tool_calls: int           # Total number of tool calls made this session
    tool_call_cap: int        # Max tool calls allowed (prevents runaway execution)
    tool_latency_ms: int      # Cumulative latency of all tool executions (ms) - estimated at this time
    tool_latency_cap_ms: int  # Max total latency budget (stops slow tool chains)

    # -------------------------------------------------------------------------
    # OBSERVABILITY: Structured Execution Trace
    # -------------------------------------------------------------------------
    # The events list provides a complete trace of agent execution.
    # Each event is a dict with at minimum: type, step, ts_ms
    #
    # Event types include:
    #   - llm_call: LLM invocation (purpose, duration_ms, token estimates)
    #   - plan_created: Short-horizon plan generated
    #   - plan_step: Execution of a planned step
    #   - plan_invalidated: Plan dropped due to changed conditions
    #   - tool_request: Tool call initiated
    #   - tool_executed: Tool execution result (ok, output, error)
    #   - think_complete: THINK node output (mode, preview)
    #   - retrieve_complete: Documents retrieved (count, previews)
    #   - memory_compressed: Memory summarization triggered
    #   - routing: Control flow decision
    #   - guardrail: Safety limit triggered
    #   - rationale: Human-readable explanation of a decision
    #
    # Events are emitted via helpers in lg_nodes.py: _event(), _rationale(), _llm_call()
    # Events are formatted for display by pretty_print.py
    events: List[Dict[str, Any]]

    # -------------------------------------------------------------------------
    # TRACE METADATA
    # -------------------------------------------------------------------------
    run_id: str       # Unique ID for this agent run/session
    started_at: float # Unix timestamp when run started (anchors ts_ms in events)

    