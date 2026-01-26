"""
tool_registry.py - Central registry for self-describing tools.

This module implements a registry pattern where tools register themselves
with metadata that enables intelligent tool selection and guardrails.

DESIGN PHILOSOPHY:
==================
Tools are "self-describing" - each tool declares not just its interface
(name, description, input schema) but also operational characteristics:

  - **cost**: Resource/API cost (low/medium/high)
  - **risk**: Safety risk level (low/medium/high)
  - **latency_ms**: Expected execution time

This metadata allows the REASON node to make informed decisions:
  - Skip high-risk tools if state["max_tool_risk"] is "low"
  - Prefer low-latency tools when speed matters
  - Factor in cost for budget-constrained scenarios

USAGE:
======
1. Define a tool in tools.py:

    from agent.tool_registry import register_tool

    register_tool({
        "name": "calculator",
        "description": "Evaluate a math expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        },
        "handler": calculator_handler,  # function that executes the tool
        "cost": "low",
        "risk": "low",
        "latency_ms": 10,
    })

2. Tools are auto-registered when tools.py is imported (see lg_graph.py)

3. The TOOL node uses get_tool() to look up and execute tools by name
"""

from typing import Any, Callable, Dict, List
from typing_extensions import TypedDict


class ToolSpec(TypedDict):
    """
    Schema for a registered tool's specification.

    Every tool must provide all these fields when registering.
    The LLM sees name, description, and input_schema to decide when to use the tool.
    The agent uses cost, risk, and latency_ms for guardrails and optimization.
    """
    # --- Interface (shown to LLM) ---
    name: str                              # Unique identifier, e.g. "calculator"
    description: str                       # What the tool does (helps LLM decide when to use it)
    input_schema: Dict[str, Any]           # JSON Schema for expected input parameters

    # --- Implementation ---
    handler: Callable[..., Dict[str, Any]] # Function to call: handler(**inputs) -> result dict

    # --- Operational metadata (for guardrails/optimization) ---
    cost: str        # Resource cost: "low" | "medium" | "high"
    risk: str        # Safety risk:   "low" | "medium" | "high"
    latency_ms: int  # Expected execution time in milliseconds


# -----------------------------------------------------------------------------
# GLOBAL REGISTRY
# -----------------------------------------------------------------------------
# All registered tools are stored here, keyed by name.
# This dict is populated at import time when tools.py is loaded.
TOOL_REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    """
    Register a tool in the global registry.

    Call this at module load time to make a tool available to the agent.
    Duplicate registrations will overwrite the previous entry.

    Args:
        spec: Complete tool specification including handler and metadata.
    """
    TOOL_REGISTRY[spec["name"]] = spec


def get_tool(name: str) -> ToolSpec:
    """
    Look up a tool by name.

    Used by the TOOL node to retrieve the handler for execution.

    Args:
        name: The tool's registered name.

    Returns:
        The full ToolSpec including the handler function.

    Raises:
        KeyError: If no tool with that name is registered.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool not found: {name}")
    return TOOL_REGISTRY[name]


def list_tools() -> List[Dict[str, Any]]:
    """
    List all registered tools (without handlers).

    Returns tool metadata suitable for showing to the LLM in prompts.
    The handler is excluded since it's not serializable/useful for the LLM.

    Returns:
        List of tool info dicts sorted alphabetically by name.
        Each dict contains: name, description, input_schema, cost, risk, latency_ms.
    """
    tools: List[Dict[str, Any]] = []
    for spec in TOOL_REGISTRY.values():
        # Exclude 'handler' - it's a function reference, not useful for LLM
        tools.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["input_schema"],
                "cost": spec["cost"],
                "risk": spec["risk"],
                "latency_ms": spec["latency_ms"],
            }
        )
    # Sort alphabetically for consistent ordering in prompts
    return sorted(tools, key=lambda t: t["name"])
