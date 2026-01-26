"""
tools.py - Tool implementations for the agent.

This module defines the actual tools the agent can use. Each tool consists of:
  1. A handler function that performs the operation
  2. A register_tool() call that adds it to the registry with metadata

ADDING NEW TOOLS:
=================
To add a new tool, follow this pattern:

    def my_tool(param1: str, param2: int) -> Dict[str, Any]:
        '''Handler function - receives inputs, returns result dict.'''
        try:
            result = do_something(param1, param2)
            return {
                "tool": "my_tool",
                "input": {"param1": param1, "param2": param2},
                "output": {"result": result},
                "ok": True,
            }
        except Exception as e:
            return {
                "tool": "my_tool",
                "input": {"param1": param1, "param2": param2},
                "output": {"error": str(e)},
                "ok": False,
            }

    register_tool({
        "name": "my_tool",
        "description": "What the tool does (shown to LLM)",
        "input_schema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."},
            },
            "required": ["param1", "param2"],
        },
        "handler": my_tool,
        "cost": "low",      # low | medium | high
        "risk": "low",      # low | medium | high
        "latency_ms": 100,  # expected execution time
    })

RETURN FORMAT:
==============
All tool handlers must return a dict with this structure:
  - tool: str        - Tool name (for logging/debugging)
  - input: dict      - The inputs that were provided
  - output: dict     - Either {"result": ...} or {"error": ...}
  - ok: bool         - True if successful, False if error

This consistent format allows the TOOL node to handle results uniformly.
"""

import ast
import operator as op
from typing import Any, Dict

from agent.tool_registry import register_tool


# =============================================================================
# CALCULATOR TOOL
# =============================================================================
# A safe math expression evaluator using Python's AST (Abstract Syntax Tree).
# This avoids the security risks of eval() by only allowing specific operations.

# Mapping of AST binary operator nodes to actual Python operator functions
_ALLOWED_BINOPS = {
    ast.Add: op.add,       # +
    ast.Sub: op.sub,       # -
    ast.Mult: op.mul,      # *
    ast.Div: op.truediv,   # /
    ast.Pow: op.pow,       # **
    ast.Mod: op.mod,       # %
}

# Mapping of AST unary operator nodes to actual Python operator functions
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,      # +x (positive)
    ast.USub: op.neg,      # -x (negative)
}


def _eval_node(node: ast.AST) -> float:
    """
    Recursively evaluate an AST node to compute a numeric result.

    This is a safe alternative to eval() - it only handles:
      - Numeric constants (int, float)
      - Binary operations (+, -, *, /, **, %)
      - Unary operations (+, -)

    Args:
        node: An AST node from parsing a math expression.

    Returns:
        The computed numeric result as a float.

    Raises:
        ValueError: If the expression contains unsupported operations
                    (function calls, variables, etc.)
    """
    # Handle numeric literals: 42, 3.14, etc.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # Handle the top-level Expression wrapper
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # Handle binary operations: 2 + 3, 10 * 5, etc.
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)

    # Handle unary operations: -5, +3, etc.
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operand = _eval_node(node.operand)
        return _ALLOWED_UNARYOPS[type(node.op)](operand)

    # Reject anything else (function calls, variables, imports, etc.)
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression.

    Examples of supported expressions:
      - "2 + 2"       -> 4.0
      - "10 / 3"      -> 3.333...
      - "2 ** 10"     -> 1024.0
      - "-5 + 3"      -> -2.0
      - "(1 + 2) * 3" -> 9.0

    Args:
        expression: A string containing a math expression.

    Returns:
        Standard tool result dict with "result" (float) or "error" (str).
    """
    try:
        # Parse the expression into an AST (catches syntax errors)
        tree = ast.parse(expression, mode="eval")
        # Recursively evaluate the AST (catches unsupported operations)
        result = _eval_node(tree)
        return {
            "tool": "calculator",
            "input": {"expression": expression},
            "output": {"result": result},
            "ok": True,
        }
    except Exception as e:
        return {
            "tool": "calculator",
            "input": {"expression": expression},
            "output": {"error": str(e)},
            "ok": False,
        }


# Register the calculator tool with metadata
register_tool(
    {
        "name": "calculator",
        "description": "Evaluate a math expression and return the numeric result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression (e.g., '2 + 2', '10 / 3', '2 ** 10')",
                }
            },
            "required": ["expression"],
        },
        "handler": calculator,
        "cost": "low",       # No external resources, just CPU
        "risk": "low",       # Safe AST-based evaluation, no code execution
        "latency_ms": 5,     # Nearly instant
    }
)


# =============================================================================
# WEB LOOKUP TOOL (STUB)
# =============================================================================
# This is a placeholder for a real web search tool.
# Replace with actual implementation (e.g., Google Search API, Tavily, etc.)

def web_lookup_stub(query: str) -> Dict[str, Any]:
    """
    Stub implementation for web search.

    This is a placeholder that always returns a fixed message.
    Replace with a real implementation that calls a search API.

    Args:
        query: The search query string.

    Returns:
        Standard tool result dict (always succeeds with stub message).
    """
    return {
        "tool": "web_lookup_stub",
        "input": {"query": query},
        "output": {"result": "Stubbed web lookup: no network enabled."},
        "ok": True,
    }


# Register the web lookup stub with realistic metadata for a real web search
register_tool(
    {
        "name": "web_lookup_stub",
        "description": "Look up information on the web (stubbed; returns placeholder text).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up on the web",
                }
            },
            "required": ["query"],
        },
        "handler": web_lookup_stub,
        "cost": "high",       # Real web search would have API costs
        "risk": "high",       # External network call, potential for abuse
        "latency_ms": 2000,   # Network latency for real implementation
    }
)
