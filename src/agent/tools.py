from __future__ import annotations

import ast
import operator as op
from typing import Any, Dict

from agent.tool_registry import register_tool

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}
_ALLOWED_UNARYOPS = {ast.UAdd: op.pos, ast.USub: op.neg}

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")

def calculator(expression: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        return {"tool": "calculator", "input": {"expression": expression}, "output": {"result": result}, "ok": True}
    except Exception as e:
        return {"tool": "calculator", "input": {"expression": expression}, "output": {"error": str(e)}, "ok": False}

register_tool(
    {
        "name": "calculator",
        "description": "Evaluate a math expression and return the numeric result.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "A math expression"}},
            "required": ["expression"],
        },
        "handler": calculator,
        "cost": "low",
        "risk": "low",
        "latency_ms": 5,
    }
)

def web_lookup_stub(query: str) -> Dict[str, Any]:
    return {
        "tool": "web_lookup_stub",
        "input": {"query": query},
        "output": {"result": "Stubbed web lookup: no network enabled."},
        "ok": True,
    }

register_tool(
    {
        "name": "web_lookup_stub",
        "description": "Look up information on the web (stubbed; returns placeholder text).",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
        "handler": web_lookup_stub,
        "cost": "high",
        "risk": "high",
        "latency_ms": 2000,
    }
)
