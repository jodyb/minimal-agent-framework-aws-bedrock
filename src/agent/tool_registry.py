from __future__ import annotations

from typing import Any, Callable, Dict, List
from typing_extensions import TypedDict

class ToolSpec(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]

    cost: str        # low|medium|high
    risk: str        # low|medium|high
    latency_ms: int

TOOL_REGISTRY: Dict[str, ToolSpec] = {}

def register_tool(spec: ToolSpec) -> None:
    TOOL_REGISTRY[spec["name"]] = spec

def get_tool(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool not found: {name}")
    return TOOL_REGISTRY[name]

def list_tools() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for spec in TOOL_REGISTRY.values():
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
    return sorted(tools, key=lambda t: t["name"])
