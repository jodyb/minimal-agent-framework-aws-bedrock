from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from typing_extensions import TypedDict

class LGState(TypedDict):
    question: str

    next: Literal["THINK", "RETRIEVE", "TOOL", "ANSWER", "STOP"]
    step_count: int
    max_steps: int

    reasoning_steps: List[str]

    knowledge: List[Dict[str, Any]]
    retrieve_count: int
    retrieve_cap: int

    tool_request: Optional[Dict[str, Any]]
    repaired_tool_request: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    tool_fail_count: int
    tool_fail_cap: int
    last_error: Optional[str]

    think_count: int

    memory_summary: str
    memory_every: int
    last_memory_at: int

    max_tool_risk: Literal["low", "medium", "high"]
