from __future__ import annotations

import sys
sys.path.insert(0, "src")

from agent.lg_graph import build_graph
from agent.pretty_print import pretty_print_run
from agent.lg_state import LGState

def main() -> None:
    question = "What is LangGraph?"  # change me

    state: LGState = {
        "question": question,
        "next": "THINK",
        "step_count": 0,
        "max_steps": 12,
        "reasoning_steps": [],
        "knowledge": [],
        "retrieve_count": 0,
        "retrieve_cap": 2,
        "tool_request": None,
        "repaired_tool_request": None,
        "tool_results": [],
        "tool_fail_count": 0,
        "tool_fail_cap": 2,
        "last_error": None,
        "think_count": 0,
        "memory_summary": "",
        "memory_every": 4,
        "last_memory_at": 0,
        "max_tool_risk": "medium",
    }

    graph = build_graph()
    final_state = graph.invoke(state)
    pretty_print_run(final_state)

if __name__ == "__main__":
    main()
