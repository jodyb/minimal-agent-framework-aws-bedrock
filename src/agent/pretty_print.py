from __future__ import annotations

from typing import Any, Dict, List

def _short(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def pretty_print_run(final_state: Dict[str, Any]) -> None:
    print("\n=== FINAL STATE ===")
    print(f"next: {final_state.get('next')}")
    print(f"step_count: {final_state.get('step_count')}")
    print(f"retrieve_count: {final_state.get('retrieve_count')}")
    print(f"tool_fail_count: {final_state.get('tool_fail_count')} last_error={final_state.get('last_error')}")
    print(f"memory_summary: {_short(final_state.get('memory_summary',''))}")

    # Tool budget summary
    tool_calls = final_state.get('tool_calls', 0)
    tool_call_cap = final_state.get('tool_call_cap', 0)
    tool_latency = final_state.get('tool_latency_ms', 0)
    tool_latency_cap = final_state.get('tool_latency_cap_ms', 0)
    latency_remaining = max(0, tool_latency_cap - tool_latency)
    print(f"\n=== TOOL BUDGET ===")
    print(f"calls: {tool_calls}/{tool_call_cap}")
    print(f"latency: {tool_latency}ms used, {latency_remaining}ms remaining (cap: {tool_latency_cap}ms)")

    print("\n=== REASONING STEPS (last 12) ===")
    steps: List[str] = final_state.get("reasoning_steps", [])
    for s in steps[-12:]:
        print("-", _short(str(s), 180))
