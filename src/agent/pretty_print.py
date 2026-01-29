from __future__ import annotations

from typing import Any, Dict, List

def _short(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."

def pretty_print_run(final_state: Dict[str, Any]) -> None:
    # Extract and display the answer prominently at the top
    print("\n=== ANSWER ===")
    answer = None

    # Look for answer in reasoning_steps (format: "ANSWER: ...")
    steps: List[str] = final_state.get("reasoning_steps", [])
    for step in reversed(steps):
        if "ANSWER:" in step:
            # Extract just the answer part
            answer = step.split("ANSWER:", 1)[1].strip()
            break

    if answer:
        print(answer)
    else:
        # No explicit answer - show termination reason
        next_state = final_state.get("next", "unknown")
        last_error = final_state.get("last_error")
        if next_state == "STOP" and last_error:
            print(f"[No answer - stopped due to error: {last_error}]")
        elif next_state == "STOP":
            print("[No answer - agent stopped]")
        else:
            print(f"[No answer - final state: {next_state}]")

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

    print("\n=== EVENTS ===")
    events: List[Dict[str, Any]] = final_state.get("events", [])
    for evt in events:
        evt_type = evt.get("type", "unknown")
        step = evt.get("step", "?")
        # Build a summary string from the remaining keys
        details = {k: v for k, v in evt.items() if k not in ("type", "step")}
        detail_str = ", ".join(f"{k}={_short(str(v), 40)}" for k, v in details.items())
        print(f"  [{step}] {evt_type}: {detail_str}")
