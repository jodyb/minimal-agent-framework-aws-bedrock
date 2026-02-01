"""
pretty_print.py - Execution trace formatting for the agent.

This module provides human-readable formatting for the agent's execution trace.
It converts the structured event log into a clear, step-by-step visualization
of what the agent did and why.

EXECUTION TRACE FORMAT:
=======================
The trace groups events by step number and shows timing deltas:

    Step 1:
      - LLM CALL: plan_generation (142ms, ~234→45 tokens)
      - PLAN CREATED: ['RETRIEVE', 'ANSWER']
      - WHY: Created plan because...

    Step 2: (+150ms)
      - PLAN STEP → RETRIEVE
      - RETRIEVE COMPLETE: 2 docs (5ms) ["preview...", "preview..."]
      - WHY: Executing plan step...

EVENT TYPES:
============
Each event type has a dedicated formatter in _format_event():

    llm_call         - LLM invocation with timing and token estimates
    plan_created     - Short-horizon plan generated
    plan_step        - Execution of a planned step
    plan_invalidated - Plan dropped due to changed conditions
    tool_request     - Tool call initiated
    tool_executed    - Tool execution result
    think_complete   - THINK node output
    retrieve_complete- Document retrieval results
    memory_compressed- Memory summarization
    routing          - Control flow decision
    guardrail        - Safety limit triggered
    rationale        - Human-readable decision explanation

EXTENDING:
==========
To add a new event type:
1. Emit the event in lg_nodes.py using _event() or similar
2. Add an `elif evt_type == "my_event":` branch in _format_event()
3. Return a formatted string with the relevant event data
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List

def group_events_by_step(events: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group events by their step number for structured display."""
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for evt in events:
        step = evt.get("step", -1)
        grouped[step].append(evt)
    return dict(sorted(grouped.items()))


def _infer_node(events: List[Dict[str, Any]]) -> str:
    """Infer which node executed based on event types."""
    for evt in events:
        evt_type = evt.get("type", "")
        if evt_type == "node_entry":
            return evt.get("node", "?")
        # REASON emits these
        if evt_type in ("routing", "guardrail", "plan_step", "plan_invalidated", "tool_request"):
            return "REASON"
        if evt_type == "llm_call" and evt.get("purpose") == "plan_generation":
            return "REASON"
        # THINK emits these
        if evt_type == "think_complete":
            return "THINK"
        if evt_type == "llm_call" and evt.get("purpose") in ("chain_of_thought", "tool_repair"):
            return "THINK"
        # RETRIEVE emits this
        if evt_type == "retrieve_complete":
            return "RETRIEVE"
        # TOOL emits this
        if evt_type == "tool_executed":
            return "TOOL"
    return "?"

def _short(s: str, n: int = 140) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _format_event(evt: Dict[str, Any]) -> str | None:
    """Format an event into a human-readable string, or None to skip."""
    evt_type = evt.get("type", "unknown")

    if evt_type == "plan_created":
        return None  # Skip - internal detail

    elif evt_type == "plan_step":
        plan_step = evt.get("plan_step", "?")
        tool = evt.get("tool")
        if tool:
            return f"PLAN STEP → {plan_step} ({tool})"
        return f"PLAN STEP → {plan_step}"

    elif evt_type == "plan_invalidated":
        return f"PLAN INVALIDATED: {evt.get('reason', 'unknown')}"

    elif evt_type == "routing":
        reason = evt.get("reason", "")
        next_node = evt.get("next", "?")
        return f"ROUTING → {next_node} ({reason})"

    elif evt_type == "tool_request":
        tool = evt.get("tool", "?")
        return f"TOOL REQUEST: {tool}"

    elif evt_type == "tool_executed":
        tool = evt.get("tool", "?")
        ok = evt.get("ok", False)
        status = "ok" if ok else "failed"
        error = evt.get("error")
        if error:
            return f"TOOL EXECUTED: {tool} ({status}: {_short(str(error), 50)})"
        return f"TOOL EXECUTED: {tool} ({status})"

    elif evt_type == "guardrail":
        guardrail = evt.get("guardrail", "?")
        return f"GUARDRAIL: {guardrail}"

    elif evt_type == "rationale":
        return f"WHY: {_short(evt.get('text', ''), 100)}"

    elif evt_type == "llm_call":
        purpose = evt.get("purpose", "?")
        duration = evt.get("duration_ms", 0)
        prompt_tokens = evt.get("prompt_tokens", 0)
        response_tokens = evt.get("response_tokens", 0)
        return f"LLM CALL: {purpose} ({duration}ms, ~{prompt_tokens} in, {response_tokens} out)"

    elif evt_type == "think_complete":
        mode = evt.get("mode", "?")
        preview = _short(evt.get("output_preview", ""), 60)
        return f"THINK COMPLETE ({mode}): \"{preview}\""

    elif evt_type == "retrieve_complete":
        doc_count = evt.get("doc_count", 0)
        duration = evt.get("duration_ms", 0)
        previews = evt.get("previews", [])
        preview_str = ", ".join(f"\"{_short(p, 30)}\"" for p in previews[:2]) if previews else "none"
        return f"RETRIEVE COMPLETE: {doc_count} docs ({duration}ms) [{preview_str}]"

    elif evt_type == "memory_compressed":
        old_steps = evt.get("old_steps", 0)
        kept_steps = evt.get("kept_steps", 0)
        return f"MEMORY COMPRESSED: {old_steps} steps → summary (kept {kept_steps})"

    else:
        # Fallback for unknown event types
        details = {k: v for k, v in evt.items() if k not in ("type", "step")}
        detail_str = ", ".join(f"{k}={_short(str(v), 40)}" for k, v in details.items())
        return f"{evt_type.upper()}: {detail_str}"

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

    print("\n=== EVENT LOG ===")
    events: List[Dict[str, Any]] = final_state.get("events", [])
    grouped_events = group_events_by_step(events)
    prev_ts = 0
    for step, step_events in grouped_events.items():
        # Get timestamp from first event in this step
        step_ts = step_events[0].get("ts_ms", 0) if step_events else 0
        delta = step_ts - prev_ts if prev_ts else 0
        timing_str = f" (+{delta}ms)" if delta > 0 else ""
        node = _infer_node(step_events)
        print(f"\nStep {step} [{node}]:{timing_str}")
        for evt in step_events:
            formatted = _format_event(evt)
            if formatted:
                print(f"  - {formatted}")
        prev_ts = step_ts

    # Compact summary
    outcome = final_state.get("next", "?")
    steps = final_state.get("step_count", 0)
    tool_calls = final_state.get("tool_calls", 0)
    tool_cap = final_state.get("tool_call_cap", 0)
    latency = final_state.get("tool_latency_ms", 0)
    latency_cap = final_state.get("tool_latency_cap_ms", 0)
    tool_fails = final_state.get("tool_fail_count", 0)
    retrieves = final_state.get("retrieve_count", 0)

    print(f"\n--- {outcome} | {steps} steps | tools: {tool_calls}/{tool_cap} | latency: {latency}/{latency_cap}ms | fails: {tool_fails} | retrieves: {retrieves} ---")

