"""
trace.py - Execution trace export for the agent.

This module provides functionality to save agent execution traces to JSON files
for later analysis, debugging, or audit purposes.

USAGE:
======
    from agent.trace import export_trace

    # After running the agent
    final_state = graph.invoke(initial_state)
    export_trace(final_state)  # Saves to ./runs/<run_id>.json

OUTPUT FORMAT:
==============
The exported JSON contains:
    {
        "run_id": "abc123",
        "question": "What is 2+2?",
        "events": [...],
        "rationale": [{"step": 1, "text": "...", "ts_ms": 123}, ...],
        "summary": {
            "next": "ANSWER",
            "step_count": 3,
            "tool_calls": 1,
            ...
        },
        "answer": "The result is 4.",
        "exported_at": "2024-01-15T10:30:00Z"
    }

RATIONALE HELPERS:
==================
Rationale is fully event-driven (type="rationale" events). Use these helpers:

    from agent.trace import extract_rationale, extract_rationale_text

    # Structured format
    rationales = extract_rationale(state)
    # [{"step": 1, "text": "Created plan...", "ts_ms": 123}, ...]

    # Simple text list
    texts = extract_rationale_text(state)
    # ["[1] Created plan...", "[2] Executing..."]
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_answer(state: Dict[str, Any]) -> Optional[str]:
    """Extract the answer from reasoning_steps if present."""
    steps = state.get("reasoning_steps", [])
    for step in reversed(steps):
        if "ANSWER:" in step:
            return step.split("ANSWER:", 1)[1].strip()
    return None


def extract_rationale(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Extract all rationale events from the execution trace.

    Rationale is now fully event-driven (type="rationale"). This helper
    extracts all rationale events with their step numbers and text.

    Args:
        state: The agent state containing events

    Returns:
        List of rationale dicts: [{"step": 1, "text": "...", "ts_ms": 123}, ...]
    """
    events = state.get("events", [])
    return [
        {"step": e.get("step"), "text": e.get("text", ""), "ts_ms": e.get("ts_ms", 0)}
        for e in events
        if e.get("type") == "rationale"
    ]


def extract_rationale_text(state: Dict[str, Any]) -> list[str]:
    """
    Extract rationale text strings from events (simple list format).

    Args:
        state: The agent state containing events

    Returns:
        List of rationale strings: ["Step 1: ...", "Step 2: ...", ...]
    """
    rationales = extract_rationale(state)
    return [f"[{r['step']}] {r['text']}" for r in rationales]


def _build_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a summary of the final state."""
    return {
        "next": state.get("next"),
        "step_count": state.get("step_count", 0),
        "retrieve_count": state.get("retrieve_count", 0),
        "think_count": state.get("think_count", 0),
        "tool_calls": state.get("tool_calls", 0),
        "tool_call_cap": state.get("tool_call_cap", 0),
        "tool_latency_ms": state.get("tool_latency_ms", 0),
        "tool_latency_cap_ms": state.get("tool_latency_cap_ms", 0),
        "tool_fail_count": state.get("tool_fail_count", 0),
        "last_error": state.get("last_error"),
        "max_steps": state.get("max_steps", 0),
        "max_tool_risk": state.get("max_tool_risk"),
        "knowledge_count": len(state.get("knowledge", [])),
        "tool_results_count": len(state.get("tool_results", [])),
    }


def export_trace(
    state: Dict[str, Any],
    output_dir: str = "./runs",
    filename: Optional[str] = None,
) -> str:
    """
    Export the execution trace to a JSON file.

    Args:
        state: The final state from the agent run
        output_dir: Directory to save traces (default: ./runs)
        filename: Optional custom filename (default: <run_id>.json)

    Returns:
        The path to the saved file
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build the trace document
    run_id = state.get("run_id", "unknown")
    trace = {
        "run_id": run_id,
        "question": state.get("question", ""),
        "events": state.get("events", []),
        "rationale": extract_rationale(state),  # Derived from events
        "summary": _build_summary(state),
        "answer": _extract_answer(state),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    # Determine output path
    if filename is None:
        filename = f"{run_id}.json"
    output_path = os.path.join(output_dir, filename)

    # Write JSON with pretty formatting
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    return output_path


def load_trace(filepath: str) -> Dict[str, Any]:
    """
    Load a previously exported trace from a JSON file.

    Args:
        filepath: Path to the trace JSON file

    Returns:
        The trace document as a dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)


def list_traces(output_dir: str = "./runs") -> list[str]:
    """
    List all trace files in the output directory.

    Args:
        output_dir: Directory containing traces (default: ./runs)

    Returns:
        List of trace file paths, sorted by modification time (newest first)
    """
    path = Path(output_dir)
    if not path.exists():
        return []

    traces = list(path.glob("*.json"))
    # Sort by modification time, newest first
    traces.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in traces]
