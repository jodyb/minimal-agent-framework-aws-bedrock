from __future__ import annotations

import json
from typing import Any, Dict, List

from agent.lg_state import LGState
from agent.llm import get_llm
from agent.retrieve import retrieve
from agent.tool_registry import get_tool, list_tools
import agent.tools  # noqa: F401 (register tools)

llm = get_llm()

def memory_node(state: LGState) -> Dict[str, Any]:
    steps = state["reasoning_steps"]
    if len(steps) - state["last_memory_at"] < state["memory_every"]:
        return {}
    old = steps[:-4] if len(steps) > 4 else []
    if not old:
        return {"last_memory_at": len(steps)}
    summary = llm.invoke("Summarize: " + "\n".join(old)).strip()
    return {"memory_summary": summary, "last_memory_at": len(steps), "reasoning_steps": steps[-4:]}

def think_node(state: LGState) -> Dict[str, Any]:
    if state.get("last_error"):
        prompt = f"""
You are repairing a tool call after a failure.
Last error: {state['last_error']}
Return ONLY valid JSON:
{{"tool": "", "args": {{}}}}
"""
        raw = llm.invoke(prompt).strip()
        try:
            repaired = json.loads(raw)
        except Exception:
            repaired = {"tool": "", "args": {}}
        return {
            "think_count": state["think_count"] + 1,
            "reasoning_steps": state["reasoning_steps"] + [f"THINK (repair) last_error={state['last_error']}"],
            "repaired_tool_request": repaired if repaired.get("tool") else None,
        }

    notes = llm.invoke("Think step-by-step: " + state["question"]).strip()
    return {"think_count": state["think_count"] + 1, "reasoning_steps": state["reasoning_steps"] + [notes]}

def retrieve_node(state: LGState) -> Dict[str, Any]:
    docs = retrieve(state["question"], k=2)
    return {
        "retrieve_count": state["retrieve_count"] + 1,
        "knowledge": state["knowledge"] + docs,
        "reasoning_steps": state["reasoning_steps"] + [f"RETRIEVE got {len(docs)} docs"],
    }

def tool_node(state: LGState) -> Dict[str, Any]:
    req = state.get("tool_request") or {}
    tool_name = req.get("tool")
    args = req.get("args", {}) or {}

    last_error = None
    failed = False

    if not tool_name:
        failed = True
        last_error = "No tool requested"
        result = {"tool": "(none)", "input": args, "output": {"error": last_error}, "ok": False}
    else:
        try:
            spec = get_tool(tool_name)
            result = spec["handler"](**args)

            if not result.get("ok", False):
                failed = True
                last_error = result.get("output", {}).get("error", "Tool returned ok=False")
            else:
                if tool_name == "calculator":
                    out = result.get("output", {}) or {}
                    val = out.get("result", None)
                    is_number = isinstance(val, (int, float))
                    if not is_number:
                        failed = True
                        last_error = f"Tool output invalid: expected number, got {type(val).__name__}"
                        result["ok"] = False
                        result["output"] = {"error": last_error}
                    else:
                        import math
                        if math.isnan(val) or math.isinf(val):
                            failed = True
                            last_error = "Tool output invalid: NaN/Inf"
                            result["ok"] = False
                            result["output"] = {"error": last_error}
        except Exception as e:
            failed = True
            last_error = str(e)
            result = {"tool": tool_name, "input": args, "output": {"error": last_error}, "ok": False}

    updates: Dict[str, Any] = {
        "tool_results": state["tool_results"] + [result],
        "reasoning_steps": state["reasoning_steps"] + [f"TOOL ran: {result.get('tool')} (ok={result.get('ok')})"],
        "tool_request": None,
        "last_error": last_error,
    }
    updates["tool_fail_count"] = state["tool_fail_count"] + 1 if failed else 0
    return updates

def answer_node(state: LGState) -> Dict[str, Any]:
    if state["tool_results"]:
        last = state["tool_results"][-1]
        if last.get("tool") == "calculator" and last.get("ok"):
            return {"reasoning_steps": state["reasoning_steps"] + [f"ANSWER: The result is {last['output']['result']}."], "next": "ANSWER"}
    if state["knowledge"]:
        return {"reasoning_steps": state["reasoning_steps"] + [f"ANSWER: {state['knowledge'][-1]['text']}"], "next": "ANSWER"}
    return {"reasoning_steps": state["reasoning_steps"] + ["ANSWER: I don't have enough evidence to answer confidently."], "next": "ANSWER"}

def reason_node(state: LGState) -> Dict[str, Any]:
    step_count = state["step_count"] + 1

    if step_count >= state["max_steps"]:
        return {"step_count": step_count, "next": "STOP", "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: max_steps reached"]}

    if state["retrieve_count"] >= state["retrieve_cap"] and not state["knowledge"]:
        return {"step_count": step_count, "next": "STOP", "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: retrieve_cap reached without evidence"]}

    if not state.get("last_error") and state.get("repaired_tool_request"):
        return {"repaired_tool_request": None}

    if state["tool_fail_count"] >= state["tool_fail_cap"]:
        has_evidence = bool(state["knowledge"]) or bool(state["tool_results"])
        if has_evidence:
            return {"step_count": step_count, "next": "ANSWER", "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → ANSWER (best-effort)"]}
        return {"step_count": step_count, "next": "STOP", "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: tool_fail_cap reached"]}

    if state["tool_fail_count"] > 0:
        repaired = state.get("repaired_tool_request")
        if repaired:
            return {"step_count": step_count, "next": "TOOL", "tool_request": repaired, "repaired_tool_request": None,
                    "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → TOOL (repaired)"]}
        return {"step_count": step_count, "next": "THINK", "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → THINK (tool failed)"]}

    # If we already have a successful tool result, go to ANSWER
    if state["tool_results"]:
        last_result = state["tool_results"][-1]
        if last_result.get("ok"):
            return {"step_count": step_count, "next": "ANSWER",
                    "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → ANSWER (tool succeeded)"]}

    q = state["question"].strip()
    if q and all((c.isdigit() or c in " +-*/().") for c in q) and any(c.isdigit() for c in q):
        return {"step_count": step_count, "next": "TOOL", "tool_request": {"tool": "calculator", "args": {"expression": q}},
                "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → TOOL (calculator)"]}

    tools_catalog = list_tools()
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    cost_rank = {"low": 0, "medium": 1, "high": 2}
    max_allowed_risk = state.get("max_tool_risk", "medium")

    filtered = [t for t in tools_catalog if risk_rank.get(t.get("risk", "high"), 2) <= risk_rank.get(max_allowed_risk, 1)]
    filtered.sort(key=lambda t: (cost_rank.get(t.get("cost", "high"), 2), int(t.get("latency_ms", 10_000)), t.get("name", "")))
    tools_catalog = filtered

    prompt = f"""
You are the REASON node.
question: {state['question']}
have_knowledge: {bool(state['knowledge'])}
have_tool_results: {bool(state['tool_results'])}
Available tools:
{tools_catalog}

Return ONLY valid JSON in exactly this shape:
{{"next":"THINK|RETRIEVE|TOOL|ANSWER|STOP","tool":"","args":{{}}}}
"""
    raw = llm.invoke(prompt).strip()
    try:
        decision = json.loads(raw)
    except Exception:
        return {"step_count": step_count, "next": "RETRIEVE" if (not state["knowledge"] and state["retrieve_count"] < state["retrieve_cap"]) else "STOP",
                "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON fallback"]}

    nxt = decision.get("next", "STOP")
    update: Dict[str, Any] = {"step_count": step_count, "next": nxt, "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → {nxt}"]}
    if nxt == "TOOL":
        update["tool_request"] = {"tool": decision.get("tool", ""), "args": decision.get("args", {}) or {}}
    return update
