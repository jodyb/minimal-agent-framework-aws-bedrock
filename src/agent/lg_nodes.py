"""
lg_nodes.py - Node implementations for the LangGraph agent.

This module contains all the node functions that make up the agent's behavior.
Each node is a function that takes state (LGState) and returns a dict of updates
to merge back into the state.

ARCHITECTURE OVERVIEW:
======================

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         REASON NODE                                  │
    │              (Control Plane / Policy Enforcement)                    │
    │                                                                      │
    │  - Decides which node to execute next (sets state["next"])          │
    │  - Creates and executes short-horizon plans (2-3 steps)             │
    │  - Enforces guardrails (max_steps, tool risk limits, caps)          │
    │  - Handles tool failure recovery                                     │
    │  - Fast-paths pure math expressions to calculator                    │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │  THINK NODE   │      │ RETRIEVE NODE │      │   TOOL NODE   │
    │               │      │               │      │               │
    │ Chain-of-     │      │ Fetch context │      │ Execute tools │
    │ thought       │      │ from knowledge│      │ with semantic │
    │ reasoning     │      │ base (RAG)    │      │ validation    │
    │               │      │               │      │               │
    │ Also handles  │      │               │      │               │
    │ tool repair   │      │               │      │               │
    └───────────────┘      └───────────────┘      └───────────────┘
            │                       │                       │
            ▼                       │                       │
    ┌───────────────┐               │                       │
    │  MEMORY NODE  │               │                       │
    │               │               │                       │
    │ Summarizes    │               │                       │
    │ old reasoning │               │                       │
    │ to prevent    │               │                       │
    │ overflow      │               │                       │
    └───────────────┘               │                       │
            │                       │                       │
            └───────────────────────┴───────────────────────┘
                                    │
                                    ▼
                           ┌───────────────┐
                           │  ANSWER NODE  │
                           │               │
                           │ Formulate     │
                           │ final answer  │
                           │ from evidence │
                           └───────────────┘

NODE RESPONSIBILITIES:
======================

REASON (reason_node):
  - The "control plane" that orchestrates the agent
  - Creates short-horizon plans (2-3 steps) via LLM
  - Executes plans step-by-step, invalidating when conditions change
  - Enforces all guardrails and limits
  - Handles tool failure recovery by routing to THINK for repair

THINK (think_node):
  - Generates chain-of-thought reasoning about the problem
  - In repair mode: proposes fixes for failed tool calls
  - Expands the agent's cognitive context

RETRIEVE (retrieve_node):
  - Fetches relevant knowledge from the retrieval system (RAG)
  - Adds retrieved documents to state["knowledge"]

TOOL (tool_node):
  - Executes the requested tool from state["tool_request"]
  - Performs semantic validation on tool outputs
  - Records success/failure for retry logic

MEMORY (memory_node):
  - Periodically summarizes old reasoning steps
  - Prevents context overflow on long-running tasks
  - Keeps recent steps intact for immediate context

ANSWER (answer_node):
  - Formulates the final answer from available evidence
  - Prefers tool results, falls back to knowledge, then admits uncertainty
"""

import json
from typing import Any, Dict, List

from agent.lg_state import LGState
from agent.llm import get_llm
from agent.retrieve import retrieve
from agent.tool_registry import get_tool, list_tools
import agent.tools  # noqa: F401 (ensures tools are registered at import time)

# Global LLM instance shared by all nodes
llm = get_llm()

# =============================================================================
# MEMORY NODE
# =============================================================================
def memory_node(state: LGState) -> Dict[str, Any]:
    """
    Periodically summarize old reasoning steps to prevent context overflow.

    This node runs after every THINK node (see lg_graph.py edges).
    It checks if enough steps have accumulated since the last summary,
    and if so, compresses older steps into a summary while keeping
    recent steps intact.

    Behavior:
      - If fewer than `memory_every` steps since last summary: do nothing
      - If not enough old steps to summarize: just update timestamp
      - Otherwise: summarize old steps, keep last 4, update memory_summary

    This prevents the reasoning_steps list from growing unbounded on
    long-running tasks, which would eventually exceed context limits.

    Returns:
        State updates: possibly memory_summary, last_memory_at, reasoning_steps
    """
    steps = state["reasoning_steps"]

    # Check if it's time to summarize (enough steps since last summary)
    if len(steps) - state["last_memory_at"] < state["memory_every"]:
        return {}  # Not yet time to summarize

    # Split into old steps (to summarize) and recent steps (to keep)
    # Keep the last 4 steps for immediate context
    old = steps[:-4] if len(steps) > 4 else []

    if not old:
        # Nothing old enough to summarize, just update the timestamp
        return {"last_memory_at": len(steps)}

    # Use LLM to compress old reasoning into a summary
    summary = llm.invoke("Summarize: " + "\n".join(old)).strip()

    return {
        "memory_summary": summary,         # Compressed history
        "last_memory_at": len(steps),      # Mark when we summarized
        "reasoning_steps": steps[-4:],     # Keep only recent steps
    }

# =============================================================================
# THINK NODE
# =============================================================================
def think_node(state: LGState) -> Dict[str, Any]:
    """
    Generate chain-of-thought reasoning or repair failed tool calls.

    This node has two modes of operation:

    1. REPAIR MODE (when last_error is set):
       - A tool call failed, and we need to fix it
       - Asks LLM to propose a corrected tool call
       - Stores the repair in repaired_tool_request for REASON to use

    2. NORMAL MODE (no last_error):
       - Generates step-by-step reasoning about the question
       - Expands the agent's cognitive context
       - The reasoning is added to reasoning_steps for traceability

    The THINK node is the agent's "cognitive expansion" mechanism - it
    allows the agent to reason about the problem before taking action.

    Returns:
        State updates: think_count, reasoning_steps, and possibly repaired_tool_request
    """
    # -------------------------------------------------------------------------
    # REPAIR MODE: Fix a failed tool call
    # -------------------------------------------------------------------------
    if state.get("last_error"):
        prompt = f"""
You are repairing a tool call after a failure.
Last error: {state['last_error']}
Return ONLY valid JSON:
{{"tool": "", "args": {{}}}}
"""
        raw = llm.invoke(prompt).strip()

        # Parse the repaired tool request (may fail if LLM returns invalid JSON)
        try:
            repaired = json.loads(raw)
        except Exception:
            repaired = {"tool": "", "args": {}}

        return {
            "think_count": state["think_count"] + 1,
            "reasoning_steps": state["reasoning_steps"] + [f"THINK (repair) last_error={state['last_error']}"],
            # Only set repaired_tool_request if we got a valid tool name
            "repaired_tool_request": repaired if repaired.get("tool") else None,
        }

    # -------------------------------------------------------------------------
    # NORMAL MODE: Chain-of-thought reasoning
    # -------------------------------------------------------------------------
    notes = llm.invoke("Think step-by-step: " + state["question"]).strip()

    return {
        "think_count": state["think_count"] + 1,
        "reasoning_steps": state["reasoning_steps"] + [notes],
    }

# =============================================================================
# RETRIEVE NODE
# =============================================================================
def retrieve_node(state: LGState) -> Dict[str, Any]:
    """
    Fetch relevant knowledge from the retrieval system (RAG).

    This node queries the knowledge base (see retrieve.py) with the
    user's question and adds any matching documents to state["knowledge"].

    The retrieved documents can then be used by:
      - REASON node to make better routing decisions
      - ANSWER node to formulate evidence-based responses

    Note: The retrieve() function is currently a stub using keyword matching.
    In production, replace it with a real vector store for semantic search.

    Returns:
        State updates: retrieve_count, knowledge, reasoning_steps
    """
    # Query the retrieval system for relevant documents
    docs = retrieve(state["question"], k=2)

    return {
        "retrieve_count": state["retrieve_count"] + 1,  # Track retrieval calls
        "knowledge": state["knowledge"] + docs,         # Accumulate knowledge
        "reasoning_steps": state["reasoning_steps"] + [f"RETRIEVE got {len(docs)} docs"],
    }

# =============================================================================
# TOOL NODE
# =============================================================================
def tool_node(state: LGState) -> Dict[str, Any]:
    """
    Execute the requested tool and record the outcome.

    This node:
      1. Reads the tool request from state["tool_request"]
      2. Looks up the tool in the registry
      3. Executes the tool's handler function
      4. Performs semantic validation on the output
      5. Records success/failure for retry logic

    SEMANTIC VALIDATION:
    --------------------
    Beyond just checking if the tool ran without exceptions, this node
    validates that the output makes sense. For example, for the calculator:
      - Result must be a number (not None or string)
      - Result must not be NaN or Infinity

    This catches cases where the tool "succeeded" but returned garbage.

    FAILURE HANDLING:
    -----------------
    When a tool fails:
      - last_error is set with the error message
      - tool_fail_count is incremented
      - REASON node will see this and may route to THINK for repair

    When a tool succeeds:
      - last_error is cleared (None)
      - tool_fail_count is reset to 0

    Returns:
        State updates: tool_results, reasoning_steps, tool_request (cleared),
                       last_error, tool_fail_count
    """
    # Extract the tool request (may be None or empty)
    req = state.get("tool_request") or {}
    tool_name = req.get("tool")
    args = req.get("args", {}) or {}

    # Track failure state
    last_error = None
    failed = False

    # -------------------------------------------------------------------------
    # CASE 1: No tool requested (shouldn't happen, but handle gracefully)
    # -------------------------------------------------------------------------
    if not tool_name:
        failed = True
        last_error = "No tool requested"
        result = {"tool": "(none)", "input": args, "output": {"error": last_error}, "ok": False}

    # -------------------------------------------------------------------------
    # CASE 2: Execute the requested tool
    # -------------------------------------------------------------------------
    else:
        try:
            # Look up the tool in the registry
            spec = get_tool(tool_name)
            # Execute the handler with the provided arguments
            result = spec["handler"](**args)

            # Check if the tool reported failure
            if not result.get("ok", False):
                failed = True
                last_error = result.get("output", {}).get("error", "Tool returned ok=False")
            else:
                # ---------------------------------------------------------
                # SEMANTIC VALIDATION: Verify output makes sense
                # ---------------------------------------------------------
                # For calculator: result must be a valid, finite number
                if tool_name == "calculator":
                    out = result.get("output", {}) or {}
                    val = out.get("result", None)
                    is_number = isinstance(val, (int, float))

                    if not is_number:
                        # Result is not a number (None, string, etc.)
                        failed = True
                        last_error = f"Tool output invalid: expected number, got {type(val).__name__}"
                        result["ok"] = False
                        result["output"] = {"error": last_error}
                    else:
                        # Check for special float values that indicate errors
                        import math
                        if math.isnan(val) or math.isinf(val):
                            failed = True
                            last_error = "Tool output invalid: NaN/Inf"
                            result["ok"] = False
                            result["output"] = {"error": last_error}

        except Exception as e:
            # Tool execution raised an exception
            failed = True
            last_error = str(e)
            result = {"tool": tool_name, "input": args, "output": {"error": last_error}, "ok": False}

    # -------------------------------------------------------------------------
    # Build state updates
    # -------------------------------------------------------------------------
    updates: Dict[str, Any] = {
        "tool_results": state["tool_results"] + [result],  # Append result to history
        "reasoning_steps": state["reasoning_steps"] + [f"TOOL ran: {result.get('tool')} (ok={result.get('ok')})"],
        "tool_request": None,     # Clear the request (it's been processed)
        "last_error": last_error,  # Set error message (or None if success)
    }

    # Update failure counter: increment on failure, reset on success
    updates["tool_fail_count"] = state["tool_fail_count"] + 1 if failed else 0

    # Track tool budget consumption
    updates["tool_calls"] = state["tool_calls"] + 1
    # Only add latency if we found a valid tool with a spec
    if tool_name and not failed:
        try:
            spec = get_tool(tool_name)
            updates["tool_latency_ms"] = state["tool_latency_ms"] + spec.get("latency_ms", 0)
        except Exception:
            pass  # Tool not found, skip latency tracking

    return updates

# =============================================================================
# ANSWER NODE
# =============================================================================
def answer_node(state: LGState) -> Dict[str, Any]:
    """
    Formulate the final answer from available evidence.

    This node constructs the final response by checking available
    evidence sources in priority order:

    1. TOOL RESULTS (highest priority):
       - If there's a successful calculator result, use that
       - Tool outputs are typically the most direct answers

    2. KNOWLEDGE (second priority):
       - If we have retrieved knowledge, use the most recent chunk
       - This provides context-based answers from RAG

    3. FALLBACK (when no evidence):
       - Admit that we don't have enough information
       - Better to be honest than to hallucinate

    The answer is appended to reasoning_steps for traceability.

    Returns:
        State updates: reasoning_steps (with answer), next (set to "ANSWER")
    """
    # -------------------------------------------------------------------------
    # Priority 1: Use tool results if available
    # -------------------------------------------------------------------------
    if state["tool_results"]:
        last = state["tool_results"][-1]
        # For calculator, format the numeric result
        if last.get("tool") == "calculator" and last.get("ok"):
            return {
                "reasoning_steps": state["reasoning_steps"] + [f"ANSWER: The result is {last['output']['result']}."],
                "next": "ANSWER",
            }

    # -------------------------------------------------------------------------
    # Priority 2: Use retrieved knowledge if available
    # -------------------------------------------------------------------------
    if state["knowledge"]:
        # Use the most recently retrieved document
        return {
            "reasoning_steps": state["reasoning_steps"] + [f"ANSWER: {state['knowledge'][-1]['text']}"],
            "next": "ANSWER",
        }

    # -------------------------------------------------------------------------
    # Fallback: Admit we don't have enough evidence
    # -------------------------------------------------------------------------
    return {
        "reasoning_steps": state["reasoning_steps"] + ["ANSWER: I don't have enough evidence to answer confidently."],
        "next": "ANSWER",
    }

# =============================================================================
# REASON NODE (Control Plane)
# =============================================================================
def reason_node(state: LGState) -> Dict[str, Any]:
    """
    The control plane that orchestrates the agent's behavior.

    This is the most important node - it decides what the agent does next.
    The REASON node is responsible for:

    1. FAST-PATH DETECTION:
       - Recognizes pure math expressions and routes directly to calculator
       - Skips planning overhead for simple, obvious cases

    2. PLAN CREATION (Short-Horizon Lookahead):
       - If no plan exists, asks LLM to create a 2-3 step plan
       - Plans are sequences like ["RETRIEVE", "THINK", "ANSWER"]
       - This provides structure while staying flexible

    3. PLAN INVALIDATION:
       - Detects when plans become stale (e.g., retrieval cap reached)
       - Drops invalid plans to allow re-planning

    4. PLAN EXECUTION:
       - Executes the next step in the plan
       - For TOOL steps, also selects which tool and arguments
       - Applies policy filters (risk, cost) to tool selection

    5. GUARDRAIL ENFORCEMENT:
       - Stops if max_steps reached (prevents infinite loops)
       - Stops if retrieve_cap reached without results
       - Handles tool failure recovery (routes to THINK for repair)
       - Enforces tool risk limits from max_tool_risk

    6. FALLBACK DECISION-MAKING:
       - When no plan exists, asks LLM for next action
       - Includes graceful fallbacks for JSON parse failures

    DECISION PRIORITY (when no plan):
    ----------------------------------
    1. Check guardrails (max_steps, caps)
    2. Handle tool failure recovery
    3. Check if we have enough evidence to answer
    4. Detect math expressions for calculator
    5. Ask LLM for routing decision

    Returns:
        State updates: step_count, next, possibly plan, tool_request, reasoning_steps
    """
    step_count = state["step_count"] + 1

    # =========================================================================
    # FAST-PATH: Pure math expression → calculator directly (skip planning)
    # =========================================================================
    # Detect expressions like "2+2", "10 / 5", "(3 + 4) * 2"
    # This avoids the overhead of planning for trivial calculations
    q = state["question"].strip()
    if q and all((c.isdigit() or c in " +-*/().") for c in q) and any(c.isdigit() for c in q):
        # If we already have a calculator result, check outcome
        if state["tool_results"]:
            last = state["tool_results"][-1]
            if last.get("tool") == "calculator":
                if last.get("ok"):
                    # Calculator succeeded - go to ANSWER
                    return {
                        "step_count": step_count,
                        "next": "ANSWER",
                        "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → ANSWER (calculator done)"],
                    }
                else:
                    # Calculator failed (e.g., division by zero) - stop
                    return {
                        "step_count": step_count,
                        "next": "STOP",
                        "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: calculator error"],
                    }

        # No result yet - check budget then call calculator
        tool_budget_exhausted = (
            state["tool_calls"] >= state["tool_call_cap"] or
            state["tool_latency_ms"] >= state["tool_latency_cap_ms"]
        )
        if tool_budget_exhausted:
            return {
                "step_count": step_count,
                "next": "STOP",
                "reasoning_steps": state["reasoning_steps"] + [
                    f"[step {step_count}] STOP: tool budget exhausted "
                    f"(calls={state['tool_calls']}/{state['tool_call_cap']}, "
                    f"latency={state['tool_latency_ms']}/{state['tool_latency_cap_ms']}ms)",
                    "ANSWER: I couldn't complete this request because the tool budget has been exhausted.",
                ],
            }
        return {
            "step_count": step_count,
            "next": "TOOL",
            "tool_request": {"tool": "calculator", "args": {"expression": q}},
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → TOOL (calculator)"],
        }

    # =========================================================================
    # TOOL FAILURE RECOVERY (must run before planning to avoid infinite loops)
    # =========================================================================
    # If a tool failed, we need to handle it before creating new plans.
    # Otherwise, plans get created and immediately invalidated in a loop.

    # Guardrail: Too many consecutive tool failures - give up
    if state["tool_fail_count"] >= state["tool_fail_cap"]:
        has_evidence = bool(state["knowledge"]) or bool(state["tool_results"])
        if has_evidence:
            return {
                "step_count": step_count,
                "next": "ANSWER",
                "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → ANSWER (best-effort after tool failures)"],
            }
        return {
            "step_count": step_count,
            "next": "STOP",
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: tool_fail_cap reached"],
        }

    # Recovery: If a tool failed, try the repaired version or go to THINK for repair
    if state["tool_fail_count"] > 0:
        repaired = state.get("repaired_tool_request")
        if repaired:
            # We have a repaired tool request from THINK - try it
            return {
                "step_count": step_count,
                "next": "TOOL",
                "tool_request": repaired,
                "repaired_tool_request": None,
                "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → TOOL (repaired)"],
            }
        # No repair yet - go to THINK to generate one
        return {
            "step_count": step_count,
            "next": "THINK",
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → THINK (tool failed, attempting repair)"],
        }

    # =========================================================================
    # PLANNING: Create a short-horizon plan if none exists
    # =========================================================================
    # Plans are 2-3 step sequences that provide structure to the agent's behavior.
    # They're "short-horizon" because we only plan a few steps ahead - this keeps
    # the agent flexible and able to adapt to new information.
    if not state["plan"]:
        plan_prompt = f"""
You are the REASON (control) node.

Create a SHORT plan (2–3 steps max) to answer the question.
Each step must be one of: THINK, RETRIEVE, TOOL, ANSWER.

Rules:
- Prefer RETRIEVE if you lack knowledge.
- Prefer TOOL only if computation or external action is required.
- END the plan with ANSWER if possible.
- Do NOT include STOP unless the question is clearly unanswerable.

Question:
{state["question"]}

Current context:
- have_knowledge: {bool(state["knowledge"])}
- have_tool_results: {bool(state["tool_results"])}

Return ONLY valid JSON in this shape:
{{ "plan": ["STEP1", "STEP2", "..."] }}
"""
        raw = llm.invoke(plan_prompt).strip()

        # Parse the plan from LLM response
        plan: List[str] = []
        try:
            obj = json.loads(raw)
            plan = obj.get("plan", []) or []
        except Exception:
            plan = []

        # Validate plan steps - only allow known node names
        valid = {"THINK", "RETRIEVE", "TOOL", "ANSWER"}
        plan = [p for p in plan if p in valid][:3]  # Max 3 steps

        if plan:
            print(f"[PLAN] Created plan: {plan}")
            return {
                "step_count": step_count,
                "plan": plan,
                "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] PLAN created: {plan}"],
            }

    # =========================================================================
    # PLAN INVALIDATION: Drop stale plans when reality changes
    # =========================================================================
    # Plans can become invalid when conditions change. We check for this and
    # drop the plan so the agent can re-plan with current information.
    if state["plan"]:
        # If next plan step is RETRIEVE but retrieval is no longer allowed, drop the plan.
        if state["plan"][0] == "RETRIEVE" and state["retrieve_count"] >= state["retrieve_cap"]:
            return {
                "step_count": step_count,
                "plan": [],
                "reasoning_steps": state["reasoning_steps"]
                + [f"[step {step_count}] PLAN invalidated: retrieve_cap reached"],
            }

        # If next plan step is TOOL but we're in a tool-failure recovery mode, drop plan.
        # We need to handle the failure first before continuing with the plan.
        if state["plan"][0] == "TOOL" and state["tool_fail_count"] > 0:
            return {
                "step_count": step_count,
                "plan": [],
                "reasoning_steps": state["reasoning_steps"]
                + [f"[step {step_count}] PLAN invalidated: tool failure recovery active"],
            }

        # If next plan step is TOOL but tool budget is exhausted, drop plan.
        if state["plan"][0] == "TOOL":
            tool_budget_exhausted = (
                state["tool_calls"] >= state["tool_call_cap"] or
                state["tool_latency_ms"] >= state["tool_latency_cap_ms"]
            )
            if tool_budget_exhausted:
                has_evidence = bool(state["knowledge"]) or bool(state["tool_results"])
                budget_msg = (
                    f"[step {step_count}] {'ANSWER' if has_evidence else 'STOP'}: tool budget exhausted "
                    f"(calls={state['tool_calls']}/{state['tool_call_cap']}, "
                    f"latency={state['tool_latency_ms']}/{state['tool_latency_cap_ms']}ms)"
                )
                steps = state["reasoning_steps"] + [budget_msg]
                if not has_evidence:
                    steps.append("ANSWER: I couldn't complete this request because the tool budget has been exhausted.")
                return {
                    "step_count": step_count,
                    "next": "ANSWER" if has_evidence else "STOP",
                    "plan": [],  # Invalidate plan
                    "reasoning_steps": steps,
                }

    # =========================================================================
    # PLAN EXECUTION: Take the next planned step
    # =========================================================================
    if state["plan"]:
        next_step = state["plan"][0]
        remaining = state["plan"][1:]

        # -----------------------------------------------------------------
        # TOOL step requires additional work: select which tool and args
        # -----------------------------------------------------------------
        if next_step == "TOOL":
            # Special case: pure math expression → use calculator directly
            q = state["question"].strip()
            if q and all((c.isdigit() or c in " +-*/().") for c in q) and any(c.isdigit() for c in q):
                return {
                    "step_count": step_count,
                    "next": "TOOL",
                    "plan": remaining,
                    "tool_request": {"tool": "calculator", "args": {"expression": q}},
                    "reasoning_steps": state["reasoning_steps"]
                    + [f"[step {step_count}] PLAN step → TOOL (calculator) (remaining={remaining})"],
                }

            # Get all registered tools
            tools_catalog = list_tools()

            # ---------------------------------------------------------
            # POLICY FILTER: Only show tools within allowed risk level
            # ---------------------------------------------------------
            risk_rank = {"low": 0, "medium": 1, "high": 2}
            cost_rank = {"low": 0, "medium": 1, "high": 2}
            max_allowed_risk = state.get("max_tool_risk", "medium")

            # Filter out tools that exceed the risk limit
            filtered = [
                t for t in tools_catalog
                if risk_rank.get(t.get("risk", "high"), 2) <= risk_rank.get(max_allowed_risk, 1)
            ]

            # Sort by cost (prefer cheaper), then latency (prefer faster), then name
            # This biases the LLM toward efficient tool choices
            filtered.sort(
                key=lambda t: (
                    cost_rank.get(t.get("cost", "high"), 2),
                    int(t.get("latency_ms", 10_000)),
                    t.get("name", ""),
                )
            )
            tools_catalog = filtered

            # Ask LLM to select a specific tool and arguments
            tool_prompt = f"""
You are selecting a tool for the next step.

Question:
{state["question"]}

Available tools (policy-filtered):
{tools_catalog}

Return ONLY valid JSON:
{{
  "tool": "tool_name",
  "args": {{ "arg": "value" }}
}}

Rules:
- tool must be in the available tools list.
- args must match the tool's input_schema.
"""
            raw = llm.invoke(tool_prompt).strip()
            tool_name = ""
            args = {}
            try:
                obj = json.loads(raw)
                tool_name = obj.get("tool", "") or ""
                args = obj.get("args", {}) or {}
            except Exception:
                tool_name = ""
                args = {}

            # If we can't pick a valid tool, fall back to THINK (safer than failing)
            if not tool_name:
                return {
                    "step_count": step_count,
                    "next": "THINK",
                    "plan": remaining,
                    "reasoning_steps": state["reasoning_steps"]
                    + [f"[step {step_count}] PLAN step TOOL failed → THINK (remaining={remaining})"],
                }

            return {
                "step_count": step_count,
                "next": "TOOL",
                "plan": remaining,
                "tool_request": {"tool": tool_name, "args": args},
                "reasoning_steps": state["reasoning_steps"]
                + [f"[step {step_count}] PLAN step → TOOL ({tool_name}) (remaining={remaining})"],
            }

        # -----------------------------------------------------------------
        # Non-TOOL steps (THINK, RETRIEVE, ANSWER) just transition directly
        # -----------------------------------------------------------------
        return {
            "step_count": step_count,
            "next": next_step,
            "plan": remaining,
            "reasoning_steps": state["reasoning_steps"]
            + [f"[step {step_count}] PLAN step → {next_step} (remaining={remaining})"],
        }

    # =========================================================================
    # GUARDRAILS: Check limits and handle edge cases
    # =========================================================================

    # Guardrail: Maximum steps reached (prevent infinite loops)
    if step_count >= state["max_steps"]:
        return {
            "step_count": step_count,
            "next": "STOP",
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: max_steps reached"],
        }

    # Guardrail: Retrieval cap reached but no knowledge found
    if state["retrieve_count"] >= state["retrieve_cap"] and not state["knowledge"]:
        return {
            "step_count": step_count,
            "next": "STOP",
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] STOP: retrieve_cap reached without evidence"],
        }

    # Clean up stale repaired tool request if there's no error
    if not state.get("last_error") and state.get("repaired_tool_request"):
        return {"repaired_tool_request": None}

    # =========================================================================
    # SUCCESS CHECK: If we have a successful tool result, go to ANSWER
    # =========================================================================
    if state["tool_results"]:
        last_result = state["tool_results"][-1]
        if last_result.get("ok"):
            return {
                "step_count": step_count,
                "next": "ANSWER",
                "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → ANSWER (tool succeeded)"],
            }

    # =========================================================================
    # MATH EXPRESSION CHECK (fallback - may be reached if planning failed)
    # =========================================================================
    q = state["question"].strip()
    if q and all((c.isdigit() or c in " +-*/().") for c in q) and any(c.isdigit() for c in q):
        return {
            "step_count": step_count,
            "next": "TOOL",
            "tool_request": {"tool": "calculator", "args": {"expression": q}},
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → TOOL (calculator)"],
        }

    # =========================================================================
    # LLM-BASED DECISION: Ask the LLM what to do next
    # =========================================================================
    # This is the fallback when we don't have a plan and no special cases apply.
    # The LLM chooses between THINK, RETRIEVE, TOOL, ANSWER, or STOP.

    # Get policy-filtered tools for the prompt
    tools_catalog = list_tools()
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    cost_rank = {"low": 0, "medium": 1, "high": 2}
    max_allowed_risk = state.get("max_tool_risk", "medium")

    # Filter and sort tools by policy
    filtered = [t for t in tools_catalog if risk_rank.get(t.get("risk", "high"), 2) <= risk_rank.get(max_allowed_risk, 1)]
    filtered.sort(key=lambda t: (cost_rank.get(t.get("cost", "high"), 2), int(t.get("latency_ms", 10_000)), t.get("name", "")))
    tools_catalog = filtered

    # Ask LLM for routing decision
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
        # JSON parse failed - use safe fallback
        # If we don't have knowledge and can still retrieve, try RETRIEVE; otherwise STOP
        return {
            "step_count": step_count,
            "next": "RETRIEVE" if (not state["knowledge"] and state["retrieve_count"] < state["retrieve_cap"]) else "STOP",
            "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON fallback"],
        }

    # Execute the LLM's decision
    nxt = decision.get("next", "STOP")
    update: Dict[str, Any] = {
        "step_count": step_count,
        "next": nxt,
        "reasoning_steps": state["reasoning_steps"] + [f"[step {step_count}] REASON → {nxt}"],
    }

    # If routing to TOOL, also set up the tool request
    if nxt == "TOOL":
        update["tool_request"] = {"tool": decision.get("tool", ""), "args": decision.get("args", {}) or {}}

    return update
