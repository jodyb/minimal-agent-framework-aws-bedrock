<!-- Repository-specific Copilot instructions for coding agents -->
# Copilot instructions — minimal-agent-framework

This repository is a minimal LangGraph-based agent framework focused on explicit
state, short-horizon planning, self-describing tools, and AWS Bedrock LLMs.
Use these notes to make targeted, safe changes.

1. Big picture
- Control plane: `REASON` (see [src/agent/lg_nodes.py](src/agent/lg_nodes.py#L460)) builds/executes short plans.
- Cognitive plane: `THINK` produces chain-of-thought and repair proposals.
- Action plane: `TOOL` runs registered tools (see [src/agent/tools.py](src/agent/tools.py)).
- Orchestration: graph defined in [src/agent/lg_graph.py](src/agent/lg_graph.py).

2. Key files to read before editing
- State schema: [src/agent/lg_state.py](src/agent/lg_state.py) — all nodes share `LGState`.
- Graph & routing: [src/agent/lg_graph.py](src/agent/lg_graph.py).
- Node logic and guardrails: [src/agent/lg_nodes.py](src/agent/lg_nodes.py).
- Tools registry: [src/agent/tool_registry.py](src/agent/tool_registry.py).
- Tools: add/modify in [src/agent/tools.py](src/agent/tools.py) (auto-registered on import).
- Bedrock LLM wrapper: [src/agent/llm.py](src/agent/llm.py).

3. Project-specific conventions
- Tools must `register_tool()` at module import time; the registry is a global dict
  in `tool_registry.py` and used by the REASON/TOOL flow.
- Tool handler return format MUST follow the repo standard: `{"tool","input","output","ok"}`
  (see examples in [src/agent/tools.py](src/agent/tools.py)).
- LLM-driven nodes expect the model to return strict JSON for programmatic parsing.
  When changing prompts or parsing, preserve the exact JSON shapes used in code
  (e.g. plan JSON in `REASON`, repaired tool request JSON in `THINK`).
- State is authoritative and intentionally explicit; prefer updating `LGState` when
  adding new fields rather than ad-hoc keys.

4. Workflows / commands
- Run locally: `uv run python main.py` or `python main.py` from repo root.
- Use Python 3.11+ (see `pyproject.toml`).
- Configure AWS Bedrock via environment variables: `BEDROCK_MODEL_ID`, `AWS_DEFAULT_REGION`,
  and standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

5. Extension points & patterns
- Adding a tool: implement handler + `register_tool({...})` in `src/agent/tools.py`.
  Keep `input_schema` accurate; the REASON node relies on schemas to generate valid args.
- Adding retrieval sources: modify `src/agent/retrieve.py` — currently a keyword stub.
- Modifying control flow: prefer editing node functions in `src/agent/lg_nodes.py` or
  the graph topology in `src/agent/lg_graph.py`; avoid changing the overall "REASON→THINK→TOOL" pattern.

6. Safety, guardrails, and testing hints
- Guardrails are enforced in `REASON` and include: `max_steps`, `retrieve_cap`,
  `tool_fail_cap`, `tool_call_cap`, `tool_latency_cap_ms`, and `max_tool_risk`.
  Tests or changes should respect these parameters or explicitly update them in `main.py` for experiments.
- When changing tools or validation logic, run `main.py` with representative `question` values
  (examples in `README.md`) to exercise calculator success, failure, and repair flows.

7. Prompt/JSON shapes to preserve (examples)
- Plan creation (REASON): returns JSON `{ "plan": ["THINK","RETRIEVE","ANSWER"] }`.
- Tool selection (REASON): returns `{ "tool": "calculator", "args": {"expression":"2+2"} }`.
- Repair proposal (THINK): returns `{ "tool": "", "args": {} }` — ensure parsable JSON only.

8. Dependencies & build notes
- See `pyproject.toml` — `langgraph`, `langchain-aws`, `boto3` are the primary deps.
- The code uses `langchain_aws.ChatBedrock` via the wrapper in `src/agent/llm.py`.

If anything in this file looks incomplete or ambiguous, tell me which section and I will expand
with examples, exact line references, or add a brief checklist for changes you want Copilot to follow.
