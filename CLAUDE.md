# Claude Code Instructions (Repo Guide)

Use Claude Code to:
- Add tools in `src/agent/tools.py` and register them declaratively.
- Improve retrieval in `src/agent/retrieve.py` (embeddings later if desired).
- Refactor nodes in `src/agent/lg_nodes.py` while keeping state + guardrails explicit.
- Add new event types in `lg_nodes.py` and formatters in `pretty_print.py`.

## Run
- `uv run python main.py`

## Architecture
- **REASON** = control plane (policy, transitions, guardrails)
- **THINK** = cognitive artifacts + tool repair proposals
- **TOOLS** = self-describing (name/description/schema + cost/risk/latency)
- **TOOL node** executes tools and records outcomes (including semantic validation)

## Observability Pattern

The agent uses a structured event system for traceability:

### Event Emission (lg_nodes.py)
- `_event(state, type="...", step=N, ...)` - emit a structured event with timestamp
- `_rationale(state, step, "why", events)` - emit a rationale event (type="rationale"), chains with existing events
- `_llm_call(state, "purpose", prompt, step)` - wrap LLM calls to track timing/tokens

### Event Formatting (pretty_print.py)
- `_format_event(evt)` - convert event dict to human-readable string
- `group_events_by_step(events)` - organize events by step number
- Add new `elif evt_type == "..."` branches for new event types

### Adding New Events
1. Emit in node: `"events": _event(state, type="my_event", step=step_count, data=...)`
2. Format in pretty_print: `elif evt_type == "my_event": return f"MY EVENT: {evt.get('data')}"`

### Trace Export (trace.py)
- `export_trace(state)` - saves run to `./runs/<run_id>.json`
- `load_trace(path)` - loads a saved trace
- `list_traces()` - lists all saved traces (newest first)
