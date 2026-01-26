# Claude Code Instructions (Repo Guide)

Use Claude Code to:
- Add tools in `src/agent/tools.py` and register them declaratively.
- Improve retrieval in `src/agent/retrieve.py` (embeddings later if desired).
- Refactor nodes in `src/agent/lg_nodes.py` while keeping state + guardrails explicit.

## Run
- `uv run python main.py`

## Architecture
- **REASON** = control plane (policy, transitions, guardrails)
- **THINK** = cognitive artifacts + tool repair proposals
- **TOOLS** = self-describing (name/description/schema + cost/risk/latency)
- **TOOL node** executes tools and records outcomes (including semantic validation)
