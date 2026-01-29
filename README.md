# Minimal Agent Framework

A lightweight agent framework built with LangGraph. Demonstrates core agent patterns including explicit state management, tool orchestration, guardrails, and policy-driven decision making.

## Features

- **Explicit State** - TypedDict-based state with full visibility into agent reasoning
- **LangGraph Integration** - Externalized control flow via graph-based orchestration
- **REASON/THINK Separation** - Control plane (policy, transitions) vs cognitive artifacts (reasoning notes)
- **Self-Describing Tools** - MCP-style registry with cost/risk/latency metadata
- **Guardrails** - Step limits, retrieval caps, tool failure tracking, tool budget limits
- **Policy-Driven Tool Selection** - Runtime control via `max_tool_risk` parameter
- **Tool Repair** - Automatic retry with LLM-guided repair on failures
- **Decision Rationale** - Human-readable explanations for why the agent made each decision
- **Observability Events** - Structured event log for auditing and debugging

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   REASON    │────▶│    THINK    │────▶│    TOOL     │
│  (control)  │◀────│  (cognition)│◀────│  (execute)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       ▼                                       ▼
┌─────────────┐                         ┌─────────────┐
│  RETRIEVE   │                         │   ANSWER    │
└─────────────┘                         └─────────────┘
```

- **REASON** - Decides next action based on state, knowledge, and policy; records decision rationale
- **THINK** - Generates reasoning notes; handles tool repair proposals
- **TOOL** - Executes tools and validates outputs; tracks budget consumption
- **RETRIEVE** - Fetches knowledge (stub; extensible for RAG)
- **ANSWER** - Produces final response

### Observability

The agent maintains two observability streams:

- **`events`** - Structured log of what happened (tool calls, plan creation, guardrail triggers)
- **`decision_rationale`** - Human-readable explanations of why each decision was made, including alternatives considered and constraints applied

## Project Structure

```
├── main.py                 # Entry point
├── pyproject.toml          # Dependencies (uv/pip)
├── src/
│   └── agent/
│       ├── lg_graph.py     # LangGraph definition
│       ├── lg_nodes.py     # Node implementations
│       ├── lg_state.py     # State schema (TypedDict)
│       ├── llm.py          # LLM interface (stub by default)
│       ├── tools.py        # Tool implementations
│       ├── tool_registry.py# Self-describing tool registry
│       ├── retrieve.py     # Knowledge retrieval
│       └── pretty_print.py # Output formatting
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- AWS account with Bedrock access enabled
- AWS CLI configured with valid credentials

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/minimal-agent-framework.git
cd minimal-agent-framework

# Create virtual environment and install dependencies
uv venv
uv sync
```

### Running

```bash
uv run python main.py
```

Edit the `question` variable in `main.py` to test different inputs:

```python
question = "12*(3+4)"           # Calculator tool
question = "What is LangGraph?" # Retrieval + answer
question = "1/0"                # Tool failure → repair → retry
```

## AWS Bedrock Setup

This project uses AWS Bedrock for LLM inference. You need valid AWS credentials with Bedrock access.

### Configure AWS Credentials

Choose one method:

```bash
# Option A: AWS CLI (recommended)
aws configure

# Option B: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### Model Configuration

The default model is Claude 3 Haiku. Override via environment variables:

```bash
export BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
export AWS_DEFAULT_REGION=us-west-2
```

Available models:
- `anthropic.claude-3-haiku-20240307-v1:0` (default, fast/cheap)
- `anthropic.claude-3-sonnet-20240229-v1:0` (balanced)
- `anthropic.claude-3-opus-20240229-v1:0` (most capable)

> **Note:** Never commit AWS credentials. Use environment variables or AWS CLI profiles.

## Extending

### Adding Tools

Add new tools in `src/agent/tools.py`:

```python
from agent.tool_registry import register_tool

def my_tool(arg: str) -> dict:
    return {"tool": "my_tool", "input": {"arg": arg}, "output": {"result": "..."}, "ok": True}

register_tool({
    "name": "my_tool",
    "description": "Does something useful.",
    "input_schema": {
        "type": "object",
        "properties": {"arg": {"type": "string"}},
        "required": ["arg"],
    },
    "handler": my_tool,
    "cost": "low",      # low | medium | high
    "risk": "low",      # low | medium | high
    "latency_ms": 100,
})
```

### Adding Retrieval Sources

Extend `src/agent/retrieve.py` with embeddings, vector stores, or external APIs.

## Configuration

Key state parameters in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 12 | Maximum reasoning steps |
| `retrieve_cap` | 2 | Maximum retrieval attempts |
| `tool_fail_cap` | 4 | Tool failures before giving up |
| `tool_call_cap` | 5 | Maximum total tool calls allowed |
| `tool_latency_cap_ms` | 10 | Maximum cumulative tool latency (ms) |
| `max_tool_risk` | "medium" | Maximum allowed tool risk level |
| `memory_every` | 4 | Steps between memory summarization |

## Example Output

Running `uv run python main.py` with a knowledge question:

```
=== ANSWER ===
A practical agent is a goal-directed state machine where an LLM helps choose transitions and code enforces guardrails.

=== FINAL STATE ===
next: ANSWER
step_count: 3
retrieve_count: 1
tool_fail_count: 0 last_error=None

=== TOOL BUDGET ===
calls: 0/5
latency: 0ms used, 10ms remaining (cap: 10ms)

=== EVENTS ===
  [1] plan_created: plan=['RETRIEVE', 'ANSWER']
  [2] plan_step: plan_step=RETRIEVE, remaining=['ANSWER']
  [3] plan_step: plan_step=ANSWER, remaining=[]

=== DECISION RATIONALE ===
  [1] Created plan ['RETRIEVE', 'ANSWER'] (have_knowledge=False, have_tool_results=False). Plan addresses question by retrieving context first then answering.
  [2] Executing plan step RETRIEVE: fetching knowledge (retrieve_count=0/2). Remaining steps: ['ANSWER'].
  [3] Executing plan step ANSWER: sufficient evidence gathered to formulate response. Remaining steps: none.
```

Running with a math expression (`question = "2 + 2"`):

```
=== ANSWER ===
The result is 4.0.

=== DECISION RATIONALE ===
  [1] Fast-path: detected math expression '2 + 2'; bypassing planning and routing directly to calculator (cost=low, latency=50ms).
```

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
