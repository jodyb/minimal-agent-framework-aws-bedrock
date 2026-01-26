from __future__ import annotations

from typing import Any, Dict, List

_CORPUS = [
    {"title": "LangGraph", "text": "LangGraph helps build stateful, multi-step LLM apps using graph-based control flow and explicit state."},
    {"title": "LangChain", "text": "LangChain provides building blocks for LLM apps (prompts, chains, retrievers, tools)."},
    {"title": "Agent pattern", "text": "A practical agent is a goal-directed state machine where an LLM helps choose transitions and code enforces guardrails."},
]

def retrieve(query: str, k: int = 2) -> List[Dict[str, Any]]:
    q = query.lower()
    scored = []
    for doc in _CORPUS:
        text = (doc["title"] + " " + doc["text"]).lower()
        score = sum(1 for tok in q.split() if tok and tok in text)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored if s > 0][:k]
