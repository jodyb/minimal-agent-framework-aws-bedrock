"""
retrieve.py - Knowledge retrieval for RAG (Retrieval-Augmented Generation).

⚠️  STUB IMPLEMENTATION
=======================
This is a minimal demo implementation using keyword matching against a
hardcoded corpus. It is NOT suitable for production use.

FOR PRODUCTION, REPLACE THIS WITH A REAL VECTOR STORE:
-------------------------------------------------------
1. **Embedding-based retrieval** - Convert documents and queries to vectors
   using an embedding model (e.g., Amazon Titan Embeddings, OpenAI embeddings)

2. **Vector database options**:
   - Amazon OpenSearch with vector search
   - Pinecone
   - Weaviate
   - Chroma (local/lightweight)
   - pgvector (PostgreSQL extension)
   - FAISS (Facebook's similarity search)

3. **Example integration with LangChain**:
   ```python
   from langchain_community.vectorstores import Chroma
   from langchain_aws import BedrockEmbeddings

   embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
   vectorstore = Chroma.from_documents(documents, embeddings)
   retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
   results = retriever.invoke(query)
   ```

HOW THE CURRENT STUB WORKS:
===========================
- Uses simple keyword overlap scoring (count of query tokens found in doc)
- Searches a tiny hardcoded corpus of 3 documents
- Returns top-k documents sorted by score
- No semantic understanding - "car" won't match "automobile"
"""

from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# HARDCODED CORPUS (replace with vector store in production)
# -----------------------------------------------------------------------------
# This tiny corpus is just for demonstration/testing purposes.
# In a real system, this would be replaced by a vector database query.
_CORPUS = [
    {
        "title": "LangGraph",
        "text": "LangGraph helps build stateful, multi-step LLM apps using graph-based control flow and explicit state."
    },
    {
        "title": "LangChain",
        "text": "LangChain provides building blocks for LLM apps (prompts, chains, retrievers, tools)."
    },
    {
        "title": "Agent pattern",
        "text": "A practical agent is a goal-directed state machine where an LLM helps choose transitions and code enforces guardrails."
    },
]


def retrieve(query: str, k: int = 2) -> List[Dict[str, Any]]:
    """
    Retrieve documents relevant to the query.

    ⚠️  STUB: Uses naive keyword matching. Replace with vector similarity search.

    Args:
        query: The search query string.
        k: Maximum number of documents to return (default: 2).

    Returns:
        List of document dicts with "title" and "text" keys,
        sorted by relevance (highest first). Only returns docs
        with at least one keyword match.

    Example:
        >>> retrieve("What is LangGraph?")
        [{"title": "LangGraph", "text": "LangGraph helps build..."}]
    """
    # Lowercase the query for case-insensitive matching
    q = query.lower()

    # Score each document by counting keyword overlaps
    scored = []
    for doc in _CORPUS:
        # Combine title and text for matching
        text = (doc["title"] + " " + doc["text"]).lower()
        # Count how many query tokens appear in the document
        # This is a very naive scoring method - embeddings would be much better
        score = sum(1 for tok in q.split() if tok and tok in text)
        scored.append((score, doc))

    # Sort by score descending (most relevant first)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-k documents that have at least one match (score > 0)
    return [d for s, d in scored if s > 0][:k]
