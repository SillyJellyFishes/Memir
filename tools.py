"""
Defines tool functions for M.E.M.I.R. LLM function-calling agentic interface.
Currently exposes memory search; can be expanded with more tools.
"""
from typing import List, Dict, Any
from fastapi import Depends
from main import collection, embedder

from typing import List, Dict, Any, Optional
from main import collection, embedder, EchoSkill, LLMSkill

# 1. Semantic memory search
def memory_search(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Semantic search over conversation memory."""
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return hits

# 2. Add a memory (note, fact, etc.)
def memory_add(document: str, metadata: dict) -> dict:
    """Add a new memory item to the database."""
    import uuid
    from datetime import datetime
    from main import _sanitize_metadata
    memory_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    meta = dict(metadata)
    meta.setdefault("created_at", now)
    meta["modified_at"] = now
    meta = _sanitize_metadata(meta)
    embedding = embedder.encode(document).tolist()
    collection.add(
        ids=[memory_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[meta],
    )
    return {"id": memory_id, "document": document, "metadata": meta}

# 3. Get a memory by ID
def memory_get(memory_id: str) -> Optional[dict]:
    results = collection.get(ids=[memory_id])
    if not results["ids"] or not results["documents"]:
        return None
    return {
        "id": results["ids"][0],
        "document": results["documents"][0],
        "metadata": results["metadatas"][0],
    }

# 4. Echo tool (repeat a message)
def echo(message: str) -> str:
    return EchoSkill().handle(message)[0]

# 5. LLM chat (direct LLM call, with optional history)
def llm_chat(message: str, history: Optional[list] = None) -> str:
    return LLMSkill().handle(message, history)[0]
