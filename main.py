import ast
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Helper functions for metadatagit add .


def _sanitize_metadata(meta):
    sanitized = {}
    for k, v in meta.items():
        if v is None:
            continue  # Remove None values completely
        if isinstance(v, (list, dict)):
            sanitized[k] = str(v)  # Store lists/dicts as string
        else:
            sanitized[k] = v
    return sanitized


def _parse_metadata(meta):
    # Parse stringified lists/dicts back for API response
    parsed = dict(meta)
    # Handle tags (list[str])
    if "tags" in parsed and isinstance(parsed["tags"], str):
        try:
            val = ast.literal_eval(parsed["tags"])
            if isinstance(val, list):
                parsed["tags"] = val
        except Exception:
            parsed["tags"] = []
    # Handle related_ids (list[str])
    if "related_ids" in parsed and isinstance(parsed["related_ids"], str):
        try:
            val = ast.literal_eval(parsed["related_ids"])
            if isinstance(val, list):
                parsed["related_ids"] = val
        except Exception:
            parsed["related_ids"] = []
    # Handle extra (dict)
    if "extra" in parsed and isinstance(parsed["extra"], str):
        try:
            val = ast.literal_eval(parsed["extra"])
            if isinstance(val, dict):
                parsed["extra"] = val
        except Exception:
            parsed["extra"] = None
    return parsed


# Initialize FastAPI
app = FastAPI()

# Initialize ChromaDB client (local, in-memory for now)
client = chromadb.Client(
    Settings(anonymized_telemetry=False, persist_directory="chroma_data")
)
collection = client.get_or_create_collection(name="memir_memory")

# Load local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# Pydantic models
class Metadata(BaseModel):
    type: str = Field(..., description="Type of memory, e.g., note, task, character")
    title: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    tags: Optional[List[str]] = None
    project_id: Optional[str] = None
    related_ids: Optional[List[str]] = None
    source: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class MemoryAddRequest(BaseModel):
    document: str
    metadata: Metadata


class MemorySearchRequest(BaseModel):
    query: str
    n_results: int = 3
    filter: Optional[Dict[str, Any]] = None  # e.g., {"type": "note"}


class MemoryItem(BaseModel):
    id: str
    document: str
    metadata: Metadata


@app.get("/")
def read_root():
    return {"message": "Welcome to M.E.M.I.R. API!"}


@app.post("/memory/add", response_model=MemoryItem)
def add_memory(item: MemoryAddRequest):
    memory_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    meta = item.metadata.dict()
    meta.setdefault("created_at", now)
    meta["modified_at"] = now
    meta = _sanitize_metadata(meta)
    embedding = embedder.encode(item.document).tolist()
    collection.add(
        ids=[memory_id],
        documents=[item.document],
        embeddings=[embedding],
        metadatas=[meta],
    )
    # Parse for API response
    return MemoryItem(
        id=memory_id, document=item.document, metadata=Metadata(**_parse_metadata(meta))
    )


@app.post("/memory/search", response_model=List[MemoryItem])
def search_memory(req: MemorySearchRequest):
    embedding = embedder.encode(req.query).tolist()
    query_args = {
        "query_embeddings": [embedding],
        "n_results": req.n_results,
    }
    if req.filter:
        query_args["where"] = req.filter
    results = collection.query(**query_args)
    items = []
    for i in range(len(results["ids"][0])):
        items.append(
            MemoryItem(
                id=results["ids"][0][i],
                document=results["documents"][0][i],
                metadata=Metadata(**_parse_metadata(results["metadatas"][0][i])),
            )
        )
    return items


@app.get("/memory/{memory_id}", response_model=MemoryItem)
def get_memory(memory_id: str):
    results = collection.get(ids=[memory_id])
    print("DEBUG: ChromaDB get result:", results)  # Diagnostic log
    # Defensive checks for ChromaDB return structure
    for key in ("ids", "documents", "metadatas"):
        if key not in results or not isinstance(results[key], list) or not results[key]:
            raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryItem(
        id=results["ids"][0],
        document=results["documents"][0],
        metadata=Metadata(**_parse_metadata(results["metadatas"][0])),
    )
