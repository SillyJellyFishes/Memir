import openai
from typing import List, Dict, Any
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_NAME = "memir-memories"

class MemoryStore:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        # Create or retrieve the vector store
        self.vector_store = openai.VectorStore.create(name=VECTOR_STORE_NAME)

    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        resp = self.vector_store.add_documents(
            documents=[{"text": text, "metadata": metadata or {}}]
        )
        return resp["ids"][0]

    def search_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        resp = self.vector_store.query(
            query=query,
            top_k=n_results
        )
        # resp["documents"] is a list of dicts with "text" and "metadata"
        # resp["ids"] is the list of document IDs
        results = []
        for idx, doc in enumerate(resp["documents"]):
            results.append({
                "id": resp["ids"][idx],
                "document": doc["text"],
                "metadata": doc.get("metadata", {})
            })
        return results

    def list_memories(self) -> List[Dict[str, Any]]:
        # OpenAI may not support full listing; if not, you may need to track IDs yourself
        # Here, we assume the API supports listing all documents
        resp = self.vector_store.list_documents()
        results = []
        for doc in resp["documents"]:
            results.append({
                "id": doc["id"],
                "document": doc["text"],
                "metadata": doc.get("metadata", {})
            })
        return results

    def remove_memory(self, memory_id: str) -> bool:
        self.vector_store.delete_documents(ids=[memory_id])
        return True
