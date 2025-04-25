import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from app.embedding import EmbeddingModel
import uuid

class MemoryStore:
    def __init__(self, persist_directory: str = "./chromadb_data"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("memories")
        self.embedder = EmbeddingModel()
        self.persist_directory = persist_directory

    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        memory_id = str(uuid.uuid4())
        embedding = self.embedder.embed(text)
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )
        return memory_id

    def search_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        # Format results for easier use
        hits = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            hits.append({
                "document": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "id": results["ids"][0][i]
            })
        return hits

    def list_memories(self) -> List[Dict[str, Any]]:
        """List all stored memories with their IDs, documents, and metadata."""
        all_ids = self.collection.get()["ids"]
        if not all_ids:
            return []
        results = self.collection.get(ids=all_ids, include=["documents", "metadatas"])
        return [
            {
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i]
            } for i in range(len(results["ids"]))
        ]

    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory by its ID. Returns True if successful, False if not found."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error deleting memory {memory_id}: {e}")
            return False
