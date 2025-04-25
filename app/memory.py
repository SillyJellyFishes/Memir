import os
from typing import List, Dict, Any
import tempfile
from openai import OpenAI
from fastapi import HTTPException

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = "vs_680bc99d6aa481918e5a726356a0281a"

client = OpenAI(api_key=OPENAI_API_KEY)

class MemoryStore:
    def __init__(self):
        self.vector_store_id = VECTOR_STORE_ID
        self.client = client

    from fastapi import HTTPException
    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> dict:
        import traceback
        try:
            # Write memory to a temp file
            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as f:
                f.write(text)
                file_path = f.name
            # Upload file to OpenAI
            with open(file_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="assistants")
            # Attach file to vector store
            self.client.vector_stores.files.create(
                vector_store_id=self.vector_store_id,
                file_id=file_obj.id
            )
            os.remove(file_path)
            return {"id": file_obj.id}
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[add_memory ERROR] {e}\n{tb}")
            raise HTTPException(status_code=500, detail=f"add_memory error: {e}")

    def search_memories(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        # Use the Responses API with the file_search tool
        resp = self.client.responses.create(
            model="gpt-4o-mini",
            input=query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
                "max_num_results": n_results
            }],
            include=["file_search_call.results"]
        )
        # Parse the output for file citations and results
        output = resp.output
        results = []
        for item in output:
            if getattr(item, "type", None) == "file_search_call":
                # If results are included, parse them
                results_attr = getattr(item, "results", None)
                if results_attr:
                    for res in results_attr:
                        results.append(res)
        return results

    def list_memories(self) -> List[Dict[str, Any]]:
        import traceback
        from fastapi import HTTPException
        try:
            # List files attached to the vector store
            resp = self.client.vector_stores.files.list(vector_store_id=self.vector_store_id)
            files = []
            for file in resp.data:
                # Print all attributes for debugging
                print(f"[VectorStoreFile] {file.__dict__}")
                file_info = {"id": getattr(file, "id", None)}
                # Add other known attributes if available
                for attr in ["created_at", "object", "size", "status", "usage_bytes", "purpose", "display_name"]:
                    if hasattr(file, attr):
                        file_info[attr] = getattr(file, attr)
                files.append(file_info)
            return files
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[list_memories ERROR] {e}\n{tb}")
            raise HTTPException(status_code=500, detail=f"list_memories error: {e}")

    def remove_memory(self, memory_id) -> bool:
        import traceback
        from fastapi import HTTPException
        try:
            # Accept either dict or string as memory_id
            if isinstance(memory_id, dict) and "id" in memory_id:
                memory_id = memory_id["id"]
            # Remove file from vector store and delete it
            self.client.vector_stores.files.delete(
                vector_store_id=self.vector_store_id,
                file_id=memory_id
            )
            self.client.files.delete(file_id=memory_id)
            return True
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[remove_memory ERROR] {e}\n{tb}")
            raise HTTPException(status_code=500, detail=f"remove_memory error: {e}")
