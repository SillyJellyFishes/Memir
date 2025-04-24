"""
Script to store/update the M.E.M.I.R. personality core in ChromaDB via the FastAPI backend.
Reads the contents of personality_template.md and posts it as a memory item with type="personality_core".
"""
import requests
import os
from datetime import datetime

API_URL = os.getenv("MEMIR_API_URL", "http://127.0.0.1:8000")
TEMPLATE_PATH = "personality_template.md"

def main():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        doc = f.read()
    metadata = {
        "type": "personality_core",
        "format": "markdown",
        "title": "M.E.M.I.R. Personality Core",
        "created_at": datetime.utcnow().isoformat(),
        "tags": ["personality", "core", "system_prompt"],
        "source": "template_script"
    }
    payload = {
        "document": doc,
        "metadata": metadata
    }
    resp = requests.post(f"{API_URL}/memory/add", json=payload)
    if resp.status_code == 200:
        print("[SUCCESS] Personality core stored in ChromaDB.")
    else:
        print(f"[ERROR] Failed to store personality core: {resp.status_code}\n{resp.text}")

if __name__ == "__main__":
    main()
