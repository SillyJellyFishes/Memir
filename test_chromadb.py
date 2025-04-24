import time

import requests

BASE_URL = "http://127.0.0.1:8000"

# Test data
memory_data = {
    "document": "The wise god Mimir guards the well of knowledge.",
    "metadata": {
        "type": "note",
        "title": "Mimir's Wisdom",
        "tags": ["norse", "wisdom"],
        "source": "test_script",
    },
}

print("[1] Adding a memory...")
add_resp = requests.post(f"{BASE_URL}/memory/add", json=memory_data)
assert add_resp.status_code == 200, f"Add failed: {add_resp.text}"
mem = add_resp.json()
print("Added:", mem)

# Wait a moment for embedding/model load if needed
if "all-MiniLM" in mem["document"]:  # crude check for first-load
    time.sleep(2)

print("\n[2] Searching for memory...")
search_data = {"query": "well of wisdom", "n_results": 2}
search_resp = requests.post(f"{BASE_URL}/memory/search", json=search_data)
assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
results = search_resp.json()
print("Search results:", results)

assert any(r["id"] == mem["id"] for r in results), "Added memory not found in search!"

print("\n[3] Retrieving memory by ID...")
get_resp = requests.get(f"{BASE_URL}/memory/{mem['id']}")
assert get_resp.status_code == 200, f"Get by ID failed: {get_resp.text}"
retrieved = get_resp.json()
print("Retrieved:", retrieved)

assert (
    retrieved["document"] == memory_data["document"]
), "Mismatch in retrieved document!"

print("\nAll endpoint tests passed!")
