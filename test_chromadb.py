import time
import uuid
import requests
import pytest

BASE_URL = "http://127.0.0.1:8000"


def test_root():
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_add_and_search_memory():
    memory_data = {
        "document": "The wise god Mimir guards the well of knowledge.",
        "metadata": {
            "type": "note",
            "title": "Mimir's Wisdom",
            "tags": ["norse", "wisdom"],
            "source": "test_script",
        },
    }
    add_resp = requests.post(f"{BASE_URL}/memory/add", json=memory_data)
    assert add_resp.status_code == 200, f"Add failed: {add_resp.text}"
    mem = add_resp.json()
    # Wait for embedding/model load if needed
    if "all-MiniLM" in mem["document"]:
        time.sleep(2)
    # Search
    search_data = {"query": "well of wisdom", "n_results": 2}
    search_resp = requests.post(f"{BASE_URL}/memory/search", json=search_data)
    assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
    results = search_resp.json()
    assert any(r["id"] == mem["id"] for r in results), "Added memory not found in search!"
    return mem


def test_get_memory():
    mem = test_add_and_search_memory()
    get_resp = requests.get(f"{BASE_URL}/memory/{mem['id']}")
    assert get_resp.status_code == 200, f"Get by ID failed: {get_resp.text}"
    retrieved = get_resp.json()
    assert retrieved["document"] == mem["document"], "Mismatch in retrieved document!"


def test_get_memory_not_found():
    fake_id = str(uuid.uuid4())
    get_resp = requests.get(f"{BASE_URL}/memory/{fake_id}")
    assert get_resp.status_code == 404, "Should return 404 for missing memory"
    assert "Memory not found" in get_resp.text


def test_chat_echo():
    chat_data = {"message": "echo: Hello, M.E.M.I.R.!"}
    chat_resp = requests.post(f"{BASE_URL}/chat/", json=chat_data)
    assert chat_resp.status_code == 200, f"Chat failed: {chat_resp.text}"
    chat_result = chat_resp.json()
    assert chat_result["response"].lower().startswith("echo:"), "Chatbot did not echo as fallback!"
    assert chat_result["history"][-2:] == [
        ["user", "echo: Hello, M.E.M.I.R.!"],
        ["bot", chat_result["response"]],
    ], "Chat history incorrect for echo!"


def test_chat_llm():
    llm_data = {"message": "What is the capital of France?"}
    llm_resp = requests.post(f"{BASE_URL}/chat/", json=llm_data)
    assert llm_resp.status_code == 200, f"LLM chat failed: {llm_resp.text}"
    llm_result = llm_resp.json()
    assert not llm_result["response"].lower().startswith("echo:"), (
        "LLM skill did not handle as expected!"
    )
    assert (
        "paris" in llm_result["response"].lower()
        or "[openrouter error" not in llm_result["response"].lower()
    ), "LLM did not return expected answer or error!"
    assert llm_result["history"][-2:] == [
        ["user", "What is the capital of France?"],
        ["bot", llm_result["response"]],
    ], "Chat history incorrect for LLM!"


def run_all():
    print("[1] Testing root endpoint...")
    test_root()
    print("[2] Testing add/search memory...")
    mem = test_add_and_search_memory()
    print("[3] Testing get memory...")
    test_get_memory()
    print("[4] Testing get memory not found...")
    test_get_memory_not_found()
    print("[5] Testing chat echo...")
    test_chat_echo()
    print("[6] Testing chat LLM...")
    test_chat_llm()
    print("\nAll endpoint tests passed!")


if __name__ == "__main__":
    run_all()
