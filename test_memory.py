from app.memory import MemoryStore
from app.openrouter_client import OpenRouterClient
import time

if __name__ == "__main__":
    store = MemoryStore()
    llm = OpenRouterClient()


    id1 = store.add_memory("Buy almond milk and eggs", {"tag": "grocery", "test": True})
    id2 = store.add_memory("Finish the AI project plan", {"tag": "work", "test": True})
    id3 = store.add_memory("Book dentist appointment", {"tag": "health", "test": True})
    print(f"Added IDs: {id1}, {id2}, {id3}")

    print("\n--- Listing all memories ---")
    for mem in store.list_memories():
        print(mem)

    print("\n--- Searching for 'grocery' ---")
    results = store.search_memories("grocery shopping")
    for hit in results:
        print(hit)

    print("\n--- Removing one memory ---")
    if results:
        to_remove = results[0]['id']
        print(f"Removing memory with ID: {to_remove}")
        store.remove_memory(to_remove)

    print("\n--- Listing all memories after removal ---")
    for mem in store.list_memories():
        print(mem)

    print("\n--- LLM Test: Summarize my memories ---")
    # Gather all memory documents for summarization
    all_texts = [mem['document'] for mem in store.list_memories()]
    if all_texts:
        prompt = "Summarize these notes in one sentence: " + " | ".join(all_texts)
        response = llm.complete(prompt)
        print("LLM summary:", response)
    else:
        print("No memories to summarize.")
