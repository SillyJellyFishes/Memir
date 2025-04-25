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

class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = None  # For future context handling

class ChatResponse(BaseModel):
    response: str
    history: Optional[list] = None

from openrouter_client import OpenRouterClient
from typing import Callable, Tuple


class Skill:
    def can_handle(self, message: str, history=None) -> bool:
        """Return True if this skill should handle the message."""
        return True  # Default: always handles (for fallback)

    def handle(self, message: str, history=None) -> Tuple[str, list]:
        """Return (reply, new_history)."""
        raise NotImplementedError


class EchoSkill(Skill):
    def can_handle(self, message: str, history=None) -> bool:
        return True  # Always handles if no other skill claims

    def handle(self, message: str, history=None) -> Tuple[str, list]:
        reply = (
            f"Echo: {message}"
            if message.strip()
            else "I'm your M.E.M.I.R. assistant! (This is a placeholder response.)"
        )
        new_history = (history or []) + [("user", message), ("bot", reply)]
        return reply, new_history


class LLMSkill(Skill):
    def __init__(self, model: str = "openai/gpt-4.1-nano"):
        self.llm = OpenRouterClient()
        self.model = model

    def can_handle(self, message: str, history=None) -> bool:
        # Only handle if not explicitly requesting echo
        return not (message.lower().strip().startswith("echo:"))

    def handle(self, message: str, history=None) -> Tuple[str, list]:
        # Fetch latest personality core from ChromaDB
        system_prompt = None
        try:
            results = collection.query(
                query_texts=["personality core"],
                n_results=1,
                where={"type": "personality_core"},
            )
            if results["documents"] and results["documents"][0]:
                # Use the latest personality core document as system prompt
                system_prompt = results["documents"][0][0].strip()
        except Exception:
            pass
        # Inject tool manifest into system prompt
        tool_desc = """
You have access to the following tools. You can chain multiple tool calls in a single turn: after you use a tool, you will see the result and can decide to use another tool if needed, repeating this process until you can provide a final answer to the user. To use a tool, respond in the format:
<function_call>tool_name</function_call><arguments>{JSON arguments}</arguments>
If you do not need a tool, just reply as normal.

- memory_search(query: str, n_results: int = 3): Search your conversation memory for relevant information.
- memory_add(document: str, metadata: dict): Store a new fact, note, or conversation in memory.
- memory_get(memory_id: str): Retrieve a specific memory by its unique ID.
- echo(message: str): Repeat a message exactly as the user sent it.
- llm_chat(message: str, history: Optional[list] = None): Have a direct conversation with the LLM, optionally providing prior context.

EXAMPLES:

User: What color did I tell you earlier?
Assistant: <function_call>memory_search</function_call><arguments>{"query": "color", "n_results": 3}</arguments>

User: Please remember this: The password is swordfish.
Assistant: <function_call>memory_add</function_call><arguments>{"document": "The password is swordfish.", "metadata": {"type": "note", "tags": ["password"]}}</arguments>

User: Get me the memory with ID 1234-abcd.
Assistant: <function_call>memory_get</function_call><arguments>{"memory_id": "1234-abcd"}</arguments>

User: Echo this: Hello world!
Assistant: <function_call>echo</function_call><arguments>{"message": "Hello world!"}</arguments>

User: Have a direct chat: What's the meaning of life?
Assistant: <function_call>llm_chat</function_call><arguments>{"message": "What's the meaning of life?"}</arguments>

User: Summarize everything I've told you and remember the summary.
Assistant: <function_call>memory_search</function_call><arguments>{"query": "all facts", "n_results": 10}</arguments>
(function_result: [list of facts])
Assistant: <function_call>llm_chat</function_call><arguments>{"message": "Summarize these facts: [list of facts]"}</arguments>
(function_result: "The summary is: ...")
Assistant: <function_call>memory_add</function_call><arguments>{"document": "The summary is: ...", "metadata": {"type": "summary"}}</arguments>
(function_result: ...)
Assistant: I've summarized and stored everything you've told me!
"""
        if not system_prompt:
            system_prompt = "You are M.E.M.I.R., a helpful, upbeat little buddy. You can use tools to recall information, store facts, repeat messages, and chat directly. You can chain multiple tool calls in a single turn, continuing to use tools as needed until you can provide a final answer.\n" + tool_desc
        else:
            system_prompt = system_prompt + "\n" + tool_desc

        # Build OpenAI-style message history
        chat_history = []
        chat_history.append({"role": "system", "content": system_prompt})
        if history:
            for role, content in history:
                chat_history.append(
                    {
                        "role": "user" if role == "user" else "assistant",
                        "content": content,
                    }
                )
        chat_history.append({"role": "user", "content": message})
        MAX_AGENT_STEPS = 5
        import re
        import json
        from tools import memory_search, memory_add, memory_get, echo, llm_chat
        tool_map = {
            "memory_search": memory_search,
            "memory_add": memory_add,
            "memory_get": memory_get,
            "echo": echo,
            "llm_chat": llm_chat,
        }
        for _ in range(MAX_AGENT_STEPS):
            reply = self.llm.chat(chat_history, model=self.model)
            print("[LLM RAW REPLY]", reply)
            # Detect function call in reply
            m = re.match(r"<function_call>(\w+)</function_call><arguments>(.*?)</arguments>", reply.strip(), re.DOTALL)
            if m:
                tool_name = m.group(1)
                args_json = m.group(2)
                try:
                    args = json.loads(args_json)
                except Exception:
                    args = {}
                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name](**args)
                    except Exception as e:
                        result = f"[Tool error: {e}]"
                else:
                    result = f"[Unknown tool: {tool_name}]"
                # Add tool result to chat history as a function message
                chat_history.append({
                    "role": "function",
                    "name": tool_name,
                    "content": str(result),
                })
                continue  # Loop again, let LLM see result
            else:
                # Not a function call: return as final reply
                return reply, [("user", message)] + (history or []) + [("bot", reply)]
        # If we reach here, agent loop exceeded max steps
        return "[Agent loop exceeded max steps]", [("user", message)] + (history or [])


class ChatbotCore:
    def __init__(self):
        self.skills = []
        self.register_skill(
            LLMSkill(model="google/gemini-2.0-flash-001")
        )  # Register LLM skill with Gemini 2.0 Flash for improved agentic behavior
        self.register_skill(EchoSkill())  # Register fallback skill last

    def register_skill(self, skill: Skill):
        self.skills.append(skill)

    def chat(self, message: str, history=None) -> (str, list):
        for skill in self.skills:
            if skill.can_handle(message, history):
                return skill.handle(message, history)
        # Should never reach here if fallback skill is registered
        return "I don't know how to respond to that.", (history or [])


chatbot_core = ChatbotCore()


class Metadata(BaseModel):
    type: str = Field(
        ..., description="Type of memory, e.g., note, task, character"
    )
    format: str = Field(
        "markdown", description="Document format: markdown, json, text, etc."
    )
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


@app.post("/chat/", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # --- Memory Recall: Always search for relevant conversation memory ---
    recall_results = collection.query(
        query_texts=[req.message],
        n_results=2,
        where={"type": "conversation"},
    )
    recall_history = []
    for i in range(len(recall_results["documents"][0])):
        doc = recall_results["documents"][0][i]
        meta = recall_results["metadatas"][0][i]
        # Defensive: handle meta as dict or str
        if isinstance(meta, dict):
            extra = meta.get("extra", {})
            if isinstance(extra, dict):
                if extra.get("user_message") != req.message:
                    recall_history.append(("memory", doc))
            else:
                recall_history.append(("memory", doc))
        else:
            recall_history.append(("memory", doc))
    merged_history = recall_history + (req.history or [])
    reply, new_history = chatbot_core.chat(req.message, merged_history)

    # --- Conversation Logging ---
    now = datetime.utcnow().isoformat()
    # Detect communication channel if possible (default to 'api')
    channel = "api"
    # Optionally, allow channel to be passed in future via req (e.g., req.channel)
    memory_metadata = {
        "type": "conversation",
        "format": "text",
        "title": f"Chat with M.E.M.I.R. ({now})",
        "created_at": now,
        "modified_at": now,
        "tags": ["conversation", "chat", channel],
        "source": channel,
        "extra": {
            "user_message": req.message,
            "bot_reply": reply,
        },
    }
    memory_document = f"User: {req.message}\nBot: {reply}"
    embedding = embedder.encode(memory_document).tolist()
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[memory_document],
        embeddings=[embedding],
        metadatas=[_sanitize_metadata(memory_metadata)],
    )
    # --- End Conversation Logging ---

    return ChatResponse(response=reply, history=new_history)


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
