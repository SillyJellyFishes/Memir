import os
from openai import OpenAI
from typing import List, Dict, Any, Optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
ASSISTANT_ID_PATH = os.path.join(os.path.dirname(__file__), "assistant_id.txt")

client = OpenAI(api_key=OPENAI_API_KEY)

# Function tool schemas
weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city and country code.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "country_code": {"type": "string", "description": "Country code (ISO 3166-1 alpha-2)"},
                "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"}
            },
            "required": ["city", "country_code"]
        }
    }
}
llm_tool_schema = {
    "type": "function",
    "function": {
        "name": "llm_complete",
        "description": "Get a completion from the LLM.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": "string", "default": "openai/gpt-4.1-nano"},
                "max_tokens": {"type": "integer", "default": 1000},
                "temperature": {"type": "number", "default": 0.7}
            },
            "required": ["prompt"]
        }
    }
}

def get_or_create_assistant(name="Memir Assistant", instructions="You are a helpful assistant.",
                           model="gpt-4o", vector_store_id: Optional[str] = None) -> str:
    """
    Create or retrieve an Assistant with file_search and function tools.
    Persist the assistant_id for reuse.
    """
    if os.path.exists(ASSISTANT_ID_PATH):
        with open(ASSISTANT_ID_PATH, "r") as f:
            return f.read().strip()
    tools = [
        {"type": "file_search", "vector_store_ids": [vector_store_id or VECTOR_STORE_ID]},
        weather_tool_schema,
        llm_tool_schema
    ]
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
        tools=tools
    )
    with open(ASSISTANT_ID_PATH, "w") as f:
        f.write(assistant.id)
    return assistant.id

def create_thread(messages: Optional[List[Dict[str, Any]]] = None) -> str:
    thread = client.beta.threads.create(messages=messages or [])
    return thread.id

def add_message(thread_id: str, role: str, content: Any, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
    msg = client.beta.threads.messages.create(
        thread_id=thread_id,
        role=role,
        content=content,
        attachments=attachments or []
    )
    return msg.id

def run_assistant(thread_id: str, assistant_id: str, instructions: Optional[str] = None) -> str:
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions
    )
    return run.id

def handle_tool_calls(thread_id: str, run_id: str, tool_call_handler_fn) -> dict:
    """
    Handles tool calls for a run. Calls tool_call_handler_fn(tool_call) for each tool call,
    submits the outputs, and returns the final run result.
    """
    import time
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run.status == "requires_action":
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                output = tool_call_handler_fn(tool_call)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })
            client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run_id,
                tool_outputs=tool_outputs
            )
        elif run.status in ("queued", "in_progress"):
            time.sleep(1)
        else:
            break
    return run.to_dict()

def get_run_status(thread_id: str, run_id: str) -> Dict[str, Any]:
    return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)

def get_messages(thread_id: str) -> List[Dict[str, Any]]:
    msgs = client.beta.threads.messages.list(thread_id=thread_id)
    return [msg.to_dict() for msg in msgs.data]

def upload_memory_file(file_path: str) -> str:
    file_obj = client.files.create(file=open(file_path, "rb"), purpose="assistants")
    # Attach to vector store
    client.vector_stores.files.create(vector_store_id=VECTOR_STORE_ID, file_id=file_obj.id)
    return file_obj.id

def list_memory_files() -> List[Dict[str, Any]]:
    resp = client.vector_stores.files.list(vector_store_id=VECTOR_STORE_ID)
    return [
        {"id": file.id, "created_at": getattr(file, "created_at", None), "status": getattr(file, "status", None)}
        for file in resp.data
    ]

# Function tool registration (weather, LLM completion) will be handled in FastAPI tool-calling logic
