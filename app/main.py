from fastapi import FastAPI, Query, HTTPException, Body
from typing import Optional
from app.weather import get_weather, get_onecall_weather
from app.openrouter_client import OpenRouterClient
from app import assistant_api
import tempfile

app = FastAPI()

llm_client = OpenRouterClient()

@app.get("/")
def root():
    return {"status": "Memir backend is live!"}

@app.get("/weather")
def weather_endpoint(
    city: str = Query("London", description="City name"),
    country_code: str = Query("CA", description="Country code (ISO 3166)", min_length=2, max_length=2),
    units: str = Query("metric", description="Units: metric or imperial"),
    city_id: Optional[int] = Query(None, description="OpenWeatherMap city ID")
):
    try:
        data = get_weather(city=city, country_code=country_code, units=units, city_id=city_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Weather data unavailable.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Assistant API Endpoints ---
@app.post("/assistant/create")
def create_assistant():
    assistant_id = assistant_api.get_or_create_assistant()
    return {"assistant_id": assistant_id}

@app.post("/thread/create")
def create_thread():
    thread_id = assistant_api.create_thread()
    return {"thread_id": thread_id}

@app.post("/thread/{thread_id}/message")
def add_message(thread_id: str, content: str = Body(...)):
    msg_id = assistant_api.add_message(thread_id, role="user", content=content)
    return {"message_id": msg_id}

from fastapi import Body
from app.tool_dispatcher import tool_call_handler


@app.post("/thread/{thread_id}/run")
def run_assistant(thread_id: str, instructions: Optional[str] = Body(None)):
    # Get or create the Assistant
    assistant_id = assistant_api.get_or_create_assistant()

    # Create a new run for the thread
    run_id = assistant_api.run_assistant(
        thread_id,
        assistant_id,
        instructions=instructions
    )

    # Handle tool calls using clean dispatcher
    run_result = assistant_api.handle_tool_calls(
        thread_id,
        run_id,
        tool_call_handler  # <-- Centralized dispatcher handles everything
    )

    return run_result


@app.get("/thread/{thread_id}/run/{run_id}/status")
def get_run_status(thread_id: str, run_id: str):
    status = assistant_api.get_run_status(thread_id, run_id)
    return status

@app.get("/thread/{thread_id}/messages")
def get_thread_messages(thread_id: str):
    messages = assistant_api.get_messages(thread_id)
    return {"messages": messages}

# --- Memory File Endpoints (Vector Store) ---
@app.post("/memory/upload")
def upload_memory_file(text: str = Body(...)):
    # Write text to a temp file and upload
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as f:
        f.write(text)
        file_path = f.name
    file_id = assistant_api.upload_memory_file(file_path)
    return {"file_id": file_id}

@app.get("/memory/list")
def list_memories():
    return {"memories": assistant_api.list_memory_files()}

# --- LLM Endpoint (Direct, also available as function tool) ---
@app.post("/llm/complete")
def llm_complete(prompt: str = Body(...), model: str = Body("openai/gpt-4.1-nano"), max_tokens: int = Body(1000), temperature: float = Body(0.7)):
    try:
        response = llm_client.complete(prompt, model=model, max_tokens=max_tokens, temperature=temperature)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/onecall")
def onecall_endpoint(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    units: str = Query("metric", description="Units: metric or imperial"),
    lang: str = Query("en", description="Language code"),
    exclude: Optional[str] = Query(None, description="Comma-separated parts to exclude (e.g., minutely,hourly)")
):
    try:
        data = get_onecall_weather(lat=lat, lon=lon, units=units, lang=lang, exclude=exclude)
        if data is None:
            raise HTTPException(status_code=404, detail="One Call weather data unavailable.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
