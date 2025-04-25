from fastapi import FastAPI, Query, HTTPException, Body
from typing import Optional
from app.weather import get_weather, get_onecall_weather
from app.memory import MemoryStore
from app.openrouter_client import OpenRouterClient

app = FastAPI()

# Initialize singletons for memory and LLM
memory_store = MemoryStore()
llm_client = OpenRouterClient()

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

# --- Memory Endpoints ---
@app.post("/memory/add")
def add_memory(text: str = Body(...), metadata: Optional[dict] = Body(None)):
    memory_id = memory_store.add_memory(text, metadata)
    return {"id": memory_id}

@app.get("/memory/search")
def search_memory(query: str = Query(...), n_results: int = Query(5)):
    results = memory_store.search_memories(query, n_results)
    return {"results": results}

@app.get("/memory/list")
def list_memories():
    return {"memories": memory_store.list_memories()}

@app.delete("/memory/remove/{memory_id}")
def remove_memory(memory_id: str):
    success = memory_store.remove_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or could not be deleted.")
    return {"status": "deleted"}

# --- LLM Endpoint ---
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
