# tool_dispatcher.py

import json
from app.weather import get_weather
from app.openrouter_client import OpenRouterClient

# Initialize external clients if needed
llm_client = OpenRouterClient()

def handle_get_weather(args):
    city = args["city"]
    country_code = args["country_code"]
    units = args.get("units", "metric")
    return get_weather(city=city, country_code=country_code, units=units)

def handle_llm_complete(args):
    prompt = args["prompt"]
    model = args.get("model", "openai/gpt-4.1-nano")
    max_tokens = args.get("max_tokens", 1000)
    temperature = args.get("temperature", 0.7)
    return llm_client.complete(prompt, model=model, max_tokens=max_tokens, temperature=temperature)

# Central tool dispatcher
TOOL_DISPATCHER = {
    "get_weather": handle_get_weather,
    "llm_complete": handle_llm_complete,
}

def tool_call_handler(tool_call):
    try:
        args = json.loads(tool_call.function.arguments)
    except Exception as e:
        return {"error": f"Failed to parse tool arguments: {e}"}
    
    handler = TOOL_DISPATCHER.get(tool_call.function.name)
    if handler:
        return handler(args)
    else:
        return {"error": f"Unknown tool: {tool_call.function.name}"}
