from app.memory import MemoryStore
from app.openrouter_client import OpenRouterClient
import re
import logging

# Set up backend logger
logging.basicConfig(
    filename="agent.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("agentic_backend")

SYSTEM_PROMPT = (
    "You are M.E.M.I.R., an agentic AI assistant with the ability to call real Python functions to interact with user memory. "
    "You must use these functions to recall, store, or manage user information. Do not make up facts—always use the memory functions to check or update information.\n\n"
    "You always have access to the full conversation history for this session. You may reference anything the user or you have said previously.\n"
    "Available functions (call by outputting the exact line as shown):\n"
    "- save_memory(\"text\")\n  Save the provided text as a new memory.\n"
    "- search_memory(\"query\")\n  Search stored memories for the most relevant information to the query.\n"
    "- list_memories()\n  List all stored memories.\n"
    "- remove_memory(\"id\")\n  Remove the memory with the given ID.\n"
    "- get_weather_forecast()\n  Get current weather, forecast, and alerts for the user's home (London, Ontario, Canada) using the One Call API. Always present both current and short-term forecast in a single response.\n"
    "- get_weather()\n  Get the current weather for the user's home location (London, Ontario, Canada).\n"
    "- get_weather(\"city\", \"country_code\")\n  Get the current weather for a city (optionally specify a country code, e.g. 'London,GB').\n\n"
    "How to use:\n"
    "- You may chain multiple function calls in a single user request if it is contextually justified (e.g., switching from current to forecast, or asking for more detail).\n"
    "- Do NOT call the same function repeatedly with the same arguments unless the user clarifies or requests an update.\n"
    "- After calling a weather function and receiving the backend result, respond to the user in plain English using the provided summary.\n"
    "- Only ask for clarification or repeat a function call if the user's request is ambiguous or they specifically ask for more details.\n"
    "- Output the function call on a single line, e.g.:\n"
    "    save_memory(\"The user's favorite color is purple.\")\n"
    "    search_memory(\"favorite color\")\n"
    "    list_memories()\n"
    "    remove_memory(\"123456\")\n"
    "- You may also use the CALL: prefix, e.g.:\n    CALL: save_memory(\"...\")\n\n"
    "Examples:\n\n"
    "User: My favorite color is purple.\n"
    "Assistant: save_memory(\"The user's favorite color is purple.\")\n"
    "(Intended output: The backend will store this memory, then you may confirm to the user.)\n\n"
    "User: What is my favorite color?\n"
    "Assistant: search_memory(\"favorite color\")\n"
    "(Backend returns: [Memory search results for 'favorite color']:\n- The user's favorite color is purple.)\n"
    "Assistant: Your favorite color is purple.\n\n"
    "User: List everything you know about me.\n"
    "Assistant: list_memories()\n"
    "(Backend returns: [All memories]:\n- The user's favorite color is purple.\n- You like cats.)\n"
    "Assistant: Here is everything I know about you: Your favorite color is purple. You like cats.\n\n"
    "User: Forget that my favorite color is purple.\n"
    "Assistant: search_memory(\"favorite color\")\n"
    "(Intended output: The backend will search for the relevant memory. If found, call remove_memory(\"id\") with the correct ID, then confirm deletion to the user.)\n\n"
    "- If a search_memory() call returns no results, you may try alternative queries or reformulate your search (e.g., synonyms, different phrasing, spelling variations).\n"
    "- You may call search_memory() up to three times per user request with different queries if you think it will help.\n"
    "- If, after several attempts, you still can’t find relevant information, respond honestly in plain English (e.g., 'I don’t know' or 'I couldn’t find that information. Would you like to tell me?').\n"
    "- After calling a function and receiving the backend result, you must respond to the user in plain English, using the information from the backend. Do not call the same function repeatedly.\n"
    "- Only use plain language after the backend has performed the requested action or provided information.\n"
    "- If you are unsure, call a function to check or update memory before answering.\n"
)

def parse_call(text):
    # Accept both 'CALL: function("arg")' and 'function("arg")'
    text = text.strip()
    # Try CALL: prefix
    match = re.match(r"CALL: (\w+)\((.*)\)", text)
    if match:
        func = match.group(1)
        arg = match.group(2)
        arg = arg.strip()
        if arg.startswith('"') and arg.endswith('"'):
            arg = arg[1:-1]
        return func, arg
    # Try just function(...)
    match = re.match(r"(\w+)\((.*)\)", text)
    if match:
        func = match.group(1)
        arg = match.group(2)
        arg = arg.strip()
        if arg.startswith('"') and arg.endswith('"'):
            arg = arg[1:-1]
        return func, arg
    return None, None

if __name__ == "__main__":
    store = MemoryStore()
    llm = OpenRouterClient()

    conversation = []
    print("Welcome to M.E.M.I.R. Agentic CLI!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        conversation.append(f"User: {user_input}")
        max_loops = 8
        loop_count = 0
        last_func_call = None
        repeat_func_count = 0
        unique_search_queries = set()
        while True:
            if loop_count > max_loops:
                # If the LLM output is a function call, handle it and continue processing until a plain English response is intended.
                while True:
                    answer = llm.complete(SYSTEM_PROMPT + "\n" + "\n".join(conversation) + "\nAssistant:")
                    answer = answer.strip()
                    # Function call pattern
                    match = re.match(r'(?:CALL: )?(\w+)\((.*)\)', answer)
                    if match:
                        func, arg = match.group(1), match.group(2)
                        logger.info(f"Function call: {func}({arg})")
                        if func == "save_memory":
                            store.add_memory(arg, {"tag": "chat", "test": False})
                            backend_message = "[Memory saved]"
                            logger.info(f"DEBUG: save_memory called with arg={arg}")
                        elif func == "search_memory":
                            results = store.search_memories(arg)
                            logger.info(f"DEBUG: search_memory('{arg}') raw results: {results}")
                            context = "\n".join([f"- {hit['document']}" for hit in results])
                            backend_message = f"[Memory search results for '{arg}']:\n{context}"
                            print(f"DEBUG: search_memory('{arg}') backend_message: {backend_message}")
                        elif func == "list_memories":
                            memories = store.list_memories()
                            logger.info(f"DEBUG: list_memories raw: {memories}")
                            context = "\n".join([f"- {mem['document']}" for mem in memories])
                            backend_message = f"[All memories]:\n{context}"
                            print(f"DEBUG: list_memories backend_message: {backend_message}")
                        elif func == "get_weather":
                            from app.weather import get_weather
                            if not arg.strip():
                                weather = get_weather()
                                city = weather.get('name', 'London')
                                country = weather.get('sys', {}).get('country', 'CA')
                            elif ',' in arg:
                                city, country = [x.strip() for x in arg.split(',', 1)]
                                weather = get_weather(city, country, city_id=None)
                            else:
                                city = arg
                                country = None
                                weather = get_weather(city, city_id=None)
                            if weather and 'weather' in weather and 'main' in weather:
                                desc = weather['weather'][0]['description'].capitalize()
                                temp = weather['main']['temp']
                                feels = weather['main'].get('feels_like', temp)
                                humidity = weather['main'].get('humidity', '?')
                                wind = weather.get('wind', {}).get('speed', '?')
                                wind_deg = weather.get('wind', {}).get('deg', '?')
                                pressure = weather['main'].get('pressure', '?')
                                summary = (
                                    f"Current weather for {city}, {country}:\n"
                                    f"- Condition: {desc}\n"
                                    f"- Temperature: {temp}°C (feels like {feels}°C)\n"
                                    f"- Humidity: {humidity}%\n"
                                    f"- Wind: {wind} m/s at {wind_deg}°\n"
                                    f"- Pressure: {pressure} hPa"
                                )
                                backend_message = summary
                            else:
                                backend_message = f"Sorry, I couldn't retrieve the weather for {city}{','+country if country else ''}."
                            logger.info(f"DEBUG: get_weather('{arg}') result: {backend_message}")
                            print(f"DEBUG: get_weather('{arg}') backend_message: {backend_message}")
                            # Output directly as assistant's reply
                            conversation.append(f"Assistant: {backend_message}")
                            continue
                        elif func == "get_weather_forecast":
                            from app.weather import get_onecall_weather
                            # Prevent repeated identical calls unless context changes
                            if (
                                hasattr(self, 'last_weather_call') and
                                self.last_weather_call == (func, None)
                            ):
                                backend_message = "You just received the latest forecast. Only request again if you want an update or different details."
                                logger.warning(f"Repeated function call '{func}()' with same context.")
                                conversation.append(f"Assistant: {backend_message}")
                                break
                            data = get_onecall_weather()
                            if not data:
                                backend_message = "Sorry, I couldn't retrieve the forecast for your location."
                            else:
                                lines = []
                                if 'weather_overview' in data:
                                    lines.append(f"Summary: {data['weather_overview']}")
                                cur = data.get('current', {})
                                if cur:
                                    desc = cur.get('weather', [{}])[0].get('description', 'N/A').capitalize()
                                    temp = cur.get('temp', '?')
                                    feels = cur.get('feels_like', temp)
                                    humidity = cur.get('humidity', '?')
                                    wind = cur.get('wind_speed', '?')
                                    wind_deg = cur.get('wind_deg', '?')
                                    pressure = cur.get('pressure', '?')
                                    lines.append(f"Current: {desc}, {temp}°C (feels like {feels}°C), humidity {humidity}%, wind {wind} m/s at {wind_deg}°, pressure {pressure} hPa.")
                                minutely = data.get('minutely', [])
                                if minutely:
                                    precip = any(m.get('precipitation', 0) > 0 for m in minutely)
                                    lines.append(f"Next hour: {'Precipitation expected' if precip else 'No precipitation expected'}.")
                                daily = data.get('daily', [])
                                from datetime import datetime
                                if daily:
                                    for i, d in enumerate(daily[:2]):
                                        day = 'Today' if i == 0 else 'Tomorrow'
                                        desc = d.get('weather', [{}])[0].get('description', 'N/A').capitalize()
                                        tmin = d.get('temp', {}).get('min', '?')
                                        tmax = d.get('temp', {}).get('max', '?')
                                        pop = d.get('pop', 0)
                                        lines.append(f"{day}: {desc}, {tmin}–{tmax}°C, {int(pop*100)}% chance of precipitation.")
                                alerts = data.get('alerts', [])
                                if alerts:
                                    for alert in alerts:
                                        lines.append(f"ALERT: {alert.get('event', 'Weather Alert')}: {alert.get('description', '')}")
                                backend_message = "\n".join(lines)
                            logger.info(f"DEBUG: get_weather_forecast result: {backend_message}")
                            print(f"DEBUG: get_weather_forecast backend_message: {backend_message}")
                            conversation.append(f"Assistant: {backend_message}")
                            # Track last weather call context
                            self.last_weather_call = (func, None)
                            continue

                        # Append backend result as next Assistant message
                        conversation.append(f"Assistant: {backend_message}")
                        # Call LLM again for next step (may be another function call)
                        prompt2 = SYSTEM_PROMPT + "\n" + "\n".join(conversation) + "\nAssistant:"
                        answer = llm.complete(prompt2)
                        continue
                    else:
                        # Not a function call, treat as plain English answer and break
                        print(f"Assistant: {answer}\n")
                        break
                break
            loop_count += 1
            print("\nThinking...")
            prompt = SYSTEM_PROMPT + "\n" + "\n".join(conversation) + "\nAssistant:"
            response = llm.complete(prompt)
            if isinstance(response, dict) and 'choices' in response:
                answer = response['choices'][0]['message']['content']
            else:
                answer = str(response)
            answer = answer.strip()
            if answer.startswith("CALL:") or re.match(r"^\w+\(.*\)$", answer):
                func, arg = parse_call(answer)
                logger.info(f"Function call: {func}({arg})")
                # Detect repeated function calls
                func_call_tuple = (func, arg)
                if func_call_tuple == last_func_call:
                    repeat_func_count += 1
                    if repeat_func_count >= 2:
                        print(f"[Aborted: Repeated function call '{func}({arg})'.]")
                        logger.warning(f"Aborted: Repeated function call '{func}({arg})'.")
                        break
                else:
                    repeat_func_count = 0
                last_func_call = func_call_tuple
                # Track unique search_memory queries
                if func == "search_memory":
                    if arg in unique_search_queries:
                        logger.info(f"Duplicate search_memory query: '{arg}'")
                    unique_search_queries.add(arg)
                    if len(unique_search_queries) > 3:
                        print("[Aborted: Too many unique search attempts. Responding: 'I don’t know'.]")
                        logger.warning("Aborted: Too many unique search attempts. Responding: 'I don’t know'.")
                        conversation.append("Assistant: I don’t know. I couldn’t find that information. Would you like to tell me?")
                        print("Assistant: I don’t know. I couldn’t find that information. Would you like to tell me?\n")
                        break
                backend_message = None
                if func == "save_memory":
                    store.add_memory(arg, {"tag": "chat", "test": False})
                    backend_message = "[Memory saved]"
                    logger.info(f"DEBUG: save_memory called with arg={arg}")
                elif func == "search_memory":
                    results = store.search_memories(arg)
                    logger.info(f"DEBUG: search_memory('{arg}') raw results: {results}")
                    context = "\n".join([f"- {hit['document']}" for hit in results])
                    backend_message = f"[Memory search results for '{arg}']:\n{context}"
                    print(f"DEBUG: search_memory('{arg}') backend_message: {backend_message}")
                elif func == "list_memories":
                    memories = store.list_memories()
                    logger.info(f"DEBUG: list_memories raw: {memories}")
                    context = "\n".join([f"- {mem['document']}" for mem in memories])
                    backend_message = f"[All memories]:\n{context}"
                    print(f"DEBUG: list_memories backend_message: {backend_message}")
                elif func == "get_weather":
                    from app.weather import get_weather
                    if not arg.strip():
                        weather = get_weather()
                        city = weather.get('name', 'London')
                        country = weather.get('sys', {}).get('country', 'CA')
                    elif ',' in arg:
                        city, country = [x.strip() for x in arg.split(',', 1)]
                        weather = get_weather(city, country, city_id=None)
                    else:
                        city = arg
                        country = None
                        weather = get_weather(city, city_id=None)
                    if weather and 'weather' in weather and 'main' in weather:
                        desc = weather['weather'][0]['description'].capitalize()
                        temp = weather['main']['temp']
                        feels = weather['main'].get('feels_like', temp)
                        humidity = weather['main'].get('humidity', '?')
                        wind = weather.get('wind', {}).get('speed', '?')
                        wind_deg = weather.get('wind', {}).get('deg', '?')
                        pressure = weather['main'].get('pressure', '?')
                        summary = (
                            f"Current weather for {city}, {country}:\n"
                            f"- Condition: {desc}\n"
                            f"- Temperature: {temp}°C (feels like {feels}°C)\n"
                            f"- Humidity: {humidity}%\n"
                            f"- Wind: {wind} m/s at {wind_deg}°\n"
                            f"- Pressure: {pressure} hPa"
                        )
                        backend_message = summary
                    else:
                        backend_message = f"Sorry, I couldn't retrieve the weather for {city}{','+country if country else ''}."
                    logger.info(f"DEBUG: get_weather('{arg}') result: {backend_message}")
                    print(f"DEBUG: get_weather('{arg}') backend_message: {backend_message}")
                    # Output directly as assistant's reply
                    conversation.append(f"Assistant: {backend_message}")
                    continue
                elif func == "get_weather_forecast":
                    from app.weather import get_onecall_weather
                    # Prevent repeated identical calls unless context changes
                    if (
                        hasattr(self, 'last_weather_call') and
                        self.last_weather_call == (func, None)
                    ):
                        backend_message = "You just received the latest forecast. Only request again if you want an update or different details."
                        logger.warning(f"Repeated function call '{func}()' with same context.")
                        conversation.append(f"Assistant: {backend_message}")
                        continue
                    data = get_onecall_weather()
                    if not data:
                        backend_message = "Sorry, I couldn't retrieve the forecast for your location."
                    else:
                        lines = []
                        if 'weather_overview' in data:
                            lines.append(f"Summary: {data['weather_overview']}")
                        cur = data.get('current', {})
                        if cur:
                            desc = cur.get('weather', [{}])[0].get('description', 'N/A').capitalize()
                            temp = cur.get('temp', '?')
                            feels = cur.get('feels_like', temp)
                            humidity = cur.get('humidity', '?')
                            wind = cur.get('wind_speed', '?')
                            wind_deg = cur.get('wind_deg', '?')
                            pressure = cur.get('pressure', '?')
                            lines.append(f"Current: {desc}, {temp}°C (feels like {feels}°C), humidity {humidity}%, wind {wind} m/s at {wind_deg}°, pressure {pressure} hPa.")
                        minutely = data.get('minutely', [])
                        if minutely:
                            precip = any(m.get('precipitation', 0) > 0 for m in minutely)
                            lines.append(f"Next hour: {'Precipitation expected' if precip else 'No precipitation expected'}.")
                        daily = data.get('daily', [])
                        from datetime import datetime
                        if daily:
                            for i, d in enumerate(daily[:2]):
                                day = 'Today' if i == 0 else 'Tomorrow'
                                desc = d.get('weather', [{}])[0].get('description', 'N/A').capitalize()
                                tmin = d.get('temp', {}).get('min', '?')
                                tmax = d.get('temp', {}).get('max', '?')
                                pop = d.get('pop', 0)
                                lines.append(f"{day}: {desc}, {tmin}–{tmax}°C, {int(pop*100)}% chance of precipitation.")
                        alerts = data.get('alerts', [])
                        if alerts:
                            for alert in alerts:
                                lines.append(f"ALERT: {alert.get('event', 'Weather Alert')}: {alert.get('description', '')}")
                        backend_message = "\n".join(lines)
                    logger.info(f"DEBUG: get_weather_forecast result: {backend_message}")
                    print(f"DEBUG: get_weather_forecast backend_message: {backend_message}")
                    conversation.append(f"Assistant: {backend_message}")
                    # Track last weather call context
                    self.last_weather_call = (func, None)
                    continue

            else:
                print(f"Assistant: {answer}\n")
                break
