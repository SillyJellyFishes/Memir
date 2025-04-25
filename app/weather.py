"""
Weather API utility for fetching current weather data using OpenWeatherMap.

Usage Example:
    # For London, Ontario, Canada:
    weather = get_weather("London", "CA")
    if weather and 'weather' in weather and 'main' in weather:
        desc = weather['weather'][0]['description']
        temp = weather['main']['temp']
        print(f"London, ON: {desc}, {temp}°C")
    else:
        print("Weather data unavailable.")
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# London, Ontario, Canada coordinates (user's precise location)
HOME_LAT = 42.968004
HOME_LON = -81.227165


def get_weather(city: str = "London", country_code: str = "CA", units: str = "metric", city_id: int = 6058560):
    """
    Fetch current weather using OpenWeatherMap /weather endpoint.
    - If city_id is provided, it takes precedence.
    - Defaults to London, Ontario, Canada (city_id=6058560).
    Returns a dict with weather info, or None on error.
    """
    if not OPENWEATHERMAP_API_KEY:
        raise ValueError("OPENWEATHERMAP_API_KEY not set in .env")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "appid": OPENWEATHERMAP_API_KEY,
        "units": units
    }
    if city_id:
        params["id"] = city_id
    else:
        q = city if not country_code else f"{city},{country_code}"
        params["q"] = q
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


def get_onecall_weather(lat: float = HOME_LAT, lon: float = HOME_LON, units: str = "metric", lang: str = "en", exclude: str = None):
    """
    Fetch current, forecast, and alerts using OpenWeatherMap One Call API 3.0.
    Returns a dict with 'current', 'hourly', 'daily', 'alerts', and 'weather_overview' if available.
    """
    if not OPENWEATHERMAP_API_KEY:
        raise ValueError("OPENWEATHERMAP_API_KEY not set in .env")
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": units,
        "lang": lang
    }
    if exclude:
        params["exclude"] = exclude
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"One Call API error: {e}")
        return None
