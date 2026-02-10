"""
Weather Tool - OpenAI Function Calling
Fetches real-time weather data using OpenWeatherMap API
"""

import os
import httpx
from typing import Dict, Any


# OpenAI Function Schemas
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specific location or coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., San Francisco, London",
                },
                "lat": {"type": "number", "description": "Latitude for precise location"},
                "lon": {"type": "number", "description": "Longitude for precise location"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)",
                },
            },
            "required": [],
        },
    },
}

LOCATION_TOOL = {
    "type": "function",
    "function": {
        "name": "get_user_location",
        "description": "Get the user's current location (City, Lat, Lon) based on their IP address. Use this when the user asks 'where am I' or 'whats my place'.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


async def get_user_location() -> str:
    """Detects user location via IP-API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ip-api.com/json/", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                import json
                return json.dumps({
                    "city": data.get("city"),
                    "region": data.get("regionName"),
                    "country": data.get("country"),
                    "lat": data.get("lat"),
                    "lon": data.get("lon"),
                    "isp": data.get("isp")
                })
    except Exception as e:
        return f'{{"error": "Location detection failed: {str(e)}"}}'
    return '{"error": "Could not detect location"}'


async def get_weather(location: str = None, lat: float = None, lon: float = None, unit: str = "celsius") -> str:
    """
    Fetch current weather data using City name OR Coordinates.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return '{"error": "Weather API key not configured."}'
    
    try:
        units = "metric" if unit == "celsius" else "imperial"
        unit_symbol = "°C" if unit == "celsius" else "°F"
        
        params = {"appid": api_key, "units": units}
        
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
            print(f"DEBUG: Fetching weather for coordinates: {lat}, {lon}")
        elif location:
            params["q"] = location
            print(f"DEBUG: Fetching weather for location: {location}")
        else:
            return '{"error": "Please provide either a city name or coordinates (lat/lon)."}'

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params=params,
                timeout=10.0
            )
            
            if response.status_code == 404:
                return f'{{"error": "The location \'{location}\' was not found. Try a more specific city like \'New Delhi, India\' or \'Mumbai\'."}}'
                
            response.raise_for_status()
            data = response.json()
            
            weather_info = {
                "location": data["name"],
                "country": data["sys"].get("country", "Unknown"),
                "temperature": data["main"]["temp"],
                "unit": unit_symbol,
                "feels_like": data["main"]["feels_like"],
                "condition": data["weather"][0]["main"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
            import json
            return json.dumps(weather_info)
            
    except Exception as e:
        print(f"ERROR in get_weather: {str(e)}")
        return f'{{"error": "Weather service failed: {str(e)}"}}'
