"""
Weather Tool - OpenAI Function Calling
Fetches real-time weather data using OpenWeatherMap API
"""

import os
import httpx
from typing import Dict, Any


# OpenAI Function Schema
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specific location. Returns temperature, conditions, humidity, and wind speed.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., San Francisco, London, Tokyo, New York",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit preference (default: celsius)",
                },
            },
            "required": ["location"],
        },
    },
}


async def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Fetch current weather data for a location using OpenWeatherMap API.
    
    Args:
        location: City name (e.g., "London", "New York")
        unit: Temperature unit - "celsius" or "fahrenheit"
    
    Returns:
        JSON string with weather data or error message
    """
    api_key = os.getenv("WEATHER_API_KEY")
    
    if not api_key:
        return '{"error": "Weather API key not configured. Please add WEATHER_API_KEY to .env file"}'
    
    try:
        # OpenWeatherMap uses 'metric' for Celsius, 'imperial' for Fahrenheit
        units = "metric" if unit == "celsius" else "imperial"
        unit_symbol = "°C" if unit == "celsius" else "°F"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "appid": api_key,
                    "units": units
                },
                timeout=10.0
            )
            
            if response.status_code == 404:
                return f'{{"error": "Location \'{location}\' not found. Please check the city name and try again."}}'
            
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant weather information
            weather_info = {
                "location": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "unit": unit_symbol,
                "feels_like": data["main"]["feels_like"],
                "condition": data["weather"][0]["main"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "wind_unit": "m/s" if unit == "celsius" else "mph"
            }
            
            import json
            return json.dumps(weather_info)
            
    except httpx.TimeoutException:
        return '{"error": "Weather service timeout. Please try again later."}'
    
    except httpx.HTTPError as e:
        return f'{{"error": "Weather service error: {str(e)}"}}'
    
    except Exception as e:
        return f'{{"error": "Unexpected error fetching weather: {str(e)}"}}'
