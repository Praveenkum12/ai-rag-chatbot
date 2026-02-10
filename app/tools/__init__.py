"""
Tools module for AI RAG Chatbot
Provides function calling capabilities for the LLM
"""

from .weather import get_weather, WEATHER_TOOL, get_user_location, LOCATION_TOOL
from .registry import AVAILABLE_TOOLS, execute_function

__all__ = [
    'get_weather',
    'WEATHER_TOOL',
    'get_user_location',
    'LOCATION_TOOL',
    'AVAILABLE_TOOLS',
    'execute_function'
]
