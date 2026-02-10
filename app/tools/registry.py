"""
Tool Registry - Central management for all function calling tools
"""

import json
from typing import Dict, Any, Callable
from .weather import get_weather, WEATHER_TOOL


# Registry of all available tools (OpenAI function schemas)
AVAILABLE_TOOLS = [
    WEATHER_TOOL,
    # Add more tools here as you create them
    # CALCULATOR_TOOL,
    # DATABASE_QUERY_TOOL,
    # etc.
]


# Function mapping - maps function names to their implementations
FUNCTION_MAP: Dict[str, Callable] = {
    "get_weather": get_weather,
    # Add more function mappings here
    # "calculate": calculate,
    # "search_database": search_database,
}


async def execute_function(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool function by name with given arguments.
    
    Args:
        function_name: Name of the function to execute
        arguments: Dictionary of arguments to pass to the function
    
    Returns:
        JSON string with function result or error message
    """
    try:
        # Check if function exists
        if function_name not in FUNCTION_MAP:
            return json.dumps({
                "error": f"Function '{function_name}' not found in registry"
            })
        
        # Get the function
        func = FUNCTION_MAP[function_name]
        
        # Execute the function
        result = await func(**arguments)
        
        return result
        
    except TypeError as e:
        return json.dumps({
            "error": f"Invalid arguments for {function_name}: {str(e)}"
        })
    
    except Exception as e:
        return json.dumps({
            "error": f"Error executing {function_name}: {str(e)}"
        })


def get_tool_info() -> Dict[str, Any]:
    """
    Get information about all available tools.
    
    Returns:
        Dictionary with tool count and names
    """
    return {
        "total_tools": len(AVAILABLE_TOOLS),
        "available_functions": list(FUNCTION_MAP.keys()),
        "tools": AVAILABLE_TOOLS
    }
