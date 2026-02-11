"""
Tool Registry - Central management for all function calling tools
"""

import json
from typing import Dict, Any, Callable
from .weather import get_weather, WEATHER_TOOL, get_user_location, LOCATION_TOOL
from .utils import get_datetime, TIME_TOOL


# Registry of all available tools (OpenAI function schemas)
AVAILABLE_TOOLS = [
    WEATHER_TOOL,
    LOCATION_TOOL,
    TIME_TOOL,
    # Add more tools here as you create them
]


# Function mapping - maps function names to their implementations
FUNCTION_MAP: Dict[str, Callable] = {
    "get_weather": get_weather,
    "get_user_location": get_user_location,
    "get_datetime": get_datetime,
    # Add more function mappings here
}


async def execute_function(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool function with validation and error handling.
    """
    try:
        # 1. Validate Function exists
        if function_name not in FUNCTION_MAP:
            print(f"ERROR: Tool '{function_name}' not found.")
            return json.dumps({"error": f"Tool '{function_name}' is not registered."})
        
        # 2. Pre-execution Validation (Edge Cases)
        if function_name == "get_weather":
            # Ensure we have at least location or coordinates, otherwise weather can't resolve
            # (Note: Our version handles IP geo, but validation adds a safety layer)
            if not any([arguments.get("location"), arguments.get("lat")]):
                print("DEBUG: Weather tool called without specific location.")

        # 3. Get function handle
        func = FUNCTION_MAP[function_name]
        
        # 4. Execute Function
        print(f"DEBUG: Executing {function_name} with args: {arguments}")
        result = await func(**arguments)
        
        # 5. Validate Result
        if result is None or result == "":
            return json.dumps({"error": f"Tool {function_name} failed to return any data."})
            
        return result
        
    except TypeError as e:
        print(f"ERROR: Argument mismatch for {function_name}: {str(e)}")
        return json.dumps({
            "error": f"Invalid parameters for {function_name}. Expected schema mismatch: {str(e)}"
        })
    
    except Exception as e:
        print(f"ERROR: System failure in tool {function_name}: {str(e)}")
        return json.dumps({
            "error": f"Internal system error during {function_name}: {str(e)}"
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
