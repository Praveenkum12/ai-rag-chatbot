"""
Utility Tools - Date, Time, and other general helpers
"""
from datetime import datetime
import json
from typing import Dict, Any

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": "Get the current date and time. Use this when the user asks 'what time is it', 'what is today's date', or 'what day is it'.",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional format: 'full' (default), 'date_only', or 'time_only'"
                }
            },
            "required": [],
        },
    },
}

async def get_datetime(format: str = "full") -> str:
    """Returns the current date and time."""
    now = datetime.now()
    
    if format == "date_only":
        result = {"date": now.strftime("%A, %B %d, %Y")}
    elif format == "time_only":
        result = {"time": now.strftime("%I:%M:%S %p")}
    else:
        result = {
            "date": now.strftime("%A, %B %d, %Y"),
            "time": now.strftime("%I:%M %p"),
            "full_iso": now.isoformat()
        }
    
    return json.dumps(result)
