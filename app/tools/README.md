# 🛠️ Function Calling Tools

This directory contains all the tools (functions) that the AI can call during conversations.

## 📁 Structure

```
app/tools/
├── __init__.py          # Module exports
├── registry.py          # Central tool registry and executor
├── weather.py           # Weather API tool
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Using the Weather Tool

```bash
# Start the server
uvicorn app.main:app --reload

# Test the weather tool
python test_weather_tool.py
```

### 2. Example API Request

```python
import requests

response = requests.post("http://127.0.0.1:8000/chat-with-tools", json={
    "question": "What's the weather in London?",
    "conversation_id": None,
    "history": []
})

print(response.json()['answer'])
# Output: "The weather in London is currently 12°C with partly cloudy skies..."
```

## 🔧 Available Tools

### 1. Weather Tool (`get_weather`)

**Description:** Fetches real-time weather data for any location

**Parameters:**

- `location` (required): City name (e.g., "London", "New York")
- `unit` (optional): "celsius" or "fahrenheit" (default: "celsius")

**Example:**

```python
# The AI will automatically call this when user asks about weather
User: "What's the temperature in Tokyo?"
AI: *calls get_weather(location="Tokyo", unit="celsius")*
AI: "The temperature in Tokyo is currently 18°C..."
```

## ➕ Adding New Tools

### Step 1: Create Tool File

Create a new file in `app/tools/` (e.g., `calculator.py`):

```python
"""
Calculator Tool - Performs mathematical calculations
"""

# Define the OpenAI function schema
CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
                }
            },
            "required": ["expression"],
        },
    },
}

# Implement the function
async def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    """
    try:
        # Use a safe evaluator (never use eval() directly!)
        import ast
        import operator

        # Define allowed operations
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        # Parse and evaluate
        node = ast.parse(expression, mode='eval')
        # ... implement safe evaluation ...

        result = {"result": calculated_value}
        import json
        return json.dumps(result)

    except Exception as e:
        import json
        return json.dumps({"error": f"Calculation error: {str(e)}"})
```

### Step 2: Register in Registry

Edit `app/tools/registry.py`:

```python
from .weather import get_weather, WEATHER_TOOL
from .calculator import calculate, CALCULATOR_TOOL  # Add this

AVAILABLE_TOOLS = [
    WEATHER_TOOL,
    CALCULATOR_TOOL,  # Add this
]

FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,  # Add this
}
```

### Step 3: Export from Module

Edit `app/tools/__init__.py`:

```python
from .weather import get_weather, WEATHER_TOOL
from .calculator import calculate, CALCULATOR_TOOL  # Add this
from .registry import AVAILABLE_TOOLS, execute_function

__all__ = [
    'get_weather',
    'WEATHER_TOOL',
    'calculate',           # Add this
    'CALCULATOR_TOOL',     # Add this
    'AVAILABLE_TOOLS',
    'execute_function'
]
```

### Step 4: Test Your Tool

```python
# Test the new tool
response = requests.post("http://127.0.0.1:8000/chat-with-tools", json={
    "question": "What is 25 * 4?",
    "conversation_id": None,
    "history": []
})

print(response.json()['answer'])
# Output: "25 * 4 equals 100"
```

## 🎯 Tool Ideas

Here are some tools you can implement:

### 1. Database Query Tool

```python
async def search_database(query: str, table: str) -> str:
    """Search database for information"""
    # Query your database
    # Return results as JSON
```

### 2. Email Tool

```python
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    # Use SMTP or email service API
    # Return confirmation
```

### 3. Web Search Tool

```python
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information"""
    # Use Google Custom Search API or similar
    # Return search results
```

### 4. Task Management Tool

```python
async def create_task(title: str, description: str, due_date: str) -> str:
    """Create a task in project management system"""
    # Integrate with Jira, Asana, etc.
    # Return task ID
```

### 5. File Operations Tool

```python
async def read_file(filepath: str) -> str:
    """Read contents of a file"""
    # Read file from allowed directory
    # Return contents
```

## 🔒 Security Best Practices

### 1. Input Validation

```python
async def my_tool(param: str) -> str:
    # Always validate inputs
    if not param or len(param) > 1000:
        return json.dumps({"error": "Invalid input"})

    # Sanitize inputs
    param = param.strip()

    # Continue with logic...
```

### 2. Error Handling

```python
async def my_tool(param: str) -> str:
    try:
        # Your logic here
        result = do_something(param)
        return json.dumps({"result": result})

    except Exception as e:
        # Never expose internal errors to users
        print(f"Internal error: {str(e)}")
        return json.dumps({"error": "Operation failed"})
```

### 3. Rate Limiting

```python
from functools import wraps
import time

# Simple rate limiter
call_times = {}

def rate_limit(max_calls=10, period=60):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            key = func.__name__

            if key not in call_times:
                call_times[key] = []

            # Remove old calls
            call_times[key] = [t for t in call_times[key] if now - t < period]

            if len(call_times[key]) >= max_calls:
                return json.dumps({"error": "Rate limit exceeded"})

            call_times[key].append(now)
            return await func(*args, **kwargs)

        return wrapper
    return decorator

@rate_limit(max_calls=5, period=60)
async def expensive_api_call(param: str) -> str:
    # This can only be called 5 times per minute
    pass
```

### 4. Permission Checks

```python
async def delete_file(filepath: str, user_id: str) -> str:
    # Check user permissions
    if not has_permission(user_id, "delete_files"):
        return json.dumps({"error": "Permission denied"})

    # Check file path is safe
    if ".." in filepath or filepath.startswith("/"):
        return json.dumps({"error": "Invalid file path"})

    # Continue with deletion...
```

## 📊 Monitoring Tool Usage

The `/chat-with-tools` endpoint returns `tools_used` in the response:

```json
{
  "answer": "The weather in London is...",
  "tools_used": [
    {
      "name": "get_weather",
      "args": { "location": "London", "unit": "celsius" }
    }
  ]
}
```

You can log this for analytics:

- Most used tools
- Tool success/failure rates
- Average execution time
- Cost per tool call

## 🐛 Debugging

### Enable Debug Logging

The tool execution already includes debug prints:

```python
print(f"DEBUG: Calling {function_name} with args: {function_args}")
```

Check your server logs to see tool execution details.

### Test Individual Tools

```python
# Test a tool directly
from app.tools import execute_function

result = await execute_function("get_weather", {
    "location": "London",
    "unit": "celsius"
})

print(result)
```

## 📚 Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenWeatherMap API Docs](https://openweathermap.org/api)
- [FastAPI Async Guide](https://fastapi.tiangolo.com/async/)

## 🎓 Learning Path

1. ✅ **Start with weather tool** (already implemented)
2. **Add calculator tool** (simple, no external API)
3. **Add database query tool** (use your existing DB)
4. **Add email tool** (integrate with SMTP)
5. **Add web search tool** (requires API key)

---

**Happy tool building!** 🚀
