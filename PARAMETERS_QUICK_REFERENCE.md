# 🎯 Function Parameters & Returns - Quick Reference

## 📍 Where is the Function?

**Location:** `app/tools/weather.py` (Line 36)

```python
async def get_weather(location: str, unit: str = "celsius") -> str:
```

---

## 📥 What Parameters Do You Pass?

### **Required:**

- **`location`** (string) - City name
  - Examples: "London", "New York", "Tokyo", "Paris"

### **Optional:**

- **`unit`** (string) - Temperature unit
  - Options: "celsius" or "fahrenheit"
  - Default: "celsius"

---

## 📤 What Do You Receive?

### **Return Type:** JSON String

### **Success Response:**

```json
{
  "location": "London",
  "country": "GB",
  "temperature": 12.5,
  "unit": "°C",
  "feels_like": 11.2,
  "condition": "Clouds",
  "description": "broken clouds",
  "humidity": 76,
  "wind_speed": 4.5,
  "wind_unit": "m/s"
}
```

### **Error Response:**

```json
{
  "error": "Location 'XYZ' not found. Please check the city name and try again."
}
```

---

## 🔄 Complete Call Chain

### **1. API Endpoint Receives:**

```python
# POST /chat-with-tools
{
    "question": "What's the weather in London?",
    "conversation_id": null,
    "history": []
}
```

### **2. OpenAI Decides:**

```python
# OpenAI returns:
{
    "function": "get_weather",
    "arguments": {
        "location": "London",
        "unit": "celsius"
    }
}
```

### **3. Registry Dispatcher Calls:**

```python
# app/tools/registry.py line 51
result = await func(**arguments)

# Which expands to:
result = await get_weather(location="London", unit="celsius")
```

### **4. Weather Function Executes:**

```python
# app/tools/weather.py line 36
async def get_weather(location: str, unit: str = "celsius") -> str:
    # 1. Get API key from .env
    api_key = os.getenv("WEATHER_API_KEY")

    # 2. Call OpenWeatherMap API
    response = await client.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": location,          # "London"
            "appid": api_key,       # Your API key
            "units": "metric"       # celsius → metric
        }
    )

    # 3. Process response
    weather_info = {
        "location": data["name"],
        "temperature": data["main"]["temp"],
        # ... more fields
    }

    # 4. Return as JSON string
    return json.dumps(weather_info)
```

### **5. Returns:**

```python
# Returns to registry.py line 51
'{"location": "London", "temperature": 12.5, "unit": "°C", ...}'
```

### **6. Registry Returns to API:**

```python
# Returns to app/api.py line 587
function_response = '{"location": "London", "temperature": 12.5, ...}'
```

### **7. OpenAI Formats:**

```python
# Second OpenAI call formats the JSON into natural language
"The weather in London is currently 12.5°C with broken clouds..."
```

### **8. Final Response:**

```python
# Returns to user
{
    "answer": "The weather in London is currently 12.5°C...",
    "tools_used": [
        {
            "name": "get_weather",
            "args": {"location": "London", "unit": "celsius"}
        }
    ]
}
```

---

## 🧪 Test Examples

### **Example 1: Basic Call**

```python
# Input
location = "London"
unit = "celsius"

# Output
{
  "location": "London",
  "temperature": 12.5,
  "unit": "°C",
  "condition": "Clouds"
}
```

### **Example 2: Fahrenheit**

```python
# Input
location = "New York"
unit = "fahrenheit"

# Output
{
  "location": "New York",
  "temperature": 68.5,
  "unit": "°F",
  "wind_unit": "mph"  # Note: Changes with unit
}
```

### **Example 3: Invalid Location**

```python
# Input
location = "InvalidCity123"
unit = "celsius"

# Output
{
  "error": "Location 'InvalidCity123' not found. Please check the city name and try again."
}
```

---

## 📊 Data Transformation

### **User Input:**

```
"What's the weather in London?"
```

### **OpenAI Extracts:**

```python
{
  "location": "London",
  "unit": "celsius"  # Default if not specified
}
```

### **Function Receives:**

```python
get_weather(location="London", unit="celsius")
```

### **API Call:**

```
GET https://api.openweathermap.org/data/2.5/weather?q=London&units=metric
```

### **API Returns:**

```json
{
  "name": "London",
  "main": { "temp": 12.5, "feels_like": 11.2, "humidity": 76 },
  "weather": [{ "main": "Clouds", "description": "broken clouds" }],
  "wind": { "speed": 4.5 }
}
```

### **Function Processes:**

```json
{
  "location": "London",
  "temperature": 12.5,
  "unit": "°C",
  "feels_like": 11.2,
  "condition": "Clouds",
  "description": "broken clouds",
  "humidity": 76,
  "wind_speed": 4.5,
  "wind_unit": "m/s"
}
```

### **Function Returns:**

```python
'{"location": "London", "temperature": 12.5, "unit": "°C", ...}'
```

### **OpenAI Formats:**

```
"The weather in London is currently 12.5°C with broken clouds.
It feels like 11.2°C, with 76% humidity and wind speeds of 4.5 m/s."
```

---

## 🔍 Key Files & Lines

| File                    | Line    | What Happens                             |
| ----------------------- | ------- | ---------------------------------------- |
| `app/api.py`            | 580-581 | Extract function name & args from OpenAI |
| `app/api.py`            | 587     | Call `execute_function()`                |
| `app/tools/registry.py` | 48      | Look up function in FUNCTION_MAP         |
| `app/tools/registry.py` | 51      | Execute: `await func(**arguments)`       |
| `app/tools/weather.py`  | 36      | Function definition                      |
| `app/tools/weather.py`  | 58-66   | HTTP call to OpenWeatherMap              |
| `app/tools/weather.py`  | 75-86   | Process API response                     |
| `app/tools/weather.py`  | 89      | Return JSON string                       |

---

## 💡 Quick Tips

1. **Parameters are extracted by OpenAI** - You don't manually pass them
2. **Function returns JSON string** - Not a Python dict
3. **Registry dispatches the call** - Looks up function by name
4. **Two API calls total:**
   - OpenAI call #1: Decide to use tool
   - OpenAI call #2: Format the result

---

**For complete flow:** Run `python parameter_flow_diagram.py`
