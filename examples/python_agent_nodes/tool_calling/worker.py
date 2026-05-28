"""
Worker Agent - Provides utility skills for the orchestrator to discover and call.

Registers:
- get_weather (skill): Returns mock weather data for a city
- calculate (skill): Performs basic math operations
- summarize (reasoner): Uses AI to summarize text

Start this agent first, then run the orchestrator.
"""

import os
from agentfield import Agent, AIConfig

app = Agent(
    node_id="utility-worker",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("MODEL", "openrouter/openai/gpt-4o-mini"),
        temperature=0.3,
    ),
)


# ============= SKILLS (DETERMINISTIC) =============


@app.skill(tags=["weather"])
def get_weather(city: str) -> dict:
    """Get the current weather for a city. Returns temperature, conditions, and humidity."""
    # Mock weather data for demo purposes
    weather_data = {
        "new york": {"temp_f": 72, "conditions": "Partly cloudy", "humidity": 65},
        "london": {"temp_f": 58, "conditions": "Overcast", "humidity": 80},
        "tokyo": {"temp_f": 81, "conditions": "Sunny", "humidity": 55},
        "paris": {"temp_f": 64, "conditions": "Light rain", "humidity": 75},
        "sydney": {"temp_f": 68, "conditions": "Clear", "humidity": 50},
    }
    key = city.lower().strip()
    data = weather_data.get(key, {"temp_f": 70, "conditions": "Clear", "humidity": 60})
    return {"city": city, **data}


@app.skill(tags=["math"])
def calculate(operation: str, a: float, b: float) -> dict:
    """Perform a basic math operation. Supports: add, subtract, multiply, divide."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    result = ops.get(operation.lower())
    if result is None:
        return {"error": f"Unknown operation: {operation}. Use: add, subtract, multiply, divide"}
    return {"operation": operation, "a": a, "b": b, "result": result}


# ============= REASONER (AI-POWERED) =============


@app.reasoner(tags=["text"])
async def summarize(text: str) -> dict:
    """Use AI to create a concise summary of the given text."""
    result = await app.ai(
        system="You are a concise summarizer. Respond with only the summary, no preamble.",
        user=f"Summarize this in 1-2 sentences:\n\n{text}",
    )
    return {"summary": str(result)}


if __name__ == "__main__":
    print("Worker Agent starting...")
    print("  Node: utility-worker")
    print("  Skills: get_weather, calculate")
    print("  Reasoners: summarize")
    app.run(port=8001)
