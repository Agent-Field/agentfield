from agentfield.vision import generate_image_openrouter
await generate_image_openrouter(
    prompt="A vertical portrait",
    model="openrouter/google/gemini-2.5-flash-image",
    size="1024x1024",
    quality="standard",
    style=None,
    response_format="url",
    image_config={"aspect_ratio": "9:16"},
)
