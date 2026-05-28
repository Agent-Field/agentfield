# Image Generation Hello World

A simple AgentField example demonstrating AI-powered image generation with **unified support for multiple providers**.

## What This Example Demonstrates

- **Multi-provider image generation** - Supports both LiteLLM and OpenRouter with the same simple API
- **Two-stage image generation workflow**:
  1. `create_prompt` - Uses AI to enhance a simple topic into a detailed prompt
  2. `generate_image` - Generates the actual image using `app.ai_with_vision()`
- **Structured outputs** with Pydantic schemas
- **Multimodal AI** - Text-to-image generation
- **Simple, boilerplate-free DX** - Switch providers by just changing the model name

## Supported Providers

The SDK automatically routes to the correct provider based on the model name:

| Provider | Model Example | Prefix |
|----------|--------------|--------|
| **LiteLLM** | `dall-e-3`, `azure/dall-e-3`, `bedrock/stability.stable-diffusion-xl` | No prefix or provider/ |
| **OpenRouter** | `openrouter/google/gemini-2.5-flash-image-preview` | `openrouter/` |

**Switch providers with zero code changes:**
```bash
# Use DALL-E (via LiteLLM)
export IMAGE_MODEL="dall-e-3"

# Use Gemini (via OpenRouter)
export IMAGE_MODEL="openrouter/google/gemini-2.5-flash-image-preview"
```

## Architecture

```
generate_artwork (entry point)
├─→ create_prompt (reasoner)
│   └─→ app.ai() - Enhances topic into detailed prompt
└─→ generate_image (reasoner)
    └─→ app.ai_with_vision() - Generates image with DALL-E
```

## Prerequisites

1. **AgentField Control Plane** running at `http://localhost:8080`
   ```bash
   cd control-plane
   go run ./cmd/af dev
   ```

2. **API Keys** (choose based on provider):
   ```bash
   # For LiteLLM (DALL-E, Azure, Bedrock, etc.)
   export OPENAI_API_KEY="sk-..."

   # For OpenRouter (Gemini, etc.)
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install from local SDK (for development)
cd ../../../sdk/python
pip install -e .
cd -
```

## Running the Agent

### Start the Agent Server

```bash
python main.py
```

The agent will:
- Register with the control plane at `http://localhost:8080`
- Start an HTTP server (auto-assigned port, typically 8081)
- Expose reasoners as REST endpoints

### Call the Agent

**Generate artwork from a simple topic:**

```bash
curl -X POST http://localhost:8081/generate_artwork \
     -H "Content-Type: application/json" \
     -d '{"topic": "sunset over mountains"}'
```

**Specify image size:**

```bash
curl -X POST http://localhost:8081/generate_artwork \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "futuristic city with flying cars",
       "size": "1792x1024"
     }'
```

**Use specific provider (OpenRouter example):**

```bash
curl -X POST http://localhost:8081/generate_artwork \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "abstract digital art",
       "model": "openrouter/google/gemini-2.5-flash-image-preview"
     }'
```

**Use specific provider (LiteLLM example):**

```bash
curl -X POST http://localhost:8081/generate_artwork \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "cyberpunk cityscape",
       "model": "dall-e-3",
       "size": "1792x1024"
     }'
```

**Available sizes:**
- `1024x1024` (square, default)
- `1792x1024` (landscape)
- `1024x1792` (portrait)

### Example Response

```json
{
  "topic": "sunset over mountains",
  "original_prompt": "A breathtaking sunset over majestic mountain peaks, with warm golden and orange hues painting the sky, wispy clouds catching the light, photorealistic style, highly detailed, 4k quality",
  "style_notes": "Photorealistic landscape with emphasis on lighting and atmospheric effects",
  "image_url": "https://oaidalleapiprodscus.blob.core.windows.net/private/...",
  "revised_prompt": "A breathtaking sunset scene...",
  "size": "1024x1024"
}
```

## Code Walkthrough

### 1. Enhanced Prompt Creation

```python
@app.reasoner()
async def create_prompt(topic: str) -> DetailedPrompt:
    """Uses AI to create a detailed DALL-E prompt"""
    return await app.ai(
        system="You are an expert at creating detailed prompts for DALL-E...",
        user=f"Create a detailed DALL-E prompt for: {topic}",
        schema=DetailedPrompt,  # Structured output
    )
```

This reasoner takes a simple topic like "sunset" and expands it into a detailed prompt with:
- Visual details (colors, lighting, composition)
- Artistic style (photorealistic, oil painting, etc.)
- Quality modifiers (highly detailed, 4k, etc.)

### 2. Image Generation

```python
@app.reasoner()
async def generate_image(prompt: str, size: str = "1024x1024") -> ImageResult:
    """Generates an image using DALL-E"""
    result = await app.ai_with_vision(
        prompt=prompt,
        size=size,
        quality="standard",
        model="dall-e-3",
    )

    return ImageResult(
        prompt_used=prompt,
        image_url=result.images[0].url,
        revised_prompt=result.images[0].revised_prompt,
    )
```

This reasoner:
- Calls DALL-E via `app.ai_with_vision()`
- Returns a `MultimodalResponse` with generated images
- Extracts the image URL and revised prompt

### 3. Main Workflow

```python
@app.reasoner()
async def generate_artwork(topic: str, size: str = "1024x1024") -> dict:
    """Orchestrates the two-stage workflow"""
    # Step 1: Enhance the prompt
    prompt_result = await create_prompt(topic)

    # Step 2: Generate the image
    image_result = await generate_image(prompt_result.prompt, size=size)

    return {
        "image_url": image_result.image_url,
        # ... more metadata
    }
```

## How Image Generation Works

### `app.ai_with_vision()` Method

The SDK's `ai_with_vision()` method provides a **unified interface** for multi-provider image generation:

**Automatic Provider Routing:**
```python
# LiteLLM route (model doesn't start with "openrouter/")
await app.ai_with_vision(
    prompt="A sunset",
    model="dall-e-3"  # Routes to LiteLLM's aimage_generation()
)

# OpenRouter route (model starts with "openrouter/")
await app.ai_with_vision(
    prompt="A sunset",
    model="openrouter/google/gemini-2.5-flash-image-preview"  # Routes to LiteLLM's acompletion() with modalities
)
```

**Key Features:**
1. **Auto-routing** based on model prefix (`openrouter/` vs others)
2. **Passes kwargs directly** to the underlying provider (no hard-coded parameters)
3. **Returns `MultimodalResponse`** with image objects (same format for both providers)
4. **Supports provider-specific parameters** via kwargs

### Implementation Details

**LiteLLM Path:**
- Uses `litellm.aimage_generation()` API
- Supports DALL-E, Azure DALL-E, Bedrock Stable Diffusion, etc.
- Parameters: `prompt`, `size`, `quality`, `style`, `response_format`, etc.

**OpenRouter Path:**
- Uses `litellm.acompletion()` with `modalities: ["image", "text"]`
- Supports Gemini image generation models
- Parameters: `prompt`, `image_config`, etc.

### Simplified Parameter Handling

This example demonstrates **boilerplate-free multi-provider support**:

- ✅ **No hard-coded `response_format`** - now configurable via parameter
- ✅ **All kwargs passed through** - full provider compatibility
- ✅ **Provider-specific patches isolated** - in `litellm_adapters.py`
- ✅ **Same API for all providers** - just change the model name

### Configuration Hierarchy

```python
# Agent-level defaults
app = Agent(
    ai_config=AIConfig(
        model="openai/gpt-4o-mini",
        temperature=0.8,
    )
)

# Method-level overrides
result = await app.ai_with_vision(
    prompt="...",
    quality="hd",  # Override quality
    size="1792x1024",  # Override size
)
```

## Advanced Usage

### Save Images Locally

```python
@app.reasoner()
async def generate_and_save(topic: str) -> dict:
    """Generate and save image locally"""
    prompt_result = await create_prompt(topic)
    result = await app.ai_with_vision(prompt=prompt_result.prompt)

    # Save the image
    result.images[0].save("output.png")

    return {"saved_to": "output.png"}
```

### Custom Image Parameters

**DALL-E (LiteLLM):**
```python
# High-quality landscape image
result = await app.ai_with_vision(
    prompt="Detailed prompt...",
    size="1792x1024",
    quality="hd",
    style="vivid",  # DALL-E 3 only
    model="dall-e-3",
)

# Base64 response instead of URL
result = await app.ai_with_vision(
    prompt="...",
    response_format="b64_json",  # Now configurable!
    model="dall-e-3",
)
```

**Gemini (OpenRouter):**
```python
# Custom aspect ratio with Gemini
result = await app.ai_with_vision(
    prompt="A futuristic city with flying cars",
    model="openrouter/google/gemini-2.5-flash-image-preview",
    image_config={"aspect_ratio": "16:9"}  # OpenRouter-specific parameter
)

# Supported aspect ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
```

### Error Handling

```python
@app.reasoner()
async def safe_generate(topic: str) -> dict:
    """Generate image with error handling"""
    try:
        prompt_result = await create_prompt(topic)
        image_result = await generate_image(prompt_result.prompt)
        return {"success": True, "image_url": image_result.image_url}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Troubleshooting

### "OpenAI API key not found"
Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### "Control plane not reachable"
Start the control plane:
```bash
cd control-plane
go run ./cmd/af dev
```

### "litellm is not installed"
Install dependencies:
```bash
pip install -r requirements.txt
```

### Rate Limiting
DALL-E has rate limits. If you hit them:
- Wait a few seconds between requests
- AgentField automatically retries with exponential backoff
- Check your OpenAI account tier and limits

## Next Steps

- **Try different topics**: Abstract concepts, specific scenes, artistic styles
- **Experiment with sizes**: Landscape (1792x1024) for panoramas, portrait (1024x1792) for vertical art
- **Adjust quality**: Use `quality="hd"` for higher quality (costs more)
- **Combine with other reasoners**: Generate images based on user input, data analysis, etc.

## Related Examples

- [hello_world](../hello_world/) - Basic agent with skills and reasoners
- [Python SDK Documentation](../../../docs/python-sdk.md) - Full SDK reference

## Learn More

- [AgentField Documentation](https://agentfield.ai/docs/learn)
- [DALL-E API Reference](https://platform.openai.com/docs/guides/images)
- [LiteLLM Documentation](https://docs.litellm.ai)
