"""
Image Generation Hello World - Multi Model Workflow Example

This keeps the agent experience approachable:
- Reasoners stay the same (prompt + image)
- A tiny CLI lets you loop through models with one command
- Outputs are just images saved inside this example folder
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
from pathlib import Path
from typing import List
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen

from agentfield import AIConfig
from agentfield import Agent
from pydantic import BaseModel

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "generated_images"


def _api_settings() -> tuple[Optional[str], Optional[str]]:
    """Pick the right credentials without extra ceremony."""

    if os.getenv("OPENROUTER_API_KEY"):
        return (
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        )
    return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_BASE"))


API_KEY, API_BASE = _api_settings()

app = Agent(
    node_id="image-generator",
    agentfield_server="http://localhost:8080",
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"),
        temperature=0.8,
        api_key=API_KEY,
        api_base=API_BASE,
    ),
)


class DetailedPrompt(BaseModel):
    prompt: str
    style_notes: str


@app.reasoner()
async def create_prompt(topic: str) -> DetailedPrompt:
    """Turn a rough idea into a vivid DALL·E/Gemini-ready prompt."""

    system_prompt = """You are an expert at creating detailed prompts for DALL-E image generation.

Take the topic, expand it into a tight but vivid description with:
- Visual details (colors, lighting, composition)
- Artistic style or medium
- Mood + atmosphere
- Quality modifiers (highly detailed, 4k, etc.)

Keep it to 1-2 sentences."""

    return await app.ai(
        system=system_prompt,
        user=f"Create a detailed image prompt for: {topic}",
        schema=DetailedPrompt,
    )


class ImageResult(BaseModel):
    prompt_used: str
    image_url: str
    revised_prompt: Optional[str] = None


@app.reasoner()
async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    model: Optional[str] = None,
) -> ImageResult:
    """Create the actual image via ai_with_vision()."""

    model = model or os.getenv("IMAGE_MODEL", "dall-e-3")
    result = await app.ai_with_vision(
        prompt=prompt,
        size=size,
        quality="standard",
        model=model,
    )

    image = result.images[0] if result.images else None
    return ImageResult(
        prompt_used=prompt,
        image_url=image.url if image else "",
        revised_prompt=image.revised_prompt if image else None,
    )


@app.reasoner()
async def generate_artwork(
    topic: str,
    size: str = "1024x1024",
    model: Optional[str] = None,
) -> dict:
    """Hello-world orchestration: prompt enhancement + image call."""

    print(f"📝 Topic: {topic}")
    prompt_result = await create_prompt(topic)
    print(f"✨ Prompt: {prompt_result.prompt}")

    image_result = await generate_image(prompt_result.prompt, size=size, model=model)
    print("✅ Image URL ready")
    return {
        "image_url": image_result.image_url,
        "prompt": prompt_result.prompt,
        "model": model or os.getenv("IMAGE_MODEL", "dall-e-3"),
    }


# ---------- Tiny helpers for the CLI ----------


def _slug(model: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in model).strip("-") or "model"


def _extension(image_url: str) -> str:
    if image_url.startswith("data:image/"):
        mime = image_url.split(";")[0].split("/")[-1]
        return f".{mime or 'png'}"
    suffix = Path(urlparse(image_url).path).suffix
    return suffix or ".png"


def _save_image(image_url: str, destination: Path) -> Path:
    """Download or decode the returned image."""

    if image_url.startswith("data:"):
        _, b64_data = image_url.split(",", 1)
        destination.write_bytes(base64.b64decode(b64_data))
        return destination

    request = Request(image_url, headers={"User-Agent": "agentfield-image-runner"})
    with urlopen(request) as response, destination.open("wb") as f:  # nosec
        f.write(response.read())
    return destination


async def run_models(
    topic: str,
    models: List[str],
    size: str,
    output_dir: Optional[str],
) -> None:
    """Loop through each model and drop images next to this example."""

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    def _quiet_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if previous_handler:
            previous_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_quiet_handler)

    output_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, model in enumerate(models, start=1):
        print("\n" + "=" * 40)
        print(f"🚀 {model}")
        try:
            result = await generate_artwork(topic, size=size, model=model)
        except Exception as exc:
            print(f"❌ {model} failed: {exc}")
            continue

        image_url = result.get("image_url")
        if not image_url:
            print("⚠️  No image returned.")
            continue

        filename = f"{idx:02d}-{_slug(model)}{_extension(image_url)}"
        saved_path = _save_image(image_url, output_path / filename)
        print(f"💾 Saved {saved_path}")

    # Give underlying HTTP clients a moment to finish cleanup before the loop closes.
    await asyncio.sleep(0.5)
    loop.set_exception_handler(previous_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Generation Hello World")
    parser.add_argument("--topic", help="Simple idea, e.g. 'sunset over mountains'")
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to test. Defaults to IMAGE_MODELS or IMAGE_MODEL.",
    )
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument(
        "--output-dir",
        help="Optional custom directory for generated images (defaults to this example folder).",
    )
    return parser.parse_args()


def resolve_models(values: Optional[List[str]]) -> List[str]:
    if values:
        return values
    env_models = os.getenv("IMAGE_MODELS")
    if env_models:
        parsed = [model.strip() for model in env_models.split(",") if model.strip()]
        if parsed:
            return parsed
    return [os.getenv("IMAGE_MODEL", "dall-e-3")]


def main() -> None:
    args = parse_args()

    if args.topic:
        models = resolve_models(args.models)
        print(f"🧪 Running {len(models)} model(s) for '{args.topic}'")
        asyncio.run(
            run_models(
                topic=args.topic,
                models=models,
                size=args.size,
                output_dir=args.output_dir,
            )
        )
        return

    # Falls back to the normal Agent server mode for parity with the docs.
    print("🎨 Image Generation Agent")
    print("📍 Node: image-generator")
    print("🌐 Control Plane: http://localhost:8080")
    app.run(auto_port=True)


if __name__ == "__main__":
    main()
