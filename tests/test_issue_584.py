from agentfield.media_providers import OpenRouterProvider
await OpenRouterProvider().generate_audio(
    text="Hello world",
    model="openrouter/openai/gpt-audio-mini",
    format="wav",
)
