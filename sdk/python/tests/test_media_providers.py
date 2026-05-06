"""
Tests for Media Providers and Unified Multimodal UX.

This module tests:
- FalProvider, LiteLLMProvider, OpenRouterProvider
- Provider routing in AgentAI (fal-ai/, openrouter/, default)
- New methods: ai_generate_video, ai_transcribe_audio
- AIConfig extensions: fal_api_key, video_model
"""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield.agent_ai import AgentAI
from agentfield.media_providers import (
    FalProvider,
    LiteLLMProvider,
    OpenRouterProvider,
    MediaProvider,
    get_provider,
    register_provider,
)
from agentfield.multimodal_response import (
    AudioOutput,
    ImageOutput,
    FileOutput,
    MultimodalResponse,
)
from agentfield.types import AIConfig


# =============================================================================
# Test Fixtures
# =============================================================================


class DummyAIConfig:
    """Dummy AIConfig for testing."""

    def __init__(self):
        self.model = "openai/gpt-4"
        self.temperature = 0.1
        self.max_tokens = 100
        self.top_p = 1.0
        self.stream = False
        self.response_format = "auto"
        self.fallback_models = []
        self.final_fallback_model = None
        self.enable_rate_limit_retry = True
        self.rate_limit_max_retries = 2
        self.rate_limit_base_delay = 0.1
        self.rate_limit_max_delay = 1.0
        self.rate_limit_jitter_factor = 0.1
        self.rate_limit_circuit_breaker_threshold = 3
        self.rate_limit_circuit_breaker_timeout = 1
        self.auto_inject_memory = []
        self.model_limits_cache = {}
        self.audio_model = "tts-1"
        self.vision_model = "dall-e-3"
        # New fields for Fal support
        self.fal_api_key = None
        self.video_model = "fal-ai/minimax-video/image-to-video"

    def copy(self, deep=False):
        return copy.deepcopy(self)

    async def get_model_limits(self, model=None):
        return {"context_length": 1000, "max_output_tokens": 100}

    def get_litellm_params(self, **overrides):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        params.update(overrides)
        return params


class StubAgent:
    """Stub Agent for testing."""

    def __init__(self):
        self.node_id = "test-agent"
        self.ai_config = DummyAIConfig()
        self.memory = SimpleNamespace()


@pytest.fixture
def agent_with_ai():
    """Create a stub agent with AI config."""
    agent = StubAgent()
    return agent


@pytest.fixture
def fal_provider():
    """Create a FalProvider instance."""
    return FalProvider(api_key="test-fal-key")


@pytest.fixture
def litellm_provider():
    """Create a LiteLLMProvider instance."""
    return LiteLLMProvider()


@pytest.fixture
def openrouter_provider():
    """Create an OpenRouterProvider instance."""
    return OpenRouterProvider()


# =============================================================================
# AIConfig Tests - New Fields
# =============================================================================


class TestAIConfigFalFields:
    """Test new Fal-related fields in AIConfig."""

    def test_aiconfig_has_fal_api_key(self):
        """AIConfig should have fal_api_key field."""
        config = AIConfig()
        assert hasattr(config, "fal_api_key")
        assert config.fal_api_key is None  # Default is None

    def test_aiconfig_has_video_model(self):
        """AIConfig should have video_model field."""
        config = AIConfig()
        assert hasattr(config, "video_model")
        assert config.video_model == "fal-ai/minimax-video/image-to-video"

    def test_aiconfig_fal_api_key_can_be_set(self):
        """fal_api_key should be settable."""
        config = AIConfig(fal_api_key="my-fal-key")
        assert config.fal_api_key == "my-fal-key"

    def test_aiconfig_video_model_can_be_overridden(self):
        """video_model should be overridable."""
        config = AIConfig(video_model="fal-ai/kling-video/v1/standard")
        assert config.video_model == "fal-ai/kling-video/v1/standard"


# =============================================================================
# FalProvider Tests
# =============================================================================


class TestFalProvider:
    """Tests for FalProvider."""

    def test_fal_provider_name(self, fal_provider):
        """FalProvider should have correct name."""
        assert fal_provider.name == "fal"

    def test_fal_provider_supported_modalities(self, fal_provider):
        """FalProvider should support image, audio, and video."""
        assert "image" in fal_provider.supported_modalities
        assert "audio" in fal_provider.supported_modalities
        assert "video" in fal_provider.supported_modalities

    def test_fal_provider_parse_image_size_preset(self, fal_provider):
        """FalProvider should parse Fal presets correctly."""
        assert fal_provider._parse_image_size("square_hd") == "square_hd"
        assert fal_provider._parse_image_size("landscape_16_9") == "landscape_16_9"
        assert fal_provider._parse_image_size("portrait_4_3") == "portrait_4_3"

    def test_fal_provider_parse_image_size_dimensions(self, fal_provider):
        """FalProvider should parse WxH dimensions correctly."""
        result = fal_provider._parse_image_size("1024x768")
        assert result == {"width": 1024, "height": 768}

        result = fal_provider._parse_image_size("512x512")
        assert result == {"width": 512, "height": 512}

    def test_fal_provider_parse_image_size_invalid_fallback(self, fal_provider):
        """FalProvider should fallback to square_hd for invalid sizes."""
        assert fal_provider._parse_image_size("invalid") == "square_hd"

    @pytest.mark.asyncio
    async def test_fal_provider_generate_image(self, fal_provider, monkeypatch):
        """FalProvider.generate_image should call fal_client correctly."""
        mock_result = {
            "images": [
                {"url": "https://fal.media/test.png", "width": 1024, "height": 1024}
            ]
        }

        mock_client = MagicMock()
        mock_client.subscribe_async = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(fal_provider, "_client", mock_client)

        result = await fal_provider.generate_image(
            prompt="A sunset",
            model="fal-ai/flux/dev",
            size="square_hd",
        )

        assert result.has_images
        assert len(result.images) == 1
        assert result.images[0].url == "https://fal.media/test.png"
        mock_client.subscribe_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_fal_provider_generate_video(self, fal_provider, monkeypatch):
        """FalProvider.generate_video should call fal_client correctly."""
        mock_result = {"video_url": "https://fal.media/video.mp4"}

        mock_client = MagicMock()
        mock_client.subscribe_async = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(fal_provider, "_client", mock_client)

        result = await fal_provider.generate_video(
            prompt="Camera pans",
            model="fal-ai/minimax-video/image-to-video",
            image_url="https://example.com/image.jpg",
        )

        assert len(result.files) == 1
        assert result.files[0].url == "https://fal.media/video.mp4"

    @pytest.mark.asyncio
    async def test_fal_provider_transcribe_audio(self, fal_provider, monkeypatch):
        """FalProvider.transcribe_audio should return transcription."""
        mock_result = {"text": "Hello world, this is a test."}

        mock_client = MagicMock()
        mock_client.subscribe_async = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(fal_provider, "_client", mock_client)

        result = await fal_provider.transcribe_audio(
            audio_url="https://example.com/audio.mp3",
            model="fal-ai/whisper",
        )

        assert result.text == "Hello world, this is a test."


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """Tests for provider registry functions."""

    def test_get_provider_fal(self):
        """get_provider should return FalProvider for 'fal'."""
        provider = get_provider("fal")
        assert isinstance(provider, FalProvider)

    def test_get_provider_litellm(self):
        """get_provider should return LiteLLMProvider for 'litellm'."""
        provider = get_provider("litellm")
        assert isinstance(provider, LiteLLMProvider)

    def test_get_provider_openrouter(self):
        """get_provider should return OpenRouterProvider for 'openrouter'."""
        provider = get_provider("openrouter")
        assert isinstance(provider, OpenRouterProvider)

    def test_get_provider_unknown_raises(self):
        """get_provider should raise for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_register_custom_provider(self):
        """register_provider should add custom providers."""

        class CustomProvider(MediaProvider):
            @property
            def name(self):
                return "custom"

            @property
            def supported_modalities(self):
                return ["image"]

            async def generate_image(self, prompt, **kwargs):
                pass

            async def generate_audio(self, text, **kwargs):
                pass

        register_provider("custom", CustomProvider)
        provider = get_provider("custom")
        assert isinstance(provider, CustomProvider)


# =============================================================================
# AgentAI Provider Routing Tests
# =============================================================================


class TestAgentAIProviderRouting:
    """Tests for provider routing in AgentAI."""

    def test_fal_provider_lazy_initialization(self, agent_with_ai):
        """_fal_provider should be lazily initialized."""
        ai = AgentAI(agent_with_ai)
        assert ai._fal_provider_instance is None

        # Access the property to trigger initialization
        with patch("agentfield.media_providers.FalProvider") as mock_fal:
            mock_fal.return_value = MagicMock()
            provider = ai._fal_provider
            assert provider is not None
            mock_fal.assert_called_once_with(api_key=None)

    def test_fal_provider_cached(self, agent_with_ai):
        """_fal_provider should be cached after first access."""
        ai = AgentAI(agent_with_ai)

        with patch("agentfield.media_providers.FalProvider") as mock_fal:
            mock_provider = MagicMock()
            mock_fal.return_value = mock_provider

            provider1 = ai._fal_provider
            provider2 = ai._fal_provider

            # Should only be created once
            assert mock_fal.call_count == 1
            assert provider1 is provider2

    @pytest.mark.asyncio
    async def test_ai_with_vision_routes_fal_ai_prefix(
        self, agent_with_ai, monkeypatch
    ):
        """ai_with_vision should route fal-ai/ models to FalProvider."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="test",
            audio=None,
            images=[ImageOutput(url="https://fal.media/test.png")],
            files=[],
        )
        mock_generate = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.generate_image = mock_generate
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        result = await ai.ai_with_vision(
            prompt="A sunset",
            model="fal-ai/flux/dev",
        )

        mock_generate.assert_called_once()
        assert result.has_images

    @pytest.mark.asyncio
    async def test_ai_with_vision_routes_fal_prefix(self, agent_with_ai, monkeypatch):
        """ai_with_vision should route fal/ models to FalProvider."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="test",
            audio=None,
            images=[ImageOutput(url="https://fal.media/test.png")],
            files=[],
        )
        mock_generate = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.generate_image = mock_generate
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        await ai.ai_with_vision(
            prompt="A sunset",
            model="fal/flux-dev",
        )

        mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ai_with_audio_routes_fal_models(self, agent_with_ai):
        """ai_with_audio should route fal-ai/ models to FalProvider."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="Hello",
            audio=AudioOutput(
                url="https://fal.media/audio.wav", data=None, format="wav"
            ),
            images=[],
            files=[],
        )
        mock_generate = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.generate_audio = mock_generate
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        result = await ai.ai_with_audio(
            "Hello world",
            model="fal-ai/kokoro-tts",
        )

        mock_generate.assert_called_once()
        assert result.has_audio


# =============================================================================
# New Methods: ai_generate_video, ai_transcribe_audio
# =============================================================================


class TestAIGenerateVideo:
    """Tests for ai_generate_video method."""

    @pytest.mark.asyncio
    async def test_ai_generate_video_uses_default_model(self, agent_with_ai):
        """ai_generate_video should use AIConfig.video_model as default."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="",
            audio=None,
            images=[],
            files=[
                FileOutput(
                    url="https://fal.media/video.mp4", data=None, mime_type="video/mp4"
                )
            ],
        )
        mock_generate = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.generate_video = mock_generate
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        await ai.ai_generate_video(prompt="A cat playing")

        # Should use default video_model
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["model"] == "fal-ai/minimax-video/image-to-video"

    @pytest.mark.asyncio
    async def test_ai_generate_video_with_image_url(self, agent_with_ai):
        """ai_generate_video should pass image_url for image-to-video."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="",
            audio=None,
            images=[],
            files=[
                FileOutput(
                    url="https://fal.media/video.mp4", data=None, mime_type="video/mp4"
                )
            ],
        )
        mock_generate = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.generate_video = mock_generate
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        await ai.ai_generate_video(
            prompt="Camera pans",
            model="fal-ai/minimax-video/image-to-video",
            image_url="https://example.com/image.jpg",
        )

        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["image_url"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    async def test_ai_generate_video_rejects_non_fal_models(self, agent_with_ai):
        """ai_generate_video should reject non-Fal and non-OpenRouter models."""
        ai = AgentAI(agent_with_ai)

        with pytest.raises(ValueError, match="No provider"):
            await ai.ai_generate_video(
                prompt="A cat",
                model="openai/video-model",
            )


class TestAITranscribeAudio:
    """Tests for ai_transcribe_audio method."""

    @pytest.mark.asyncio
    async def test_ai_transcribe_audio_default_model(self, agent_with_ai):
        """ai_transcribe_audio should default to fal-ai/whisper."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="Hello world",
            audio=None,
            images=[],
            files=[],
        )
        mock_transcribe = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.transcribe_audio = mock_transcribe
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        result = await ai.ai_transcribe_audio(audio_url="https://example.com/audio.mp3")

        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["model"] == "fal-ai/whisper"
        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_ai_transcribe_audio_with_language(self, agent_with_ai):
        """ai_transcribe_audio should pass language hint."""
        ai = AgentAI(agent_with_ai)

        mock_response = MultimodalResponse(
            text="Hola mundo", audio=None, images=[], files=[]
        )
        mock_transcribe = AsyncMock(return_value=mock_response)

        # Patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.transcribe_audio = mock_transcribe
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        await ai.ai_transcribe_audio(
            audio_url="https://example.com/spanish.mp3",
            model="fal-ai/whisper",
            language="es",
        )

        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["language"] == "es"

    @pytest.mark.asyncio
    async def test_ai_transcribe_audio_rejects_non_fal_models(self, agent_with_ai):
        """ai_transcribe_audio should reject non-Fal models."""
        ai = AgentAI(agent_with_ai)

        with pytest.raises(ValueError, match="only supports Fal.ai models"):
            await ai.ai_transcribe_audio(
                audio_url="https://example.com/audio.mp3",
                model="openai/whisper",
            )


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestUnifiedMultimodalUX:
    """Integration tests for unified multimodal UX pattern."""

    @pytest.mark.asyncio
    async def test_image_generation_routes_correctly(self, agent_with_ai, monkeypatch):
        """Different model prefixes should route to correct providers."""
        ai = AgentAI(agent_with_ai)

        # Track which methods are called
        calls = []

        async def mock_fal_generate(*args, **kwargs):
            calls.append(("fal", kwargs.get("model")))
            return MultimodalResponse(
                text="",
                audio=None,
                images=[ImageOutput(url="https://fal.media/img.png")],
                files=[],
            )

        # Setup mocks - patch the instance attribute directly
        mock_provider = MagicMock()
        mock_provider.generate_image = mock_fal_generate
        mock_provider.supported_modalities = ["image", "audio", "video"]
        mock_provider.name = "fal"
        ai._fal_provider_instance = mock_provider
        ai._media_router_instance = None

        # Test fal-ai/ prefix
        await ai.ai_with_vision(prompt="test", model="fal-ai/flux/dev")
        assert ("fal", "fal-ai/flux/dev") in calls

        # Test fal/ prefix
        calls.clear()
        await ai.ai_with_vision(prompt="test", model="fal/recraft-v3")
        assert ("fal", "fal/recraft-v3") in calls

    def test_all_new_methods_exist(self, agent_with_ai):
        """Agent should have all new multimodal methods."""
        ai = AgentAI(agent_with_ai)

        # Check methods exist
        assert hasattr(ai, "ai_generate_video")
        assert hasattr(ai, "ai_transcribe_audio")
        assert hasattr(ai, "_fal_provider")

        # Check they're callable
        assert callable(ai.ai_generate_video)
        assert callable(ai.ai_transcribe_audio)
