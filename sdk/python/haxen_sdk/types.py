from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentStatus(str, Enum):
    """Agent lifecycle status enum matching the Go backend"""

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class MCPServerHealth:
    """MCP server health information for heartbeat reporting"""

    alias: str
    status: str
    tool_count: int = 0
    port: Optional[int] = None
    process_id: Optional[int] = None
    started_at: Optional[str] = None
    last_health_check: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HeartbeatData:
    """Enhanced heartbeat data with status and MCP information"""

    status: AgentStatus
    mcp_servers: List[MCPServerHealth]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "mcp_servers": [server.to_dict() for server in self.mcp_servers],
            "timestamp": self.timestamp,
        }


@dataclass
class MemoryConfig:
    auto_inject: List[str]
    memory_retention: str
    cache_results: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReasonerDefinition:
    id: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    memory_config: Optional[MemoryConfig] = None  # Optional for now, can be added later

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.memory_config is not None:
            data["memory_config"] = self.memory_config.to_dict()
        return data


@dataclass
class SkillDefinition:
    id: str
    input_schema: Dict[str, Any]
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionHeaders:
    """
    Simple helper for constructing execution headers when initiating Haxen calls.

    This replaces the wide workflow context structure with the minimal information
    required by the run-based execution pipeline.
    """

    run_id: str
    session_id: Optional[str] = None
    actor_id: Optional[str] = None
    parent_execution_id: Optional[str] = None

    def to_headers(self) -> Dict[str, str]:
        headers = {"X-Run-ID": self.run_id}
        if self.parent_execution_id:
            headers["X-Parent-Execution-ID"] = self.parent_execution_id
        if self.session_id:
            headers["X-Session-ID"] = self.session_id
        if self.actor_id:
            headers["X-Actor-ID"] = self.actor_id
        return headers


@dataclass
class WebhookConfig:
    """Webhook registration details for async executions."""

    url: str
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"url": self.url}
        if self.secret:
            payload["secret"] = self.secret
        if self.headers:
            payload["headers"] = self.headers
        return payload


class AIConfig(BaseModel):
    """
    Configuration for AI calls, defining default models, temperatures, and other parameters.
    These settings can be overridden at the method call level.

    Leverages LiteLLM's standard environment variable handling for API keys:
    - OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_OPENAI_API_KEY, etc.
    - LiteLLM automatically detects and uses these standard environment variables

    All fields have sensible defaults, so you can create an AIConfig with minimal configuration:

    Examples:
        # Minimal configuration - uses all defaults
        AIConfig()

        # Override just the API key
        AIConfig(api_key="your-key")

        # Override specific models for multimodal tasks
        AIConfig(audio_model="tts-1-hd", vision_model="dall-e-3")
    """

    model: str = Field(
        default="gpt-4o",
        description="Default LLM model to use (e.g., 'gpt-4o', 'claude-3-sonnet').",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Creativity level (0.0-2.0)."
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum response length."
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Controls diversity via nucleus sampling.",
    )
    stream: bool = Field(default=False, description="Enable streaming response.")
    response_format: Literal["auto", "json", "text"] = Field(
        default="auto", description="Desired response format."
    )

    # Multimodal settings - updated with better defaults for TTS
    vision_model: str = Field(
        default="dall-e-3", description="Model for vision/image generation tasks."
    )
    audio_model: str = Field(
        default="tts-1",
        description="Model for audio generation (tts-1, tts-1-hd, gpt-4o-mini-tts).",
    )
    image_quality: Literal["low", "high"] = Field(
        default="high", description="Quality for image generation/processing."
    )
    audio_format: str = Field(
        default="wav", description="Default format for audio output (wav, mp3)."
    )

    # Behavior settings
    timeout: int = Field(default=60, description="Timeout for AI calls in seconds.")
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed AI calls."
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds."
    )

    # Rate limiting configuration
    rate_limit_max_retries: int = Field(
        default=20,
        description="Maximum number of retries for rate limit errors (allows up to ~20 minutes of retries).",
    )
    rate_limit_base_delay: float = Field(
        default=1.0,
        description="Base delay for rate limit exponential backoff in seconds.",
    )
    rate_limit_max_delay: float = Field(
        default=300.0,
        description="Maximum delay for rate limit backoff in seconds (5 minutes).",
    )
    rate_limit_jitter_factor: float = Field(
        default=0.25,
        description="Jitter factor for rate limit backoff (±25% randomization).",
    )
    rate_limit_circuit_breaker_threshold: int = Field(
        default=10,
        description="Number of consecutive rate limit failures before opening circuit breaker.",
    )
    rate_limit_circuit_breaker_timeout: int = Field(
        default=300, description="Circuit breaker timeout in seconds (5 minutes)."
    )
    enable_rate_limit_retry: bool = Field(
        default=True, description="Enable automatic retry for rate limit errors."
    )

    # Cost controls
    max_cost_per_call: Optional[float] = Field(
        default=None, description="Maximum cost per AI call in USD."
    )
    daily_budget: Optional[float] = Field(
        default=None, description="Daily budget for AI calls in USD."
    )

    # Memory integration (defaults for auto-injection)
    auto_inject_memory: List[str] = Field(
        default_factory=list,
        description="List of memory scopes to auto-inject (e.g., ['workflow', 'session']).",
    )
    preserve_context: bool = Field(
        default=True,
        description="Whether to preserve conversation context across calls.",
    )
    context_window: int = Field(
        default=10, description="Number of previous messages to include in context."
    )

    # LiteLLM configuration - these get passed directly to litellm.completion()
    api_key: Optional[str] = Field(
        default=None, description="API key override (if not using env vars)"
    )
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")
    api_version: Optional[str] = Field(
        default=None, description="API version (for Azure)"
    )
    organization: Optional[str] = Field(
        default=None, description="Organization ID (for OpenAI)"
    )

    # Additional LiteLLM parameters that can be overridden
    litellm_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters to pass to LiteLLM"
    )
    fallback_models: List[str] = Field(
        default_factory=list,
        description="List of models to fallback to if primary fails.",
    )

    # Model limits caching for optimization
    model_limits_cache: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Cached model limits to avoid repeated API calls",
    )
    avg_chars_per_token: int = Field(
        default=4, description="Average characters per token for approximation"
    )
    max_input_tokens: Optional[int] = Field(
        default=None,
        description="Maximum input context tokens (overrides auto-detection)",
    )

    # Pydantic V2: allow fields that start with `model_`
    model_config = {"protected_namespaces": ()}

    # Fallback model context mappings for when LiteLLM detection fails
    _MODEL_CONTEXT_LIMITS = {
        # OpenRouter Gemini models
        "openrouter/google/gemini-2.5-flash-lite": 1048576,  # 1M tokens
        "openrouter/google/gemini-2.5-flash": 1048576,  # 1M tokens
        "openrouter/google/gemini-2.5-pro": 2097152,  # 2M tokens
        "openrouter/google/gemini-1.5-pro": 2097152,  # 2M tokens
        "openrouter/google/gemini-1.5-flash": 1048576,  # 1M tokens
        # Direct Gemini models
        "gemini-2.5-flash": 1048576,
        "gemini-2.5-pro": 2097152,
        "gemini-1.5-pro": 2097152,
        "gemini-1.5-flash": 1048576,
        # OpenAI models
        "openrouter/openai/gpt-4.1-mini": 128000,
        "openrouter/openai/gpt-4o": 128000,
        "openrouter/openai/gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        # Claude models
        "openrouter/anthropic/claude-3.5-sonnet": 200000,
        "openrouter/anthropic/claude-3-opus": 200000,
        "claude-3.5-sonnet": 200000,
        "claude-3-opus": 200000,
    }

    async def get_model_limits(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch and cache model limits to avoid repeated API calls.

        Args:
            model: Model to get limits for (defaults to self.model)

        Returns:
            Dict containing context_length and max_output_tokens
        """
        target_model = model or self.model

        # Return cached limits if available
        if target_model in self.model_limits_cache:
            return self.model_limits_cache[target_model]

        try:
            import litellm

            # Fetch model info once and cache it
            info = litellm.get_model_info(target_model)
            limits = {
                "context_length": getattr(
                    info, "max_tokens", 131072
                ),  # Default fallback
                "max_output_tokens": getattr(info, "max_output_tokens", None),
            }

            # Cache the limits
            self.model_limits_cache[target_model] = limits
            return limits

        except Exception:
            # Fallback to conservative defaults if model info fetch fails
            fallback_limits = {
                "context_length": 8192,  # Conservative default
                "max_output_tokens": 4096,
            }
            self.model_limits_cache[target_model] = fallback_limits
            return fallback_limits

    def trim_by_chars(self, text: str, limit: int, head_ratio: float = 0.2) -> str:
        """
        Trim text by character count using head/tail ratio to preserve important content.

        Args:
            text: Text to trim
            limit: Character limit
            head_ratio: Ratio of content to keep from the beginning (0.0-1.0)

        Returns:
            Trimmed text with head and tail preserved
        """
        if len(text) <= limit:
            return text

        head_chars = int(limit * head_ratio)
        tail_chars = int(limit * (1 - head_ratio))

        head = text[:head_chars]
        tail = text[-tail_chars:]

        return head + "\n…TRIMMED…\n" + tail

    def get_safe_prompt_chars(
        self, model: Optional[str] = None, max_output_tokens: Optional[int] = None
    ) -> int:
        """
        Calculate safe character limit for prompts based on cached model limits.

        Args:
            model: Model to calculate for (defaults to self.model)
            max_output_tokens: Override for max output tokens

        Returns:
            Safe character limit for prompts
        """
        # This is a synchronous method that uses cached limits
        target_model = model or self.model

        # Use cached limits if available, otherwise use conservative defaults
        if target_model in self.model_limits_cache:
            limits = self.model_limits_cache[target_model]
            max_ctx = limits["context_length"]
            max_out = max_output_tokens or limits["max_output_tokens"] or 0
        else:
            # Conservative defaults if not cached yet
            max_ctx = 8192
            max_out = max_output_tokens or 4096

        # Calculate safe prompt character limit
        safe_prompt_chars = (max_ctx - max_out) * self.avg_chars_per_token
        return max(safe_prompt_chars, 1000)  # Ensure minimum viable prompt size

    def get_litellm_params(
        self, messages: Optional[List[Dict]] = None, **overrides
    ) -> Dict[str, Any]:
        """
        Get parameters formatted for LiteLLM, with runtime overrides and smart token management.
        LiteLLM handles environment variable detection automatically.
        """
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
            "timeout": self.timeout,
            "num_retries": self.retry_attempts,
        }

        # Add optional parameters if set
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_version:
            params["api_version"] = self.api_version
        if self.organization:
            params["organization"] = self.organization

        # Add response format if not auto
        if self.response_format != "auto":
            params["response_format"] = {"type": self.response_format}

        # Add any additional litellm params
        params.update(self.litellm_params)

        # Apply runtime overrides (highest priority)
        params.update(overrides)

        # Smart max_tokens calculation if messages provided and not explicitly set
        if messages and "max_tokens" not in overrides and not self.max_tokens:
            target_model = params.get("model", self.model)
            if target_model in self.model_limits_cache:
                limits = self.model_limits_cache[target_model]
                max_ctx = limits["context_length"]
                max_out = limits["max_output_tokens"]

                # Estimate prompt tokens from character count
                prompt_text = ""
                for msg in messages:
                    if isinstance(msg.get("content"), str):
                        prompt_text += msg["content"]
                    elif isinstance(msg.get("content"), list):
                        for item in msg["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                prompt_text += item.get("text", "")

                estimated_prompt_tokens = len(prompt_text) // self.avg_chars_per_token

                # Calculate safe max_tokens
                if max_out:
                    safe_max_tokens = min(
                        max_out, max_ctx - estimated_prompt_tokens - 100
                    )  # 100 token buffer
                else:
                    safe_max_tokens = max_ctx - estimated_prompt_tokens - 100

                if safe_max_tokens > 0:
                    params["max_tokens"] = safe_max_tokens

        # Remove None values
        return {k: v for k, v in params.items() if v is not None}

    def copy(
        self,
        *,
        include: Optional[Any] = None,
        exclude: Optional[Any] = None,
        update: Optional[Dict[str, Any]] = None,
        deep: bool = False,
    ) -> "AIConfig":
        """Create a copy of the configuration"""
        return super().copy(include=include, exclude=exclude, update=update, deep=deep)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return self.model_dump()

    @classmethod
    def from_env(cls, **overrides) -> "AIConfig":
        """
        Create AIConfig with smart defaults, letting LiteLLM handle env vars.
        This is the recommended way to create configs in production.
        """
        config = cls(**overrides)
        return config


@dataclass
class MemoryValue:
    """Represents a memory value stored in the Haxen system."""

    key: str
    data: Any
    scope: str
    scope_id: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryValue":
        return cls(**data)


@dataclass
class MemoryChangeEvent:
    """Represents a memory change event for reactive programming."""

    scope: str
    scope_id: str
    key: str
    action: str  # "set" or "delete"
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryChangeEvent":
        return cls(**data)
