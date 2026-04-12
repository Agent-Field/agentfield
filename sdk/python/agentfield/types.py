from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class AgentStatus(str, Enum):
    """Agent lifecycle status enum matching the Go backend"""

    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class HeartbeatData:
    """Enhanced heartbeat data with status information"""

    status: AgentStatus
    timestamp: str
    version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
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
    Simple helper for constructing execution headers when initiating AgentField calls.

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


# -----------------------------------------------------------------------------
# Discovery API Models
# -----------------------------------------------------------------------------


@dataclass
class DiscoveryPagination:
    limit: int
    offset: int
    has_more: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryPagination":
        return cls(
            limit=int(data.get("limit", 0)),
            offset=int(data.get("offset", 0)),
            has_more=bool(data.get("has_more", False)),
        )


@dataclass
class ReasonerCapability:
    id: str
    description: Optional[str]
    tags: List[str]
    input_schema: Optional[Dict[str, Any]]
    output_schema: Optional[Dict[str, Any]]
    examples: Optional[List[Dict[str, Any]]]
    invocation_target: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasonerCapability":
        return cls(
            id=data.get("id", ""),
            description=data.get("description"),
            tags=list(data.get("tags") or []),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            examples=[dict(x) for x in data.get("examples") or []] or None,
            invocation_target=data.get("invocation_target", ""),
        )


@dataclass
class SkillCapability:
    id: str
    description: Optional[str]
    tags: List[str]
    input_schema: Optional[Dict[str, Any]]
    invocation_target: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCapability":
        return cls(
            id=data.get("id", ""),
            description=data.get("description"),
            tags=list(data.get("tags") or []),
            input_schema=data.get("input_schema"),
            invocation_target=data.get("invocation_target", ""),
        )


@dataclass
class AgentCapability:
    agent_id: str
    base_url: str
    version: str
    health_status: str
    deployment_type: str
    last_heartbeat: str
    reasoners: List[ReasonerCapability] = field(default_factory=list)
    skills: List[SkillCapability] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapability":
        return cls(
            agent_id=data.get("agent_id", ""),
            base_url=data.get("base_url", ""),
            version=data.get("version", ""),
            health_status=data.get("health_status", ""),
            deployment_type=data.get("deployment_type", ""),
            last_heartbeat=data.get("last_heartbeat", ""),
            reasoners=[
                ReasonerCapability.from_dict(r) for r in data.get("reasoners") or []
            ],
            skills=[SkillCapability.from_dict(s) for s in data.get("skills") or []],
        )


@dataclass
class DiscoveryResponse:
    discovered_at: str
    total_agents: int
    total_reasoners: int
    total_skills: int
    pagination: DiscoveryPagination
    capabilities: List[AgentCapability]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveryResponse":
        return cls(
            discovered_at=str(data.get("discovered_at", "")),
            total_agents=int(data.get("total_agents", 0)),
            total_reasoners=int(data.get("total_reasoners", 0)),
            total_skills=int(data.get("total_skills", 0)),
            pagination=DiscoveryPagination.from_dict(data.get("pagination") or {}),
            capabilities=[
                AgentCapability.from_dict(cap) for cap in data.get("capabilities") or []
            ],
        )


@dataclass
class CompactCapability:
    id: str
    agent_id: str
    target: str
    tags: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompactCapability":
        return cls(
            id=data.get("id", ""),
            agent_id=data.get("agent_id", ""),
            target=data.get("target", ""),
            tags=list(data.get("tags") or []),
        )


@dataclass
class CompactDiscoveryResponse:
    discovered_at: str
    reasoners: List[CompactCapability]
    skills: List[CompactCapability]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompactDiscoveryResponse":
        return cls(
            discovered_at=str(data.get("discovered_at", "")),
            reasoners=[
                CompactCapability.from_dict(r) for r in data.get("reasoners") or []
            ],
            skills=[CompactCapability.from_dict(s) for s in data.get("skills") or []],
        )


@dataclass
class DiscoveryResult:
    format: str
    raw: str
    json: Optional[DiscoveryResponse] = None
    compact: Optional[CompactDiscoveryResponse] = None
    xml: Optional[str] = None


class HarnessConfig(BaseModel):
    provider: str = Field(
        ...,
        description='Coding agent provider: "claude-code" | "codex" | "gemini" | "opencode"',
    )
    model: str = Field(default="sonnet", description="Default model identifier.")
    max_turns: int = Field(default=30, description="Maximum agent iterations.")
    max_budget_usd: Optional[float] = Field(
        default=None, description="Cost cap in USD."
    )
    max_retries: int = Field(
        default=3, description="Maximum retry attempts for transient errors."
    )
    initial_delay: float = Field(
        default=1.0, description="Initial retry delay in seconds."
    )
    max_delay: float = Field(
        default=30.0, description="Maximum retry delay in seconds."
    )
    backoff_factor: float = Field(default=2.0, description="Retry backoff multiplier.")
    tools: List[str] = Field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        description="Default allowed tools.",
    )
    permission_mode: Optional[str] = Field(
        default=None, description='Permission mode: "plan" | "auto" | None'
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Default system prompt."
    )
    env: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the agent."
    )
    cwd: Optional[str] = Field(default=None, description="Default working directory.")
    project_dir: Optional[str] = Field(
        default=None,
        description=(
            "Project directory for the coding agent to explore (e.g. a target "
            "repository path). Maps to --dir in opencode. When set, cwd is used "
            "only for output file placement while project_dir controls the "
            "agent's working context."
        ),
    )
    codex_bin: str = Field(default="codex", description="Path to codex binary.")
    gemini_bin: str = Field(default="gemini", description="Path to gemini binary.")
    opencode_bin: str = Field(
        default="opencode", description="Path to opencode binary."
    )


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
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Creativity level (0.0-2.0). If None, uses model's default.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum response length. If None, uses model's default.",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Controls diversity via nucleus sampling. If None, uses model's default.",
    )
    stream: Optional[bool] = Field(
        default=None,
        description="Enable streaming response. If None, uses model's default.",
    )
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

    # Fal.ai settings
    fal_api_key: Optional[str] = Field(
        default=None,
        description="Fal.ai API key. If not set, uses FAL_KEY environment variable.",
    )
    video_model: str = Field(
        default="fal-ai/minimax-video/image-to-video",
        description="Default model for video generation.",
    )

    @computed_field
    @property
    def image_model(self) -> str:
        """Alias for vision_model - clearer name for image generation model."""
        return self.vision_model

    # Behavior settings
    timeout: Optional[int] = Field(
        default=None,
        description="Timeout for AI calls in seconds. If None, uses LiteLLM's default.",
    )
    retry_attempts: Optional[int] = Field(
        default=None,
        description="Number of retry attempts for failed AI calls. If None, uses LiteLLM's default.",
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds."
    )

    # Rate limiting configuration
    rate_limit_max_retries: int = Field(
        default=5,
        description="Maximum number of retries for rate limit errors.",
    )
    rate_limit_base_delay: float = Field(
        default=0.5,
        description="Base delay for rate limit exponential backoff in seconds.",
    )
    rate_limit_max_delay: float = Field(
        default=30.0,
        description="Maximum delay for rate limit backoff in seconds.",
    )
    rate_limit_jitter_factor: float = Field(
        default=0.25,
        description="Jitter factor for rate limit backoff (±25% randomization).",
    )
    rate_limit_circuit_breaker_threshold: int = Field(
        default=5,
        description="Number of consecutive rate limit failures before opening circuit breaker.",
    )
    rate_limit_circuit_breaker_timeout: int = Field(
        default=30, description="Circuit breaker timeout in seconds."
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

        fallback_context = self._MODEL_CONTEXT_LIMITS.get(target_model)

        try:
            import litellm

            litellm.suppress_debug_info = True
            # Fetch model info once and cache it
            info = litellm.get_model_info(target_model)

        except Exception:
            info = None  # Ensure info is undefined outside except

        if info is not None:
            context_length = (
                getattr(info, "max_tokens", None) or fallback_context or 131072
            )
            max_output = getattr(info, "max_output_tokens", None) or getattr(
                info, "max_completion_tokens", None
            )
        else:
            context_length = fallback_context or 8192
            max_output = None

        if not max_output:
            # Default to a conservative completion window capped at 32K
            max_output = min(32768, max(2048, context_length // 4))

        limits = {
            "context_length": context_length,
            "max_output_tokens": max_output,
        }

        self.model_limits_cache[target_model] = limits
        return limits

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

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # OpenAI Responses API expects max_completion_tokens instead of max_tokens
        model_name = params.get("model") or self.model
        provider = (
            model_name.split("/", 1)[0] if model_name and "/" in model_name else None
        )
        if provider == "openai" and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        return params

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
    """Represents a memory value stored in the AgentField system."""

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

    id: Optional[str] = None
    type: Optional[str] = None
    timestamp: Optional[str] = None
    scope: str = ""
    scope_id: str = ""
    key: str = ""
    action: str = ""
    data: Optional[Any] = None
    previous_data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def new_value(self) -> Optional[Any]:
        """Backward compatibility alias for data."""
        return self.data

    @property
    def old_value(self) -> Optional[Any]:
        """Backward compatibility alias for previous_data."""
        return self.previous_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryChangeEvent":
        return cls(
            id=data.get("id"),
            type=data.get("type"),
            timestamp=data.get("timestamp"),
            scope=data.get("scope", ""),
            scope_id=data.get("scope_id", ""),
            key=data.get("key", ""),
            action=data.get("action", ""),
            data=data.get("data"),
            previous_data=data.get("previous_data"),
            metadata=data.get("metadata") or {},
        )
