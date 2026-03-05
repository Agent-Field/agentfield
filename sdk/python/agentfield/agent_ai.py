from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

if TYPE_CHECKING:
    from agentfield.multimodal_response import MultimodalResponse
    from agentfield.tool_calling import ToolCallConfig

import requests
from agentfield.agent_utils import AgentUtils
from agentfield.logger import log_debug, log_error, log_warn
from agentfield.rate_limiter import StatelessRateLimiter
from httpx import HTTPStatusError
from pydantic import BaseModel

# Lazy loading for heavy LLM libraries to reduce memory footprint
# These are only imported when AI features are actually used
_litellm = None
_openai = None


def _get_litellm():
    """Lazy import of litellm - only loads when AI features are used."""
    global _litellm
    if _litellm is None:
        try:
            import litellm

            litellm.suppress_debug_info = True
            _litellm = litellm
        except Exception:  # pragma: no cover

            class _LiteLLMStub:
                pass

            _litellm = _LiteLLMStub()
    return _litellm


def _get_openai():
    """Lazy import of openai - only loads when AI features are used."""
    global _openai
    if _openai is None:
        try:
            import openai

            _openai = openai
        except Exception:  # pragma: no cover

            class _OpenAIStub:
                class OpenAI:
                    pass

            _openai = _OpenAIStub()
    return _openai


# Backward compatibility: expose as module-level but with lazy loading
class _LazyModule:
    """Lazy module proxy that defers import until attribute access."""

    def __init__(self, loader):
        self._loader = loader
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = self._loader()
        return getattr(self._module, name)


litellm = _LazyModule(_get_litellm)
openai = _LazyModule(_get_openai)


class AgentAI:
    """AI/LLM Integration functionality for AgentField Agent"""

    def __init__(self, agent_instance):
        """
        Initialize AgentAI with a reference to the main agent instance.

        Args:
            agent_instance: The main Agent instance
        """
        self.agent = agent_instance
        self._initialization_complete = False
        self._rate_limiter = None
        self._fal_provider_instance = None

    @property
    def _fal_provider(self):
        """
        Lazy-initialized Fal provider for image, audio, and video generation.

        Returns:
            FalProvider: Configured Fal.ai provider instance
        """
        if self._fal_provider_instance is None:
            from agentfield.media_providers import FalProvider

            self._fal_provider_instance = FalProvider(
                api_key=self.agent.ai_config.fal_api_key
            )
        return self._fal_provider_instance

    def _get_rate_limiter(self) -> StatelessRateLimiter:
        """
        Get or create the rate limiter instance based on current configuration.

        Returns:
            StatelessRateLimiter: Configured rate limiter instance
        """
        if self._rate_limiter is None:
            config = self.agent.ai_config
            self._rate_limiter = StatelessRateLimiter(
                max_retries=config.rate_limit_max_retries,
                base_delay=config.rate_limit_base_delay,
                max_delay=config.rate_limit_max_delay,
                jitter_factor=config.rate_limit_jitter_factor,
                circuit_breaker_threshold=config.rate_limit_circuit_breaker_threshold,
                circuit_breaker_timeout=config.rate_limit_circuit_breaker_timeout,
            )
        return self._rate_limiter

    async def _ensure_model_limits_cached(self):
        """
        Ensure model limits are cached for the current model configuration.
        This is called once during the first AI call to avoid startup delays.
        """
        if not self._initialization_complete:
            try:
                # Cache limits for the default model
                await self.agent.ai_config.get_model_limits()

                # Cache limits for multimodal models if different
                if self.agent.ai_config.audio_model != self.agent.ai_config.model:
                    await self.agent.ai_config.get_model_limits(
                        self.agent.ai_config.audio_model
                    )

                if self.agent.ai_config.vision_model != self.agent.ai_config.model:
                    await self.agent.ai_config.get_model_limits(
                        self.agent.ai_config.vision_model
                    )

                self._initialization_complete = True

            except Exception as e:
                log_debug(f"Failed to cache model limits: {e}")
                # Continue with fallback defaults
                self._initialization_complete = True

    async def ai(
        self,
        *args: Any,
        system: Optional[str] = None,
        user: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        response_format: Optional[Union[Literal["auto", "json", "text"], Dict]] = None,
        context: Optional[Dict] = None,
        memory_scope: Optional[List[str]] = None,
        tools: Optional[
            Union[
                Literal["discover"],
                ToolCallConfig,
                Dict[str, Any],
                List[Any],
            ]
        ] = None,
        max_turns: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """
        Universal AI method supporting multimodal inputs with intelligent type detection.

        This method provides a flexible interface for interacting with various LLMs,
        supporting text, image, audio, and file inputs. It intelligently detects
        input types and applies a hierarchical configuration system.

        Args:
            *args: Flexible inputs - text, images, audio, files, or mixed content.
                   - str: Text content, URLs, or file paths (auto-detected).
                   - bytes: Binary data (images, audio, documents).
                   - dict: Structured input with explicit keys (e.g., {"image": "url"}).
                   - list: Multimodal conversation or content list.

            system (str, optional): System prompt for AI behavior.
            user (str, optional): User message (alternative to positional args).
            schema (Type[BaseModel], optional): Pydantic model for structured output validation.
            model (str, optional): Override default model (e.g., "gpt-4", "claude-3").
            temperature (float, optional): Creativity level (0.0-2.0).
            max_tokens (int, optional): Maximum response length.
            stream (bool, optional): Enable streaming response.
            response_format (str, optional): Desired response format ('auto', 'json', 'text').
            context (Dict, optional): Additional context data to pass to the LLM.
            memory_scope (List[str], optional): Memory scopes to inject (e.g., ['workflow', 'session', 'reasoner']).
            tools: Tool definitions for LLM tool calling. Accepts:
                - "discover": auto-discover all tools from the control plane
                - DiscoveryResponse: use pre-fetched discovery results
                - list of capabilities: ReasonerCapability/SkillCapability/AgentCapability
                - list of dicts: raw OpenAI-format tool schemas
                - ToolCallConfig or dict: discover with filtering/progressive options
            max_turns (int, optional): Maximum LLM turns in the tool-call loop (default: 10).
            max_tool_calls (int, optional): Maximum total tool calls allowed (default: 25).
            **kwargs: Additional provider-specific parameters to pass to the LLM.

        Returns:
            Any: The AI response - raw text, structured object (if schema), or a stream.

        Examples:
            # Simple text input
            response = await app.ai("Summarize this document.")

            # System and user prompts
            response = await app.ai(
                system="You are a helpful assistant.",
                user="What is the capital of France?"
            )

            # Multimodal input with auto-detection (image URL and text)
            response = await app.ai(
                "Describe this image:",
                "https://example.com/image.jpg"
            )

            # Multimodal input with file path (audio)
            response = await app.ai(
                "Transcribe this audio:",
                "./audio.mp3"
            )

            # Structured output with Pydantic schema
            class SentimentResult(BaseModel):
                sentiment: str
                confidence: float

            result = await app.ai(
                "Analyze the sentiment of 'I love this product!'",
                schema=SentimentResult
            )

            # Override default AI configuration parameters
            response = await app.ai(
                "Generate a creative story.",
                model="gpt-4-turbo",
                temperature=0.9,
                max_tokens=500,
                stream=True
            )

            # Complex multimodal conversation
            response = await app.ai([
                {"role": "system", "content": "You are a visual assistant."},
                {"role": "user", "content": "What do you see here?"},
                "https://example.com/chart.png",
                {"role": "user", "content": "Can you explain the trend?"}
            ])
        """
        # Apply hierarchical configuration: Agent defaults < Method overrides < Runtime overrides
        final_config = self.agent.ai_config.copy(deep=True)

        # Default enable rate limit retry unless explicitly set to False
        if (
            not hasattr(final_config, "enable_rate_limit_retry")
            or final_config.enable_rate_limit_retry is None
        ):
            final_config.enable_rate_limit_retry = True

        # Apply method-level overrides
        if model:
            final_config.model = model
        if temperature is not None:
            final_config.temperature = temperature
        if max_tokens is not None:
            final_config.max_tokens = max_tokens
        if stream is not None:
            final_config.stream = stream
        if response_format is not None:
            if isinstance(response_format, str):
                final_config.response_format = response_format

        # TODO: Integrate memory injection based on memory_scope and self.memory_config
        # For now, just pass context if provided
        if context:
            # This would be where memory data is merged into the context
            pass

        # Prepare messages for LiteLLM
        messages = []

        # If a schema is provided, augment the system prompt with strict schema adherence instructions and schema context
        if schema:
            # Generate a readable JSON schema string using the modern Pydantic API
            try:
                schema_dict = schema.model_json_schema()
                schema_json = json.dumps(schema_dict, indent=2)
            except Exception:
                schema_json = str(schema)
            schema_instruction = (
                "IMPORTANT: You must exactly adhere to the output schema provided below. "
                "Do not add or omit any fields. Output must be valid JSON matching the schema. "
                "If a field is required in the schema, it must be present in the output. "
                "If a field is not in the schema, do NOT include it in the output. "
                "Here is the output schema you must follow:\n"
                f"{schema_json}\n"
                "Repeat: Output ONLY valid JSON matching the schema above. Do not include any extra text or explanation."
            )
            # Merge with any user-provided system prompt
            if system:
                system_prompt = f"{system}\n\n{schema_instruction}"
            else:
                system_prompt = schema_instruction
            messages.append({"role": "system", "content": system_prompt})
        else:
            if system:
                messages.append({"role": "system", "content": system})

        # Handle flexible user input with intelligent processing
        if user:
            messages.append({"role": "user", "content": user})
        elif args:
            processed_content = self._process_multimodal_args(args)
            if processed_content:
                messages.extend(processed_content)

        litellm_module = litellm if hasattr(litellm, "acompletion") else None

        # Ensure model limits are cached (done once per instance)
        await self._ensure_model_limits_cached()

        # Apply prompt trimming using LiteLLM's token-aware utility when available.
        utils_module = (
            getattr(litellm_module, "utils", None) if litellm_module else None
        )
        token_counter = (
            getattr(utils_module, "token_counter", None) if utils_module else None
        )
        trim_messages = (
            getattr(utils_module, "trim_messages", None) if utils_module else None
        )

        if token_counter is None:

            def token_counter(model: str, messages: List[dict]) -> int:
                return len(json.dumps(messages))

        if trim_messages is None:

            def trim_messages(
                messages: List[dict], model: str, max_tokens: int
            ) -> List[dict]:
                return messages

        # Determine model context length using multiple fallback strategies
        model_context_length = None

        # Strategy 1: Use explicit max_input_tokens from config
        if hasattr(final_config, "max_input_tokens") and final_config.max_input_tokens:
            model_context_length = final_config.max_input_tokens

        # Strategy 3: Use fallback model mappings
        if not model_context_length and hasattr(final_config, "_MODEL_CONTEXT_LIMITS"):
            candidate_limit = final_config._MODEL_CONTEXT_LIMITS.get(final_config.model)
            if candidate_limit:
                model_context_length = candidate_limit

        # Strategy 4: Conservative fallback with warning
        if not model_context_length:
            model_context_length = 10192  # More reasonable than 4096

        # Calculate safe input token limit: context_length - max_output_tokens - buffer
        output_tokens = (
            final_config.max_tokens or 7096
        )  # Default output if not specified
        buffer_tokens = 100  # Small buffer for safety

        safe_input_limit = max(
            1000, model_context_length - output_tokens - buffer_tokens
        )

        # Validate the calculation makes sense
        if safe_input_limit < 1000:
            safe_input_limit = 1000

        # Count actual prompt tokens using LiteLLM's token counter
        try:
            actual_prompt_tokens = token_counter(
                model=final_config.model, messages=messages
            )
        except Exception as e:
            log_debug(f"Could not count prompt tokens, proceeding with trimming: {e}")
            actual_prompt_tokens = (
                safe_input_limit + 1
            )  # Force trimming if we can't count

        # Only trim if necessary based on actual token count
        if actual_prompt_tokens > safe_input_limit:
            trimmed_messages = trim_messages(
                messages, final_config.model, max_tokens=safe_input_limit
            )
            if len(trimmed_messages) != len(messages) or any(
                m1 != m2 for m1, m2 in zip(messages, trimmed_messages)
            ):
                messages = trimmed_messages
        else:
            pass

        # Prepare LiteLLM parameters using the config's method
        # This leverages LiteLLM's standard environment variable handling and smart token management
        litellm_params = final_config.get_litellm_params(
            messages=messages,
            **kwargs,  # Runtime overrides have highest priority
        )

        # Ensure messages are always included in the final params
        litellm_params["messages"] = messages

        if schema:
            # Convert Pydantic model to JSON schema format for LiteLLM
            # This workaround prevents "Object of type ModelMetaclass is not JSON serializable" error
            # See: https://github.com/BerriAI/litellm/issues/6830
            litellm_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": schema.model_json_schema(),
                    "name": schema.__name__,
                    "strict": True,
                },
            }

        # Tool-calling loop: if tools= is provided, enter the discover->call loop
        if tools is not None:
            # Streaming is not supported with tool-calling
            if final_config.stream:
                raise ValueError(
                    "Streaming is not supported with tool-calling. "
                    "Use tools= OR stream=True, not both."
                )

            from agentfield.tool_calling import (
                ToolCallResponse,
                _build_tool_config,
                execute_tool_call_loop,
            )

            tool_schemas, tool_config, needs_lazy = _build_tool_config(
                tools, self.agent
            )

            # Apply per-call overrides
            if max_turns is not None:
                tool_config.max_turns = max_turns
            if max_tool_calls is not None:
                tool_config.max_tool_calls = max_tool_calls

            async def _tool_loop_completion(params):
                """Make an LLM call with rate limiting and model fallbacks."""
                if litellm_module is None:
                    raise ImportError(
                        "litellm is not installed. Please install it with `pip install litellm`."
                    )

                async def _make_call():
                    return await litellm_module.acompletion(**params)

                async def _call_with_fallbacks():
                    fallback_models = getattr(final_config, "fallback_models", None)
                    if not fallback_models and getattr(
                        final_config, "final_fallback_model", None
                    ):
                        fallback_models = [final_config.final_fallback_model]

                    if fallback_models:
                        all_models = [params.get("model", final_config.model)] + list(
                            fallback_models
                        )
                        last_exception = None
                        for m in all_models:
                            try:
                                params["model"] = m
                                return await _make_call()
                            except Exception as e:
                                log_debug(
                                    f"Tool loop: model {m} failed with {e}, trying next..."
                                )
                                last_exception = e
                                continue
                        if last_exception:
                            raise last_exception
                    return await _make_call()

                if final_config.enable_rate_limit_retry:
                    rate_limiter = self._get_rate_limiter()
                    return await rate_limiter.execute_with_retry(_call_with_fallbacks)
                return await _call_with_fallbacks()

            resp, trace = await execute_tool_call_loop(
                agent=self.agent,
                messages=messages,
                tools=tool_schemas,
                config=tool_config,
                needs_lazy_hydration=needs_lazy,
                litellm_params=litellm_params,
                make_completion=_tool_loop_completion,
            )

            if schema:
                try:
                    content = resp.choices[0].message.content
                    json_data = json.loads(str(content))
                    return schema(**json_data)
                except (json.JSONDecodeError, ValueError):
                    pass

            return ToolCallResponse(resp, trace)

        # Define the LiteLLM call function for rate limiter
        async def _make_litellm_call():
            if litellm_module is None:
                raise ImportError(
                    "litellm is not installed. Please install it with `pip install litellm`."
                )
            return await litellm_module.acompletion(**litellm_params)

        async def _execute_with_fallbacks():
            # Check for configured fallback models in AI config
            fallback_models = getattr(final_config, "fallback_models", None)
            if not fallback_models and getattr(
                final_config, "final_fallback_model", None
            ):
                # If only a final model is provided, treat it as a fallback list of one
                fallback_models = [final_config.final_fallback_model]

            if fallback_models:
                # Ensure each fallback call has a valid provider
                all_models = [final_config.model] + list(fallback_models)
                last_exception = None
                for m in all_models:
                    try:
                        if "/" not in m:
                            log_debug(
                                f"Skipping model {m} - no provider specified in model name"
                            )
                            raise ValueError(
                                f"Invalid model spec: '{m}'. Must include provider prefix, e.g. 'openai/gpt-4'."
                            )
                        litellm_params["model"] = m
                        return await _make_litellm_call()
                    except Exception as e:
                        log_debug(
                            f"Model {m} failed with {e}, trying next fallback if available..."
                        )
                        last_exception = e
                        continue
                # If all models fail, re-raise the last exception
                if last_exception:
                    raise last_exception
            else:
                # No fallbacks configured, just make the call
                if "/" not in final_config.model:
                    raise ValueError(
                        f"Invalid model spec: '{final_config.model}'. Must include provider prefix, e.g. 'openai/gpt-4'."
                    )
                return await _make_litellm_call()

        # Maximum retries for transient parse failures (malformed JSON from LLM)
        max_parse_retries = 2

        async def _execute_and_parse():
            """Execute LLM call and parse response. Raised ValueError triggers parse retry."""
            if final_config.enable_rate_limit_retry:
                rate_limiter = self._get_rate_limiter()
                try:
                    resp = await rate_limiter.execute_with_retry(
                        _execute_with_fallbacks
                    )
                except Exception as e:
                    log_debug(f"LiteLLM call failed after retries: {e}")
                    raise
            else:
                try:
                    resp = await _execute_with_fallbacks()
                except HTTPStatusError as e:
                    log_debug(
                        f"LiteLLM HTTP call failed: {e.response.status_code} - {e.response.text}"
                    )
                    raise
                except requests.exceptions.RequestException as e:
                    log_debug(f"LiteLLM network call failed: {e}")
                    if e.response is not None:
                        log_debug(f"Response status: {e.response.status_code}")
                        log_debug(f"Response text: {e.response.text}")
                    raise
                except Exception as e:
                    log_debug(f"LiteLLM call failed: {e}")
                    raise

            if final_config.stream:
                return resp

            from .multimodal_response import detect_multimodal_response

            multimodal_response = detect_multimodal_response(resp)

            if schema:
                try:
                    json_data = json.loads(str(multimodal_response.text))
                    return schema(**json_data)
                except (json.JSONDecodeError, ValueError) as parse_error:
                    log_error(f"Failed to parse JSON response: {parse_error}")
                    log_debug(f"Raw response: {multimodal_response.text}")
                    json_match = re.search(
                        r"\{.*\}", str(multimodal_response.text), re.DOTALL
                    )
                    if json_match:
                        try:
                            json_data = json.loads(json_match.group())
                            return schema(**json_data)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    raise ValueError(
                        f"Could not parse structured response: {multimodal_response.text}"
                    )

            return multimodal_response

        # Retry on parse failures (malformed LLM JSON output)
        last_parse_error = None
        for attempt in range(max_parse_retries + 1):
            try:
                return await _execute_and_parse()
            except ValueError as e:
                if schema and "Could not parse structured response" in str(e):
                    last_parse_error = e
                    if attempt < max_parse_retries:
                        log_debug(
                            f"Parse retry {attempt + 1}/{max_parse_retries}: LLM returned malformed JSON, retrying..."
                        )
                        continue
                raise
        raise last_parse_error

    def _process_multimodal_args(self, args: tuple) -> List[Dict[str, Any]]:
        """Process multimodal arguments into LiteLLM-compatible message format"""
        from agentfield.multimodal import Audio, File, Image, Text

        messages = []
        user_content = []

        for arg in args:
            # Handle our multimodal input classes first
            if isinstance(arg, Text):
                user_content.append({"type": "text", "text": arg.text})

            elif isinstance(arg, Image):
                if isinstance(arg.image_url, dict):
                    user_content.append(
                        {"type": "image_url", "image_url": arg.image_url}
                    )
                else:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": arg.image_url, "detail": "high"},
                        }
                    )

            elif isinstance(arg, Audio):
                # Handle audio input according to LiteLLM GPT-4o-audio pattern
                user_content.append(
                    {"type": "input_audio", "input_audio": arg.input_audio}
                )

            elif isinstance(arg, File):
                # For now, treat files as text references
                if isinstance(arg.file, dict):
                    file_info = arg.file
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"[File: {file_info.get('url', 'unknown')}]",
                        }
                    )
                else:
                    user_content.append({"type": "text", "text": f"[File: {arg.file}]"})

            else:
                # Fall back to automatic detection for raw inputs
                detected_type = AgentUtils.detect_input_type(arg)

                if detected_type == "text":
                    user_content.append({"type": "text", "text": arg})

                elif detected_type == "image_url":
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": arg, "detail": "high"},
                        }
                    )

                elif detected_type == "image_file":
                    # Convert file to base64 data URL
                    try:
                        import base64

                        with open(arg, "rb") as f:
                            image_data = base64.b64encode(f.read()).decode()
                        ext = os.path.splitext(arg)[1].lower()
                        mime_type = AgentUtils.get_mime_type(ext)
                        data_url = f"data:{mime_type};base64,{image_data}"
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            }
                        )
                    except Exception as e:
                        log_warn(f"Could not read image file {arg}: {e}")
                        user_content.append(
                            {"type": "text", "text": f"[Image file: {arg}]"}
                        )

                elif detected_type == "audio_file":
                    # Convert audio file to LiteLLM input_audio format
                    try:
                        import base64

                        with open(arg, "rb") as f:
                            audio_data = base64.b64encode(f.read()).decode()

                        # Detect format from extension
                        ext = os.path.splitext(arg)[1].lower().lstrip(".")
                        audio_format = (
                            ext if ext in ["wav", "mp3", "flac", "ogg"] else "wav"
                        )

                        user_content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data,
                                    "format": audio_format,
                                },
                            }
                        )
                    except Exception as e:
                        log_warn(f"Could not read audio file {arg}: {e}")
                        user_content.append(
                            {
                                "type": "text",
                                "text": f"[Audio file: {os.path.basename(arg)}]",
                            }
                        )

                elif detected_type == "document_file":
                    # For documents, we might need to extract text
                    # For now, just reference the file
                    user_content.append(
                        {
                            "type": "text",
                            "text": f"[Document file: {os.path.basename(arg)}]",
                        }
                    )

                elif detected_type == "image_base64":
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": arg, "detail": "high"},
                        }
                    )

                elif detected_type == "audio_base64":
                    # Extract format and data from data URL
                    try:
                        if arg.startswith("data:audio/"):
                            # Parse data URL: data:audio/wav;base64,<data>
                            header, data = arg.split(",", 1)
                            format_part = header.split(";")[0].split("/")[1]
                            user_content.append(
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": data,
                                        "format": format_part,
                                    },
                                }
                            )
                        else:
                            user_content.append(
                                {"type": "text", "text": "[Audio data provided]"}
                            )
                    except Exception as e:
                        log_warn(f"Could not process audio base64: {e}")
                        user_content.append(
                            {"type": "text", "text": "[Audio data provided]"}
                        )

                elif detected_type == "image_bytes":
                    # Convert bytes to base64 data URL
                    try:
                        import base64

                        image_data = base64.b64encode(arg).decode()
                        # Try to detect image type from bytes
                        if arg.startswith(b"\xff\xd8\xff"):
                            mime_type = "image/jpeg"
                        elif arg.startswith(b"\x89PNG"):
                            mime_type = "image/png"
                        elif arg.startswith(b"GIF8"):
                            mime_type = "image/gif"
                        else:
                            mime_type = "image/png"  # Default

                        data_url = f"data:{mime_type};base64,{image_data}"
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            }
                        )
                    except Exception as e:
                        log_warn(f"Could not process image bytes: {e}")
                        user_content.append(
                            {"type": "text", "text": "[Image data provided]"}
                        )

                elif detected_type == "audio_bytes":
                    # Convert audio bytes to input_audio format
                    try:
                        import base64

                        audio_data = base64.b64encode(arg).decode()
                        # Try to detect format from bytes
                        if arg.startswith(b"RIFF") and b"WAVE" in arg[:12]:
                            audio_format = "wav"
                        elif arg.startswith(b"ID3") or arg.startswith(b"\xff\xfb"):
                            audio_format = "mp3"
                        else:
                            audio_format = "wav"  # Default

                        user_content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data,
                                    "format": audio_format,
                                },
                            }
                        )
                    except Exception as e:
                        log_warn(f"Could not process audio bytes: {e}")
                        user_content.append(
                            {"type": "text", "text": "[Audio data provided]"}
                        )

                elif detected_type == "structured_input":
                    # Handle dict with explicit keys
                    if "system" in arg:
                        messages.append({"role": "system", "content": arg["system"]})
                    if "user" in arg:
                        user_content.append({"type": "text", "text": arg["user"]})
                    # Handle other structured content
                    for key in [
                        "text",
                        "image",
                        "image_url",
                        "audio",
                    ]:
                        if key in arg:
                            if key == "text":
                                user_content.append({"type": "text", "text": arg[key]})
                            elif key in ["image", "image_url"]:
                                if isinstance(arg[key], dict):
                                    user_content.append(
                                        {"type": "image_url", "image_url": arg[key]}
                                    )
                                else:
                                    user_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": arg[key],
                                                "detail": "high",
                                            },
                                        }
                                    )
                            elif key == "audio":
                                if isinstance(arg[key], dict):
                                    user_content.append(
                                        {"type": "input_audio", "input_audio": arg[key]}
                                    )
                                else:
                                    # Assume it's a file path or URL
                                    user_content.append(
                                        {"type": "text", "text": f"[Audio: {arg[key]}]"}
                                    )

                elif detected_type == "message_dict":
                    # Handle message format dict
                    messages.append(arg)

                elif detected_type == "conversation_list":
                    # Handle list of messages
                    messages.extend(arg)

                elif detected_type == "multimodal_list":
                    # Handle mixed list of content
                    for item in arg:
                        if isinstance(item, str):
                            user_content.append({"type": "text", "text": item})
                        elif isinstance(item, dict):
                            if "role" in item:
                                messages.append(item)
                            else:
                                # Process as structured input
                                sub_messages = self._process_multimodal_args((item,))
                                messages.extend(sub_messages)

                elif detected_type == "dict":
                    # Generic dict - convert to text representation
                    import json

                    user_content.append(
                        {"type": "text", "text": f"Data: {json.dumps(arg, indent=2)}"}
                    )

                else:
                    # Fallback for unknown types
                    user_content.append({"type": "text", "text": str(arg)})

        # Add user content as a message if we have any
        if user_content:
            if len(user_content) == 1 and user_content[0]["type"] == "text":
                # Simplify single text content
                messages.append({"role": "user", "content": user_content[0]["text"]})
            else:
                # Multiple content types
                messages.append({"role": "user", "content": user_content})

        return messages

    async def ai_with_audio(
        self,
        *args: Any,
        voice: str = "alloy",
        format: str = "wav",
        model: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        AI method optimized for audio output generation.

        Automatically detects the model type and uses the appropriate LiteLLM function:
        - For TTS models (tts-1, tts-1-hd, gpt-4o-mini-tts): Uses litellm.speech()
        - For audio-capable chat models (gpt-4o-audio-preview): Uses litellm.completion() with audio modalities

        Args:
            *args: Input arguments (text prompts, etc.)
            voice: Voice to use for audio generation (alloy, echo, fable, onyx, nova, shimmer)
            format: Audio format (wav, mp3, etc.)
            model: Model to use (defaults to tts-1)
            **kwargs: Additional parameters

        Returns:
            MultimodalResponse with audio content

        Example:
            audio_result = await agent.ai_with_audio("Say hello warmly", voice="alloy")
            audio_result.audio.save("greeting.wav")
        """
        # Use TTS model as default (more reliable than gpt-4o-audio-preview)
        if model is None:
            model = (
                self.agent.ai_config.audio_model
            )  # Use configured audio model (defaults to tts-1)

        # Route based on model prefix - Fal TTS models
        if model.startswith("fal-ai/") or model.startswith("fal/"):
            # Combine all text inputs
            text_input = " ".join(str(arg) for arg in args if isinstance(arg, str))
            if not text_input:
                text_input = "Hello, this is a test audio message."

            return await self._fal_provider.generate_audio(
                text=text_input,
                model=model,
                voice=voice,
                format=format,
                **kwargs,
            )

        # Check if mode="openai_direct" is specified
        if mode == "openai_direct":
            # Use direct OpenAI client with streaming response
            return await self._generate_openai_direct_audio(
                *args,
                voice=voice,
                format=format,
                model=model or "gpt-4o-mini-tts",
                **kwargs,
            )

        # Check if this is a TTS model that needs the speech endpoint
        tts_models = ["tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
        if model in tts_models:
            # Use LiteLLM speech function for TTS models
            return await self._generate_tts_audio(
                *args, voice=voice, format=format, model=model, **kwargs
            )
        else:
            # Use chat completion with audio modalities for other models
            audio_params = {
                "modalities": ["text", "audio"],
                "audio": {"voice": voice, "format": format},
            }
            final_kwargs = {**audio_params, **kwargs}
            return await self.ai(*args, model=model, **final_kwargs)

    async def _generate_tts_audio(
        self,
        *args: Any,
        voice: str = "alloy",
        format: str = "wav",
        model: str = "tts-1",
        **kwargs,
    ) -> Any:
        """
        Generate audio using LiteLLM's speech function for TTS models.
        """
        from agentfield.multimodal_response import (
            AudioOutput,
            MultimodalResponse,
        )

        litellm_module = litellm
        if not hasattr(litellm_module, "aspeech"):
            raise ImportError(
                "litellm is not installed. Please install it with `pip install litellm` to use TTS features."
            )

        # Combine all text inputs
        text_input = " ".join(str(arg) for arg in args if isinstance(arg, str))
        if not text_input:
            text_input = "Hello, this is a test audio message."

        try:
            # Get API configuration
            config = self.agent.ai_config.get_litellm_params()

            # Use LiteLLM speech function
            response = await litellm_module.aspeech(
                model=model,
                input=text_input,
                voice=voice,
                response_format=format,
                api_key=config.get("api_key"),
                **kwargs,
            )

            # Convert binary response to base64 string for AudioOutput
            import base64

            try:
                # Try different methods to get binary content
                if hasattr(response, "content"):
                    binary_content = response.content
                elif hasattr(response, "read"):
                    binary_content = response.read()
                elif hasattr(response, "__iter__"):
                    # For HttpxBinaryResponseContent, iterate to get bytes
                    binary_content = b"".join(response)
                else:
                    # Last resort - convert to string and encode
                    binary_content = str(response).encode("utf-8")

                audio_data = base64.b64encode(binary_content).decode("utf-8")
            except Exception as e:
                log_error(f"Failed to process audio response: {e}")
                # Use a placeholder for now
                audio_data = ""

            # Create AudioOutput directly
            audio_output = AudioOutput(data=audio_data, format=format, url=None)

            # Create MultimodalResponse directly
            return MultimodalResponse(
                text=text_input,
                audio=audio_output,
                images=[],
                files=[],
                raw_response=response,
            )

        except Exception as e:
            # Fallback to text-only MultimodalResponse
            log_error(f"TTS generation failed: {e}")
            return MultimodalResponse(
                text=text_input,
                audio=None,
                images=[],
                files=[],
                raw_response=text_input,
            )

    async def _generate_openai_direct_audio(
        self,
        *args: Any,
        voice: str = "alloy",
        format: str = "wav",
        model: str = "gpt-4o-mini-tts",
        **kwargs,
    ) -> Any:
        """
        Generate audio using OpenAI client directly with streaming response.
        This method supports OpenAI-specific parameters like 'instructions' and 'speed'.

        All kwargs are passed through to OpenAI SDK. The SDK will validate parameters
        and reject unsupported ones.

        Common OpenAI parameters:
        - instructions: Guide the model's speaking style
        - speed: Speech speed (0.25 to 4.0)
        - response_format: Audio format (mp3, opus, aac, flac, wav, pcm)
        """
        import base64
        import tempfile
        from pathlib import Path

        from agentfield.multimodal_response import AudioOutput, MultimodalResponse
        from openai import OpenAI

        # Combine all text inputs
        text_input = " ".join(str(arg) for arg in args if isinstance(arg, str))
        if not text_input:
            text_input = "Hello, this is a test audio message."

        try:
            # Get API configuration
            config = self.agent.ai_config.get_litellm_params()
            api_key = config.get("api_key")

            if not api_key:
                raise ValueError("OpenAI API key not found in configuration")

            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            # Prepare base parameters for OpenAI speech API
            speech_params = {
                "model": model,
                "voice": voice,
                "input": text_input,
            }

            # Map format parameter to response_format if not already in kwargs
            if "response_format" not in kwargs and format:
                speech_params["response_format"] = format

            # Pass all kwargs through to OpenAI SDK
            # Let OpenAI SDK handle parameter validation
            speech_params.update(kwargs)

            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(
                suffix=f".{format}", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)

            try:
                # Use OpenAI streaming response
                with client.audio.speech.with_streaming_response.create(
                    **speech_params
                ) as response:
                    response.stream_to_file(temp_path)

                # Read the audio file and convert to base64
                with open(temp_path, "rb") as audio_file:
                    binary_content = audio_file.read()
                    audio_data = base64.b64encode(binary_content).decode("utf-8")

                # Create AudioOutput
                audio_output = AudioOutput(data=audio_data, format=format, url=None)

                # Create MultimodalResponse
                return MultimodalResponse(
                    text=text_input,
                    audio=audio_output,
                    images=[],
                    files=[],
                    raw_response=response,
                )

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            # Fallback to text-only MultimodalResponse
            log_error(f"OpenAI direct audio generation failed: {e}")
            return MultimodalResponse(
                text=text_input,
                audio=None,
                images=[],
                files=[],
                raw_response=text_input,
            )

    async def ai_with_vision(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: Optional[str] = None,
        model: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ) -> Any:
        """
        AI method optimized for image generation.

        Supports both LiteLLM and OpenRouter providers:
        - LiteLLM: Use model names like "dall-e-3", "azure/dall-e-3", "bedrock/stability.stable-diffusion-xl"
        - OpenRouter: Use model names with "openrouter/" prefix like "openrouter/google/gemini-2.5-flash-image-preview"

        Args:
            prompt: Text prompt for image generation
            size: Image size (256x256, 512x512, 1024x1024, 1792x1024, 1024x1792)
            quality: Image quality (standard, hd)
            style: Image style (vivid, natural) for DALL-E 3
            model: Model to use (defaults to dall-e-3)
            response_format: Response format ('url' or 'b64_json'). Defaults to 'url'
            **kwargs: Additional provider-specific parameters

        Returns:
            MultimodalResponse with image content

        Examples:
            # LiteLLM (DALL-E)
            result = await agent.ai_with_vision("A sunset over mountains")
            result.images[0].save("sunset.png")

            # OpenRouter (Gemini)
            result = await agent.ai_with_vision(
                "A futuristic city",
                model="openrouter/google/gemini-2.5-flash-image-preview",
                image_config={"aspect_ratio": "16:9"}
            )

            # Get base64 data directly
            result = await agent.ai_with_vision("A sunset", response_format="b64_json")
        """
        from agentfield import vision

        # Use image generation model if not specified
        if model is None:
            model = "dall-e-3"  # Default image model

        # Route based on model prefix
        if model.startswith("fal-ai/") or model.startswith("fal/"):
            # Fal: Use FalProvider for Flux, SDXL, Recraft, etc.
            return await self._fal_provider.generate_image(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                **kwargs,
            )
        elif model.startswith("openrouter/"):
            # OpenRouter: Use chat completions API with image modality
            return await vision.generate_image_openrouter(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                style=style,
                response_format=response_format,
                **kwargs,
            )
        else:
            # LiteLLM: Use image generation API
            return await vision.generate_image_litellm(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                style=style,
                response_format=response_format,
                **kwargs,
            )

    async def ai_with_multimodal(
        self,
        *args: Any,
        modalities: Optional[List[str]] = None,
        audio_config: Optional[Dict] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        AI method for explicit multimodal input/output control.

        Args:
            *args: Mixed multimodal inputs
            modalities: List of desired output modalities (["text", "audio", "image"])
            audio_config: Audio configuration if audio modality requested
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            MultimodalResponse with requested modalities

        Example:
            result = await agent.ai_with_multimodal(
                "Describe this image and provide audio narration",
                image_from_url("https://example.com/image.jpg"),
                modalities=["text", "audio"],
                audio_config={"voice": "nova", "format": "wav"}
            )
        """
        multimodal_params = {}

        if modalities:
            multimodal_params["modalities"] = modalities

        if audio_config and "audio" in (modalities or []):
            multimodal_params["audio"] = audio_config

        # Use multimodal-capable model if not specified
        if model is None and modalities and "audio" in modalities:
            model = "gpt-4o-audio-preview"

        # Merge with user kwargs
        final_kwargs = {**multimodal_params, **kwargs}

        return await self.ai(*args, model=model, **final_kwargs)

    async def ai_generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        style: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ) -> "MultimodalResponse":
        """
        Generate an image from a text prompt.

        This is a dedicated method for image generation with a clearer name
        than ai_with_vision. Returns a MultimodalResponse containing the
        generated image(s).

        Supported Providers:
        - LiteLLM: DALL-E models like "dall-e-3", "dall-e-2"
        - OpenRouter: Models like "openrouter/google/gemini-2.5-flash-image-preview"
        - Fal.ai: Models like "fal-ai/flux/dev", "fal-ai/flux/schnell", "fal-ai/recraft-v3"

        Args:
            prompt: Text description of the image to generate
            model: Model to use (defaults to AIConfig.vision_model, typically "dall-e-3")
            size: Image dimensions (e.g., "1024x1024", "1792x1024") or Fal presets
                  ("square_hd", "landscape_16_9", "portrait_4_3")
            quality: Image quality ("standard" or "hd")
            style: Image style for DALL-E 3 ("vivid" or "natural")
            response_format: Output format ("url" or "b64_json")
            **kwargs: Provider-specific parameters (e.g., image_config for OpenRouter)

        Returns:
            MultimodalResponse: Response object with .images list containing ImageOutput objects.
                - Use response.has_images to check if generation succeeded
                - Use response.images[0].save("path.png") to save the image
                - Use response.images[0].get_bytes() to get raw image bytes

        Examples:
            # Basic image generation
            result = await app.ai_generate_image("A sunset over mountains")
            if result.has_images:
                result.images[0].save("sunset.png")

            # OpenRouter with Gemini
            result = await app.ai_generate_image(
                "A futuristic cityscape at night",
                model="openrouter/google/gemini-2.5-flash-image-preview",
                image_config={"aspect_ratio": "16:9"}
            )

            # High quality DALL-E 3
            result = await app.ai_generate_image(
                "A photorealistic portrait",
                model="dall-e-3",
                quality="hd",
                style="natural"
            )

            # Fal.ai Flux (fast, high quality)
            result = await app.ai_generate_image(
                "A cyberpunk cityscape",
                model="fal-ai/flux/dev",
                size="landscape_16_9",
                num_images=2
            )

            # Fal.ai Flux Schnell (fastest)
            result = await app.ai_generate_image(
                "A serene Japanese garden",
                model="fal-ai/flux/schnell",
                size="square_hd"
            )
        """
        # Use configured vision/image model as default
        if model is None:
            model = self.agent.ai_config.vision_model

        return await self.ai_with_vision(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            style=style,
            response_format=response_format,
            **kwargs,
        )

    async def ai_generate_audio(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "alloy",
        format: str = "wav",
        speed: float = 1.0,
        **kwargs,
    ) -> "MultimodalResponse":
        """
        Generate audio/speech from text (Text-to-Speech).

        This is a dedicated method for audio generation with a clearer name
        than ai_with_audio. Returns a MultimodalResponse containing the
        generated audio.

        Supported Providers:
        - LiteLLM: OpenAI TTS models like "tts-1", "tts-1-hd", "gpt-4o-mini-tts"
        - Fal.ai: TTS models like "fal-ai/kokoro/..." (custom deployments)

        Args:
            text: Text to convert to speech
            model: TTS model to use (defaults to AIConfig.audio_model, typically "tts-1")
            voice: Voice to use ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
            format: Audio format ("wav", "mp3", "opus", "aac", "flac", "pcm")
            speed: Speech speed multiplier (0.25 to 4.0)
            **kwargs: Provider-specific parameters

        Returns:
            MultimodalResponse: Response object with .audio containing AudioOutput.
                - Use response.has_audio to check if generation succeeded
                - Use response.audio.save("path.wav") to save the audio
                - Use response.audio.get_bytes() to get raw audio bytes
                - Use response.audio.play() to play the audio (requires pygame)

        Examples:
            # Basic speech generation
            result = await app.ai_generate_audio("Hello, how are you today?")
            if result.has_audio:
                result.audio.save("greeting.wav")

            # High-quality TTS with custom voice
            result = await app.ai_generate_audio(
                "Welcome to the presentation.",
                model="tts-1-hd",
                voice="nova",
                format="mp3"
            )

            # Adjust speech speed
            result = await app.ai_generate_audio(
                "This is spoken slowly.",
                speed=0.75
            )
        """
        # Use configured audio model as default
        if model is None:
            model = self.agent.ai_config.audio_model

        return await self.ai_with_audio(
            text,
            model=model,
            voice=voice,
            format=format,
            speed=speed,
            **kwargs,
        )

    async def ai_generate_video(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        duration: Optional[float] = None,
        **kwargs,
    ) -> "MultimodalResponse":
        """
        Generate video from text or image.

        This method generates videos using Fal.ai's video generation models.
        Supports both text-to-video and image-to-video generation.

        Supported Providers:
        - Fal.ai: Models like "fal-ai/minimax-video/image-to-video",
          "fal-ai/kling-video/v1/standard", "fal-ai/luma-dream-machine"

        Args:
            prompt: Text description for the video
            model: Video model to use (defaults to AIConfig.video_model)
            image_url: Optional input image URL for image-to-video models
            duration: Video duration in seconds (model-dependent)
            **kwargs: Provider-specific parameters

        Returns:
            MultimodalResponse: Response with .files containing the video.
                - Use response.files[0].save("video.mp4") to save
                - Use response.files[0].url to get the video URL

        Examples:
            # Image to video
            result = await app.ai_generate_video(
                "Camera slowly pans across the landscape",
                model="fal-ai/minimax-video/image-to-video",
                image_url="https://example.com/image.jpg"
            )
            result.files[0].save("output.mp4")

            # Text to video
            result = await app.ai_generate_video(
                "A cat playing with yarn",
                model="fal-ai/kling-video/v1/standard"
            )

            # Luma Dream Machine
            result = await app.ai_generate_video(
                "A dreamy underwater scene",
                model="fal-ai/luma-dream-machine"
            )
        """
        if model is None:
            model = self.agent.ai_config.video_model

        # Currently only Fal supports video generation
        if not (model.startswith("fal-ai/") or model.startswith("fal/")):
            raise ValueError(
                f"Video generation currently only supports Fal.ai models. "
                f"Use models like 'fal-ai/minimax-video/image-to-video'. Got: {model}"
            )

        return await self._fal_provider.generate_video(
            prompt=prompt,
            model=model,
            image_url=image_url,
            duration=duration,
            **kwargs,
        )

    async def ai_transcribe_audio(
        self,
        audio_url: str,
        model: str = "fal-ai/whisper",
        language: Optional[str] = None,
        **kwargs,
    ) -> "MultimodalResponse":
        """
        Transcribe audio to text (Speech-to-Text).

        This method transcribes audio files to text using Fal.ai's Whisper models.

        Supported Providers:
        - Fal.ai: Models like "fal-ai/whisper", "fal-ai/wizper" (2x faster)

        Args:
            audio_url: URL to audio file to transcribe
            model: STT model to use (defaults to "fal-ai/whisper")
            language: Optional language hint (e.g., "en", "es", "fr")
            **kwargs: Provider-specific parameters

        Returns:
            MultimodalResponse: Response with .text containing the transcription.
                - Use response.text to get the transcribed text

        Examples:
            # Basic transcription
            result = await app.ai_transcribe_audio(
                "https://example.com/audio.mp3"
            )
            print(result.text)

            # With language hint
            result = await app.ai_transcribe_audio(
                "https://example.com/spanish_audio.mp3",
                model="fal-ai/whisper",
                language="es"
            )

            # Fast transcription with Wizper
            result = await app.ai_transcribe_audio(
                "https://example.com/audio.mp3",
                model="fal-ai/wizper"
            )
        """
        # Currently only Fal supports transcription
        if not (model.startswith("fal-ai/") or model.startswith("fal/")):
            raise ValueError(
                f"Audio transcription currently only supports Fal.ai models. "
                f"Use 'fal-ai/whisper' or 'fal-ai/wizper'. Got: {model}"
            )

        return await self._fal_provider.transcribe_audio(
            audio_url=audio_url,
            model=model,
            language=language,
            **kwargs,
        )
