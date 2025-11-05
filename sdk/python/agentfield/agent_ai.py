import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Type, Union

import requests
from agentfield.agent_utils import AgentUtils
from agentfield.logger import log_debug, log_error, log_warn
from agentfield.rate_limiter import StatelessRateLimiter
from httpx import HTTPStatusError
from pydantic import BaseModel

# Expose module-level symbols for patching in tests
try:
    import litellm as litellm  # type: ignore
except Exception:  # pragma: no cover - test environments may not have litellm

    class _LiteLLMStub:
        pass

    litellm = _LiteLLMStub()  # type: ignore

try:
    import openai as openai  # type: ignore
except Exception:  # pragma: no cover - test environments may not have openai

    class _OpenAIStub:
        class OpenAI:
            pass

    openai = _OpenAIStub()  # type: ignore


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

        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is not installed. Please install it with `pip install litellm`."
            )

        # Ensure model limits are cached (done once per instance)
        await self._ensure_model_limits_cached()

        # Apply prompt trimming using LiteLLM's token-aware utility instead of char-based limits
        try:
            from litellm.utils import token_counter, trim_messages
        except ImportError:
            raise ImportError(
                "litellm is required for token-aware prompt trimming. Please install it with `pip install litellm`"
            )

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
            # Use LiteLLM's native Pydantic model support for structured outputs
            litellm_params["response_format"] = schema

        litellm_params["transforms"] = ["middle-out"]

        # Define the LiteLLM call function for rate limiter
        async def _make_litellm_call():
            return await litellm.acompletion(**litellm_params)

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
                        return await litellm.acompletion(**litellm_params)
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

        if final_config.enable_rate_limit_retry:
            rate_limiter = self._get_rate_limiter()
            try:
                response = await rate_limiter.execute_with_retry(
                    _execute_with_fallbacks
                )
            except Exception as e:
                log_debug(f"LiteLLM call failed after retries: {e}")
                raise
        else:
            try:
                response = await _execute_with_fallbacks()
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

        # Process the response
        if final_config.stream:
            # For streaming, return the generator
            return response
        else:
            # Import multimodal response detection
            from .multimodal_response import detect_multimodal_response

            # Detect and wrap multimodal content
            multimodal_response = detect_multimodal_response(response)

            if schema:
                # For schema responses, try to parse from text content
                try:
                    json_data = json.loads(str(multimodal_response.text))
                    return schema(**json_data)
                except (json.JSONDecodeError, ValueError) as parse_error:
                    log_error(f"Failed to parse JSON response: {parse_error}")
                    log_debug(f"Raw response: {multimodal_response.text}")
                    # Fallback: try to extract JSON from the response
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

            # Return MultimodalResponse for backward compatibility and enhanced features
            return multimodal_response

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
        import litellm
        from agentfield.multimodal_response import (
            AudioOutput,
            MultimodalResponse,
        )

        # Combine all text inputs
        text_input = " ".join(str(arg) for arg in args if isinstance(arg, str))
        if not text_input:
            text_input = "Hello, this is a test audio message."

        try:
            # Get API configuration
            config = self.agent.ai_config.get_litellm_params()

            # Use LiteLLM speech function
            response = await litellm.aspeech(
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
        This method supports the 'instructions' parameter that LiteLLM doesn't support.
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

            # Prepare parameters for OpenAI speech API
            speech_params = {
                "model": model,
                "voice": voice,
                "input": text_input,
            }

            # Add instructions if provided in kwargs
            if "instructions" in kwargs:
                speech_params["instructions"] = kwargs["instructions"]

            # Add speed parameter if provided and is a valid float
            if "speed" in kwargs:
                try:
                    speech_params["speed"] = float(kwargs["speed"])
                except (ValueError, TypeError):
                    pass  # Skip invalid speed values

            # Handle format parameter (map to response_format)
            # Only use supported formats
            supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
            if format and format in supported_formats:
                speech_params["response_format"] = format
            elif (
                "response_format" in kwargs
                and kwargs["response_format"] in supported_formats
            ):
                speech_params["response_format"] = kwargs["response_format"]

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
        **kwargs,
    ) -> Any:
        """
        AI method optimized for image generation.

        Args:
            prompt: Text prompt for image generation
            size: Image size (256x256, 512x512, 1024x1024, 1792x1024, 1024x1792)
            quality: Image quality (standard, hd)
            style: Image style (vivid, natural) for DALL-E 3
            model: Model to use (defaults to image generation model)
            **kwargs: Additional parameters

        Returns:
            MultimodalResponse with image content

        Example:
            image_result = await agent.ai_with_vision("A sunset over mountains", size="1024x1024")
            image_result.images[0].save("sunset.png")
        """
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is not installed. Please install it with `pip install litellm`."
            )

        # Use image generation model if not specified
        if model is None:
            model = "dall-e-3"  # Default image model

        # Prepare image generation parameters
        image_params = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "quality": quality,
            "response_format": "url",  # Can be 'url' or 'b64_json'
            **kwargs,
        }

        if style and model == "dall-e-3":
            image_params["style"] = style

        try:
            # Use LiteLLM's image generation function
            response = await litellm.aimage_generation(**image_params)

            # Import multimodal response detection
            from agentfield.multimodal_response import detect_multimodal_response

            # Detect and wrap multimodal content
            return detect_multimodal_response(response)

        except Exception as e:
            log_error(f"Image generation failed: {e}")
            raise

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
