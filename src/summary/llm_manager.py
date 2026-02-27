"""
LLMManager - Provides plugins access to fast and reasoning LLM clients.

Plugins choose which client to use based on their needs:
- fast_client: For quick/cheap requests (e.g., rapid summarization)
- reasoning_client: For complex analysis (e.g., context summaries)

Note: LLMManager provides raw AsyncOpenAI clients. Plugins handle their own
response schema restrictions.
"""

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple, Awaitable
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

logger = logging.getLogger(__name__)


class MessageFormatMode(str, Enum):
    """Message format modes for different LLM providers."""
    SYSTEM_PROMPT = "system"  # Use system role for system prompt
    USER_PREFIX = "user"      # Convert system prompt to user message with prefix


# Type alias for build_messages callback
BuildMessagesCallback = Callable[[str, str], List[Dict[str, str]]]


class HealthMetrics:
    """
    Tracks health metrics for LLM requests.
    
    Created once with LLMManager. Stream_id is set when each stream starts.
    """
    
    def __init__(
        self,
        timeout_seconds: float = 240.0,
        result_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        monitoring_callback: Optional[Callable[[dict, str], Awaitable[None]]] = None
    ):
        self.stream_id = None  # Set via set_stream_id() on stream start
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._reset_counters()
        
        # Callbacks for publishing
        self._result_callback = result_callback
        self._monitoring_callback = monitoring_callback
        
        # Scheduler
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        
        # Stream-level counters (persist for entire stream)
        self._stream_total_requests = 0
        self._stream_successful_requests = 0
        self._stream_parse_failures = 0
        self._stream_timeout_failures = 0
        self._stream_truncation_failures = 0
        self._stream_plugin_counts: Dict[str, Dict[str, int]] = {}
        self._stream_model_type_counts: Dict[str, Dict[str, int]] = {}
        self._stream_start_time = datetime.now(timezone.utc)
    
    def _reset_counters(self):
        self._total_requests = 0
        self._successful_requests = 0
        self._parse_failures = 0
        self._timeout_failures = 0
        self._truncation_failures = 0
        self._plugin_counts: Dict[str, Dict[str, int]] = {}
        self._model_type_counts: Dict[str, Dict[str, int]] = {}  # reasoning vs fast
        self._window_start = datetime.now(timezone.utc)
    
    def set_stream_id(self, stream_id: Optional[str]):
        """Set stream_id when stream starts (for monitoring payload)."""
        self.stream_id = stream_id
    
    def set_publish_callbacks(
        self,
        result_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        monitoring_callback: Optional[Callable[[dict, str], Awaitable[None]]] = None
    ):
        """Set callbacks for publishing health messages."""
        self._result_callback = result_callback
        self._monitoring_callback = monitoring_callback
    
    def record(
        self,
        plugin_name: str,
        success: bool,
        failure_type: Optional[str] = None,
        response_text: str = "",
        elapsed_time: float = 0.0,
        model_type: Optional[str] = None
    ):
        """Record a request outcome.
        
        Args:
            plugin_name: Name of the plugin (e.g., 'rapid_summary', 'context_summary')
            success: Whether the request was successful
            failure_type: Type of failure if not successful ('parse_failure' or 'timeout_failure')
            response_text: The response text from the LLM
            elapsed_time: Time taken for the request
            model_type: Type of model ('reasoning' or 'fast'). If None, inferred from plugin_name.
        """
        # Infer model_type from plugin_name if not provided
        if model_type is None:
            # context_summary uses reasoning model, rapid_summary uses fast model
            model_type = "reasoning" if "context" in plugin_name else "fast"
        
        with self._lock:
            # === Window-level tracking ===
            self._total_requests += 1
            
            # Track by plugin_name
            if plugin_name not in self._plugin_counts:
                self._plugin_counts[plugin_name] = {
                    "total": 0, "success": 0, "parse_failure": 0, "timeout_failure": 0, "truncation_failure": 0
                }
            
            stats = self._plugin_counts[plugin_name]
            stats["total"] += 1
            
            # Track by model_type (reasoning vs fast)
            if model_type not in self._model_type_counts:
                self._model_type_counts[model_type] = {
                    "total": 0, "success": 0, "parse_failure": 0, "timeout_failure": 0, "truncation_failure": 0
                }
            
            model_stats = self._model_type_counts[model_type]
            model_stats["total"] += 1
            
            if success:
                self._successful_requests += 1
                stats["success"] += 1
                model_stats["success"] += 1
            else:
                if failure_type == "parse_failure":
                    self._parse_failures += 1
                    stats["parse_failure"] += 1
                    model_stats["parse_failure"] += 1
                elif failure_type == "timeout_failure":
                    self._timeout_failures += 1
                    stats["timeout_failure"] += 1
                    model_stats["timeout_failure"] += 1
                elif failure_type == "truncation_failure":
                    self._truncation_failures += 1
                    stats["truncation_failure"] += 1
                    model_stats["truncation_failure"] += 1
            
            # === Stream-level tracking ===
            self._stream_total_requests += 1
            
            # Track by plugin_name
            if plugin_name not in self._stream_plugin_counts:
                self._stream_plugin_counts[plugin_name] = {
                    "total": 0, "success": 0, "parse_failure": 0, "timeout_failure": 0, "truncation_failure": 0
                }
            
            stream_stats = self._stream_plugin_counts[plugin_name]
            stream_stats["total"] += 1
            
            # Track by model_type
            if model_type not in self._stream_model_type_counts:
                self._stream_model_type_counts[model_type] = {
                    "total": 0, "success": 0, "parse_failure": 0, "timeout_failure": 0, "truncation_failure": 0
                }
            
            stream_model_stats = self._stream_model_type_counts[model_type]
            stream_model_stats["total"] += 1
            
            if success:
                self._stream_successful_requests += 1
                stream_stats["success"] += 1
                stream_model_stats["success"] += 1
            else:
                if failure_type == "parse_failure":
                    self._stream_parse_failures += 1
                    stream_stats["parse_failure"] += 1
                    stream_model_stats["parse_failure"] += 1
                elif failure_type == "timeout_failure":
                    self._stream_timeout_failures += 1
                    stream_stats["timeout_failure"] += 1
                    stream_model_stats["timeout_failure"] += 1
                elif failure_type == "truncation_failure":
                    self._stream_truncation_failures += 1
                    stream_stats["truncation_failure"] += 1
                    stream_model_stats["truncation_failure"] += 1
    
    def _get_window_metrics(self) -> dict:
        """Get current window (1-minute) metrics."""
        total = self._total_requests
        rate = (self._successful_requests / total * 100) if total > 0 else 0.0
        
        # Calculate model_type breakdown with success rates
        model_breakdown = {}
        for model_type, stats in self._model_type_counts.items():
            model_total = stats["total"]
            model_rate = (stats["success"] / model_total * 100) if model_total > 0 else 0.0
            model_breakdown[model_type] = {
                "total": model_total,
                "successful": stats["success"],
                "failed": {
                    "total": stats["parse_failure"] + stats["timeout_failure"] + stats.get("truncation_failure", 0),
                    "parse_failure": stats["parse_failure"],
                    "timeout_failure": stats["timeout_failure"],
                    "truncation_failure": stats.get("truncation_failure", 0)
                },
                "success_rate_percentage": round(model_rate, 2)
            }
        
        return {
            "window_start_time": self._window_start.isoformat(),
            "window_duration_seconds": 60,
            "total_requests": total,
            "successful_requests": self._successful_requests,
            "failed_requests": {
                "total": self._parse_failures + self._timeout_failures + self._truncation_failures,
                "parse_failure": self._parse_failures,
                "timeout_failure": self._timeout_failures,
                "truncation_failure": self._truncation_failures
            },
            "success_rate_percentage": round(rate, 2),
            "plugin_breakdown": dict(self._plugin_counts),
            "model_type_breakdown": model_breakdown
        }
    
    def _get_stream_metrics(self) -> dict:
        """Get stream-level metrics (entire stream lifetime)."""
        total = self._stream_total_requests
        rate = (self._stream_successful_requests / total * 100) if total > 0 else 0.0
        
        # Calculate model_type breakdown with success rates
        model_breakdown = {}
        for model_type, stats in self._stream_model_type_counts.items():
            model_total = stats["total"]
            model_rate = (stats["success"] / model_total * 100) if model_total > 0 else 0.0
            model_breakdown[model_type] = {
                "total": model_total,
                "successful": stats["success"],
                "failed": {
                    "total": stats["parse_failure"] + stats["timeout_failure"] + stats.get("truncation_failure", 0),
                    "parse_failure": stats["parse_failure"],
                    "timeout_failure": stats["timeout_failure"],
                    "truncation_failure": stats.get("truncation_failure", 0)
                },
                "success_rate_percentage": round(model_rate, 2)
            }
        
        return {
            "stream_start_time": self._stream_start_time.isoformat(),
            "total_requests": total,
            "successful_requests": self._stream_successful_requests,
            "failed_requests": {
                "total": self._stream_parse_failures + self._stream_timeout_failures + self._stream_truncation_failures,
                "parse_failure": self._stream_parse_failures,
                "timeout_failure": self._stream_timeout_failures,
                "truncation_failure": self._stream_truncation_failures
            },
            "success_rate_percentage": round(rate, 2),
            "plugin_breakdown": dict(self._stream_plugin_counts),
            "model_type_breakdown": model_breakdown
        }
    
    def get_metrics(self) -> dict:
        """Get both window and stream metrics."""
        with self._lock:
            return {
                "window": self._get_window_metrics(),
                "stream": self._get_stream_metrics()
            }
    
    async def publish(self) -> dict:
        """Publish health message and reset."""
        base_message = {
            "type": "summary_processing_health",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self.get_metrics()
        }
        
        # Send via result_callback (NO stream_id)
        if self._result_callback:
            try:
                await self._result_callback(base_message)
            except Exception as e:
                logger.warning(f"Failed to publish health via result_callback: {e}")
        
        # Send via monitoring_callback (WITH stream_id)
        if self._monitoring_callback:
            try:
                monitoring_message = {**base_message, "stream_id": self.stream_id}
                await self._monitoring_callback(monitoring_message, "summary_processing_health")
            except Exception as e:
                logger.warning(f"Failed to publish health via monitoring_callback: {e}")
        
        # Reset window counters only (stream counters persist for entire stream)
        with self._lock:
            self._total_requests = 0
            self._successful_requests = 0
            self._parse_failures = 0
            self._timeout_failures = 0
            self._truncation_failures = 0
            self._plugin_counts = {}
            self._model_type_counts = {}
            self._window_start = datetime.now(timezone.utc)
        
        return base_message
    
    async def start_scheduler(self):
        """Start 60-second publishing loop."""
        self._stop_event = asyncio.Event()
        self._scheduler_task = asyncio.create_task(self._run_scheduler())
    
    async def stop_scheduler(self):
        """Stop the scheduler."""
        if self._stop_event:
            self._stop_event.set()
        if self._scheduler_task:
            await self._scheduler_task
    
    async def _run_scheduler(self):
        while not self._stop_event.is_set():
            await asyncio.sleep(60)
            if not self._stop_event.is_set():
                await self.publish()


class LLMClient:
    """Wrapper around AsyncOpenAI client with model-specific message building.
    
    Provides a simplified interface for making chat completions calls without
    needing to manage message formatting or pass LLMManager references.
    """
    
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        build_messages_callback: BuildMessagesCallback,
        health_metrics: Optional[HealthMetrics] = None,
        plugin_name: str = "unknown"
    ):
        """Initialize LLMClient with client, model, and message builder.
        
        Args:
            client: AsyncOpenAI client for API calls
            model: Model name to use in completions.create calls
            build_messages_callback: Callback function that takes (system_prompt, user_content)
                                      and returns a list of message dictionaries
            health_metrics: Optional HealthMetrics instance for tracking request outcomes
            plugin_name: Name of the plugin using this client (for metrics)
        """
        self._client = client
        self._model = model
        self._build_messages_callback = build_messages_callback
        self._health_metrics = health_metrics
        self._plugin_name = plugin_name
    
    @property
    def model(self) -> str:
        """Returns the model name."""
        return self._model
    
    @property
    def client(self) -> AsyncOpenAI:
        """Returns the underlying AsyncOpenAI client."""
        return self._client
    
    def build_messages(
        self,
        system_prompt: str,
        user_content: str
    ) -> List[Dict[str, str]]:
        """Build messages using the configured callback.
        
        Args:
            system_prompt: The system prompt text
            user_content: The user message content
            
        Returns:
            List of message dictionaries with proper format for the model
        """
        return self._build_messages_callback(system_prompt, user_content)
    
    async def create_completion(
        self,
        system_prompt: str = "",
        user_content: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Tuple[str, str, int, int, int]:
        """Create a chat completion with automatic message building.
        
        Args:
            system_prompt: The system prompt text (used if messages not provided)
            user_content: The user message content (used if messages not provided)
            messages: Optional pre-built messages list. If provided, bypasses message building
            **kwargs: Additional arguments passed to chat.completions.create
            
        Returns:
            Tuple of (reasoning, content, input_tokens, output_tokens, reasoning_tokens)
            - reasoning: The reasoning content from the model (for reasoning models)
            - content: The text content from the first choice's message
            - input_tokens: Number of tokens in the input
            - output_tokens: Number of tokens in the output
            - reasoning_tokens: Number of reasoning tokens in the input (for reasoning models)
        """
        start_time = time.perf_counter()
        response_text = ""
        
        # Use provided messages or build them from system_prompt and user_content
        if messages is None:
            messages = self.build_messages(system_prompt, user_content)
        
        try:
            # Get timeout from health_metrics if available
            timeout = self._health_metrics.timeout_seconds if self._health_metrics else 240.0
            
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    **kwargs
                ),
                timeout=timeout
            )
            
            # Extract content and finish_reason
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            reasoning = response.choices[0].message.reasoning if hasattr(response.choices[0].message, "reasoning") else ""
            response_text = content

            # Extract token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            logger.info(f"LLMClient.create_completion token usage: {response.usage if response.usage else 'N/A'}")
            # Extract reasoning tokens from completion_tokens_details (available for reasoning models like o1, o3-mini)
            reasoning_tokens = 0
            if response.usage and hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens') else 0
            
            # Record based on finish_reason
            if self._health_metrics:
                if finish_reason == "length":
                    # Response was truncated due to max_tokens limit - record as failure
                    self._health_metrics.record(
                        plugin_name=self._plugin_name,
                        success=False,
                        failure_type="truncation_failure",
                        response_text=response_text,
                        elapsed_time=time.perf_counter() - start_time
                    )
                else:
                    self._health_metrics.record(
                        plugin_name=self._plugin_name,
                        success=True,
                        response_text=response_text,
                        elapsed_time=time.perf_counter() - start_time
                    )
            
            return (reasoning, content, input_tokens, output_tokens, reasoning_tokens)
            
        except asyncio.TimeoutError:
            # Record timeout failure
            if self._health_metrics:
                self._health_metrics.record(
                    plugin_name=self._plugin_name,
                    success=False,
                    failure_type="timeout_failure",
                    elapsed_time=time.perf_counter() - start_time
                )
            raise
            
        except json.JSONDecodeError:
            # Record parse failure
            if self._health_metrics:
                self._health_metrics.record(
                    plugin_name=self._plugin_name,
                    success=False,
                    failure_type="parse_failure",
                    response_text=response_text,
                    elapsed_time=time.perf_counter() - start_time
                )
            raise
            
        except Exception as e:
            # Record as parse failure
            if self._health_metrics:
                self._health_metrics.record(
                    plugin_name=self._plugin_name,
                    success=False,
                    failure_type="parse_failure",
                    response_text=response_text,
                    elapsed_time=time.perf_counter() - start_time
                )
            raise


class LLMManager:
    """Manages LLM clients for plugins."""
    
    def __init__(
        self,
        fast_base_url: str,
        fast_api_key: str,
        reasoning_base_url: str,
        reasoning_api_key: str,
        rapid_model: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        message_format_mode: Optional[MessageFormatMode] = None,
        request_timeout_seconds: float = 240.0,
        health_result_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
        health_monitoring_callback: Optional[Callable[[dict, str], Awaitable[None]]] = None,
    ):
        """Initialize LLMManager with base URLs, API keys, and optional model names.
        
        Args:
            fast_base_url: Base URL for the fast/cheap endpoint (e.g., rapid summarization)
            fast_api_key: API key for the fast endpoint
            reasoning_base_url: Base URL for the reasoning/complex endpoint
            reasoning_api_key: API key for the reasoning endpoint
            rapid_model: Optional model name for fast requests. If None, will be auto-detected.
            reasoning_model: Optional model name for reasoning requests. If None, will be auto-detected.
            message_format_mode: Optional message format mode. If None, reads from environment variable.
            request_timeout_seconds: Timeout for LLM requests (default 240s)
            health_result_callback: Callback for publishing health messages (no stream_id)
            health_monitoring_callback: Callback for publishing health messages (with stream_id)
        """
        self._fast_base_url = fast_base_url.rstrip("/")
        self._fast_api_key = fast_api_key
        self._reasoning_base_url = reasoning_base_url.rstrip("/")
        self._reasoning_api_key = reasoning_api_key
        self._rapid_model = rapid_model
        self._reasoning_model = reasoning_model
        
        # Initialize message_format_mode from parameter or environment
        if message_format_mode is not None:
            self._message_format_mode = message_format_mode
        else:
            # Default to SYSTEM_PROMPT, can be overridden via properties
            self._message_format_mode = MessageFormatMode.SYSTEM_PROMPT
        
        # Create the clients
        self._fast_client = AsyncOpenAI(
            api_key=fast_api_key or "dummy",
            base_url=fast_base_url
        )
        self._reasoning_client = AsyncOpenAI(
            api_key=reasoning_api_key or "dummy",
            base_url=reasoning_base_url
        )
        
        # Create build_messages callbacks based on message format modes
        self._rapid_build_messages = self._create_build_messages_callback(
            self.rapid_message_format_mode
        )
        self._reasoning_build_messages = self._create_build_messages_callback(
            self.reasoning_message_format_mode
        )
        
        # Create LLMClient instances (will be finalized after model detection)
        self._rapid_llm_client: Optional[LLMClient] = None
        self._reasoning_llm_client: Optional[LLMClient] = None
        
        # Health metrics tracking
        self._request_timeout = request_timeout_seconds
        self._health_metrics = HealthMetrics(
            timeout_seconds=request_timeout_seconds,
            result_callback=health_result_callback,
            monitoring_callback=health_monitoring_callback
        )
    
    def set_stream_id(self, stream_id: Optional[str]):
        """Set stream_id when stream starts (for monitoring payload)."""
        self._health_metrics.set_stream_id(stream_id)
    
    def start_scheduler(self):
        """Start the health metrics publishing scheduler."""
        return self._health_metrics.start_scheduler()
    
    def stop_scheduler(self):
        """Stop the health metrics publishing scheduler."""
        return self._health_metrics.stop_scheduler()
    
    def _create_build_messages_callback(self, mode: MessageFormatMode) -> BuildMessagesCallback:
        """Create a build_messages callback based on the format mode.
        
        Args:
            mode: The message format mode (SYSTEM_PROMPT or USER_PREFIX)
            
        Returns:
            A callback function that builds messages in the appropriate format
        """
        if mode == MessageFormatMode.SYSTEM_PROMPT:
            return lambda system, user: [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        else:  # USER_PREFIX
            return lambda system, user: [
                {"role": "user", "content": f"[SYSTEM PROMPT]\n{system}\n\n[USER CONTENT]\n{user}"}
            ]
    
    def _create_llm_clients(self) -> None:
        """Create LLMClient instances after models are detected.
        
        Should be called after initialize() has detected/Set the model names.
        """
        self._rapid_llm_client = LLMClient(
            client=self._fast_client,
            model=self._rapid_model,
            build_messages_callback=self._rapid_build_messages,
            health_metrics=self._health_metrics,
            plugin_name="rapid_summary"
        )
        self._reasoning_llm_client = LLMClient(
            client=self._reasoning_client,
            model=self._reasoning_model,
            build_messages_callback=self._reasoning_build_messages,
            health_metrics=self._health_metrics,
            plugin_name="context_summary"
        )
    
    @property
    def fast_client(self) -> AsyncOpenAI:
        """Returns the fast client for quick/cheap requests."""
        return self._fast_client
    
    @property
    def reasoning_client(self) -> AsyncOpenAI:
        """Returns the reasoning client for complex requests."""
        return self._reasoning_client
    
    @property
    def rapid_llm_client(self) -> LLMClient:
        """Returns the LLMClient for rapid/fast requests."""
        if self._rapid_llm_client is None:
            raise RuntimeError("LLMClient not initialized. Call initialize() first.")
        return self._rapid_llm_client
    
    @property
    def reasoning_llm_client(self) -> LLMClient:
        """Returns the LLMClient for reasoning requests."""
        if self._reasoning_llm_client is None:
            raise RuntimeError("LLMClient not initialized. Call initialize() first.")
        return self._reasoning_llm_client
    
    @property
    def rapid_model(self) -> str:
        """Returns the model name for fast requests."""
        return self._rapid_model
    
    @property
    def reasoning_model(self) -> str:
        """Returns the model name for reasoning requests."""
        return self._reasoning_model
    
    @property
    def rapid_message_format_mode(self) -> MessageFormatMode:
        """Returns the message format mode for rapid model based on environment variable."""
        env_value = os.getenv("LOCAL_RAPID_MODEL_USES_SYSTEM_PROMPT", "yes").lower()
        if env_value in ["no"]:
            return MessageFormatMode.USER_PREFIX
        return MessageFormatMode.SYSTEM_PROMPT
    
    @property
    def reasoning_message_format_mode(self) -> MessageFormatMode:
        """Returns the message format mode for reasoning model based on environment variable."""
        env_value = os.getenv("LOCAL_REASONING_MODEL_USES_SYSTEM_PROMPT", "yes").lower()
        if env_value in ["no"]:
            return MessageFormatMode.USER_PREFIX
        return MessageFormatMode.SYSTEM_PROMPT
    
    @property
    def message_format_mode(self) -> MessageFormatMode:
        """Returns the message format mode based on environment variable.
        
        Reads from LOCAL_SUMMARY_MODEL_USES_SYSTEM_PROMPT environment variable.
        Defaults to SYSTEM_PROMPT if not set or set to "yes".
        """
        env_value = os.getenv("LOCAL_SUMMARY_MODEL_USES_SYSTEM_PROMPT", "yes").lower()
        if env_value in ["no"]:
            return MessageFormatMode.USER_PREFIX
        return MessageFormatMode.SYSTEM_PROMPT
    
    async def fetch_loaded_model(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> str:
        """
        Fetch the currently loaded model from the /models endpoint using OpenAI library.
        
        Uses the OpenAI Python library's models.list() method to retrieve available models
        and returns the first one (typically the loaded/primary model).
        
        Args:
            base_url: Optional base URL for the API (defaults to fast_base_url)
            api_key: Optional API key (defaults to fast_api_key)
        
        Returns:
            The model ID string from the OpenAI models list response
            
        Raises:
            RuntimeError: If the models.list() call fails or returns no models
        """
        use_base_url = base_url or self._fast_base_url
        use_api_key = api_key or self._fast_api_key
        
        try:
            logger.info(f"Fetching available models from {use_base_url}")
            
            # Create a temporary client for the request
            temp_client = AsyncOpenAI(
                api_key=use_api_key or "dummy",
                base_url=use_base_url
            )
            response = await temp_client.models.list()
            
            if response.data and len(response.data) > 0:
                # Return the first model (typically the loaded/primary model)
                model_id = response.data[0].id
                logger.info(f"Detected loaded model: {model_id}")
                return model_id
            else:
                logger.warning(f"No models returned from {use_base_url}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return None
    
    async def initialize(self) -> Optional[str]:
        """
        Initialize the LLMManager and auto-detect models if needed.
        
        If rapid_model or reasoning_model is None/empty, fetches the loaded model
        from the respective /models endpoint.
        
        Also calls warm_up() to verify both endpoints are responsive.
        
        Returns:
            The detected model if any auto-detection occurred, None otherwise
            
        Raises:
            RuntimeError: If warm_up fails (reasoning or rapid endpoint is unresponsive)
        """
        detected = None
        
        # Auto-detect rapid model if not specified
        if not self._rapid_model:
            logger.info("No rapid_model specified, fetching loaded model from rapid endpoint")
            try:
                self._rapid_model = await self.fetch_loaded_model(
                    base_url=self._fast_base_url,
                    api_key=self._fast_api_key
                )
                logger.info(f"Auto-detected rapid model: {self._rapid_model}")
                detected = self._rapid_model
            except Exception as e:
                logger.warning(f"Failed to auto-detect rapid model: {e}")
                raise RuntimeError(f"Failed to auto-detect rapid model: {e}")
        
        # Auto-detect reasoning model if not specified
        if not self._reasoning_model:
            logger.info("No reasoning_model specified, fetching loaded model from reasoning endpoint")
            try:
                self._reasoning_model = await self.fetch_loaded_model(
                    base_url=self._reasoning_base_url,
                    api_key=self._reasoning_api_key
                )
                logger.info(f"Auto-detected reasoning model: {self._reasoning_model}")
                if detected is None:
                    detected = self._reasoning_model
                else:
                    detected += f", {self._reasoning_model}"
            except Exception as e:
                logger.warning(f"Failed to auto-detect reasoning model: {e}")
                raise RuntimeError(f"Failed to auto-detect reasoning model: {e}")
        
        # Create LLMClient instances after model detection
        self._create_llm_clients()
        
        # Warm up both endpoints to verify they are responsive
        # This will raise RuntimeError if either endpoint fails
        await self.warm_up()
        
        return detected
    
    async def warm_up(self) -> None:
        """
        Send warm-up requests to verify both endpoints are responsive.
        
        Tests the reasoning endpoint and rapid/fast endpoint by sending minimal
        test requests. Raises RuntimeError if either endpoint fails.
        
        Raises:
            RuntimeError: If either the reasoning or rapid endpoint fails to respond
        """
        # Test reasoning model endpoint
        if self._reasoning_model:
            try:
                logger.info(f"Testing reasoning model endpoint: {self._reasoning_base_url}")
                response = await self._reasoning_client.chat.completions.create(
                    model=self._reasoning_model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info("Reasoning model warm-up request successful - model is responsive")
            except Exception as e:
                error_msg = f"Reasoning model warm-up failed: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        # Test rapid/fast model endpoint
        if self._rapid_model:
            try:
                logger.info(f"Testing rapid model endpoint: {self._fast_base_url}")
                response = await self._fast_client.chat.completions.create(
                    model=self._rapid_model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                logger.info("Rapid model warm-up request successful - model is responsive")
            except Exception as e:
                error_msg = f"Rapid model warm-up failed: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
    
    def build_messages(
        self,
        system_prompt: str,
        user_content: str
    ) -> List[Dict[str, str]]:
        """
        Build messages list with system prompt and user content.
        
        Args:
            system_prompt: The system prompt text
            user_content: The user message content
            
        Returns:
            List of message dictionaries with system and user roles
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def update_params(
        self,
        reasoning_base_url: Optional[str] = None,
        reasoning_api_key: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        fast_base_url: Optional[str] = None,
        fast_api_key: Optional[str] = None,
        fast_model: Optional[str] = None,
    ):
        """Update LLMManager parameters dynamically.
        
        Args:
            reasoning_base_url: New base URL for the reasoning API
            reasoning_api_key: New API key for the reasoning API
            reasoning_model: New model name for the reasoning API
            fast_base_url: New base URL for the fast API
            fast_api_key: New API key for the fast API
            fast_model: New model name for the fast API
        """
        # Update fast client settings
        if fast_base_url is not None:
            self._fast_base_url = fast_base_url.rstrip("/")
            # Recreate the client with new base URL
            self._fast_client = AsyncOpenAI(
                api_key=self._fast_api_key or "dummy",
                base_url=self._fast_base_url
            )
            logger.info(f"Updated fast base URL to {self._fast_base_url}")
        
        if fast_api_key is not None:
            self._fast_api_key = fast_api_key
            # Recreate the client with new API key
            self._fast_client = AsyncOpenAI(
                api_key=self._fast_api_key or "dummy",
                base_url=self._fast_base_url
            )
            logger.info("Updated fast API key")
        
        if fast_model is not None:
            self._rapid_model = fast_model
            # Update the LLMClient if it exists
            if self._rapid_llm_client is not None:
                self._rapid_llm_client._model = fast_model
            logger.info(f"Updated fast model to {fast_model}")
        
        # Update reasoning client settings
        if reasoning_base_url is not None:
            self._reasoning_base_url = reasoning_base_url.rstrip("/")
            # Recreate the client with new base URL
            self._reasoning_client = AsyncOpenAI(
                api_key=self._reasoning_api_key or "dummy",
                base_url=self._reasoning_base_url
            )
            logger.info(f"Updated reasoning base URL to {self._reasoning_base_url}")
        
        if reasoning_api_key is not None:
            self._reasoning_api_key = reasoning_api_key
            # Recreate the client with new API key
            self._reasoning_client = AsyncOpenAI(
                api_key=self._reasoning_api_key or "dummy",
                base_url=self._reasoning_base_url
            )
            logger.info("Updated reasoning API key")
        
        if reasoning_model is not None:
            self._reasoning_model = reasoning_model
            # Update the LLMClient if it exists
            if self._reasoning_llm_client is not None:
                self._reasoning_llm_client._model = reasoning_model
            logger.info(f"Updated reasoning model to {reasoning_model}")