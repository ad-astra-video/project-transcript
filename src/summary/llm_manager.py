"""
LLMManager - Provides plugins access to fast and reasoning LLM clients.

Plugins choose which client to use based on their needs:
- fast_client: For quick/cheap requests (e.g., rapid summarization)
- reasoning_client: For complex analysis (e.g., context summaries)

Note: LLMManager provides raw AsyncOpenAI clients. Plugins handle their own
response schema restrictions.
"""

import logging
import os
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

logger = logging.getLogger(__name__)


class MessageFormatMode(str, Enum):
    """Message format modes for different LLM providers."""
    SYSTEM_PROMPT = "system"  # Use system role for system prompt
    USER_PREFIX = "user"      # Convert system prompt to user message with prefix


# Type alias for build_messages callback
BuildMessagesCallback = Callable[[str, str], List[Dict[str, str]]]


class LLMClient:
    """Wrapper around AsyncOpenAI client with model-specific message building.
    
    Provides a simplified interface for making chat completions calls without
    needing to manage message formatting or pass LLMManager references.
    """
    
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        build_messages_callback: BuildMessagesCallback
    ):
        """Initialize LLMClient with client, model, and message builder.
        
        Args:
            client: AsyncOpenAI client for API calls
            model: Model name to use in completions.create calls
            build_messages_callback: Callback function that takes (system_prompt, user_content)
                                      and returns a list of message dictionaries
        """
        self._client = client
        self._model = model
        self._build_messages_callback = build_messages_callback
    
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
        # Use provided messages or build them from system_prompt and user_content
        if messages is None:
            messages = self.build_messages(system_prompt, user_content)
        
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs
        )
        
        # Extract content
        content = response.choices[0].message.content or ""
        reasoning = response.choices[0].message.reasoning if hasattr(response.choices[0].message, "reasoning") else ""

        # Extract token usage
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Extract reasoning tokens from prompt_tokens_details (available for reasoning models like o1, o3-mini)
        reasoning_tokens = 0
        if response.usage and hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
            reasoning_tokens = response.usage.prompt_tokens_details.reasoning_tokens if hasattr(response.usage.prompt_tokens_details, 'reasoning_tokens') else 0
        
        return (reasoning, content, input_tokens, output_tokens, reasoning_tokens)


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
            build_messages_callback=self._rapid_build_messages
        )
        self._reasoning_llm_client = LLMClient(
            client=self._reasoning_client,
            model=self._reasoning_model,
            build_messages_callback=self._reasoning_build_messages
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