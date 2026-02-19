"""
Content type detection task module.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Optional

from .task import ContentTypeDetectionTask
from .prompts import CONTENT_TYPE_DETECTION_PROMPT

logger = logging.getLogger(__name__)

class ContentTypeDetectionPlugin:
    """Plugin for content type detection."""
    
    def __init__(self, window_manager, llm_manager, result_callback, summary_client=None, **kwargs):
        self._window_manager = window_manager
        self._llm = llm_manager
        self._max_tokens = 350
        self._temperature = 0.2
        self._content_type_context_limit = 2000  # Character limit for content type detection
        self._result_callback = result_callback
        self._summary_client = summary_client  # Keep for non-LLM operations
        self._auto_detect = True
        self._in_progress = False
        
        # Track content type state for previous_content_type
        self._current_content_type = "UNKNOWN"
        self._user_content_type_override = None
        
        # Create task once at init and reuse for all processing
        self._task = ContentTypeDetectionTask(
            llm_client=self._llm.reasoning_llm_client,
            max_tokens=self._max_tokens,
            temperature=self._temperature
        )
    
    def set_content_type_override(self, content_type: str):
        """Set user override for content type.
        
        When set, this content type will be used regardless of auto-detection.
        Set to None to clear override.
        
        Args:
            content_type: Content type to use or None to clear
        """
        self._user_content_type_override = content_type
        if content_type:
            logger.info(f"Content type user override set to: {content_type}")
        else:
            logger.info("Content type user override cleared")
    
    def _should_detect(self, summary_window_id: int) -> bool:
        """Check if should run content type detection for a summary window."""
        return self._auto_detect and not self._in_progress
    
    async def process(self, summary_window_id: int, **kwargs):
        """Process a summary window for content type detection.
        
        Args:
            summary_window_id: The ID of the summary window
            **kwargs: Additional parameters (summary_window_id, window_manager, etc.)
        """
       
        if not self._should_detect(summary_window_id):
            return {}
        
        self._in_progress = True
        try:
            # Check for user override first
            if self._user_content_type_override:
                previous_content_type = self._current_content_type
                payload = {
                    "type": "content_type_detection",
                    "content_type": self._user_content_type_override,
                    "confidence": 1.0,
                    "source": "USER_OVERRIDE",
                    "previous_content_type": previous_content_type,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()
                }
                
                self._current_content_type = self._user_content_type_override
                self._auto_detect = False
                await self._result_callback(payload)
                
                # Emit on_content_type_detected event to notify other plugins
                if self._summary_client:
                    await self._summary_client._notify_plugins(
                        "on_content_type_detected",
                        content_type=self._user_content_type_override,
                        confidence=1.0,
                        source="USER_OVERRIDE",
                        reasoning="User override"
                    )
                
                return payload
            
            # Use get_recent_windows_text with the configured context limit
            text = self._window_manager.get_recent_windows_text(self._content_type_context_limit)
            
            # Use pre-created task instead of creating new one each time
            result = await self._task.detect_content_type(text, context_length=self._content_type_context_limit)
            
            # Track previous content type for change detection
            previous_content_type = self._current_content_type
            
            payload = {
                "type": "content_type_detection",
                "content_type": result.content_type,
                "confidence": result.confidence,
                "source": "AUTO_DETECTED",
                "previous_content_type": previous_content_type,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }
            
            # Update current content type
            self._current_content_type = result.content_type
            self._auto_detect = False
            await self._result_callback(payload)
            
            # Emit on_content_type_detected event to notify other plugins
            if self._summary_client:
                await self._summary_client._notify_plugins(
                    "on_content_type_detected",
                    content_type=result.content_type,
                    confidence=result.confidence,
                    source="AUTO_DETECTED",
                    reasoning=result.reasoning
                )
            
            return payload
        finally:
            self._in_progress = False
    
    def on_update_params(
        self,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_temperature: Optional[float] = None,
        content_type_context_limit: Optional[int] = None,
    ):
        """Handle on_update_params event from SummaryClient.
        
        Args:
            reasoning_max_tokens: New max tokens for reasoning API
            reasoning_temperature: New temperature for reasoning API
            content_type_context_limit: New character limit for content type detection
        """
        if reasoning_max_tokens is not None:
            self._max_tokens = reasoning_max_tokens
            self._task.max_tokens = reasoning_max_tokens
            logger.info(f"Updated reasoning_max_tokens to {reasoning_max_tokens}")
        
        if reasoning_temperature is not None:
            self._temperature = reasoning_temperature
            self._task.temperature = reasoning_temperature
            logger.info(f"Updated reasoning_temperature to {reasoning_temperature}")
        
        if content_type_context_limit is not None:
            self._content_type_context_limit = content_type_context_limit
            logger.info(f"Updated content_type_context_limit to {content_type_context_limit}")


def init_plugin(plugin_name: str, window_manager, llm_manager, result_callback: Callable, summary_client=None):
    """Initialize the plugin and register with summary_client."""
    global _window_manager, _summary_client, _result_callback
        
    plugin_instance = ContentTypeDetectionPlugin(
        window_manager=window_manager,
        llm_manager=llm_manager,
        result_callback=result_callback,
        summary_client=summary_client
    )
    
    if summary_client:
        summary_client.register_plugin_event_sub(
            plugin_name=plugin_name,
            plugin_instance=plugin_instance,
            events={
                "summary_window_available": plugin_instance.process,
                "on_update_params": plugin_instance.on_update_params
            }
        )


__all__ = [
    "ContentTypeDetectionTask",
    "ContentTypeDetectionPlugin",
    "init_plugin",
    "CONTENT_TYPE_DETECTION_PROMPT",
    # Types and schemas
    "ContentType",
    "ContentTypeSource",
    "ContentTypeState",
    "ContentTypeDetectionSchema",
]