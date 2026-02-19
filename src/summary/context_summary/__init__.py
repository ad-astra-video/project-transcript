"""
Context summary task module.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List, Optional

from .task import ContextSummaryTask
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_OUTPUT_CONSTRAINTS, CONTENT_TYPE_RULE_MODIFIERS

logger = logging.getLogger(__name__)



class ContentTypeStateHolder:
    """Simple holder for content type state to pass to tasks."""
    def __init__(self, content_type: str = "UNKNOWN", confidence: float = 0.0, source: str = "INITIAL"):
        self.content_type = content_type
        self.confidence = confidence
        self.source = source


class ContextSummaryPlugin:
    """Plugin for context-based summarization."""
    
    def __init__(self, window_manager, llm_manager, result_callback, summary_client=None,
                 initial_summary_delay_seconds: float = 15.0, **kwargs):
        self._window_manager = window_manager
        self._llm = llm_manager
        self._result_callback = result_callback
        self._summary_client = summary_client  # Keep for non-LLM operations
        
        self._initial_summary_delay_seconds = initial_summary_delay_seconds
        self._has_performed_summary = False
        
        # Content type state - managed by this plugin
        self._content_type_state = ContentTypeStateHolder()
        self._sentiment_enabled: bool = False
        self._participants_enabled: bool = False
        
        # Create task once at init and reuse for all processing
        # Note: message_format_mode is now managed by LLMClient
        self._task = ContextSummaryTask(
            llm_client=self._llm.reasoning_llm_client,
            content_type_state=self._content_type_state,
            window_manager=self._window_manager,
        )
    
    async def handle_content_type_detected(self, content_type: str, confidence: float, source: str, reasoning: str):
        """Handle on_content_type_detected event from content_type_detection plugin.
        
        Updates the content type state so that subsequent
        summarization uses the correct content type rules and prompts.
        
        Args:
            content_type: The detected content type
            confidence: Detection confidence score
            source: Source of detection (e.g., "AUTO_DETECTED")
            reasoning: LLM's reasoning for the detection
        """
        self._content_type_state = ContentTypeStateHolder(
            content_type=content_type,
            confidence=confidence,
            source=source
        )
                
        logger.info(f"ContextSummaryPlugin updated content type to: {content_type} "
                    f"(sentiment_enabled={self._sentiment_enabled}, participants_enabled={self._participants_enabled})")
    
    def _should_process(self, summary_window_id: int) -> bool:
        """Check if should process this summary window."""
        if not self._has_performed_summary:
            if self._window_manager._first_window_timestamp is not None:
                # Find the window and get its timestamp_start
                window_start = self._window_manager._summary_windows[-1].timestamp_start
                elapsed = window_start - self._window_manager._first_window_timestamp
                if elapsed < self._initial_summary_delay_seconds:
                    logger.info(f"Delaying first summary - only {elapsed:.1f}s elapsed")
                    return False
        return True
    
    async def process(self, summary_window_id: int, **kwargs):
        """Process a summary window.
        
        Args:
            summary_window_id: The ID of the summary window
            **kwargs: Additional parameters
        """
        
        if not self._should_process(summary_window_id):
            return {}
        
        # Get window timing
        window_start = self._window_manager.get_window_start(summary_window_id)
        window_end = self._window_manager.get_window_end(summary_window_id)
        
        # Use pre-created task - it handles context building, text retrieval, and result processing internally
        result = await self._task.process_context_summary(summary_window_id)
        
        segments = [
            {
                "id": f"{summary_window_id}-0",
                "summary_type": "context_summary",
                "background_context": result.get("reasoning_content", ""),
                "summary": result.get("summary_text", "{}"),
            }
        ]
        
        payload = {
            "type": "context_summary",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timing": {
                "summary_window_id": summary_window_id,
                "transcription_window_ids": self._window_manager.get_window_transcription_ids(summary_window_id),
                "media_window_start_ms": int(window_start * 1000),
                "media_window_end_ms": int(window_end * 1000)
            },
            "llm_usage": {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0)
            },
            "segments": segments,
        }
        
        self._has_performed_summary = True
        
        # Send result via callback (like RapidSummaryPlugin does)
        await self._result_callback(payload)
        
        # Emit event with timestamp for other plugins (e.g., rapid_summary)
        if self._summary_client:
            await self._summary_client._notify_plugins(
                "on_context_summary_complete",
                summary_window_id=summary_window_id,
                timestamp=window_end
            )
        
        return payload


    def on_update_params(
        self,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_temperature: Optional[float] = None,
        reasoning_system_prompt: Optional[str] = None,
        initial_summary_delay_seconds: Optional[float] = None,
    ):
        """Handle on_update_params event from SummaryClient.
        
        Args:
            reasoning_max_tokens: New max tokens for reasoning API
            reasoning_temperature: New temperature for reasoning API
            reasoning_system_prompt: New system prompt for reasoning API
            initial_summary_delay_seconds: New delay before first summary
        """
        if reasoning_max_tokens is not None:
            self._task.max_tokens = reasoning_max_tokens
            logger.info(f"Updated reasoning_max_tokens to {reasoning_max_tokens}")
        
        if reasoning_temperature is not None:
            self._task.temperature = reasoning_temperature
            logger.info(f"Updated reasoning_temperature to {reasoning_temperature}")
        
        if reasoning_system_prompt is not None:
            self._task.system_prompt = reasoning_system_prompt
            logger.info("Updated reasoning_system_prompt")
        
        if initial_summary_delay_seconds is not None:
            self._initial_summary_delay_seconds = initial_summary_delay_seconds
            logger.info(f"Updated initial_summary_delay_seconds to {initial_summary_delay_seconds}")


def init_plugin(plugin_name: str, window_manager, llm_manager, result_callback: Callable, summary_client=None):
    """Initialize the plugin and register with summary_client."""
    initial_delay = summary_client.initial_summary_delay_seconds if summary_client else 0.0
    
    plugin_instance = ContextSummaryPlugin(
        window_manager=window_manager,
        llm_manager=llm_manager,
        result_callback=result_callback,
        summary_client=summary_client,
        initial_summary_delay_seconds=initial_delay
    )
    
    if summary_client:
        summary_client.register_plugin_event_sub(
            plugin_name=plugin_name,
            plugin_instance=plugin_instance,
            events={
                "summary_window_available": plugin_instance.process,
                "on_content_type_detected": plugin_instance.handle_content_type_detected,
                "on_update_params": plugin_instance.on_update_params
            }
        )


__all__ = [
    "ContextSummaryTask",
    "ContextSummaryPlugin",
    "init_plugin",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_OUTPUT_CONSTRAINTS",
    "CONTENT_TYPE_RULE_MODIFIERS",
    # Types and schemas
    "InsightType",
    "ClassificationField",
    "InsightResponseItemSchema",
    "InsightsResponseSchema",
]