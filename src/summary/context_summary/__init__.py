"""
Context summary task module.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List, Optional

from .task import ContextSummaryTask, Topic
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_OUTPUT_CONSTRAINTS, CONTENT_TYPE_RULE_MODIFIERS

logger = logging.getLogger(__name__)


def _format_context_summary_for_context(result: Dict[str, Any]) -> str:
    """Format context_summary result for AI context with insight IDs and continuation markers."""
    import json
    
    summary_text = result.get("summary_text", "{}")
    
    try:
        summary_data = json.loads(summary_text)
        insights = summary_data.get("insights", [])
        
        formatted_insights = []
        for insight in insights:
            # Include insight ID so LLM can reference it in continuation_of/correction_of
            insight_id = insight.get("insight_id", 0)
            id_hint = f"[#{insight_id}]" if insight_id else ""
            
            # Add continuation/correction markers if present
            markers = []
            continuation_of = insight.get("continuation_of")
            correction_of = insight.get("correction_of")
            if continuation_of:
                markers.append(f"CONTINUATION of insight #{continuation_of}")
            if correction_of:
                markers.append(f"CORRECTION of insight #{correction_of}")
            
            marker_text = f" ({', '.join(markers)})" if markers else ""
            
            formatted_insights.append(
                f"- **{insight.get('insight_type', 'NOTES')}** {id_hint}: "
                f"{insight.get('insight_text', '')}{marker_text}"
            )
        
        return f"### Insights\n" + "\n".join(formatted_insights)
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logger.warning("Failed to parse context_summary result: %s", str(e))
        return summary_text


class ContentTypeStateHolder:
    """Simple holder for content type state to pass to tasks."""
    def __init__(self, content_type: str = "UNKNOWN", confidence: float = 0.0, source: str = "INITIAL"):
        self.content_type = content_type
        self.confidence = confidence
        self.source = source


class ContextSummaryPlugin:
    """Plugin for context-based summarization."""
    
    def __init__(self, window_manager, llm_manager, result_callback, summary_client=None,
                 initial_summary_delay_seconds: float = 15.0, send_monitoring_event_callback=None, **kwargs):
        self._window_manager = window_manager
        self._llm = llm_manager
        self._result_callback = result_callback
        self._summary_client = summary_client  # Keep for non-LLM operations
        self._send_monitoring_event_callback = send_monitoring_event_callback
        
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
            rapid_llm_client=self._llm.rapid_llm_client,
            content_type_state=self._content_type_state,
            window_manager=self._window_manager,
        )
    
    async def _send_monitoring_event(self, event_data: Dict[str, Any], event_type: str):
        """Send a monitoring event if callback is configured."""
        if self._send_monitoring_event_callback:
            try:
                await self._send_monitoring_event_callback(event_data, event_type)
            except Exception as e:
                logger.warning(f"Failed to send monitoring event: {e}")
    
    def set_monitoring_callback(self, callback):
        """Set the monitoring event callback after initialization."""
        self._send_monitoring_event_callback = callback
    
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
        
        # Update the task's reference to the content type state
        # This ensures the task sees the updated content type when processing
        self._task._content_type_state = self._content_type_state
                
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
        
        # Get word count from the summary window
        window_text = self._window_manager.get_window_text(summary_window_id)
        word_count = len(window_text.split()) if window_text else 0
        
        # Emit start monitoring event
        await self._send_monitoring_event({
            "summary_window_id": summary_window_id,
            "word_count": word_count,
            "window_start_ms": int(window_start * 1000),
            "window_end_ms": int(window_end * 1000),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }, "context_summary_start")
        
        start_time = time.perf_counter()
        success = True
        
        try:
            # Use pre-created task - it handles context building, text retrieval, and result processing internally
            result = await self._task.process_context_summary(summary_window_id)
            
            # Store result in window for other plugins to access
            self._window_manager.store_plugin_result(
                window_id=summary_window_id,
                plugin_name="context_summary",
                result=result,
                include_in_context=True,
            )
            
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
        except Exception as e:
            success = False
            raise
        finally:
            # Emit complete monitoring event
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._send_monitoring_event({
                "summary_window_id": summary_window_id,
                "word_count": word_count,
                "duration_ms": duration_ms,
                "success": success,
                "content_type": self._content_type_state.content_type,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }, "context_summary_complete")


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


def init_plugin(plugin_name: str, window_manager, llm_manager, result_callback: Callable,
                summary_client=None, send_monitoring_event_callback=None):
    """Initialize the plugin and register with summary_client."""
    # Register format callback for plugin results
    window_manager.register_plugin_format_callback(
        "context_summary",
        _format_context_summary_for_context,
    )
    
    initial_delay = summary_client.initial_summary_delay_seconds if summary_client else 0.0
    
    plugin_instance = ContextSummaryPlugin(
        window_manager=window_manager,
        llm_manager=llm_manager,
        result_callback=result_callback,
        summary_client=summary_client,
        initial_summary_delay_seconds=initial_delay,
        send_monitoring_event_callback=send_monitoring_event_callback
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
    "Topic",
    "InsightResponseItemSchema",
    "InsightsResponseSchema",
]