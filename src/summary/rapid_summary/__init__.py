"""
Rapid summary task module.
"""

import logging
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List, Optional

from .task import RapidSummaryTask

logger = logging.getLogger(__name__)


def _format_rapid_summary_for_context(result: Dict[str, Any]) -> str:
    """Format rapid_summary result for AI context."""
    try:
        segments = result.get("segments", [])
        
        if segments:
            summary = segments[0].get("summary", "")
            return f"### Rapid Summary\n{summary}"
        return "No summary available"
    except (TypeError, IndexError, KeyError) as e:
        logger.warning("Failed to format rapid_summary result: %s", str(e))
        return "No summary available"


# Module-level references (set by init_plugin)
_window_manager = None
_summary_client = None
_result_callback = None


class RapidSummaryPlugin:
    """Plugin for rapid summarization."""
    
    def __init__(self, window_manager, llm_manager, result_callback, summary_client=None,
                 send_monitoring_event_callback=None, **kwargs):
        self._window_manager = window_manager
        self._llm = llm_manager
        self._result_callback = result_callback
        self._summary_client = summary_client  # Keep for non-LLM operations
        self._send_monitoring_event_callback = send_monitoring_event_callback
        
        # Track the highest context_summary timestamp (for out-of-order windows)
        self._context_summary_timestamp: float = 0.0
        
        # Create task once at init and reuse for all processing
        self._task = RapidSummaryTask(
            llm_client=self._llm.rapid_llm_client,
            rapid_summary_response_json_schema=None,
            window_manager=window_manager,  # Pass window_manager for prior context
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
    
    def _get_window_by_id(self, window_id: int):
        """Get window by ID from window_manager._summary_windows."""
        for window in self._window_manager._summary_windows:
            if window.window_id == window_id:
                return window
        return None
    
    async def process(
        self,
        summary_window_id: int,
        **kwargs
    ):
        """Process a summary window for rapid summary.
        
        Args:
            summary_window_id: The ID of the summary window
            **kwargs: Additional parameters
        """
        # Use stored references
        window_manager = self._window_manager
        result_callback = self._result_callback
        
        # Check if rapid summary is enabled - need summary_client for this
        if not self._llm.fast_client:
            return {}
        
        # Get context from tracked timestamp forward (set by on_context_summary_complete event)
        if self._context_summary_timestamp > 0:
            rapid_context, rapid_window_ids = self._window_manager.get_text_and_window_ids_since_timestamp(
                self._context_summary_timestamp
            )
        else:
            rapid_context = ""
            rapid_window_ids = []
        
        # Get the summary window by accessing _windows directly
        window = self._get_window_by_id(summary_window_id)
        if not window:
            logger.warning(f"Could not find summary window {summary_window_id}")
            return {}
        
        window_start = window.timestamp_start
        window_end = window.timestamp_end
        window_text = window.text
        
        # Get transcription window IDs from the summary window
        transcription_window_ids = window_manager.get_window_transcription_ids(summary_window_id)
        if not transcription_window_ids:
            # Fallback for backwards compatibility
            transcription_window_ids = [summary_window_id]
        
        # Convert window text to segments format
        segments = [{"text": window_text, "start": window_start, "end": window_end}]
        
        # Get word count
        word_count = len(window_text.split()) if window_text else 0
        
        start_time = time.perf_counter()
        success = True
        
        try:
            # Use pre-created task instead of creating new one each time
            # Pass the window so it can store scribe notes in plugin results
            logger.info(f"Starting rapid summary for window {summary_window_id}, text length: {len(window_text)}")
            result = await self._task.build_rapid_summary_payload(
                segments=segments,
                transcription_window_id=summary_window_id,
                window_start_ts=window_start,
                window_end_ts=window_end,
                context_since_last_summary=rapid_context,
                transcription_window_ids=transcription_window_ids,
                summary_window=window,  # Pass window for plugin storage
            )
            
            # Note: Result is now stored in plugin results via summary_window.store_result()
            # The old store_plugin_result call is no longer needed since task.py handles it
            
            logger.info(f"Rapid summary completed, result: {result}")
            await result_callback(result)
            return result
        except Exception as e:
            success = False
            logger.error(f"Rapid summary error: {e}", exc_info=True)
            return {}
        finally:
            # Emit complete monitoring event
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._send_monitoring_event({
                "summary_window_id": summary_window_id,
                "word_count": word_count,
                "duration_ms": duration_ms,
                "success": success,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }, "rapid_summary_complete")


    async def handle_context_summary_complete(self, summary_window_id: int, timestamp: float):
        """Handle on_context_summary_complete event from context_summary plugin.
        
        Track the highest timestamp since windows can arrive out of order.
        
        Args:
            summary_window_id: The ID of the processed summary window
            timestamp: The window_end timestamp from context_summary
        """
        if timestamp > self._context_summary_timestamp:
            self._context_summary_timestamp = timestamp
            logger.debug(f"Updated context summary timestamp to {timestamp}")
    
    def on_update_params(
        self,
        fast_max_tokens: Optional[int] = None,
        fast_temperature: Optional[float] = None,
    ):
        """Handle on_update_params event from SummaryClient.
        
        Args:
            fast_max_tokens: New max tokens for fast summary API
            fast_temperature: New temperature for fast summary API
        """
        if fast_max_tokens is not None:
            self._task.max_tokens = fast_max_tokens
            logger.info(f"Updated fast_max_tokens to {fast_max_tokens}")
        
        if fast_temperature is not None:
            self._task.temperature = fast_temperature
            logger.info(f"Updated fast_temperature to {fast_temperature}")
    
    def reset(self):
        """Reset tracked timestamp for new stream."""
        self._context_summary_timestamp = 0.0
        logger.debug("RapidSummaryPlugin reset - context timestamp cleared")


def init_plugin(plugin_name: str, window_manager, llm_manager, result_callback: Callable,
                summary_client=None, send_monitoring_event_callback=None):
    """Initialize the plugin and register with summary_client."""
    global _window_manager, _summary_client, _result_callback
    
    # Register format callback for plugin results
    window_manager.register_plugin_format_callback(
        "rapid_summary",
        _format_rapid_summary_for_context,
    )
    
    _window_manager = window_manager
    _summary_client = summary_client  # Keep for registration only
    _result_callback = result_callback
    
    plugin_instance = RapidSummaryPlugin(
        window_manager=window_manager,
        llm_manager=llm_manager,
        result_callback=result_callback,
        summary_client=summary_client,
        send_monitoring_event_callback=send_monitoring_event_callback
    )
    
    if summary_client:
        summary_client.register_plugin_event_sub(
            plugin_name=plugin_name,
            plugin_instance=plugin_instance,
            events={
                "summary_window_available": plugin_instance.process,
                "on_context_summary_complete": plugin_instance.handle_context_summary_complete,
                "on_update_params": plugin_instance.on_update_params
            }
        )


__all__ = [
    "RapidSummaryTask",
    "RapidSummaryPlugin",
    "init_plugin",
]