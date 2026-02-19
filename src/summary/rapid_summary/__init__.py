"""
Rapid summary task module.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List, Optional

from .task import RapidSummaryTask

logger = logging.getLogger(__name__)

# Module-level references (set by init_plugin)
_window_manager = None
_summary_client = None
_result_callback = None


class RapidSummaryPlugin:
    """Plugin for rapid summarization."""
    
    def __init__(self, window_manager, llm, result_callback, summary_client=None, **kwargs):
        self._window_manager = window_manager
        self._llm = llm
        self._result_callback = result_callback
        self._summary_client = summary_client  # Keep for non-LLM operations
        
        # Create task once at init and reuse for all processing
        self._task = RapidSummaryTask(
            llm_client=self._llm.rapid_llm_client,
            rapid_summary_response_json_schema=None,
        )
    
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
        
        # Get context for rapid summary - need to get from somewhere
        # For now, pass empty context - plugins can enhance this
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
        
        try:
            # Use pre-created task instead of creating new one each time
            result = await self._task.build_rapid_summary_payload(
                segments=segments,
                transcription_window_id=summary_window_id,
                window_start_ts=window_start,
                window_end_ts=window_end,
                context_since_last_summary=rapid_context,
                transcription_window_ids=transcription_window_ids
            )
            
            # Update tracker so next rapid summary knows what text has been processed
            self._summary_client._update_context_summary_tracker(window_end)
            
            await result_callback(result)
            return result
        except Exception as e:
            logger.error(f"Rapid summary error: {e}")
            return {}


def init_plugin(plugin_name: str, window_manager, llm, result_callback: Callable, summary_client=None):
    """Initialize the plugin and register with summary_client."""
    global _window_manager, _summary_client, _result_callback
    
    _window_manager = window_manager
    _summary_client = summary_client  # Keep for registration only
    _result_callback = result_callback
    
    plugin_instance = RapidSummaryPlugin(
        window_manager=window_manager,
        llm=llm,
        result_callback=result_callback,
        summary_client=summary_client,
    )
    
    if summary_client:
        summary_client.register_plugin_event_sub(
            plugin_name=plugin_name,
            plugin_instance=plugin_instance,
            events={
                "summary_window_available": plugin_instance.process
            }
        )


__all__ = [
    "RapidSummaryTask",
    "RapidSummaryPlugin",
    "init_plugin",
]