"""
Transcript summary plugin.

Subscribes to `summary_window_available` to buffer summary window info,
and `fast_summary_available` to buffer fast_summary_items.
Only fires an LLM call after 2 fast_summary_available events have been received,
accumulating all data between executions. The result is a growing markdown summary
of the whole transcript sent back to the client as a `transcript_summary` event.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .task import TranscriptSummaryTask

logger = logging.getLogger(__name__)


class TranscriptSummaryPlugin:
    """Maintains a cumulative markdown summary driven by fast_summary events."""

    def __init__(
        self,
        window_manager,
        llm_manager,
        result_callback: Callable,
        summary_client=None,
        send_monitoring_event_callback=None,
        **kwargs,
    ) -> None:
        self._window_manager = window_manager
        self._llm = llm_manager
        self._result_callback = result_callback
        self._summary_client = summary_client
        self._send_monitoring_event_callback = send_monitoring_event_callback

        # Running state — updated after every successful LLM call
        self._current_summary: Optional[Dict[str, Any]] = None

        # Buffer for fast_summary_items from multiple fast_summary_available events
        self._buffered_fast_summary_items: List[Dict[str, Any]] = []

        # Track the latest summary window info (updated by summary_window_available)
        self._latest_summary_window_id: Optional[int] = None
        self._latest_window_start_ms: int = 0
        self._latest_window_end_ms: int = 0

        # Track processing state for queue mechanism
        self._is_processing: bool = False

        # Prevent concurrent LLM calls from corrupting _current_summary
        self._lock = asyncio.Lock()

        self._task = TranscriptSummaryTask(
            llm_client=self._llm.reasoning_llm_client,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_summary_window(
        self,
        summary_window_id: int,
        **kwargs,
    ):
        """Store summary window info for later use.

        Subscribed to 'summary_window_available'.
        """
        self._latest_summary_window_id = summary_window_id
        # Get window timing info from window_manager (convert seconds to milliseconds)
        self._latest_window_start_ms = int(self._window_manager.get_window_start(summary_window_id) * 1000)
        self._latest_window_end_ms = int(self._window_manager.get_window_end(summary_window_id) * 1000)

    async def handle_transcription_window(self, transcription_window_id: int, **kwargs):
        """Buffer each incoming transcription window ID.

        Subscribed to 'transcription_window_available'.
        Note: This is kept for backward compatibility but not actively used
        in the new throttled pattern (we use summary_window text instead).
        """
        pass  # No longer needed - we use summary_window text now

    async def process(
        self,
        summary_window_id: int,
        fast_summary_items: List[str],
        window_start_ms: int,
        window_end_ms: int,
        **kwargs,
    ):
        """Update the running transcript summary using the latest fast summary.

        Subscribed to 'fast_summary_available'.
        Buffers fast_summary_items and only runs LLM after 2 events have been received.

        Args:
            summary_window_id: Summary window that triggered this fast summary.
            fast_summary_items: Bullet-point strings from the rapid_summary LLM.
            window_start_ms: Media start time of the summary window (ms).
            window_end_ms: Media end time of the summary window (ms).
        """
        if not self._llm.reasoning_client:
            return

        # Buffer the incoming data regardless of processing state
        self._buffered_fast_summary_items.append({
            "fast_summary_items": fast_summary_items,
            "summary_window_id": summary_window_id,
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
        })

        logger.info(
            "transcript_summary: buffered event, buffer_size=%d",
            len(self._buffered_fast_summary_items),
        )

        # Process if we have enough events and are not currently processing
        await self._process_buffered_events()

    async def _process_buffered_events(self):
        """Process buffered events if conditions are met.

        Runs LLM when:
        - Buffer has 2 or more events AND
        - No previous LLM call is still in progress

        After processing, checks if new data arrived while processing
        and processes again if needed.
        """
        while True:
            # Check conditions: need at least 2 events and not currently processing
            if len(self._buffered_fast_summary_items) < 2 or self._is_processing:
                break

            # Set processing flag
            self._is_processing = True

            # Get all buffered data
            buffered_data = list(self._buffered_fast_summary_items)
            self._buffered_fast_summary_items.clear()

            # Build per-segment dicts preserving individual window timing and text
            segments: List[Dict[str, Any]] = []
            for data in buffered_data:
                seg_window_id = data["summary_window_id"]
                window_text = self._window_manager.get_window_text(seg_window_id) or ""
                segments.append({
                    "fast_summary_items": data["fast_summary_items"],
                    "window_start_ms": data["window_start_ms"],
                    "window_end_ms": data["window_end_ms"],
                    "window_text": window_text,
                })

            # Derive overall timing from earliest start and latest end across all segments
            summary_window_id = buffered_data[-1]["summary_window_id"]
            window_start_ms = min(d["window_start_ms"] for d in buffered_data)
            window_end_ms = max(d["window_end_ms"] for d in buffered_data)
            total_items = sum(len(d["fast_summary_items"]) for d in buffered_data)

            logger.info(
                "transcript_summary: processing summary_window=%d, "
                "buffered_events=%d, total_fast_summary_items=%d",
                summary_window_id,
                len(buffered_data),
                total_items,
            )

            try:
                async with self._lock:
                    prior_summary = self._current_summary

                result = await self._task.process(
                    segments=segments,
                    current_summary=prior_summary,
                )

                async with self._lock:
                    self._current_summary = result

                payload = {
                    "type": "transcript_summary",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "timing": {
                        "summary_window_id": summary_window_id,
                        "buffered_event_count": len(buffered_data),
                        "media_window_start_ms": window_start_ms,
                        "media_window_end_ms": window_end_ms,
                    },
                    "llm_usage": {
                        "input_tokens": result.get("input_tokens", 0),
                        "output_tokens": result.get("output_tokens", 0),
                    },
                    "summary": result["summary"],
                    "key_points": result["key_points"],
                    "topics": result["topics"],
                }

                await self._result_callback(payload)

            except Exception as exc:
                logger.error("transcript_summary error: %s", exc, exc_info=True)
                
                # Re-buffer the items to the front of the buffer for retry
                # Insert at the beginning to maintain order (oldest first)
                for data in reversed(buffered_data):
                    self._buffered_fast_summary_items.insert(0, data)
                
                logger.info(
                    "transcript_summary: re-buffered %d events for retry",
                    len(buffered_data),
                )
            finally:
                # Clear processing flag
                self._is_processing = False

                # Check if new data arrived while processing - loop to process if needed
                # (the while loop will check buffer size and process again if needed)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_update_params(
        self,
        fast_max_tokens: Optional[int] = None,
        fast_temperature: Optional[float] = None,
        **kwargs,
    ):
        """Handle on_update_params event."""
        if fast_max_tokens is not None:
            self._task.max_tokens = fast_max_tokens
            logger.info("transcript_summary: updated max_tokens to %d", fast_max_tokens)
        if fast_temperature is not None:
            self._task.temperature = fast_temperature
            logger.info("transcript_summary: updated temperature to %f", fast_temperature)

    def reset(self):
        """Clear all state for a new stream."""
        self._buffered_fast_summary_items.clear()
        self._latest_summary_window_id = None
        self._latest_window_start_ms = 0
        self._latest_window_end_ms = 0
        self._is_processing = False
        self._current_summary = None
        logger.debug("TranscriptSummaryPlugin reset")


def init_plugin(
    plugin_name: str,
    window_manager,
    llm_manager,
    result_callback: Callable,
    summary_client=None,
    send_monitoring_event_callback=None,
):
    """Auto-discover entry point — called by SummaryClient._discover_plugins()."""
    plugin_instance = TranscriptSummaryPlugin(
        window_manager=window_manager,
        llm_manager=llm_manager,
        result_callback=result_callback,
        summary_client=summary_client,
        send_monitoring_event_callback=send_monitoring_event_callback,
    )

    if summary_client:
        summary_client.register_plugin_event_sub(
            plugin_name=plugin_name,
            plugin_instance=plugin_instance,
            events={
                "summary_window_available": plugin_instance.handle_summary_window,
                "transcription_window_available": plugin_instance.handle_transcription_window,
                "fast_summary_available": plugin_instance.process,
                "on_update_params": plugin_instance.on_update_params,
            },
        )


__all__ = ["TranscriptSummaryPlugin", "TranscriptSummaryTask", "init_plugin"]
