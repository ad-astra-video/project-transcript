"""
Transcript summary plugin.

Subscribes to `transcription_window_available` to buffer incoming window IDs,
then fires an LLM call each time `fast_summary_available` is emitted by the
rapid_summary plugin.  The result is a growing markdown summary of the whole
transcript sent back to the client as a `transcript_summary` event.
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

        # IDs of transcription windows received since the last fast_summary_available
        self._buffered_window_ids: List[int] = []

        # Prevent concurrent LLM calls from corrupting _current_summary
        self._lock = asyncio.Lock()

        self._task = TranscriptSummaryTask(
            llm_client=self._llm.reasoning_llm_client,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_transcription_window(self, transcription_window_id: int, **kwargs):
        """Buffer each incoming transcription window ID.

        Subscribed to 'transcription_window_available'.
        """
        self._buffered_window_ids.append(transcription_window_id)

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

        Args:
            summary_window_id: Summary window that triggered this fast summary.
            fast_summary_items: Bullet-point strings from the rapid_summary LLM.
            window_start_ms: Media start time of the summary window (ms).
            window_end_ms: Media end time of the summary window (ms).
        """
        if not self._llm.reasoning_client:
            return

        async with self._lock:
            # Snapshot and clear the buffer atomically
            buffered_ids = list(self._buffered_window_ids)
            self._buffered_window_ids.clear()

        # Retrieve merged text for all buffered transcription windows
        if buffered_ids:
            window_texts = self._window_manager.get_merged_text_for_transcription_ids(buffered_ids)
        else:
            window_texts = ""

        logger.info(
            "transcript_summary: processing summary_window=%d, "
            "buffered_transcription_windows=%d, fast_summary_items=%d",
            summary_window_id,
            len(buffered_ids),
            len(fast_summary_items),
        )

        try:
            async with self._lock:
                prior_summary = self._current_summary

            result = await self._task.process(
                window_texts=window_texts,
                fast_summary_items=fast_summary_items,
                current_summary=prior_summary,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )

            async with self._lock:
                self._current_summary = result

            payload = {
                "type": "transcript_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {
                    "summary_window_id": summary_window_id,
                    "transcription_window_ids": buffered_ids,
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
        self._buffered_window_ids.clear()
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
                "transcription_window_available": plugin_instance.handle_transcription_window,
                "fast_summary_available": plugin_instance.process,
                "on_update_params": plugin_instance.on_update_params,
            },
        )


__all__ = ["TranscriptSummaryPlugin", "TranscriptSummaryTask", "init_plugin"]
