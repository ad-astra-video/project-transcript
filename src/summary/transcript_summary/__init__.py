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
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

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

        self._context_window_minutes = self._read_int_env(
            "TRANSCRIPT_SUMMARY_CONTEXT_WINDOW_MINUTES", default=10, minimum=1
        )
        self._context_window_ms = self._context_window_minutes * 60 * 1000
        self._max_segment_text_chars = self._read_int_env(
            "TRANSCRIPT_SUMMARY_MAX_SEGMENT_TEXT_CHARS", default=4000, minimum=200
        )

        # Running state — updated after every successful LLM call
        self._current_summary: Optional[Dict[str, Any]] = None

        # Full session accumulation of summary sections (client-facing history)
        self._accumulated_sections: List[Dict[str, Any]] = []

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

        # Retry guard: discard buffered data after this many consecutive failures
        self._MAX_RETRIES = 3
        self._consecutive_failures: int = 0

        self._task = TranscriptSummaryTask(
            llm_client=self._llm.reasoning_llm_client,
            verification_llm_client=self._llm.rapid_llm_client,
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
                    prior_summary = self._current_summary or {
                        "sections": list(self._accumulated_sections),
                        "key_points": [],
                        "topics": [],
                    }

                result = await self._task.process(
                    segments=segments,
                    current_summary=prior_summary,
                    context_window_ms=self._context_window_ms,
                    max_segment_text_chars=self._max_segment_text_chars,
                )

                async with self._lock:
                    # The latest section (highest end_ms) is the only one the
                    # LLM was allowed to shorten.  Pass its end_ms so the
                    # merge logic can protect all other sections from
                    # unintended shortening.
                    _latest_end_ms: Optional[int] = None
                    if self._accumulated_sections:
                        _latest_end_ms = max(
                            int(s.get("end_ms", 0))
                            for s in self._accumulated_sections
                        )
                    self._accumulated_sections = self._merge_sections(
                        self._accumulated_sections,
                        result.get("sections", []),
                        latest_end_ms=_latest_end_ms,
                    )

                    self._current_summary = {
                        "sections": list(self._accumulated_sections),
                        "key_points": result.get("key_points", []),
                        "topics": result.get("topics", []),
                    }

                # Reset failure counter on success
                self._consecutive_failures = 0

                summary_markdown = self._render_summary_markdown(self._accumulated_sections)

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
                        "verification_input_tokens": result.get("verification_input_tokens", 0),
                        "verification_output_tokens": result.get("verification_output_tokens", 0),
                    },
                    "summary": summary_markdown,
                    "key_points": result.get("key_points", []),
                    "topics": result.get("topics", []),
                }

                await self._result_callback(payload)

            except Exception as exc:
                logger.error("transcript_summary error: %s", exc, exc_info=True)

                self._consecutive_failures += 1

                # Determine whether this error can ever succeed with the same input.
                # Context-length / bad-request errors from the LLM API are non-retryable
                # because the identical payload will always produce the same error.
                non_retryable = self._is_non_retryable(exc)

                if non_retryable or self._consecutive_failures > self._MAX_RETRIES:
                    if non_retryable:
                        logger.warning(
                            "transcript_summary: non-retryable error (context/token limit) — "
                            "discarding %d buffered events",
                            len(buffered_data),
                        )
                    else:
                        logger.warning(
                            "transcript_summary: exceeded max retries (%d) — "
                            "discarding %d buffered events",
                            self._MAX_RETRIES,
                            len(buffered_data),
                        )
                    self._consecutive_failures = 0
                    # Do NOT re-buffer; the data is discarded.
                else:
                    # Re-buffer the items to the front of the buffer for retry.
                    # Insert at the beginning to maintain order (oldest first).
                    for data in reversed(buffered_data):
                        self._buffered_fast_summary_items.insert(0, data)

                    logger.info(
                        "transcript_summary: re-buffered %d events for retry "
                        "(attempt %d/%d)",
                        len(buffered_data),
                        self._consecutive_failures,
                        self._MAX_RETRIES,
                    )
            finally:
                # Clear processing flag
                self._is_processing = False

                # Check if new data arrived while processing - loop to process if needed
                # (the while loop will check buffer size and process again if needed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_non_retryable(exc: Exception) -> bool:
        """Return True if *exc* is an error that will never succeed with the same input.

        Specifically catches OpenAI context-length / bad-request errors where the
        payload is too large for the model's context window.
        """
        if _OPENAI_AVAILABLE:
            if isinstance(exc, openai.BadRequestError):
                # OpenAI uses error code 'context_length_exceeded' or surfaces it
                # via the message when the prompt is too long.
                msg = str(exc).lower()
                code = getattr(exc, "code", "") or ""
                return (
                    "context_length_exceeded" in code
                    or "context_length_exceeded" in msg
                    or "maximum context length" in msg
                    or "too many tokens" in msg
                    or "reduce the length" in msg
                )
        # Fallback: check the error message text for common phrases regardless
        # of whether the openai package is installed.
        msg = str(exc).lower()
        return (
            "context_length_exceeded" in msg
            or "maximum context length" in msg
            or "too many tokens" in msg
            or "reduce the length" in msg
        )

    @staticmethod
    def _read_int_env(name: str, default: int, minimum: int = 0) -> int:
        """Read int env var with validation and fallback."""
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("transcript_summary: invalid int for %s=%r, using default=%d", name, raw, default)
            return default
        if value < minimum:
            logger.warning(
                "transcript_summary: %s=%d below minimum=%d, using default=%d",
                name,
                value,
                minimum,
                default,
            )
            return default
        return value

    @staticmethod
    def _format_timestamp(ms: int) -> str:
        """Format milliseconds as H:MM:SS for markdown headings."""
        total_seconds = ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    @classmethod
    def _render_summary_markdown(cls, sections: List[Dict[str, Any]]) -> str:
        """Render full accumulated sections as markdown for client payload."""
        if not sections:
            return ""

        ordered = sorted(
            sections,
            key=lambda sec: (int(sec.get("start_ms", 0)), int(sec.get("end_ms", 0))),
        )
        blocks: List[str] = []
        for section in ordered:
            heading = (section.get("heading") or "Untitled Topic").strip()
            heading = cls._TRAILING_TS_RE.sub("", heading).strip()
            start_ms = int(section.get("start_ms", 0))
            end_ms = int(section.get("end_ms", 0))
            content = (section.get("content") or "").strip()
            blocks.append(
                f"## {heading} [{cls._format_timestamp(start_ms)} – {cls._format_timestamp(end_ms)}]\n\n{content}"
            )
        return "\n\n".join(blocks).strip()

    @staticmethod
    def _ranges_overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
        """Compute overlap ratio over the shorter interval length."""
        overlap_start = max(a_start, b_start)
        overlap_end = min(a_end, b_end)
        if overlap_end <= overlap_start:
            return 0.0
        overlap = overlap_end - overlap_start
        a_len = max(1, a_end - a_start)
        b_len = max(1, b_end - b_start)
        return overlap / min(a_len, b_len)

    # Regex to strip a trailing timestamp range the LLM may echo into heading fields.
    _TRAILING_TS_RE = re.compile(
        r"\s*\[\d+:\d{2}:\d{2}\s*[–—-]\s*\d+:\d{2}:\d{2}\]\s*$"
    )

    @classmethod
    def _merge_sections(
        cls,
        existing_sections: List[Dict[str, Any]],
        new_sections: List[Dict[str, Any]],
        latest_end_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Merge updated sections into accumulated sections.

        The LLM is only permitted to shorten or reword the *latest* section
        (the one with ``end_ms`` matching *latest_end_ms*).  For every other
        section, the incoming version is accepted only when its word count is
        at least as large as the existing version — this prevents the LLM from
        silently trimming sections it was told to preserve verbatim.

        Args:
            existing_sections: Accumulated sections from prior cycles.
            new_sections: Sections returned by the LLM in the current cycle.
            latest_end_ms: ``end_ms`` of the latest section that is eligible
                for compression.  When *None*, all incoming sections are
                accepted unconditionally (backward-compatible default).
        """
        merged: List[Dict[str, Any]] = [dict(section) for section in existing_sections]

        for incoming in new_sections:
            in_heading = (incoming.get("heading") or "").strip().lower()
            in_start = int(incoming.get("start_ms", 0))
            in_end = int(incoming.get("end_ms", in_start))
            in_end = max(in_start, in_end)

            best_index = None
            best_score = 0.0

            for idx, existing in enumerate(merged):
                ex_heading = (existing.get("heading") or "").strip().lower()
                ex_start = int(existing.get("start_ms", 0))
                ex_end = int(existing.get("end_ms", ex_start))
                ex_end = max(ex_start, ex_end)

                overlap_ratio = cls._ranges_overlap_ratio(in_start, in_end, ex_start, ex_end)
                heading_match = bool(in_heading and ex_heading and in_heading == ex_heading)
                score = overlap_ratio + (0.5 if heading_match else 0.0)

                if score > best_score and (overlap_ratio >= 0.5 or heading_match):
                    best_score = score
                    best_index = idx

            raw_heading = (incoming.get("heading") or "Untitled Topic").strip()
            normalized_incoming = {
                "heading": cls._TRAILING_TS_RE.sub("", raw_heading).strip(),
                "start_ms": in_start,
                "end_ms": in_end,
                "content": incoming.get("content", ""),
            }

            if best_index is None:
                merged.append(normalized_incoming)
            else:
                existing_sec = merged[best_index]
                ex_end_ms = int(existing_sec.get("end_ms", 0))

                # Determine whether this section is the latest (compression-eligible)
                is_latest = (
                    latest_end_ms is not None
                    and abs(ex_end_ms - latest_end_ms) < 30_000  # 30 s tolerance
                )

                if is_latest or latest_end_ms is None:
                    # Latest section or no protection requested — accept as-is
                    merged[best_index] = normalized_incoming
                else:
                    # Non-latest section: accept only if not shortened
                    existing_words = len(existing_sec.get("content", "").split())
                    incoming_words = len(normalized_incoming.get("content", "").split())

                    if incoming_words >= existing_words:
                        merged[best_index] = normalized_incoming
                    else:
                        # Preserve existing content; still update end_ms if extended
                        if in_end > ex_end_ms:
                            merged[best_index]["end_ms"] = in_end
                        logger.debug(
                            "transcript_summary: kept existing content for non-latest "
                            "section '%s' (%d words vs incoming %d words)",
                            existing_sec.get("heading", ""),
                            existing_words,
                            incoming_words,
                        )

        merged.sort(key=lambda sec: (int(sec.get("start_ms", 0)), int(sec.get("end_ms", 0))))
        return merged

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_update_params(
        self,
        fast_max_tokens: Optional[int] = None,
        fast_temperature: Optional[float] = None,
        context_window_minutes: Optional[int] = None,
        max_segment_text_chars: Optional[int] = None,
        **kwargs,
    ):
        """Handle on_update_params event."""
        if fast_max_tokens is not None:
            self._task.max_tokens = fast_max_tokens
            logger.info("transcript_summary: updated max_tokens to %d", fast_max_tokens)
        if fast_temperature is not None:
            self._task.temperature = fast_temperature
            logger.info("transcript_summary: updated temperature to %f", fast_temperature)
        if context_window_minutes is not None:
            if context_window_minutes >= 1:
                self._context_window_minutes = context_window_minutes
                self._context_window_ms = context_window_minutes * 60 * 1000
                logger.info(
                    "transcript_summary: updated context_window_minutes to %d",
                    context_window_minutes,
                )
            else:
                logger.warning(
                    "transcript_summary: ignored invalid context_window_minutes=%s",
                    context_window_minutes,
                )
        if max_segment_text_chars is not None:
            if max_segment_text_chars >= 200:
                self._max_segment_text_chars = max_segment_text_chars
                logger.info(
                    "transcript_summary: updated max_segment_text_chars to %d",
                    max_segment_text_chars,
                )
            else:
                logger.warning(
                    "transcript_summary: ignored invalid max_segment_text_chars=%s",
                    max_segment_text_chars,
                )

    def reset(self):
        """Clear all state for a new stream."""
        self._buffered_fast_summary_items.clear()
        self._latest_summary_window_id = None
        self._latest_window_start_ms = 0
        self._latest_window_end_ms = 0
        self._is_processing = False
        self._current_summary = None
        self._accumulated_sections.clear()
        self._consecutive_failures = 0
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
