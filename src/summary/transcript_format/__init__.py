"""Transcript format plugin.

Formats the actual raw transcript (summary window text) into polished prose.
Uses content-type detection timeline to apply different style guidelines over time.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .task import TranscriptFormatTask

logger = logging.getLogger(__name__)


class TranscriptFormatPlugin:
    """Incrementally formats raw transcript windows with content-type-aware styles."""

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

        self._task = TranscriptFormatTask(
            llm_client=self._llm.rapid_llm_client,
            max_tokens=2048,
            temperature=0.1,
            model_context_tokens=16000,
            input_token_budget=12000,
            chunk_overlap_chars=500,
        )

        self._content_type_changes: List[Dict[str, Any]] = []
        self._latest_content_type: str = "UNKNOWN"
        self._latest_confidence: float = 0.0

        self._formatted_by_window: Dict[int, Dict[str, Any]] = {}
        self._dirty_start_window_id: Optional[int] = None
        self._is_processing: bool = False
        self._lock = asyncio.Lock()

    async def _send_monitoring_event(self, event_data: Dict[str, Any], event_type: str):
        if self._send_monitoring_event_callback:
            try:
                await self._send_monitoring_event_callback(event_data, event_type)
            except Exception as exc:
                logger.warning("Failed to send monitoring event: %s", exc)

    def _mark_dirty_from(self, window_id: Optional[int]) -> None:
        if window_id is None:
            if self._window_manager._summary_windows:
                self._dirty_start_window_id = self._window_manager._summary_windows[0].window_id
            return
        if self._dirty_start_window_id is None:
            self._dirty_start_window_id = window_id
        else:
            self._dirty_start_window_id = min(self._dirty_start_window_id, window_id)

    def _record_content_type_change(self, summary_window_id: Optional[int], content_type: str, confidence: float) -> None:
        if summary_window_id is None:
            self._latest_content_type = content_type
            self._latest_confidence = confidence
            return

        start_ms = int(self._window_manager.get_window_start(summary_window_id) * 1000)
        end_ms = int(self._window_manager.get_window_end(summary_window_id) * 1000)
        if end_ms < start_ms:
            end_ms = start_ms

        self._content_type_changes.append(
            {
                "summary_window_id": summary_window_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "content_type": content_type,
                "confidence": confidence,
            }
        )
        self._content_type_changes.sort(key=lambda item: (int(item.get("start_ms", 0)), int(item.get("summary_window_id", 0))))

        self._latest_content_type = content_type
        self._latest_confidence = confidence

    def _resolve_content_type_for_window(self, window_start_ms: int) -> str:
        chosen = "UNKNOWN"
        for change in self._content_type_changes:
            if int(change.get("start_ms", 0)) <= window_start_ms:
                chosen = str(change.get("content_type", "UNKNOWN"))
            else:
                break
        if chosen == "UNKNOWN" and self._latest_content_type:
            return self._latest_content_type
        return chosen

    def _render_document(self) -> Dict[str, Any]:
        ordered = sorted(self._formatted_by_window.values(), key=lambda item: int(item.get("start_ms", 0)))
        blocks: List[str] = []
        styles_used: List[str] = []
        previous_type = None

        for item in ordered:
            current_type = str(item.get("content_type", "UNKNOWN"))
            if previous_type is not None and current_type != previous_type:
                blocks.append("---")
            heading = str(item.get("heading") or "Formatted Transcript").strip()
            start_ms = int(item.get("start_ms", 0))
            end_ms = int(item.get("end_ms", 0))
            text = str(item.get("formatted_text") or "").strip()
            blocks.append(f"## {heading} [{self._format_timestamp(start_ms)} – {self._format_timestamp(end_ms)}]\n\n{text}")
            previous_type = current_type
            if current_type not in styles_used:
                styles_used.append(current_type)

        return {
            "formatted_document": "\n\n".join(blocks).strip(),
            "format_styles_used": styles_used,
            "chunk_count": sum(int(item.get("chunk_count", 0)) for item in ordered),
            "window_count": len(ordered),
        }

    @staticmethod
    def _format_timestamp(ms: int) -> str:
        total_seconds = max(0, ms // 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    async def _process_dirty_windows(self) -> None:
        async with self._lock:
            if self._is_processing:
                return
            self._is_processing = True

        try:
            while True:
                start_id = self._dirty_start_window_id
                if start_id is None:
                    break
                self._dirty_start_window_id = None

                windows = sorted(getattr(self._window_manager, "_summary_windows", []), key=lambda w: w.window_id)
                windows = [window for window in windows if window.window_id >= start_id]
                if not windows:
                    continue

                running_input_tokens = 0
                running_output_tokens = 0
                last_window_id = windows[-1].window_id

                for window in windows:
                    raw_text = (window.text or "").strip()
                    if not raw_text:
                        continue

                    content_type = self._resolve_content_type_for_window(int(window.timestamp_start * 1000))

                    prior_tail = ""
                    previous_window_id = window.window_id - 1
                    previous_formatted = self._formatted_by_window.get(previous_window_id)
                    if previous_formatted:
                        prior_tail = str(previous_formatted.get("formatted_text", ""))[-1000:]

                    result = await self._task.format_text(
                        raw_text=raw_text,
                        content_type=content_type,
                        prior_formatted_tail=prior_tail,
                    )

                    running_input_tokens += int(result.get("input_tokens", 0))
                    running_output_tokens += int(result.get("output_tokens", 0))

                    self._formatted_by_window[window.window_id] = {
                        "summary_window_id": window.window_id,
                        "start_ms": int(window.timestamp_start * 1000),
                        "end_ms": int(window.timestamp_end * 1000),
                        "content_type": content_type,
                        "heading": result.get("heading", "Formatted Transcript"),
                        "formatted_text": result.get("formatted_text", ""),
                        "chunk_count": int(result.get("chunk_count", 0)),
                    }

                    self._window_manager.store_plugin_result(
                        window_id=window.window_id,
                        plugin_name="transcript_format",
                        result=self._formatted_by_window[window.window_id],
                        include_in_context=False,
                    )

                rendered = self._render_document()
                payload = {
                    "type": "transcript_format",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "timing": {
                        "summary_window_id": last_window_id,
                    },
                    "llm_usage": {
                        "input_tokens": running_input_tokens,
                        "output_tokens": running_output_tokens,
                    },
                    "formatted_document": rendered["formatted_document"],
                    "content_type_timeline": [
                        {
                            "summary_window_id": int(change.get("summary_window_id", 0)),
                            "from_ms": int(change.get("start_ms", 0)),
                            "to_ms": int(change.get("end_ms", 0)),
                            "content_type": str(change.get("content_type", "UNKNOWN")),
                            "confidence": float(change.get("confidence", 0.0)),
                        }
                        for change in self._content_type_changes
                    ],
                    "chunk_count": rendered["chunk_count"],
                    "window_count": rendered["window_count"],
                    "format_styles_used": rendered["format_styles_used"],
                }

                await self._result_callback(payload)
                await self._send_monitoring_event(
                    {
                        "summary_window_id": last_window_id,
                        "chunk_count": rendered["chunk_count"],
                        "window_count": rendered["window_count"],
                        "timestamp_utc": payload["timestamp_utc"],
                    },
                    "transcript_format_complete",
                )
        finally:
            async with self._lock:
                self._is_processing = False

    async def handle_summary_window(self, summary_window_id: int, **kwargs):
        self._mark_dirty_from(summary_window_id)
        await self._process_dirty_windows()

    async def handle_content_type_detected(
        self,
        content_type: str,
        confidence: float,
        source: str,
        reasoning: str,
        summary_window_id: Optional[int] = None,
        **kwargs,
    ):
        self._record_content_type_change(summary_window_id, str(content_type), float(confidence))
        self._mark_dirty_from(summary_window_id)
        await self._process_dirty_windows()

    def on_update_params(
        self,
        fast_max_tokens: Optional[int] = None,
        fast_temperature: Optional[float] = None,
        transcript_format_chunk_input_tokens: Optional[int] = None,
        **kwargs,
    ):
        if fast_max_tokens is not None:
            self._task.max_tokens = fast_max_tokens
        if fast_temperature is not None:
            self._task.temperature = fast_temperature
        if transcript_format_chunk_input_tokens is not None and transcript_format_chunk_input_tokens > 2000:
            self._task.input_token_budget = min(
                transcript_format_chunk_input_tokens,
                self._task.model_context_tokens - self._task.max_tokens - 1000,
            )

    def reset(self):
        self._content_type_changes.clear()
        self._latest_content_type = "UNKNOWN"
        self._latest_confidence = 0.0
        self._formatted_by_window.clear()
        self._dirty_start_window_id = None
        self._is_processing = False


def init_plugin(
    plugin_name: str,
    window_manager,
    llm_manager,
    result_callback: Callable,
    summary_client=None,
    send_monitoring_event_callback=None,
):
    plugin_instance = TranscriptFormatPlugin(
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
                "on_content_type_detected": plugin_instance.handle_content_type_detected,
                "on_update_params": plugin_instance.on_update_params,
            },
        )


__all__ = ["TranscriptFormatPlugin", "TranscriptFormatTask", "init_plugin"]
