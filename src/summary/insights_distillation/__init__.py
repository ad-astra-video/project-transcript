"""Insights distillation plugin.

Waits for BOTH context_summary and transcript_summary completion events for
same summary window ID, then distills deeper insights.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .task import InsightsDistillationTask

logger = logging.getLogger(__name__)


class InsightsDistillationPlugin:
    """Plugin that distills deeper insights after both summaries complete."""

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

        self._latest_insights: List[str] = []
        self._accumulated_detailed_insights: List[Dict[str, Any]] = []
        self._last_context_summary_result: Optional[Dict[str, Any]] = None
        self._last_transcript_summary_result: Optional[Dict[str, Any]] = None
        self._pending_by_window: Dict[int, Set[str]] = {}
        self._processing_windows: Set[int] = set()
        self._lock = asyncio.Lock()

        self._task = InsightsDistillationTask(llm_client=self._llm.reasoning_llm_client)

    def get_latest_insights(self) -> List[str]:
        """Return latest distilled insights for transcript payload attachment."""
        return list(self._latest_insights)

    async def handle_context_summary_complete(self, summary_window_id: int, **kwargs):
        """Mark context summary completion for a window and maybe process."""
        await self._mark_and_maybe_process(summary_window_id, "context_summary")

    async def handle_transcript_summary_complete(self, summary_window_id: int, **kwargs):
        """Mark transcript summary completion for a window and maybe process."""
        await self._mark_and_maybe_process(summary_window_id, "transcript_summary")

    async def _mark_and_maybe_process(self, summary_window_id: int, marker: str):
        should_process = False

        async with self._lock:
            markers = self._pending_by_window.setdefault(summary_window_id, set())
            markers.add(marker)

            if (
                "context_summary" in markers
                and "transcript_summary" in markers
                and summary_window_id not in self._processing_windows
            ):
                self._processing_windows.add(summary_window_id)
                should_process = True

        if not should_process:
            return

        try:
            await self._process_window(summary_window_id)
        finally:
            async with self._lock:
                self._processing_windows.discard(summary_window_id)
                self._pending_by_window.pop(summary_window_id, None)

    async def _process_window(self, summary_window_id: int):
        window = self._window_manager.get_window(summary_window_id)
        if not window:
            logger.warning(
                "insights_distillation: missing window for summary_window_id=%s",
                summary_window_id,
            )
            return

        context_summary_result = window.get_result("context_summary")
        transcript_summary_result = window.get_result("transcript_summary")

        if not context_summary_result or not transcript_summary_result:
            logger.info(
                "insights_distillation: waiting data unavailable for summary_window_id=%s",
                summary_window_id,
            )
            return

        try:
            result = await self._task.process(
                context_summary_result=context_summary_result,
                transcript_summary_result=transcript_summary_result,
                previous_context_summary_result=self._last_context_summary_result,
                previous_transcript_summary_result=self._last_transcript_summary_result,
                prior_insights=list(self._accumulated_detailed_insights),
            )
        except Exception as e:
            logger.error(
                "insights_distillation: error processing window %s: %s",
                summary_window_id,
                e,
            )
            return

        self._latest_insights = result.get("insights", [])
        self._accumulated_detailed_insights = result.get("detailed_insights", [])
        self._last_context_summary_result = context_summary_result
        self._last_transcript_summary_result = transcript_summary_result

        self._window_manager.store_plugin_result(
            window_id=summary_window_id,
            plugin_name="insights_distillation",
            result=result,
            include_in_context=False,
        )

        payload = {
            "type": "insights_distillation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timing": {
                "summary_window_id": summary_window_id,
            },
            "llm_usage": {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            },
            "insights": result.get("insights", []),
            "detailed_insights": result.get("detailed_insights", []),
        }
        await self._result_callback(payload)

        if self._summary_client:
            await self._summary_client._notify_plugins(
                "on_insights_distillation_complete",
                summary_window_id=summary_window_id,
            )

    def on_update_params(
        self,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_temperature: Optional[float] = None,
        **kwargs,
    ):
        """Handle dynamic parameter updates."""
        if reasoning_max_tokens is not None:
            self._task.max_tokens = reasoning_max_tokens
            logger.info("insights_distillation: updated max_tokens to %d", reasoning_max_tokens)

        if reasoning_temperature is not None:
            self._task.temperature = reasoning_temperature
            logger.info("insights_distillation: updated temperature to %s", reasoning_temperature)

    def reset(self):
        """Reset tracked state for new stream."""
        self._latest_insights = []
        self._accumulated_detailed_insights = []
        self._last_context_summary_result = None
        self._last_transcript_summary_result = None
        self._pending_by_window.clear()
        self._processing_windows.clear()


def init_plugin(
    plugin_name: str,
    window_manager,
    llm_manager,
    result_callback: Callable,
    summary_client=None,
    send_monitoring_event_callback=None,
):
    """Auto-discover entry point."""
    plugin_instance = InsightsDistillationPlugin(
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
                "on_context_summary_complete": plugin_instance.handle_context_summary_complete,
                "on_transcript_summary_complete": plugin_instance.handle_transcript_summary_complete,
                "on_update_params": plugin_instance.on_update_params,
            },
        )


__all__ = ["InsightsDistillationPlugin", "InsightsDistillationTask", "init_plugin"]
