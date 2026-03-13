"""
Actions extraction plugin.

Subscribes to `fast_summary_available` and uses the fast LLM to extract
concrete future actions and follow-up items from the rapid-summary bullet
points produced for each summary window.

Extraction criteria are adapted to the current content type (received via
`on_content_type_detected`) so that the model is appropriately strict: very
picky for lectures/podcasts/news, and explicitly-commitment-based for meetings.

Accumulated actions and follow-ups across the entire session are made
available to other plugins (notably `transcript_summary`) via the
`on_actions_extracted` event and via plugin result storage.

No raw transcript text is passed to the LLM — only the already-distilled
bullet points from rapid_summary — keeping latency minimal.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .task import ActionsExtractionTask

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize an item string for deduplication comparisons."""
    return text.strip().lower().rstrip(".")


class ActionsExtractionPlugin:
    """Extracts and accumulates actions / follow-ups across the session."""

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

        # Current content type — updated via on_content_type_detected
        self._content_type: str = "UNKNOWN"

        # Session-level accumulators
        self._accumulated_actions: List[str] = []
        self._accumulated_follow_ups: List[str] = []

        # Normalised sets for O(1) deduplication
        self._seen_actions_norm: set = set()
        self._seen_follow_ups_norm: set = set()

        self._task = ActionsExtractionTask(
            llm_client=self._llm.reasoning_llm_client,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_content_type_detected(
        self,
        content_type: str,
        confidence: float = 0.0,
        source: str = "AUTO_DETECTED",
        reasoning: str = "",
        **kwargs,
    ) -> None:
        """Update extraction criteria when the content type changes.

        Subscribed to ``on_content_type_detected``.
        """
        previous = self._content_type
        self._content_type = content_type or "UNKNOWN"
        if previous != self._content_type:
            logger.info(
                "actions_extraction: content type changed %s → %s (confidence=%.2f)",
                previous,
                self._content_type,
                confidence,
            )

    async def process(
        self,
        summary_window_id: int,
        fast_summary_items: List[str],
        **kwargs,
    ) -> None:
        """Extract actions/follow-ups from incoming rapid-summary bullet points.

        Subscribed to ``fast_summary_available``.
        """
        if not self._llm.reasoning_client:
            return

        if not fast_summary_items:
            return

        try:
            result = await self._task.process(
                fast_summary_items=fast_summary_items,
                content_type=self._content_type,
            )
        except Exception as exc:
            logger.error("actions_extraction: LLM call failed: %s", exc, exc_info=True)
            return

        new_actions = self._deduplicate(
            result.get("actions", []),
            self._seen_actions_norm,
            self._accumulated_actions,
        )
        new_follow_ups = self._deduplicate(
            result.get("follow_ups", []),
            self._seen_follow_ups_norm,
            self._accumulated_follow_ups,
        )

        if not new_actions and not new_follow_ups:
            logger.debug(
                "actions_extraction: no new items for window %d (content_type=%s)",
                summary_window_id,
                self._content_type,
            )
            return

        logger.info(
            "actions_extraction: window=%d content_type=%s new_actions=%d new_follow_ups=%d "
            "total_actions=%d total_follow_ups=%d",
            summary_window_id,
            self._content_type,
            len(new_actions),
            len(new_follow_ups),
            len(self._accumulated_actions),
            len(self._accumulated_follow_ups),
        )

        # Store snapshot in window for potential future context use
        self._window_manager.store_plugin_result(
            window_id=summary_window_id,
            plugin_name="actions_extraction",
            result={
                "actions": list(self._accumulated_actions),
                "follow_ups": list(self._accumulated_follow_ups),
            },
            include_in_context=False,
        )

        # Emit result payload to the client
        payload = {
            "type": "actions_extraction",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timing": {"summary_window_id": summary_window_id},
            "llm_usage": {
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            },
            "content_type": self._content_type,
            "actions": list(self._accumulated_actions),
            "follow_ups": list(self._accumulated_follow_ups),
        }
        await self._result_callback(payload)

        # Notify downstream plugins (e.g. transcript_summary) with full accumulated lists
        if self._summary_client:
            await self._summary_client._notify_plugins(
                "on_actions_extracted",
                summary_window_id=summary_window_id,
                actions=list(self._accumulated_actions),
                follow_ups=list(self._accumulated_follow_ups),
            )

    def on_update_params(
        self,
        reasoning_max_tokens: Optional[int] = None,
        reasoning_temperature: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Handle on_update_params event."""
        if reasoning_max_tokens is not None:
            self._task.max_tokens = reasoning_max_tokens
        if reasoning_temperature is not None:
            self._task.temperature = reasoning_temperature

    def reset(self) -> None:
        """Clear all session state for a new stream."""
        self._content_type = "UNKNOWN"
        self._accumulated_actions.clear()
        self._accumulated_follow_ups.clear()
        self._seen_actions_norm.clear()
        self._seen_follow_ups_norm.clear()
        logger.debug("ActionsExtractionPlugin reset")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        incoming: List[str],
        seen_norm: set,
        accumulated: List[str],
    ) -> List[str]:
        """Add unique items from *incoming* into *accumulated* in-place.

        Returns the list of newly added items.
        """
        added: List[str] = []
        for item in incoming:
            item = item.strip()
            if not item:
                continue
            norm = _normalize(item)
            if norm not in seen_norm:
                seen_norm.add(norm)
                accumulated.append(item)
                added.append(item)
        return added


def init_plugin(
    plugin_name: str,
    window_manager,
    llm_manager,
    result_callback: Callable,
    summary_client=None,
    send_monitoring_event_callback=None,
) -> None:
    """Auto-discover entry point — called by SummaryClient._discover_plugins()."""
    plugin_instance = ActionsExtractionPlugin(
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
                "fast_summary_available": plugin_instance.process,
                "on_content_type_detected": plugin_instance.handle_content_type_detected,
                "on_update_params": plugin_instance.on_update_params,
            },
        )


__all__ = ["ActionsExtractionPlugin", "ActionsExtractionTask", "init_plugin"]
