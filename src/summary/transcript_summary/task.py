"""
Transcript summary task implementation.

Maintains a growing, cumulative markdown summary of the transcript by combining
buffered transcription windows with the latest fast summary bullets.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..llm_manager import LLMClient
from .prompts import TRANSCRIPT_SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TranscriptSummarySchema(BaseModel):
    """Schema for the LLM response — full updated overview document plus refreshed lists."""

    summary: str
    key_points: List[str]
    topics: List[str]


class TranscriptSummaryTask:
    """
    Task that maintains a growing markdown summary of the full transcript.

    Called whenever a fast summary becomes available. It combines the buffered
    transcription window text with the new fast summary bullets and the existing
    running summary to produce an updated cumulative document.
    """

    _json_schema: Dict[str, Any] = TranscriptSummarySchema.model_json_schema()

    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens: int = 16384,
        temperature: float = 0.1,
    ) -> None:
        self._llm_client = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _format_timestamp(ms: int) -> str:
        """Format milliseconds as H:MM:SS for use in timing references."""
        total_seconds = ms // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def _build_user_content(
        self,
        window_texts: str,
        fast_summary_items: List[str],
        current_summary: Optional[Dict[str, Any]],
        window_start_ms: int,
        window_end_ms: int,
    ) -> str:
        """Construct the prompt content for this update cycle."""

        segment_timing = (
            f"[{self._format_timestamp(window_start_ms)} – {self._format_timestamp(window_end_ms)}]"
        )

        # ── Section 1: current running overview ─────────────────────────────
        if current_summary and current_summary.get("summary"):
            running_md = current_summary["summary"]
        else:
            running_md = "(No overview yet — this is the first segment.)"

        # ── Section 2: current key points ───────────────────────────────────
        existing_kp = current_summary.get("key_points", []) if current_summary else []
        kp_text = "\n".join(f"- {kp}" for kp in existing_kp) if existing_kp else "(None yet)"

        # ── Section 3: current topics ────────────────────────────────────────
        existing_topics = current_summary.get("topics", []) if current_summary else []
        topics_text = "\n".join(f"- {t}" for t in existing_topics) if existing_topics else "(None yet)"

        # ── Section 4: fast summary bullets ─────────────────────────────────
        bullets = (
            "\n".join(f"- {item}" for item in fast_summary_items)
            if fast_summary_items
            else "(No fast summary notes for this segment.)"
        )

        # ── Section 5: raw transcription text ────────────────────────────────
        raw_text = window_texts.strip() if window_texts else "(No new transcription text.)"

        return (
            "## Current Overview Document\n\n"
            f"{running_md}\n\n"
            "## Current Key Points\n\n"
            f"{kp_text}\n\n"
            "## Current Topics\n\n"
            f"{topics_text}\n\n"
            f"## New Segment  {segment_timing}\n\n"
            "### Fast Summary Notes\n\n"
            f"{bullets}\n\n"
            "### Raw Transcription\n\n"
            f"{raw_text}"
        )

    async def process(
        self,
        window_texts: str,
        fast_summary_items: List[str],
        current_summary: Optional[Dict[str, Any]],
        window_start_ms: int = 0,
        window_end_ms: int = 0,
    ) -> Dict[str, Any]:
        """Update the running transcript overview.

        Args:
            window_texts: Merged text from all buffered transcription windows.
            fast_summary_items: Bullet-point items from the latest rapid summary.
            current_summary: The previous result dict, or None on the first call.
            window_start_ms: Media start time of the new segment in milliseconds.
            window_end_ms: Media end time of the new segment in milliseconds.

        Returns:
            Dict with keys: summary, key_points, topics, input_tokens, output_tokens.
        """
        if not self._llm_client:
            raise RuntimeError("Transcript summary LLM client not initialized")

        user_content = self._build_user_content(
            window_texts,
            fast_summary_items,
            current_summary,
            window_start_ms,
            window_end_ms,
        )

        _, content, input_tokens, output_tokens, _ = await self._llm_client.create_completion(
            system_prompt=TRANSCRIPT_SUMMARY_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=self.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "transcript_summary",
                    "schema": self._json_schema,
                },
            },
        )

        # Preserve prior lists as fallback in case parsing fails
        prior_key_points: List[str] = current_summary.get("key_points", []) if current_summary else []
        prior_topics: List[str] = current_summary.get("topics", []) if current_summary else []

        try:
            json_content = content.replace("```json", "").replace("```", "").strip()
            parsed = TranscriptSummarySchema.model_validate_json(json_content)
        except Exception as exc:
            logger.warning(
                "Failed to parse transcript_summary response: %s — raw content: %.200s",
                exc,
                content,
            )
            return {
                "summary": content,
                "key_points": prior_key_points,
                "topics": prior_topics,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return {
            "summary": parsed.summary,
            "key_points": parsed.key_points,
            "topics": parsed.topics,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


__all__ = ["TranscriptSummaryTask", "TranscriptSummarySchema"]
