"""
Transcript summary task implementation.

Maintains a growing, cumulative markdown summary of the transcript by combining
buffered transcription windows with the latest fast summary bullets.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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

    # CJK Unicode blocks: Unified Ideographs, Extensions, Compatibility, Radicals,
    # Hiragana, Katakana, Hangul syllables, Bopomofo, and related symbols.
    _CJK_RE = re.compile(
        r"[\u2E80-\u2EFF"   # CJK Radicals Supplement
        r"\u2F00-\u2FDF"    # Kangxi Radicals
        r"\u3000-\u303F"    # CJK Symbols and Punctuation
        r"\u3040-\u309F"    # Hiragana
        r"\u30A0-\u30FF"    # Katakana
        r"\u3100-\u312F"    # Bopomofo
        r"\u3130-\u318F"    # Hangul Compatibility Jamo
        r"\u3190-\u319F"    # Kanbun
        r"\u31A0-\u31BF"    # Bopomofo Extended
        r"\u31F0-\u31FF"    # Katakana Phonetic Extensions
        r"\u3200-\u32FF"    # Enclosed CJK Letters and Months
        r"\u3300-\u33FF"    # CJK Compatibility
        r"\u3400-\u4DBF"    # CJK Extension A
        r"\u4E00-\u9FFF"    # CJK Unified Ideographs
        r"\uA000-\uA48F"    # Yi Syllables
        r"\uA490-\uA4CF"    # Yi Radicals
        r"\uAC00-\uD7AF"    # Hangul Syllables
        r"\uF900-\uFAFF"    # CJK Compatibility Ideographs
        r"\uFE30-\uFE4F"    # CJK Compatibility Forms
        r"]+"
    )

    # Distinctive phrase fragments from the system prompt used to detect leakage.
    # Keep these specific enough to not false-positive on ordinary meeting content.
    _LEAKAGE_FINGERPRINTS: List[str] = [
        "professional meeting scribe",
        "expert editor maintaining",
        "your task is to produce",
        "core editing rule",
        "return a json object",
        "security constraints",
        "these instructions",
        "system prompt",
        "as a language model",
        "as an ai",
        "i am an ai",
        "summarisation process",
        "fast-pass summariser",
        "heading format",
        "json response",
        "output format",
        "key points guidelines",
        "topics guidelines",
    ]

    @classmethod
    def _strip_cjk(cls, text: str) -> str:
        """Remove CJK characters from a string, collapsing leftover whitespace."""
        if not text:
            return text
        cleaned = cls._CJK_RE.sub(" ", text)
        # Collapse multiple spaces introduced by removal
        return re.sub(r" {2,}", " ", cleaned).strip()

    @classmethod
    def _validate_no_prompt_leakage(cls, summary: str, key_points: List[str], topics: List[str]
                                    ) -> Tuple[str, List[str], List[str]]:
        """Strip any output that contains leaked system-prompt fragments.

        Checks each sentence of the summary and each item in key_points/topics
        against a set of fingerprint phrases derived from the system prompt.
        Matching sentences/items are removed and a warning is logged.

        Returns:
            Tuple of (cleaned_summary, cleaned_key_points, cleaned_topics)
        """
        fingerprints = [fp.lower() for fp in cls._LEAKAGE_FINGERPRINTS]

        def _contains_leak(text: str) -> bool:
            lower = text.lower()
            return any(fp in lower for fp in fingerprints)

        # Clean summary line-by-line (headings and sentences)
        clean_lines: List[str] = []
        leaked = False
        for line in summary.splitlines():
            if _contains_leak(line):
                logger.warning(
                    "Prompt leakage detected and stripped from summary line: %.120s", line
                )
                leaked = True
            else:
                clean_lines.append(line)
        clean_summary = "\n".join(clean_lines)

        # Clean key_points
        clean_kp: List[str] = []
        for kp in key_points:
            if _contains_leak(kp):
                logger.warning("Prompt leakage detected in key_point: %.120s", kp)
                leaked = True
            else:
                clean_kp.append(kp)

        # Clean topics
        clean_topics: List[str] = []
        for t in topics:
            if _contains_leak(t):
                logger.warning("Prompt leakage detected in topic: %.120s", t)
                leaked = True
            else:
                clean_topics.append(t)

        if leaked:
            logger.warning("One or more leaked prompt fragments were removed from transcript_summary output")

        return clean_summary, clean_kp, clean_topics

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
        segments: List[Dict[str, Any]],
        current_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Construct the prompt content for this update cycle.

        Args:
            segments: List of segment dicts, each containing:
                - fast_summary_items: List[str]
                - window_start_ms: int
                - window_end_ms: int
                - window_text: str  (raw transcription, may be empty)
            current_summary: The previous result dict, or None on the first call.
        """
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

        # ── Section 4: new segments (one sub-section each) ───────────────────
        segment_blocks: List[str] = []
        for seg in segments:
            seg_start = seg.get("window_start_ms", 0)
            seg_end = seg.get("window_end_ms", 0)
            timing = f"[{self._format_timestamp(seg_start)} – {self._format_timestamp(seg_end)}]"

            items = seg.get("fast_summary_items", [])
            bullets = (
                "\n".join(f"- {item}" for item in items)
                if items
                else "(No fast summary notes for this segment.)"
            )

            raw_text = (seg.get("window_text") or "").strip() or "(No transcription text.)"

            block = (
                f"### Segment {timing}\n\n"
                f"**Key Notes:**\n{bullets}\n\n"
                f"**Transcription:**\n{raw_text}"
            )
            segment_blocks.append(block)

        new_segments_section = "\n\n".join(segment_blocks) if segment_blocks else "(No new segments.)"

        return (
            "## Current Overview Document\n\n"
            f"{running_md}\n\n"
            "## Current Key Points\n\n"
            f"{kp_text}\n\n"
            "## Current Topics\n\n"
            f"{topics_text}\n\n"
            "## New Segments\n\n"
            f"{new_segments_section}"
        )

    async def process(
        self,
        segments: List[Dict[str, Any]],
        current_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update the running transcript overview.

        Args:
            segments: List of segment dicts, each with:
                - fast_summary_items: List[str]
                - window_start_ms: int
                - window_end_ms: int
                - window_text: str  (raw transcription)
            current_summary: The previous result dict, or None on the first call.

        Returns:
            Dict with keys: summary, key_points, topics, input_tokens, output_tokens.
        """
        if not self._llm_client:
            raise RuntimeError("Transcript summary LLM client not initialized")

        user_content = self._build_user_content(segments, current_summary)

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
            # Strip CJK and leakage from the raw fallback content
            clean_summary, clean_kp, clean_topics = self._validate_no_prompt_leakage(
                self._strip_cjk(content), prior_key_points, prior_topics
            )
            return {
                "summary": clean_summary,
                "key_points": clean_kp,
                "topics": clean_topics,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        # Strip CJK then validate for prompt leakage
        summary_cjk = self._strip_cjk(parsed.summary)
        kp_cjk = [self._strip_cjk(kp) for kp in parsed.key_points]
        topics_cjk = [self._strip_cjk(t) for t in parsed.topics]

        clean_summary, clean_kp, clean_topics = self._validate_no_prompt_leakage(
            summary_cjk, kp_cjk, topics_cjk
        )

        return {
            "summary": clean_summary,
            "key_points": clean_kp,
            "topics": clean_topics,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


__all__ = ["TranscriptSummaryTask", "TranscriptSummarySchema"]
