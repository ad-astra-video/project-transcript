"""
Transcript summary task implementation.

Maintains a growing, cumulative markdown summary of the transcript by combining
buffered transcription windows with the latest fast summary bullets.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, field_validator, model_validator

from ..llm_manager import LLMClient
from .prompts import TRANSCRIPT_SUMMARY_SYSTEM_PROMPT, build_system_prompt

logger = logging.getLogger(__name__)

class TranscriptSummarySection(BaseModel):
    """Single summary section with explicit media timing."""

    heading: str
    start_ms: int
    end_ms: int
    content: str

    @field_validator("start_ms", "end_ms", mode="before")
    @classmethod
    def _clamp_non_negative(cls, v: Any) -> int:
        """Coerce to int and clamp to >= 0; reject negative LLM outputs."""
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return max(0, v)

    @model_validator(mode="after")
    def _end_gte_start(self) -> "TranscriptSummarySection":
        """Ensure end_ms is never less than start_ms."""
        if self.end_ms < self.start_ms:
            self.end_ms = self.start_ms
        return self

class TranscriptSummarySchema(BaseModel):
    """Schema for the LLM response — sectioned overview plus refreshed lists."""

    sections: List[TranscriptSummarySection]
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
        max_tokens: int = 131072,
        temperature: float = 0.3,
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

    # Matches zero-only timing ranges embedded in key_points/topics strings, e.g.
    # "## key_points [0:00:00 – 0:00:00]" — artefacts produced by the LLM when it
    # has no valid timing context for an item.  En-dash, em-dash, and hyphen-minus
    # are all covered; surrounding whitespace is optional.
    _ZERO_RANGE_RE = re.compile(
        r"\[\s*0:00:00\s*[\u2013\u2014\-]\s*0:00:00\s*\]"
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

    @classmethod
    def _discard_zero_timed_items(cls, items: List[str]) -> List[str]:
        """Discard key_points or topics items that carry a [0:00:00 – 0:00:00] range.

        When the LLM has no valid timing anchor it emits a zero-range timestamp
        (e.g. "## key_points [0:00:00 – 0:00:00]") which shows up in the frontend
        as a heading with meaningless timing.  These are degenerate artefacts and
        are silently dropped here.
        """
        filtered: List[str] = []
        for item in items:
            if cls._ZERO_RANGE_RE.search(item):
                logger.debug(
                    "transcript_summary: discarding zero-timed item: %.120s", item
                )
            else:
                filtered.append(item)
        return filtered

    @staticmethod
    def _compute_caps(latest_end_ms: int) -> Tuple[int, int]:
        """Return (key_points_cap, topics_cap) scaled by 30-minute blocks.

        Baseline caps (0–30 min): key_points=10, topics=8.
        Each additional 30-minute block adds +3 key points and +2 topics.
        Hard caps: key_points=35, topics=24 (covers recordings up to ~9 hours).

        Args:
            latest_end_ms: The furthest window_end_ms across all current segments.

        Returns:
            Tuple of (key_points_cap, topics_cap).
        """
        blocks = max(1, math.ceil(latest_end_ms / 1_800_000))  # 1 block = 30 min
        key_points_cap = min(10 + (blocks - 1) * 3, 35)
        topics_cap = min(8 + (blocks - 1) * 2, 24)
        return key_points_cap, topics_cap

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
        context_window_ms: int,
        max_segment_text_chars: int,
    ) -> str:
        """Construct the prompt content for this update cycle.

        Args:
            segments: List of segment dicts, each containing:
                - fast_summary_items: List[str]
                - window_start_ms: int
                - window_end_ms: int
                - window_text: str  (raw transcription, may be empty)
            current_summary: The previous result dict, or None on the first call.
            context_window_ms: Duration of prior summary context to include.
            max_segment_text_chars: Max transcription chars per new segment.
        """
        # ── Section 1: current running overview sections (windowed) ─────────
        current_sections = current_summary.get("sections", []) if current_summary else []
        latest_end_ms = max((seg.get("window_end_ms", 0) for seg in segments), default=0)
        cutoff_ms = max(0, latest_end_ms - context_window_ms)

        windowed_sections: List[Dict[str, Any]] = [
            section
            for section in current_sections
            if int(section.get("end_ms", 0)) >= cutoff_ms
        ]

        # Always include the most recently completed section even when its end_ms
        # falls before the cutoff.  Without this, the LLM has no continuity anchor
        # for the topic that was actively being discussed at the window boundary —
        # it would create a duplicate section instead of extending the correct one.
        if current_sections:
            most_recent = max(current_sections, key=lambda s: int(s.get("end_ms", 0)))
            most_recent_end = int(most_recent.get("end_ms", 0))
            if most_recent_end < cutoff_ms:
                # Prepend so chronological order (oldest→newest) is preserved.
                windowed_sections = [most_recent] + [
                    s for s in windowed_sections if s is not most_recent
                ]
                logger.debug(
                    "transcript_summary: pinned most-recent section '%s' (end_ms=%d) "
                    "into sliding window (cutoff_ms=%d)",
                    most_recent.get("heading", ""),
                    most_recent_end,
                    cutoff_ms,
                )

        if windowed_sections:
            # Identify the latest (most-recent) section so we can tag it for
            # the LLM — only this section is eligible for compression.
            latest_end = max(
                int(s.get("end_ms", 0)) for s in windowed_sections
            )

            section_blocks = []
            for section in windowed_sections:
                section_start = int(section.get("start_ms", 0))
                section_end = int(section.get("end_ms", 0))
                section_heading = section.get("heading", "Untitled Topic")
                section_content = (section.get("content") or "").strip() or "(No section content.)"

                # Tag the section with the highest end_ms so the LLM knows it
                # may compress/reword this one.
                latest_tag = (
                    " [LATEST — may be compressed]"
                    if section_end == latest_end
                    else ""
                )

                # Include raw ms values alongside the human-readable timestamps so
                # the LLM can extend end_ms by direct copy rather than converting
                # from H:MM:SS (which is lossy and error-prone).
                section_blocks.append(
                    f"### {section_heading}{latest_tag} "
                    f"[{self._format_timestamp(section_start)} – {self._format_timestamp(section_end)}] "
                    f"(start_ms: {section_start}, end_ms: {section_end})\n\n"
                    f"{section_content}"
                )
            running_sections_text = "\n\n".join(section_blocks)
        else:
            running_sections_text = "(No overview sections yet — this is the first segment.)"

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
            # Expose raw ms values so the LLM can copy window_end_ms directly into
            # a section's end_ms field when extending an existing section, or use
            # window_start_ms as start_ms when opening a new section.
            timing = (
                f"[{self._format_timestamp(seg_start)} – {self._format_timestamp(seg_end)}] "
                f"(window_start_ms: {seg_start}, window_end_ms: {seg_end})"
            )

            items = seg.get("fast_summary_items", [])
            bullets = (
                "\n".join(f"- {item}" for item in items)
                if items
                else "(No fast summary notes for this segment.)"
            )

            raw_text = (seg.get("window_text") or "").strip()
            if max_segment_text_chars > 0 and len(raw_text) > max_segment_text_chars:
                raw_text = raw_text[-max_segment_text_chars:]
            raw_text = raw_text or "(No transcription text.)"

            block = (
                f"### Segment {timing}\n\n"
                f"**Key Notes:**\n{bullets}\n\n"
                f"**Transcription:**\n{raw_text}"
            )
            segment_blocks.append(block)

        new_segments_section = "\n\n".join(segment_blocks) if segment_blocks else "(No new segments.)"

        return (
            "## Current Overview Sections (Sliding Window)\n\n"
            f"{running_sections_text}\n\n"
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
        context_window_ms: int,
        max_segment_text_chars: int,
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
            Dict with keys: sections, key_points, topics, input_tokens, output_tokens.
        """
        if not self._llm_client:
            raise RuntimeError("Transcript summary LLM client not initialized")

        latest_end_ms = max((seg.get("window_end_ms", 0) for seg in segments), default=0)
        key_points_cap, topics_cap = self._compute_caps(latest_end_ms)
        logger.debug(
            "transcript_summary: latest_end_ms=%d  key_points_cap=%d  topics_cap=%d",
            latest_end_ms, key_points_cap, topics_cap,
        )

        user_content = self._build_user_content(
            segments=segments,
            current_summary=current_summary,
            context_window_ms=context_window_ms,
            max_segment_text_chars=max_segment_text_chars,
        )

        _, content, input_tokens, output_tokens, _ = await self._llm_client.create_completion(
            system_prompt=build_system_prompt(key_points_cap, topics_cap),
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

        # Preserve prior content as fallback in case parsing fails
        prior_sections: List[Dict[str, Any]] = current_summary.get("sections", []) if current_summary else []
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
            clean_kp     = self._discard_zero_timed_items(clean_kp)
            clean_topics = self._discard_zero_timed_items(clean_topics)
            return {
                "sections": prior_sections,
                "key_points": clean_kp,
                "topics": clean_topics,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        # Strip CJK from section fields and validate section contents for prompt leakage
        cleaned_sections: List[Dict[str, Any]] = []
        _PLACEHOLDER_HEADINGS = {"untitled topic", "untitled", ""}
        for section in parsed.sections:
            section_heading = self._strip_cjk(section.heading).strip()
            section_content = self._strip_cjk(section.content).strip()
            clean_content, _, _ = self._validate_no_prompt_leakage(
                section_content, [], []
            )
            clean_content = clean_content.strip()

            # Discard degenerate sections: placeholder headings with no real content
            # are artefacts of the LLM not having enough context for a proper section.
            heading_is_placeholder = section_heading.lower() in _PLACEHOLDER_HEADINGS
            if heading_is_placeholder and not clean_content:
                logger.debug(
                    "transcript_summary: discarding degenerate section "
                    "(heading=%r, start_ms=%d, end_ms=%d)",
                    section_heading,
                    section.start_ms,
                    section.end_ms,
                )
                continue

            cleaned_sections.append(
                {
                    "heading": section_heading or "Overview",
                    "start_ms": int(section.start_ms),
                    "end_ms": int(section.end_ms),
                    "content": clean_content,
                }
            )

        kp_cjk = [self._strip_cjk(kp).lstrip("- ").strip() for kp in parsed.key_points]
        topics_cjk = [self._strip_cjk(t).lstrip("- ").strip() for t in parsed.topics]

        _, clean_kp, clean_topics = self._validate_no_prompt_leakage(
            "", kp_cjk, topics_cjk
        )
        clean_kp     = self._discard_zero_timed_items(clean_kp)
        clean_topics = self._discard_zero_timed_items(clean_topics)

        return {
            "sections": cleaned_sections,
            "key_points": clean_kp,
            "topics": clean_topics,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


TranscriptSummarySchema.model_rebuild()


__all__ = ["TranscriptSummaryTask", "TranscriptSummarySchema", "TranscriptSummarySection"]
