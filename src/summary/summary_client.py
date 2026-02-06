"""
Summary client for LLM-based transcription cleaning and summarization.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from enum import Enum

from .prompts import CONTENT_TYPE_DETECTION_PROMPT, CONTENT_TYPE_RULE_MODIFIERS, SYSTEM_PROMPT, SYSTEM_PROMPT_OUTPUT_CONSTRAINTS

logger = logging.getLogger(__name__)


@dataclass
class SummarySegment:
    """Represents a summarized/cleaned segment."""
    summary_type: str
    background_context: str
    summary: str
    timestamp_start: float
    timestamp_end: float

@dataclass
class WindowInsight:
    """Insight extracted from a summary window."""
    insight_type: str
    insight_text: str
    window_id: int
    timestamp_start: float
    timestamp_end: float
    classification: str = "~"

class InsightType(str, Enum):
    """Enumeration of possible insight types."""
    ACTION = "ACTION"
    DECISION = "DECISION"
    QUESTION = "QUESTION"
    KEY_POINT = "KEY POINT"
    RISK = "RISK"
    SENTIMENT = "SENTIMENT"
    NOTES = "NOTES"

class ClassificationField(str, Enum):
    """Classification markers for insights - general and reusable across all insight types."""
    POSITIVE = "+"
    NEUTRAL = "~"
    NEGATIVE = "-"

class MessageFormatMode(str, Enum):
    """Message format modes for different LLM providers."""
    SYSTEM_PROMPT = "system"  # Use system role for system prompt
    USER_PREFIX = "user"      # Convert system prompt to user message with prefix


class ContentType(str, Enum):
    """Enumeration of supported content types."""
    GENERAL_MEETING = "GENERAL_MEETING"
    TECHNICAL_TALK = "TECHNICAL_TALK"
    LECTURE_OR_TALK = "LECTURE_OR_TALK"
    INTERVIEW = "INTERVIEW"
    PODCAST = "PODCAST"
    STREAMER_MONOLOGUE = "STREAMER_MONOLOGUE"
    NEWS_UPDATE = "NEWS_UPDATE"
    GAMEPLAY_COMMENTARY = "GAMEPLAY_COMMENTARY"
    CUSTOMER_SUPPORT = "CUSTOMER_SUPPORT"
    DEBATE = "DEBATE"
    UNKNOWN = "UNKNOWN"


class ContentTypeSource(str, Enum):
    """Source of content type."""
    USER_OVERRIDE = "USER_OVERRIDE"
    AUTO_DETECTED = "AUTO_DETECTED"
    INITIAL = "INITIAL"


@dataclass
class ContentTypeState:
    """State for content type tracking."""
    content_type: str = ContentType.UNKNOWN.value
    confidence: float = 0.0
    source: str = ContentTypeSource.INITIAL.value
    last_detection_text: str = ""  # Last N chars used for detection
    context_length: int = 2000  # Current context length for detection

class ContentTypeDetectionSchema(BaseModel):
    """Schema for content type detection response."""
    content_type: ContentType
    confidence: float
    reasoning: str

class InsightResponseItemSchema(BaseModel):
    """Schema for a single insight item."""
    insight_type: InsightType
    insight_text: str
    confidence: float
    classification: ClassificationField


class InsightsResponseSchema(BaseModel):
    """Schema for insights response from LLM."""
    analysis: str
    insights: List[InsightResponseItemSchema]

@dataclass
class SummaryWindow:
    """A 5-second summary window with text and insights."""
    window_id: int
    text: str  # Non-overlapping text for this window
    insights: List[WindowInsight]
    timestamp_start: float
    timestamp_end: float
    char_count: int  # Length of text for context limit calculation
    processed: bool = False  # Track if window has been processed by LLM

class WindowManager:
    """Manages summary windows and their text/insights."""
    
    def __init__(self, max_chars: int = 5000, windows_to_accumulate: int = 2):
        self._windows: List[SummaryWindow] = []  # Ordered oldest -> newest
        self._char_count: int = 0
        self._next_window_id: int = 0
        self.max_chars = max_chars
        self.windows_to_accumulate = windows_to_accumulate  # Number of windows to exclude from accumulated text
        self._first_window_timestamp: Optional[float] = None  # Track first window timestamp for self-contained delay logic
    
    def add_window(self, text: str, timestamp_start: float, timestamp_end: float) -> int:
        """
        Add a new window, dropping oldest if over char limit.
        Also tracks first window timestamp for self-contained delay logic.
        
        Args:
            text: Text content for this window
            timestamp_start: Start timestamp in seconds
            timestamp_end: End timestamp in seconds
        
        Returns:
            window_id of the added window
        """
        window_id = self._next_window_id
        
        # Track first window timestamp for initial delay (self-contained logic)
        if self._first_window_timestamp is None:
            self._first_window_timestamp = timestamp_start
            logger.info(f"First transcript at {timestamp_start:.3f}s - will delay first summary by {self.initial_summary_delay_seconds}s")
        
        # Check if adding would exceed limit
        new_char_count = self._char_count + len(text)
        
        # Drop oldest windows until under limit
        while new_char_count > self.max_chars and self._windows:
            oldest = self._windows.pop(0)
            self._char_count -= oldest.char_count
            new_char_count = self._char_count + len(text)
        
        # Create and add window
        window = SummaryWindow(
            window_id=window_id,
            text=text,
            insights=[],
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            char_count=len(text)
        )
        self._windows.append(window)
        self._char_count += len(text)
        self._next_window_id += 1

        logger.debug(f"Added window {window_id}, char_count={self._char_count}, total_windows={len(self._windows)}")

        return window_id
    
    def get_accumulated_text(self) -> str:
        """
        Get accumulated window text concatenated, excluding the last N windows.
        
        The excluded windows are those that will be processed in the next summary call,
        so this returns only the historical context that should be used for understanding
        references in the current window.
        """
        if len(self._windows) <= self.windows_to_accumulate:
            return ""
        return " ".join(w.text for w in self._windows[:-self.windows_to_accumulate])
    
    def get_all_windows_text(self) -> str:
        """
        Get text from all windows, ordered from most recent to oldest.
        
        Returns:
            Concatenated text from all windows, newest to oldest
        """
        if not self._windows:
            return ""
        
        # Get text from all windows, newest first
        all_text_parts = [w.text for w in reversed(self._windows)]
        return " ".join(all_text_parts)
    
    def get_window_insights(self, window_id: int) -> List[WindowInsight]:
        """Get insights for a specific window."""
        for window in self._windows:
            if window.window_id == window_id:
                return window.insights
        return []
    
    def add_insight_to_window(self, window_id: int, insight: WindowInsight):
        """Add an insight to a specific window."""
        for window in self._windows:
            if window.window_id == window_id:
                window.insights.append(insight)
                return
    
    def drop_window(self, window_id: int):
        """Drop a window and its insights."""
        for i, window in enumerate(self._windows):
            if window.window_id == window_id:
                self._char_count -= window.char_count
                self._windows.pop(i)
                return
    
    def get_all_insights(self) -> List[WindowInsight]:
        """Get all insights from all windows."""
        all_insights = []
        for window in self._windows:
            all_insights.extend(window.insights)
        return all_insights
    
    def clear(self):
        """Clear all windows."""
        self._windows.clear()
        self._char_count = 0
    
    def __len__(self):
        return len(self._windows)
    
    def get_unprocessed_text(self) -> str:
        """
        Get text from all unprocessed windows, concatenated.
        Used when waiting for initial delay - all windows are unprocessed.
        
        Returns:
            Concatenated text from all unprocessed windows
        """
        if not self._windows:
            return ""
        
        # Get text from all unprocessed windows
        unprocessed_text_parts = []
        for window in self._windows:
            if not window.processed:
                unprocessed_text_parts.append(window.text)
        
        return " ".join(unprocessed_text_parts)
    
    def mark_all_windows_processed(self):
        """Mark all current windows as processed. Called after first summary."""
        for window in self._windows:
            window.processed = True


class SummaryClient:
    """Client for LLM-based transcription cleaning and summarization."""
    
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://byoc-transcription-vllm:5000/v1",
        history_length: int = 0,
        model: str = "Nanbeige/Nanbeige4-3B-Thinking-2511",
        max_tokens: int = 3072,
        temperature: float = 0.1,
        system_prompt: str = SYSTEM_PROMPT,
        windows_to_accumulate: int = 2,
        initial_summary_delay_seconds: float = 30.0
    ):
        """
        Initialize the summary client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the OpenAI-compatible API
            history_length: Number of previous segments to include in context (0 = all history)
            model: Model name to use for summarization
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: System prompt for the LLM
            windows_to_accumulate: Number of windows to exclude from accumulated text (default: 2)
            initial_summary_delay_seconds: Seconds to wait before first summary (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.history_length = history_length
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.insights_response_json_schema = InsightsResponseSchema.model_json_schema()
        self.content_type_response_json_schema = ContentTypeDetectionSchema.model_json_schema()
        
        # Load message format mode from environment, fallback to default
        env_value = os.getenv("LOCAL_SUMMARY_MODEL_USES_SYSTEM_PROMPT", "yes").lower()
        if env_value in ["no"]:
            self.message_format_mode: MessageFormatMode = MessageFormatMode.USER_PREFIX
            logger.info("Using USER_PREFIX message format for summaries")
        else:
            self.message_format_mode: MessageFormatMode = MessageFormatMode.SYSTEM_PROMPT
            logger.info("Using SYSTEM_PROMPT message format for summaries")

        # Initialize OpenAI async client
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        # Window-based state management with configurable accumulation
        self._window_manager: WindowManager = WindowManager(windows_to_accumulate=windows_to_accumulate)
        
        # Track last processed timestamp per window
        self._window_last_timestamp: Dict[int, float] = {}
        
        # Track whether we've performed the first summary call
        self._has_performed_summary: bool = False
        
        # Lock for thread safety (though we're single-threaded in async)
        self._lock: Optional[asyncio.Lock] = None
        
        # Concurrency limiter for summary calls (set via env var)
        try:
            max_concurrent = int(os.getenv("MAX_CONCURRENT_SUMMARIES", "15"))
        except Exception:
            max_concurrent = 15
        self.max_concurrent_summaries: int = max(1, max_concurrent)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrent_summaries)
        
        # In-flight tracking for graceful shutdown
        self.in_flight_windows: set[int] = set()  # Track window IDs being processed
        
        # Skipped segments buffer for merging with next window
        self._skipped_segments_buffer: Optional[Dict[str, Any]] = None  # Store skipped segments for merging
        
        # Content type detection state
        self._content_type_state: ContentTypeState = ContentTypeState()
        self._user_content_type_override: Optional[str] = None
        
        # Store last raw LLM responses for debugging
        self._last_summary_raw_response: Optional[str] = None
        
        # Window tracking state (moved from TranscriberState for co-location)
        self.summary_window_counter: int = 0
        self.summary_window_count: int = 0
        self.summary_skip_every_n: int = 2
        
        # Initial summary delay configuration
        self.initial_summary_delay_seconds: float = initial_summary_delay_seconds
    
    async def initialize(self):
        """
        Initialize the lock for async operations.
        
        With the OpenAI client, we no longer need to fetch models from /models endpoint.
        """
        logger.info("SummaryClient.initialize called")
        
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        logger.info(f"SummaryClient initialized with model: {self.model}")
    
    async def startup_summary(self) -> bool:
        """
        Send a startup summary request with only the system prompt.
        This serves as a warm-up request to check model availability.
        
        Returns:
            True if the request succeeded, False otherwise
        """
        logger.info("SummaryClient.startup_summary - sending warm-up request with system prompt only")
        
        async def _do_startup_request() -> bool:
            user_content = "This is a startup warm-up request."
            messages = self._build_messages(
                system_prompt=self.system_prompt,
                user_content=user_content
            )
            
            try:
                logger.info(f"Sending startup warm-up request to model: {self.model}")
                
                # Send minimal request with just system prompt
                response: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,  # Minimal response, just to verify model is responsive
                    temperature=0.0,   # Deterministic response
                )
                
                if response.choices and len(response.choices) > 0:
                    logger.info(f"Startup warm-up successful, model responded")
                    return True
                else:
                    logger.warning(f"Startup warm-up received empty response")
                    return False
                    
            except Exception as e:
                logger.error(f"Startup warm-up failed: {e}")
                return False
        
        semaphore = getattr(self, "_semaphore", None)
        if semaphore is not None:
            async with semaphore:
                return await _do_startup_request()
        else:
            return await _do_startup_request()
    
    def update_windows_to_accumulate(self, value: int):
        """
        Update the number of windows to accumulate for context.
        
        Args:
            value: Number of windows to accumulate (minimum 1)
        """
        self._window_manager.windows_to_accumulate = max(1, value)
        logger.info(f"Updated windows_to_accumulate to {self._window_manager.windows_to_accumulate}")
    
    def update_summary_skip_every_n(self, value: int):
        """
        Update the summary skip rate.
        
        Args:
            value: Number of windows to skip between processing (minimum 1)
        """
        self.summary_skip_every_n = max(1, value)
        logger.info(f"Updated summary_skip_every_n to {self.summary_skip_every_n}")
    
    def get_all_windows_text(self) -> str:
        """
        Get text from all windows, ordered from most recent to oldest.
        
        Returns:
            Concatenated text from all windows, newest to oldest
        """
        return self._window_manager.get_all_windows_text()
    
    def update_params(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        history_length: Optional[int] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        message_format_mode: Optional[MessageFormatMode] = None,
        windows_to_accumulate: Optional[int] = None,
        initial_summary_delay_seconds: Optional[float] = None
    ):
        """
        Update client parameters dynamically.
        
        Args:
            base_url: New base URL for the API
            api_key: New API key
            history_length: New history length
            model: New model name
            max_tokens: New max tokens
            temperature: New temperature
            system_prompt: New system prompt
            message_format_mode: New message format mode
            windows_to_accumulate: New number of windows to accumulate for context
            initial_summary_delay_seconds: New delay before first summary (default: 30.0)
        """
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        if api_key is not None:
            self.api_key = api_key
        if history_length is not None:
            self.history_length = history_length
        # Allow setting model to None to re-fetch from /models
        if model is not None:
            self.model = model
        elif model is None:
            # Explicitly allow setting model to None to re-fetch
            self.model = None
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if message_format_mode is not None:
            self.message_format_mode = message_format_mode
        if windows_to_accumulate is not None:
            self.update_windows_to_accumulate(windows_to_accumulate)
        if initial_summary_delay_seconds is not None:
            self.initial_summary_delay_seconds = initial_summary_delay_seconds
            logger.info(f"Updated initial_summary_delay_seconds to {initial_summary_delay_seconds}")
        
        logger.info(f"SummaryClient params updated: base_url={self.base_url}, history_length={self.history_length}, model={self.model}")
    
    def get_new_text_for_window(
        self,
        segments: List[Dict[str, Any]],
        window_id: int
    ) -> str:
        """
        Get non-overlapping text from segments for a specific window.
        
        Args:
            segments: List of segment dictionaries with 'words' containing word timestamps
            window_id: The window we're processing text for
        
        Returns:
            Non-overlapping text for this window
        """
        last_ts = self._window_last_timestamp.get(window_id, 0)
        new_text_parts = []
        
        for segment in segments:
            seg_start = segment.get("start_ms", segment.get("start", 0))
            seg_end = segment.get("end_ms", segment.get("end", 0))
            
            if seg_end <= last_ts:
                # Segment entirely before last processed - skip
                continue
            
            if seg_start > last_ts:
                # Segment entirely after last processed - include all
                text = segment.get("text", "")
                if text:
                    new_text_parts.append(text)
            else:
                # Segment overlaps - include only the new portion
                text = segment.get("text", "")
                if text:
                    words = text.split()
                    # Calculate how many words to skip based on timestamp overlap
                    seg_duration = seg_end - seg_start
                    if seg_duration > 0:
                        overlap_ratio = (last_ts - seg_start) / seg_duration
                        skip_count = int(len(words) * overlap_ratio)
                        new_words = words[skip_count:]
                        if new_words:
                            new_text_parts.append(" ".join(new_words))
        
        # Update last timestamp for this window
        if segments:
            last_seg_end = segments[-1].get("end_ms", segments[-1].get("end", 0))
            self._window_last_timestamp[window_id] = last_seg_end
        
        return " ".join(new_text_parts)
    
    def _build_context(self) -> str:
        """Build context string from WindowManager accumulated text."""
        accumulated_text = self._window_manager.get_accumulated_text()
        if accumulated_text:
            return f"{accumulated_text}"
        return ""
    
    def _format_content_type_rules(self, content_type: str) -> str:
        """
        Format content type rules as prompt text for injection into system prompt.
        
        Args:
            content_type: The active content type string
            
        Returns:
            Formatted rules string for inclusion in system prompt
        """
        if content_type not in CONTENT_TYPE_RULE_MODIFIERS:
            return ""
        
        rules = CONTENT_TYPE_RULE_MODIFIERS[content_type]
        
        # Format the rules into a clear, actionable prompt section
        formatted = f"""

## CONTENT TYPE RULES: {content_type}

### Priority Insight Types (Emphasize):
{chr(10).join(f'- {t}' for t in rules["emphasize"])}

### Suppressed Insight Types (Deemphasize):
{chr(10).join(f'- {t}' for t in rules["deemphasize"])}

### Processing Guidelines:
- Sentiment Tracking: {"ENABLED" if rules["sentiment_enabled"] else "DISABLED"}
- Action Strictness: {rules["action_strictness"].upper()}
- Notes Frequency: {rules["notes_frequency"].upper()}

### Application Rules:
- Apply these modifiers strictly to guide extraction focus
- Do NOT compensate by inventing other insight types
- Prefer omission over speculation when rules suppress insight types
"""
        return formatted
    
    def _build_messages(
        self,
        system_prompt: str,
        context: str = "",
        user_content: str = ""
    ) -> List[Dict[str, str]]:
        """
        Build messages list with configurable format.
        
        Args:
            system_prompt: The system prompt text
            context: Optional context from previous segments
            user_content: The user message content
            
        Returns:
            List of message dictionaries in the correct format
        """
        messages = []
        
        # Get active content type for dynamic rule injection
        content_type, confidence, source = self.get_effective_content_type()
        logger.debug(f"Building messages with content_type={content_type}, confidence={confidence}, source={source}")
        
        # Format content type rules for this content type
        content_type_rules = self._format_content_type_rules(content_type)
        
        if self.message_format_mode == MessageFormatMode.SYSTEM_PROMPT:
            # Build system prompt with dynamic content type rules
            combined_system = system_prompt
            
            # Add content type rules after system prompt, before output constraints
            if content_type_rules:
                combined_system += content_type_rules
            
            # Add output constraints ALWAYS before PRIOR CONTEXT
            combined_system += f"\n{SYSTEM_PROMPT_OUTPUT_CONSTRAINTS}"
            
            if context:
                combined_system += f"\n\n## PRIOR CONTEXT\nThe following context is from previous transcript windows. Use it for understanding references only. Do not extract new insights from this context unless the current window transcript provides new information that builds on or contradicts it.\n\n{context}"
            
            combined_system += "\n\nAnalyze the following current window transcript text and report only new or changed insights since the previous context.:"

            messages.append({"role": "system", "content": combined_system})
            messages.append({"role": "user", "content": user_content})
        
        else:  # USER_PREFIX mode
            # Build combined content with dynamic rules
            combined_content = f"[SYSTEM PROMPT]\n{system_prompt}"
            
            # Add content type rules
            if content_type_rules:
                combined_content += content_type_rules
            
            # Add output constraints ALWAYS
            combined_content += f"\n{SYSTEM_PROMPT_OUTPUT_CONSTRAINTS}"
            
            combined_content += f"\n\n[USER CONTENT]\n{user_content}"
            
            if context:
                combined_content = f"[SYSTEM PROMPT]\n{system_prompt}"
                
                if content_type_rules:
                    combined_content += content_type_rules
                
                # Output constraints always added
                combined_content += f"\n{SYSTEM_PROMPT_OUTPUT_CONSTRAINTS}"
                combined_content += f"\n\n[CONTEXT]\n{context}\n\n[USER CONTENT]\n{user_content}"
            
            messages.append({"role": "user", "content": combined_content})
        
        return messages
    
    async def summarize_text(self, text: str, context: str = "") -> tuple[str, str]:
        """
        Send text to LLM for cleaning and summarization.
        
        Args:
            text: Text to summarize/clean
            context: Additional context from previous segments
            
        Returns:
            Tuple of (summary_text, reasoning_content)
            - summary_text: The JSON response content from LLM
            - reasoning_content: The reasoning content from LLM (may be empty)
        """
        logger.debug(f"summarize_text called with text length={len(text)}, context length={len(context)}")
        
        async def _do_request() -> tuple[str, str]:
            messages = self._build_messages(
                system_prompt=self.system_prompt,
                context=context,
                user_content=text
            )
            
            try:
                logger.debug(f"Sending to LLM for analysis")
                
                # Use OpenAI client directly
                response: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_schema", "json_schema": {"name": "insights", "schema": self.insights_response_json_schema}},
                )
                
                # Extract summary text and reasoning content from response
                if response.choices and len(response.choices) > 0:
                    summary_text = response.choices[0].message.content or ""
                    summary_text_raw = summary_text  # Store raw response for logging
                    summary_text = summary_text.replace("```json", "").replace("```", "").strip()

                    reasoning_content = response.choices[0].message.reasoning or ""

                    logger.info(f"summarize_text received response, length={len(summary_text)}")
                    
                    # Store raw response for potential logging when no insights found
                    self._last_summary_raw_response = summary_text_raw
                    
                    return summary_text.strip(), reasoning_content.strip()
                
                logger.info("summarize_text received empty response")
                self._last_summary_raw_response = None
                return "", ""
                
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                raise

        semaphore = getattr(self, "_semaphore", None)
        if semaphore is not None:
            async with semaphore:
                return await _do_request()
        else:
            return await _do_request()
    
    async def process_segments(
        self,
        summary_type: str,
        segments: List[Dict[str, Any]],
        window_id: int,
        window_start: float,
        window_end: float
    ) -> List[SummarySegment]:
        """
        Process transcription segments for a 5-second summary window.
        
        Args:
            segments: List of transcription segments
            window_id: Unique identifier for this summary window
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
        
        Returns:
            List of SummarySegment objects with actionable summary information
        """
        logger.info(f"SummaryClient.process_segments called with {len(segments)} segments for window {window_id}")
        
        # Get non-overlapping text for this window
        new_text = self.get_new_text_for_window(segments, window_id)
        
        if not new_text:
            logger.info(f"No new text for window {window_id}")
            return []
        
        logger.info(f"Got {len(new_text)} chars of new text for window {window_id}")
        
        # Add window to WindowManager (sets _first_window_timestamp if first window)
        self._window_manager.add_window(new_text, window_start, window_end)
        
        # Check initial delay (self-contained, no STATE import)
        if not self._has_performed_summary:
            if self._window_manager._first_window_timestamp is not None:
                elapsed = window_start - self._window_manager._first_window_timestamp
                logger.debug(f"Initial delay check: elapsed={elapsed:.1f}s, delay={self.initial_summary_delay_seconds}s")
                if elapsed < self.initial_summary_delay_seconds:
                    logger.info(f"Delaying first summary - only {elapsed:.1f}s elapsed (need {self.initial_summary_delay_seconds}s)")
                    return []  # Skip LLM call, accumulate more text
                else:
                    logger.info(f"Initial delay passed - proceeding with first summary after {elapsed:.1f}s")
        
        # Get context from accumulated text (processed windows only)
        context = self._build_context()
        
        # Get unprocessed text from all windows (for first summary, this includes all accumulated text)
        unprocessed_text = self._window_manager.get_unprocessed_text()
        
        # Determine what text to send to LLM for analysis
        if not self._has_performed_summary:
            # First summary: analyze ALL accumulated text from unprocessed windows
            content_to_analyze = unprocessed_text
            logger.info(f"First summary: analyzing {len(content_to_analyze)} chars from {len(self._window_manager._windows)} windows")
        else:
            # Subsequent summaries: only analyze new text
            content_to_analyze = new_text
            logger.info(f"Subsequent summary: analyzing {len(content_to_analyze)} chars new text")
        
        # Send text + context to LLM
        logger.info(f"Sending {len(content_to_analyze)} chars to analyze + {len(context)} chars context to LLM")
        summary_text, reasoning_content = await self.summarize_text(content_to_analyze, context)
        
        logger.info(f"Processed window {window_id}, summary length={len(summary_text)}")
        
        # Parse JSON and extract analysis for background_context
        background_context = ""
        parsed_data = None
        
        if summary_text:
            try:
                parsed_data = json.loads(summary_text)
                analysis_field = parsed_data.get("analysis", "")
                if analysis_field:
                    background_context = analysis_field
                    logger.debug(f"Using analysis field as background_context (length={len(analysis_field)})")
                else:
                    background_context = reasoning_content
                    logger.debug("No analysis field found, using reasoning_content as fallback")
            except json.JSONDecodeError:
                background_context = reasoning_content
                logger.debug("JSON parse failed, using reasoning_content as fallback")
        
        # Extract insights using parsed data (no summary_text parameter)
        insights = []
        if parsed_data:
            insights = self._extract_insights(parsed_data, window_id, window_start, window_end)
            for insight in insights:
                self._window_manager.add_insight_to_window(window_id, insight)
        
        # Create summary segment
        summary_segment = SummarySegment(
            summary_type=summary_type,
            background_context=background_context,
            summary=summary_text,
            timestamp_start=window_start,
            timestamp_end=window_end,
        )
        
        # Mark that we have performed at least one summary
        self._has_performed_summary = True
        
        # Mark all windows as processed after first summary
        self._window_manager.mark_all_windows_processed()
        logger.info("Marked all windows as processed after first summary")
        
        logger.info(f"Returning summary segment for window {window_id}")
        
        return [summary_segment]
    
    def _extract_insights(
        self,
        parsed_data: Dict,
        window_id: int,
        window_start: float,
        window_end: float
    ) -> List[WindowInsight]:
        """
        Extract insights from parsed JSON data and return as WindowInsight objects.
        
        Args:
            parsed_data: Parsed JSON data from LLM response
            window_id: The window these insights belong to
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
        
        Returns:
            List of WindowInsight objects
        """
        insights = []
        
        if not parsed_data:
            return insights
        
        # Work directly with parsed data - no JSON parsing needed
        data = parsed_data
        logger.debug("Using parsed JSON data for insights extraction")
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    insight_type = item.get("insight_type", "NOTES")
                    insight_text = item.get("insight_text", "") or item.get("insight", "") or item.get("text", "")
                    
                    # Use classification directly from schema (no parsing needed)
                    classification = item.get("classification", "[~]")
                    
                    if insight_text:
                        insights.append(WindowInsight(
                            insight_type=insight_type,
                            insight_text=insight_text,
                            window_id=window_id,
                            timestamp_start=window_start,
                            timestamp_end=window_end,
                            classification=classification
                        ))
        elif isinstance(data, dict):
            # Single object with insight types
            for key, value in data.items():
                if key == "insights" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            insight_type = item.get("insight_type", item.get("type", "NOTES"))
                            insight_text = item.get("insight_text", "") or item.get("insight", "") or item.get("text", "")
                            
                            # Use classification directly from schema (no parsing needed)
                            classification = item.get("classification", "[~]")
                            
                            if insight_text:
                                insights.append(WindowInsight(
                                    insight_type=insight_type,
                                    insight_text=insight_text,
                                    window_id=window_id,
                                    timestamp_start=window_start,
                                    timestamp_end=window_end,
                                    classification=classification
                                ))
        
        return insights
    
    def reset(self):
        """Reset all accumulated state."""
        self._window_manager.clear()
        self.in_flight_windows.clear()
        self._skipped_segments_buffer = None
        # Reset content type state
        self._content_type_state = ContentTypeState()
        self._user_content_type_override = None
        # Reset raw response storage
        self._last_summary_raw_response = None
        # Reset window tracking state
        self.summary_window_counter = 0
        self.summary_window_count = 0
    
    def set_content_type(self, content_type: str, confidence: float = 0.0, source: str = "AUTO_DETECTED"):
        """
        Set content type state.
        
        Args:
            content_type: Content type to set
            confidence: Confidence level (0.0-1.0)
            source: Source of content type (USER_OVERRIDE, AUTO_DETECTED)
        """
        valid_types = [ct.value for ct in ContentType]
        if content_type not in valid_types:
            logger.warning(f"Invalid content type: {content_type}")
            return
        
        self._content_type_state = ContentTypeState(
            content_type=content_type,
            confidence=confidence,
            source=source
        )
        logger.info(f"Content type set to: {content_type} (source: {source}, confidence: {confidence:.2f})")
    
    def set_content_type_override(self, content_type: Optional[str]):
        """
        Set user override for content type.
        
        When set, this content type will be used regardless of auto-detection.
        Set to None to clear override.
        
        Args:
            content_type: Content type to use or None to clear
        """
        if content_type is None:
            self._user_content_type_override = None
            logger.info("Content type user override cleared")
        else:
            valid_types = [ct.value for ct in ContentType]
            if content_type in valid_types:
                self._user_content_type_override = content_type
                logger.info(f"Content type user override set to: {content_type}")
            else:
                logger.warning(f"Invalid content type for override: {content_type}")
    
    def get_content_type_state(self) -> dict:
        """
        Get current content type state.
        
        Returns:
            Dict with content_type, confidence, source, and user_override
        """
        effective_type = self._user_content_type_override or self._content_type_state.content_type
        effective_source = "USER_OVERRIDE" if self._user_content_type_override else self._content_type_state.source
        effective_confidence = 1.0 if self._user_content_type_override else self._content_type_state.confidence
        
        return {
            "content_type": effective_type,
            "confidence": effective_confidence,
            "source": effective_source,
            "user_override": self._user_content_type_override
        }
    
    def get_effective_content_type(self) -> tuple[str, float, str]:
        """
        Get effective content type considering user override.
        
        Returns:
            Tuple of (content_type, confidence, source)
        """
        if self._user_content_type_override:
            return (self._user_content_type_override, 1.0, "USER_OVERRIDE")
        return (
            self._content_type_state.content_type,
            self._content_type_state.confidence,
            self._content_type_state.source
        )
    
    def add_in_flight_window(self, window_id: int):
        """Add window ID to in-flight tracking."""
        self.in_flight_windows.add(window_id)
    
    def remove_in_flight_window(self, window_id: int):
        """Remove window ID from in-flight tracking."""
        self.in_flight_windows.discard(window_id)
    
    def get_pending_count(self) -> int:
        """Get count of pending summary requests."""
        return len(self.in_flight_windows)
    
    def has_skipped_segments(self) -> bool:
        """Check if there are stored skipped segments for merging."""
        return self._skipped_segments_buffer is not None
    
    def get_skipped_segments(self) -> Optional[Dict[str, Any]]:
        """Get stored skipped segments and clear buffer."""
        skipped = self._skipped_segments_buffer
        self._skipped_segments_buffer = None
        return skipped
    
    async def store_skipped_segments(
        self,
        segments: List[Dict[str, Any]],
        window_id: int,
        window_start: float,
        window_end: float
    ):
        """
        Store skipped segments for merging with next window.
        
        Args:
            segments: List of transcription segments
            window_id: Unique identifier for this window
            window_start: Start timestamp
            window_end: End timestamp
        """
        # Get non-overlapping text
        new_text = self.get_new_text_for_window(segments, window_id)
        
        if not new_text:
            return
        
        # Store for merging with next window
        self._skipped_segments_buffer = {
            "segments": segments,
            "window_id": window_id,
            "window_start": window_start,
            "window_end": window_end,
            "text": new_text
        }
        
        logger.info(f"Stored skipped window {window_id} for merging, {len(new_text)} chars")
    
    async def process_content_type_detection(
        self,
        window_id: int,
        window_start: float,
        window_end: float
    ) -> Optional[ContentTypeState]:
        """
        Process content type detection for a window.
        
        Called by the summary worker when content type detection is needed.
        Returns the ContentTypeState so the worker can handle result sending.
        
        Args:
            window_id: Unique identifier for this window
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
        
        Returns:
            ContentTypeState if detection succeeded, None if skipped (user override)
        """
        # Check for user override first
        if self._user_content_type_override:
            logger.info(f"Skipping content type detection - user override active: {self._user_content_type_override}")
            return None
        
        # Get accumulated text from all windows (newest to oldest)
        context_text = self._window_manager.get_all_windows_text()
        
        # Get current context length from state, or use default
        context_length = self._content_type_state.context_length
        if context_length < 2000:
            context_length = 2000
        
        # Don't exceed available text
        max_context_length = len(context_text)
        if context_length > max_context_length:
            context_length = max_context_length
        
        # Get context text (last N chars)
        context_to_use = context_text[-context_length:] if len(context_text) > context_length else context_text
        
        # Format the context text for the user message
        user_content = f"""## TRANSCRIPT CONTEXT

Transcript Text (Last {context_length} characters):
{context_to_use}

---

Please analyze the transcript context above and output content type detection as JSON with fields: content_type, confidence, and reasoning."""

        messages = self._build_messages(
            system_prompt=CONTENT_TYPE_DETECTION_PROMPT,
            user_content=user_content
        )
        
        try:
            logger.info(f"Running content type detection (context_length={context_length})")
            
            try:
                response: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=250,
                    temperature=0.1,
                    response_format={"type": "json_schema", "json_schema": {"name": "content_type_detection", "schema": self.content_type_response_json_schema}},
                )
            except Exception as e:
                logger.error(f"Content type detection LLM call failed: {e}")
                raise

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content or ""
                
                # Validate response before parsing
                if not content or not content.strip():
                    return self._content_type_state
                
                # Check for partial/incomplete JSON
                content_stripped = content.strip()
                starts_with_brace = content_stripped.startswith('{')
                ends_with_brace = content_stripped.endswith('}')
                
                if not starts_with_brace or not ends_with_brace:
                    return self._content_type_state
                
                result = self._parse_content_type_response(content)
                
                # Check if UNKNOWN - increase context length for next run
                if result.content_type == ContentType.UNKNOWN.value:
                    logger.info(f"Content type UNKNOWN detected (confidence: {result.confidence:.2f})")
                    
                    # Extract reasoning from response if available
                    try:
                        data = json.loads(content)
                        reasoning = data.get("reasoning", "No reasoning provided")
                        logger.info(f"UNKNOWN reasoning: {reasoning}")
                    except Exception:
                        logger.info("Could not extract reasoning from UNKNOWN response")
                    
                    # Increase context length by 500 for next run (don't exceed max)
                    new_context_length = min(context_length + 500, max_context_length)
                    if new_context_length != context_length:
                        logger.info(f"Increasing context length to {new_context_length} for next run")
                    context_length = new_context_length
                
                # Update state with new context length
                self._content_type_state = ContentTypeState(
                    content_type=result.content_type,
                    confidence=result.confidence,
                    source=ContentTypeSource.AUTO_DETECTED.value,
                    last_detection_text=context_to_use,
                    context_length=context_length
                )
                
                logger.info(f"Content type detected: {result.content_type} (confidence: {result.confidence:.2f})")
                
                return self._content_type_state
            
            # Return current state if LLM returns empty response
            return self._content_type_state
            
        except Exception as e:
            logger.error(f"Content type detection error: {e}")
            return self._content_type_state
    
    def _parse_content_type_response(self, content: str) -> ContentTypeState:
        """Parse LLM response for content type detection."""
        try:
            data = json.loads(content)
            
            # CRITICAL: Check if data is a string (not a dict) - this would cause KeyError!
            if isinstance(data, str):
                # Return UNKNOWN since we can't parse this
                return ContentTypeState(
                    content_type=ContentType.UNKNOWN.value,
                    confidence=0.0,
                    source=ContentTypeSource.AUTO_DETECTED.value
                )
            
            # Try direct key access first (this is where KeyError might occur)
            try:
                content_type = data["content_type"]
            except KeyError:
                content_type = ContentType.UNKNOWN.value
            
            confidence = float(data.get("confidence", 0.0))
            
        except json.JSONDecodeError:
            content_type = ContentType.UNKNOWN.value
            confidence = 0.0
        except KeyError:
            content_type = ContentType.UNKNOWN.value
            confidence = 0.0
        except Exception:
            content_type = ContentType.UNKNOWN.value
            confidence = 0.0
        
        # Validate content type
        valid_types = [ct.value for ct in ContentType]
        if content_type not in valid_types:
            content_type = ContentType.UNKNOWN.value
        
        return ContentTypeState(
            content_type=content_type,
            confidence=confidence,
            source=ContentTypeSource.AUTO_DETECTED.value
        )