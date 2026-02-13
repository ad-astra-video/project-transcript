"""
Summary client for LLM-based transcription cleaning and summarization.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable, Awaitable
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
    insight_id: int = 0  # Unique identifier assigned by system (not LLM)
    insight_type: str = ""
    insight_text: str = ""
    confidence: float = 0.0  # Confidence score from LLM (0.0-1.0)
    window_id: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    classification: str = "~"
    continuation_of: Optional[int] = None  # Previous insight ID this continues
    correction_of: Optional[int] = None  # Previous insight ID this corrects
    
    # Excludes timestamp_start and timestamp_end
    #   this is used for sending over json data channel which includes timing for all insights sent
    def as_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "insight_text": self.insight_text,
            "confidence": self.confidence,
            "classification": self.classification,
            "continuation_of": self.continuation_of,
            "correction_of": self.correction_of,
        }

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


# Callback type for monitoring events: takes event data dict and event type string (async)
MonitoringCallback = Callable[[Dict[str, Any], str], Awaitable[None]]


@dataclass
class ContentTypeState:
    """State for content type tracking."""
    content_type: str = ContentType.UNKNOWN.value
    confidence: float = 0.0
    source: str = ContentTypeSource.INITIAL.value
    last_detection_text: str = ""  # Last N chars used for detection
    context_length: int = 2000  # Current context length for detection
    sentiment_enabled: bool = False  # Whether sentiment tracking is enabled for this content type

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
    continuation_of: Optional[int] = None  # Previous insight ID this continues
    correction_of: Optional[int] = None  # Previous insight ID this corrects


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
    
    def __init__(
        self,
        context_limit: int = 50000,  # Max characters for accumulated text
        raw_text_context_limit: int = 2000,  # Max characters for raw text in LLM context
        transcription_windows_per_summary_window: int = 4  # Number of transcription windows per summary window
    ):
        self._windows: List[SummaryWindow] = []  # Ordered oldest -> newest
        self._char_count: int = 0
        self._next_window_id: int = 0
        self.context_limit = context_limit  # Max characters for accumulated text
        self.raw_text_context_limit = raw_text_context_limit  # Max characters for raw text in LLM context
        self.transcription_windows_per_summary_window = transcription_windows_per_summary_window  # Number of transcription windows per summary window
        self._first_window_timestamp: Optional[float] = None  # Track first window timestamp for self-contained delay logic
        self._next_insight_id: int = 0  # Counter for unique insight IDs
    
    def add_window(self, text: str, timestamp_start: float, timestamp_end: float,
                   window_id: Optional[int] = None) -> int:
        """
        Add a new window, dropping oldest if over char limit.
        Also tracks first window timestamp for self-contained delay logic.
        
        Args:
            text: Text content for this window
            timestamp_start: Start timestamp in seconds
            timestamp_end: End timestamp in seconds
            window_id: Optional external window ID. If provided, uses this as the
                       authoritative ID. If None, generates next internal ID.
        
        Returns:
            window_id of the added window (either provided or generated)
        """
        if window_id is not None:
            actual_window_id = window_id
            # Update internal counter to stay in sync if needed
            # The counter is for internal tracking, not exposed externally
            self._next_window_id += 1
        else:
            actual_window_id = self._next_window_id
            self._next_window_id += 1
        
        # Track first window timestamp for initial delay (self-contained logic)
        if self._first_window_timestamp is None:
            self._first_window_timestamp = timestamp_start
            logger.info(f"First transcript at {timestamp_start:.3f}s - initial delay logic active")
        
        # Check if adding would exceed limit
        new_char_count = self._char_count + len(text)
        
        # Drop oldest windows until under limit
        while new_char_count > self.context_limit and self._windows:
            oldest = self._windows.pop(0)
            self._char_count -= oldest.char_count
            new_char_count = self._char_count + len(text)
        
        # Create and add window
        window = SummaryWindow(
            window_id=actual_window_id,
            text=text,
            insights=[],
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            char_count=len(text)
        )
        self._windows.append(window)
        self._char_count += len(text)

        logger.debug(f"Added window {actual_window_id}, char_count={self._char_count}, total_windows={len(self._windows)}")

        return actual_window_id
    
    def get_accumulated_text_and_insights(self) -> tuple[str, List[WindowInsight], int, float]:
        """
        Get accumulated text and insights from all windows except the last one.
        
        The last window is the "current" window being analyzed - its text is sent
        as new_text to the LLM. All previous windows form the prior context.
        
        Raw text is limited to raw_text_context_limit (default: 2000 chars) by only
        adding text from windows while current length is under the limit.
        
        Returns:
            Tuple of (accumulated_text_string, list_of_insights, text_length, insights_per_window_metric)
        """
        if len(self._windows) <= 1:
            logger.debug(f"Not enough windows for accumulation: {len(self._windows)} <= 1")
            return "", [], 0, 0.0
        
        # All windows except the last one (current window being analyzed)
        accumulated_windows = self._windows[:-1]
        text_parts = []
        current_text_length = 0
        insights = []
        
        for window in accumulated_windows:
            # Only add text if we're still under the limit
            if window.text and current_text_length < self.raw_text_context_limit:
                text_parts.append(window.text)
                current_text_length += len(window.text)
            
            # Collect insights (always include all insights regardless of text limit)
            if window.insights:
                insights.extend(window.insights)
        
        accumulated_text = " ".join(text_parts)
        
        # Calculate metrics once
        num_windows = len(accumulated_windows)
        total_insights = len(insights)
        text_length = len(accumulated_text)
        insights_per_window = total_insights / num_windows if num_windows > 0 else 0.0
        
        # Log the insights per window metric
        logger.info(
            f"Returning accumulated text from {num_windows} windows with {total_insights} total insights. "
            f"Text length: {text_length} chars (limit: {self.raw_text_context_limit}). "
            f"Insights per window metric: {insights_per_window:.2f}"
        )
        
        return accumulated_text, insights, text_length, insights_per_window
    
    def get_all_windows_text(self) -> str:
        """
        Get text from all windows, ordered from most recent to oldest.
        
        Returns:
            Concatenated text from all windows, newest to oldest
        """
        if not self._windows:
            return ""
        
        # Get text from all windows, newest first
        all_text_parts = [w.text for w in self._windows]
        return " ".join(all_text_parts)
    
    def get_window_insights(self, window_id: int) -> List[WindowInsight]:
        """Get insights for a specific window."""
        for window in self._windows:
            if window.window_id == window_id:
                return window.insights
        return []
    
    def _get_next_insight_id(self) -> int:
        """
        Increment and return the next unique insight ID.
        First call returns 1, second returns 2, etc.
        
        Returns:
            Unique integer ID for the next insight
        """
        self._next_insight_id += 1
        return self._next_insight_id
    
    def add_insight_to_window(self, window_id: int, insight: WindowInsight) -> int:
        """
        Add a single insight to a window. Searches windows in reverse order
        (newest first) for efficiency since recent windows are more likely targets.
        
        Args:
            window_id: The window to add insight to
            insight: The WindowInsight to add
        
        Returns:
            The insight_id of the added insight, or -1 if window not found
        """
        for window in reversed(self._windows):
            if window.window_id == window_id:
                window.insights.append(insight)
                logger.debug(f"Added insight {insight.insight_id} to window {window_id}")
                return insight.insight_id
        
        # Window not found - log error with diagnostic info
        available_ids = [w.window_id for w in self._windows]
        logger.error(
            f"Failed to add insight {insight.insight_id} to window {window_id} - window not found. "
            f"Available window IDs: {available_ids}, Total windows: {len(self._windows)}"
        )
        return -1  # Window not found
    
    def clear(self):
        """Clear all windows and reset internal counters for fresh stream."""
        self._windows.clear()
        self._char_count = 0
        # Reset all internal counters for fresh stream
        self._next_window_id = 0
        self._next_insight_id = 0
        self._first_window_timestamp = None
        logger.debug("WindowManager cleared - all counters reset")
    
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
        max_tokens: int = 4096,
        temperature: float = 0.1,
        system_prompt: str = SYSTEM_PROMPT,
        transcription_windows_per_summary_window: int = 4,
        raw_text_context_limit: int = 2000,
        initial_summary_delay_seconds: float = 10.0,
        send_monitoring_event_callback: Optional[MonitoringCallback] = None
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
            transcription_windows_per_summary_window: Number of transcription windows merged per summary window (default: 4)
            raw_text_context_limit: Max characters for raw text in LLM context (default: 2000)
            initial_summary_delay_seconds: Seconds to wait before first summary (default: 10.0)
            send_monitoring_event_callback: Optional callback for sending monitoring events
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
        self._window_manager: WindowManager = WindowManager(
            transcription_windows_per_summary_window=transcription_windows_per_summary_window,
            raw_text_context_limit=raw_text_context_limit
        )
        
        # Store raw_text_context_limit at client level for runtime access
        self.raw_text_context_limit = raw_text_context_limit
        
        # Track last processed timestamp (global, not per-window)
        self._last_processed_timestamp: float = 0.0
        
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
        
        # Auto content type detection trigger - runs on first buffered window
        self._auto_detect_content_type_detection: bool = True
        
        # In-flight flag to prevent concurrent content type detection requests
        self._content_type_detection_in_progress: bool = False
        
        # Store last raw LLM responses for debugging
        self._last_summary_raw_response: Optional[str] = None
        
        # Initial summary delay configuration
        self.initial_summary_delay_seconds: float = initial_summary_delay_seconds
        
        # Temporary buffer for accumulating transcription segments before processing
        # Segments are buffered and only processed when _transcription_window_counter % transcription_windows_per_summary_window == 0
        self._temp_segment_buffer: List[Dict[str, Any]] = []  # Segments waiting to be processed
        self._temp_buffer_timing: List[Dict[str, float]] = []  # Timing info for buffered segments
        self._temp_buffer_window_ids: List[int] = []  # transcription_window_ids for buffered segments
        self._transcription_window_counter: int = 0  # Track total transcription windows for modulo check
        
        # Monitoring event callback
        self._send_monitoring_event_callback: Optional[MonitoringCallback] = send_monitoring_event_callback
    
    async def initialize(self) -> Optional[str]:
        """
        Initialize the lock for async operations and detect model if needed.
        
        If model is empty/None, fetches the loaded model from /models endpoint using
        the OpenAI library's models.list() method.
        
        Returns:
            The detected model ID if model was auto-detected, None otherwise
        """
        logger.info("SummaryClient.initialize called")
        
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        # Auto-detect model if not specified
        if not self.model:
            logger.info("No model specified, fetching loaded model from /models endpoint")
            detected_model = await self.fetch_loaded_model()
            self.model = detected_model
            logger.info(f"Auto-detected model: {self.model}")
            return detected_model
        
        logger.info(f"SummaryClient initialized with model: {self.model}")
        return None
    
    async def fetch_loaded_model(self) -> str:
        """
        Fetch the currently loaded model from the /models endpoint using OpenAI library.
        
        Uses the OpenAI Python library's models.list() method to retrieve available models
        and returns the first one (typically the loaded/primary model).
        
        Returns:
            The model ID string from the OpenAI models list response
            
        Raises:
            RuntimeError: If the models.list() call fails or returns no models
        """
        try:
            logger.info(f"Fetching available models from {self.base_url}")
            
            # Use OpenAI Python library's built-in models.list() method
            response = await self.client.models.list()
            
            if response.data and len(response.data) > 0:
                # Return the first model (typically the loaded/primary model)
                model_id = response.data[0].id
                logger.info(f"Detected loaded model: {model_id}")
                return model_id
            else:
                raise RuntimeError("No models available from OpenAI API")
                
        except Exception as e:
            logger.error(f"Failed to fetch models using OpenAI library: {e}")
            raise RuntimeError(f"Model detection failed: {e}")
    
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
    
    def _add_to_temp_buffer(
        self,
        segments: List[Dict[str, Any]],
        transcription_window_id: int,
        window_start: float,
        window_end: float
    ):
        """
        Add segments to temporary buffer for later processing.
        
        Segments are accumulated until _transcription_window_counter % transcription_windows_per_summary_window == 0,
        at which point all buffered segments are flushed and processed together.
        
        Args:
            segments: List of transcription segment dictionaries
            transcription_window_id: Unique identifier for this transcription window
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
        """
        self._temp_segment_buffer.append(segments)
        self._temp_buffer_timing.append({"window_start": window_start, "window_end": window_end})
        self._temp_buffer_window_ids.append(transcription_window_id)
        # Note: _transcription_window_counter is incremented in process_segments before calling this
        logger.debug(
            f"Added transcription window {transcription_window_id} to temp buffer "
            f"(counter={self._transcription_window_counter}, buffer_size={len(self._temp_segment_buffer)})"
        )
    
    def _flush_temp_buffer(self) -> tuple[List[Dict[str, Any]], int, float, float, List[int]]:
        """
        Flush all buffered segments and return merged data.
        
        Returns:
            Tuple of:
            - merged_segments: All segments from buffer merged into single list
            - last_window_id: The transcription_window_id of the last buffered window
            - combined_start: Start timestamp (from first buffered window)
            - combined_end: End timestamp (from last buffered window)
            - accumulated_ids: All transcription_window_ids from buffered windows
        """
        if not self._temp_segment_buffer:
            return [], 0, 0.0, 0.0, []
        
        # Merge all buffered segments
        merged_segments = []
        for seg_list in self._temp_segment_buffer:
            merged_segments.extend(seg_list)
        
        # Get combined timing (first start, last end)
        first_timing = self._temp_buffer_timing[0]
        last_timing = self._temp_buffer_timing[-1]
        combined_start = first_timing["window_start"]
        combined_end = last_timing["window_end"]
        
        # Get all accumulated window IDs
        accumulated_ids = self._temp_buffer_window_ids.copy()
        last_window_id = accumulated_ids[-1] if accumulated_ids else 0
        
        # Clear buffer
        self._temp_segment_buffer.clear()
        self._temp_buffer_timing.clear()
        self._temp_buffer_window_ids.clear()
        
        logger.debug(
            f"Flushed temp buffer: {len(merged_segments)} segments, "
            f"timing=[{combined_start:.3f}s - {combined_end:.3f}s], "
            f"window_ids={accumulated_ids}"
        )
        
        return merged_segments, last_window_id, combined_start, combined_end, accumulated_ids
    
    def update_windows_to_accumulate(self, value: int):
        """
        Update the number of windows to accumulate for context.
        
        Args:
            value: Number of windows to accumulate (minimum 1)
        """
        self._window_manager.transcription_windows_per_summary_window = max(1, value)
        logger.info(f"Updated transcription_windows_per_summary_window to {self._window_manager.transcription_windows_per_summary_window}")
        
        # Check if buffer should be flushed with new accumulation setting
        if self._temp_segment_buffer:
            remaining = self._transcription_window_counter % self._window_manager.transcription_windows_per_summary_window
            if remaining == 0:
                logger.info(
                    f"Buffer ready to flush with new transcription_windows_per_summary_window={value}, "
                    f"next process_segments call will process {len(self._temp_segment_buffer)} buffered windows"
                )
    
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
        transcription_windows_per_summary_window: Optional[int] = None,
        raw_text_context_limit: Optional[int] = None,
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
            transcription_windows_per_summary_window: New number of transcription windows per summary window
            raw_text_context_limit: New max characters for raw text in LLM context
            initial_summary_delay_seconds: New delay before first summary (default: 10.0)
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
        if transcription_windows_per_summary_window is not None:
            self.update_windows_to_accumulate(transcription_windows_per_summary_window)
        if raw_text_context_limit is not None:
            self.raw_text_context_limit = raw_text_context_limit
            self._window_manager.raw_text_context_limit = raw_text_context_limit
            logger.info(f"Updated raw_text_context_limit to {raw_text_context_limit}")
        if initial_summary_delay_seconds is not None:
            self.initial_summary_delay_seconds = initial_summary_delay_seconds
            logger.info(f"Updated initial_summary_delay_seconds to {initial_summary_delay_seconds}")
        
        logger.info(f"SummaryClient params updated: base_url={self.base_url}, history_length={self.history_length}, model={self.model}")
    
    def get_new_text_for_summary_window(
        self,
        segments: List[Dict[str, Any]]
    ) -> str:
        """
        Get non-overlapping text from segments for a transcription window.
        
        Uses self._last_processed_timestamp (global) as the reference for overlap detection.
        Updates self._last_processed_timestamp with the end time of the last segment
        for use in subsequent window processing.
        
        All timestamps are in milliseconds (consistent with segment timing).
        
        Args:
            segments: List of segment dictionaries with 'words' containing word timestamps
        
        Returns:
            Non-overlapping text for this window
        """
        last_ts = self._last_processed_timestamp  # Use global cross-window timestamp
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
        
        # Update _last_processed_timestamp for cross-window tracking (in milliseconds)
        if segments:
            last_seg_end = segments[-1].get("end_ms", segments[-1].get("end", 0))
            self._last_processed_timestamp = last_seg_end
        
        return " ".join(new_text_parts)
    
    def _build_context(self, include_insights: bool = True) -> tuple[str, List[WindowInsight], int, float]:
        """
        Build context string with text and optionally insights from accumulated windows.
        Single-pass: loops through accumulated windows once to get both text and insights.
        
        Returns:
            Tuple of (formatted_context_string, list_of_insights_from_accumulated_windows, text_length, insights_per_window_metric)
        """
        # Single-pass accumulation from same windows
        accumulated_text, insights, text_length, insights_per_window = self._window_manager.get_accumulated_text_and_insights()
        
        parts = []
        
        # Add accumulated text
        if accumulated_text:
            parts.append(f"## PRIOR TEXT\n{accumulated_text}")
        
        # Add insights from same accumulated windows
        if include_insights and insights:
            insights_text = self._format_insights_for_context(insights)
            parts.append(f"## PRIOR INSIGHTS\n{insights_text}")
        
        context_string = "\n\n".join(parts) if parts else ""
        return context_string, insights, text_length, insights_per_window
    
    def _format_insights_for_context(self, insights: List[WindowInsight]) -> str:
        """Format insights for inclusion in Prior Context with IDs and timing hints."""
        formatted = []
        for insight in insights:
            # Format with ID and timing hints
            # ID is included so LLM can reference it in continuation_of/correction_of
            id_hint = f"[#{insight.insight_id}]"
            timing_hint = f"[{insight.timestamp_start:.1f}s - {insight.timestamp_end:.1f}s]"
            
            # Add continuation/correction markers if present
            markers = []
            if insight.continuation_of:
                markers.append(f"CONTINUATION of insight #{insight.continuation_of}")
            if insight.correction_of:
                markers.append(f"CORRECTION of insight #{insight.correction_of}")
            
            marker_text = f" ({', '.join(markers)})" if markers else ""
            
            formatted.append(
                f"- **{insight.insight_type}** {id_hint} {timing_hint}: "
                f"{insight.insight_text}{marker_text}"
            )
        
        return "\n".join(formatted)
    
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
        
        # Build RISK guidance section if present
        risk_guidance = rules.get("risk_guidance", "")
        risk_section = ""
        if risk_guidance:
            risk_section = f"""
### RISK Definition (Content-Type-Specific):
{risk_guidance}"""
        
        # Build KEY POINT guidance section if present
        key_point_guidance = rules.get("key_point_guidance", "")
        key_point_section = ""
        if key_point_guidance:
            key_point_section = f"""
### KEY POINT Guidance:
{key_point_guidance}"""
        
        # Build STORY guidance section if present
        story_guidance = rules.get("story_guidance", "")
        story_section = ""
        if story_guidance:
            story_section = f"""
### STORY Handling:
{story_guidance}"""
        
        # Format the rules into a clear, actionable prompt section
        formatted = f"""

## CONTENT TYPE RULES: {content_type}

### Priority Insight Types (Emphasize):
{chr(10).join(f'- {t}' for t in rules["emphasize"])}

### Suppressed Insight Types (Deemphasize):
{chr(10).join(f'- {t}' for t in rules["deemphasize"])}

{risk_section}

{key_point_section}

{story_section}

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
        
        #messages.append({"role": "assistant", "content": "<think>Sure, "})
        return messages
    
    async def summarize_text(self, text: str, context: str = "") -> tuple[str, str, int]:
        """
        Send text to LLM for cleaning and summarization.
        
        Args:
            text: Text to summarize/clean
            context: Additional context from previous segments
            
        Returns:
            Tuple of (summary_text, reasoning_content, input_tokens)
            - summary_text: The JSON response content from LLM
            - reasoning_content: The reasoning content from LLM (may be empty)
            - input_tokens: Number of input tokens sent to LLM
        """
        logger.debug(f"summarize_text called with text length={len(text)}, context length={len(context)}")
        
        async def _do_request() -> tuple[str, str, int]:
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
                    response_format={"type": "json_schema", "json_schema": {"name": "insights", "schema": self.insights_response_json_schema}}
                )

                # Extract input tokens from response usage
                input_tokens = 0
                if response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                
                # Extract summary text and reasoning content from response
                if response.choices and len(response.choices) > 0:
                    summary_text = response.choices[0].message.content or ""
                    summary_text_raw = summary_text  # Store raw response for logging
                    summary_text = summary_text.replace("```json", "").replace("```", "").strip()

                    reasoning_content = response.choices[0].message.reasoning or ""

                    logger.info(f"summarize_text received response, length={len(summary_text)}, input_tokens={input_tokens}")
                    
                    # Store raw response for potential logging when no insights found
                    self._last_summary_raw_response = summary_text_raw
                    
                    return summary_text.strip(), reasoning_content.strip(), input_tokens
                
                logger.info("summarize_text received empty response")
                self._last_summary_raw_response = None
                return "", "", 0
                
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
        transcription_window_id: int,
        window_start: float,
        window_end: float
    ) -> Dict[str, Any]:
        """
        Process transcription segments and return complete payload.
        
        Uses modulo-based accumulation: segments are buffered until
        _transcription_window_counter % transcription_windows_per_summary_window == 0, then
        all buffered segments are merged and processed together.
        
        Args:
            segments: List of transcription segments
            transcription_window_id: ID from whisper client
            window_start: Start timestamp of the current window
            window_end: End timestamp of the current window
        
        Returns:
            Complete payload dictionary with summary_window_id and transcription_window_ids list
        """
        logger.debug(f"process_segments called with {len(segments)} segments for transcription_window_id={transcription_window_id}")
        
        # Increment transcription window counter for modulo check
        #  Not using transcription_window_id for counting because it may not be strictly sequential
        self._transcription_window_counter += 1
        
        # Check if we should process now based on modulo
        should_process = (self._transcription_window_counter % self._window_manager.transcription_windows_per_summary_window == 0)
        
        if not should_process:
            # Window will be buffered - add to buffer first
            self._add_to_temp_buffer(segments, transcription_window_id, window_start, window_end)
            logger.info(
                f"Buffered transcription window {transcription_window_id} "
                f"(counter={self._transcription_window_counter}, modulo={self._transcription_window_counter % self._window_manager.transcription_windows_per_summary_window})"
            )
            
            # Check if this is the first buffered window (buffer just went from empty to non-empty)
            # AND we need auto-detection
            if len(self._temp_segment_buffer) == 1 and self._auto_detect_content_type_detection:
                # Skip if content type detection is already in progress (prevents concurrent LLM calls)
                if self._content_type_detection_in_progress:
                    logger.debug("Content type detection already in progress - skipping duplicate request")
                    return {"type": "context_summary", "segments": []}
                
                # Check user override FIRST - send immediately without elapsed time check
                if self._user_content_type_override:
                    # User override provided - send detection message with override
                    self.set_content_type(self._user_content_type_override, 1.0, "USER_OVERRIDE")
                    self._auto_detect_content_type_detection = False
                    return {
                        "type": "content_type_detection",
                        "content_type": self._user_content_type_override,
                        "confidence": 1.0,
                        "source": "USER_OVERRIDE",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()
                    }
                
                # Check elapsed time threshold (75% of initial delay) for auto-detection
                elapsed = 0.0
                if self._window_manager._first_window_timestamp is not None:
                    elapsed = window_start - self._window_manager._first_window_timestamp
                
                detection_threshold = self.initial_summary_delay_seconds * 0.75
                
                if elapsed >= detection_threshold:
                    # Run detection on accumulated text (includes current window)
                    result = await self.process_content_type_detection(
                        window_start=window_start,
                        window_end=window_end
                    )
                    if result:
                        # Update state and signal result to be sent
                        self.set_content_type(result.content_type, result.confidence, result.source)
                        self._auto_detect_content_type_detection = False
                        return {
                            "type": "content_type_detection",
                            "content_type": result.content_type,
                            "confidence": result.confidence,
                            "source": "AUTO_DETECTED",
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()
                        }
            
            # Either buffer already had items, or elapsed time not reached threshold, or no detection needed
            return {"type": "context_summary", "segments": []}
        
        # Time to process - flush buffer and merge with current segments
        logger.info(
            f"Processing buffered windows (counter={self._transcription_window_counter}, modulo=0)"
        )
        
        # Flush temp buffer to get merged segments from previous windows
        buffered_segments, buffered_last_id, buffered_start, buffered_end, buffered_ids = self._flush_temp_buffer()
        
        # Combine buffered segments with current segments
        # Buffer timing: buffered_start (first buffered) to buffered_end (last buffered)
        # Current timing: window_start to window_end
        # Combined: buffered_start to window_end
        if buffered_segments:
            # Merge: buffered + current
            merged_segments = buffered_segments + segments
            final_window_start = buffered_start
            final_window_end = window_end
            # transcription_window_ids: buffered IDs + current ID
            transcription_window_ids = buffered_ids + [transcription_window_id]
            logger.debug(
                f"Merged {len(buffered_segments)} buffered segments with {len(segments)} current segments, "
                f"timing=[{final_window_start:.3f}s - {final_window_end:.3f}s], ids={transcription_window_ids}"
            )
        else:
            # No buffered segments, use current only
            merged_segments = segments
            final_window_start = window_start
            final_window_end = window_end
            transcription_window_ids = [transcription_window_id]
            logger.debug(f"No buffered segments, using {len(segments)} current segments")
        
        new_text = ""
        # Add text from current segments (use get_new_text_for_summary_window)
        if merged_segments:
            new_text = self.get_new_text_for_summary_window(merged_segments)
        
        if not new_text:
            logger.info(f"No new text for transcription_window_id={transcription_window_id}")
            return {"type": "context_summary", "segments": []}
        
        logger.info(f"Got {len(new_text)} chars of new text for {len(transcription_window_ids)} transcription window(s)")
        
        # Add window to WindowManager
        # summary_window_id only increments when we actually process (not when buffering)
        summary_window_id = self._window_manager.add_window(
            new_text, final_window_start, final_window_end
        )
        logger.debug(f"summary_window_id={summary_window_id} for transcription_window_ids={transcription_window_ids}")
        
        # Check initial delay (self-contained, no STATE import)
        if not self._has_performed_summary:
            if self._window_manager._first_window_timestamp is not None:
                elapsed = final_window_start - self._window_manager._first_window_timestamp
                logger.debug(f"Initial delay check: elapsed={elapsed:.1f}s, delay={self.initial_summary_delay_seconds}s")
                if elapsed < self.initial_summary_delay_seconds:
                    logger.info(f"Delaying first summary - only {elapsed:.1f}s elapsed (need {self.initial_summary_delay_seconds}s)")
                    return {"type": "context_summary", "segments": []}  # Skip LLM call, accumulate more text
                else:
                    logger.info(f"Initial delay passed - proceeding with first summary after {elapsed:.1f}s")
        
        # Get context and prior insights from accumulated windows (also returns metrics)
        context, prior_insights, context_text_length, insights_per_window = self._build_context(include_insights=True)
        
        # Get unprocessed text from all windows (for first summary, this includes all accumulated text)
        unprocessed_text = self._window_manager.get_unprocessed_text()
        
        # Determine what text to send to LLM for analysis
        if not self._has_performed_summary:
            # First summary: analyze ALL accumulated text from unprocessed windows
            content_to_analyze = unprocessed_text
        else:
            # Subsequent summaries: only analyze new text
            content_to_analyze = new_text
            logger.info(f"Subsequent summary: analyzing {len(content_to_analyze)} chars new text")
        
        # Send text + context (with insights) to LLM
        logger.info(f"Sending {len(content_to_analyze)} chars to analyze + {context_text_length} chars context (with {len(prior_insights)} prior insights) to LLM")
        summary_text, reasoning_content, input_tokens = await self.summarize_text(content_to_analyze, context)
        
        # Calculate accumulated windows count (exclude current window being processed)
        num_accumulated_windows = len(self._window_manager._windows) - 1
        
        # Send monitoring event with input token count and insights per window metric
        await self._send_monitoring_event({
            "summary_window_id": summary_window_id,
            "transcription_window_ids": transcription_window_ids,
            "input_tokens": input_tokens,
            "text_chars": len(content_to_analyze),
            "context_chars": context_text_length,
            "prior_insights_count": len(prior_insights),
            "accumulated_windows_count": num_accumulated_windows,
            "insights_per_window": round(insights_per_window, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }, "summary_tokens")
        
        logger.info(f"Processed window (summary_window_id={summary_window_id}, transcription_window_ids={transcription_window_ids}), summary length={len(summary_text)}, input_tokens={input_tokens}")
        
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
                    parsed_data.pop("analysis")  # Remove analysis from parsed_data to avoid duplication in summary
                else:
                    background_context = reasoning_content
                    logger.debug("No analysis field found, using reasoning_content as fallback")
            except json.JSONDecodeError:
                background_context = reasoning_content
                logger.debug("JSON parse failed, using reasoning_content as fallback")
        
        # Extract insights and get updated parsed_data with assigned insight_ids
        insights = []
        if parsed_data:
            # Use summary_window_id from WindowManager
            # Pass prior_insights for reference validation
            parsed_data = self._extract_insights(
                parsed_data, summary_window_id, final_window_start, final_window_end, prior_insights
            )
            # Get insights from the window (they were added during _extract_insights)
            insights = self._window_manager.get_window_insights(summary_window_id)
            logger.info(f"Extracted {len(insights)} insights for window {summary_window_id}")
        
        # Update summary_text with assigned insight_ids
        updated_summary_text = json.dumps(parsed_data) if parsed_data else summary_text
        
        # Create summary segment
        summary_segment = SummarySegment(
            summary_type=summary_type,
            background_context=background_context,
            summary=updated_summary_text,
            timestamp_start=final_window_start,
            timestamp_end=final_window_end,
        )
        
        # Mark that we have performed at least one summary
        self._has_performed_summary = True
        
        # Re-enable auto-detection for next buffered window (after buffer flushes) if no override
        if not self._user_content_type_override:
            self._auto_detect_content_type_detection = True
        
        # Mark all windows as processed after first summary
        self._window_manager.mark_all_windows_processed()
        
        # Build complete payload with summary_window_id and transcription_window_ids list
        payload = {
            "type": "context_summary",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timing": {
                "summary_window_id": summary_window_id,
                "transcription_window_ids": transcription_window_ids,
                "media_window_start_ms": int(final_window_start * 1000),
                "media_window_end_ms": int(final_window_end * 1000)
            },
            "llm_usage": {
                "input_tokens": input_tokens
            },
            "segments": [
                {
                    "id": f"{summary_window_id}-0",
                    "summary_type": summary_segment.summary_type,
                    "background_context": summary_segment.background_context,
                    "summary": summary_segment.summary,
                }
            ]
        }
        
        logger.info(f"Returning complete payload with summary_window_id={summary_window_id}, transcription_window_ids={transcription_window_ids}")
        
        return payload
    
    def _parse_reference_id(self, value: Any, field_name: str) -> Optional[int]:
        """
        Parse and validate a reference ID field from LLM response.
        
        Args:
            value: The raw value from LLM response
            field_name: Name of the field for logging purposes
        
        Returns:
            Valid integer ID or None if invalid/missing
        """
        if value is None:
            return None
        
        try:
            # Handle string representations (LLM may return "42" instead of 42)
            if isinstance(value, str):
                parsed = int(value)
                if parsed <= 0:
                    logger.warning(f"Invalid {field_name}: '{value}' (must be positive integer)")
                    return None
                return parsed
            
            # Handle numeric types
            if isinstance(value, (int, float)):
                if value <= 0 or not float(value).is_integer():
                    logger.warning(f"Invalid {field_name}: {value} (must be positive integer)")
                    return None
                return int(value)
            
            # Log unexpected types
            logger.warning(f"Unexpected {field_name} type: {type(value).__name__} (value: {value})")
            return None
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse {field_name}: {value} ({e})")
            return None
    
    def _validate_insight_reference(
        self,
        ref_id: Optional[int],
        prior_insights: List['WindowInsight'],
        field_name: str
    ) -> Optional[int]:
        """
        Validate that a reference ID exists in prior insights.
        
        Args:
            ref_id: The reference ID to validate
            prior_insights: List of prior insights to check against
            field_name: Name of the field for logging purposes
        
        Returns:
            Valid reference ID or None if invalid/not found
        """
        if ref_id is None:
            return None
        
        # Check if the referenced insight exists
        valid_ids = {insight.insight_id for insight in prior_insights}
        if ref_id not in valid_ids:
            logger.warning(
                f"Invalid {field_name}: {ref_id} not found in prior insights. "
                f"Valid IDs: {sorted(valid_ids)[:10]}"  # Limit to first 10 for readability
            )
            return None
        
        return ref_id
    
    def _extract_insights(
        self,
        parsed_data: Dict,
        window_id: int,
        window_start: float,
        window_end: float,
        prior_insights: Optional[List['WindowInsight']] = None
    ) -> Dict:
        """
        Extract insights from parsed JSON data, assign IDs, and insert into parsed_data.
        Filters out SENTIMENT insights when sentiment_enabled is False for the content type.
        
        Args:
            parsed_data: Parsed JSON data from LLM response (dict with "insights" key or list)
            window_id: The window these insights belong to
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
            prior_insights: Optional list of prior insights for reference validation
        
        Returns:
            Parsed data with insights array updated to include assigned insight_ids
        """
        if not parsed_data:
            return parsed_data
        
        # Handle both dict with "insights" key and list format
        if isinstance(parsed_data, list):
            # List format: insights are the list items themselves
            insights_list = parsed_data
            is_list_format = True
        else:
            # Dict format: insights are in "insights" key
            insights_list = parsed_data.get("insights", [])
            is_list_format = False
        
        # Check if sentiment is enabled for current content type
        sentiment_enabled = self.is_sentiment_enabled()
        
        insights_for_summary = []
        
        for item in insights_list:
            if not isinstance(item, dict):
                continue
            
            # Check if this is a SENTIMENT insight that should be filtered
            insight_type = item.get("insight_type", "")
            if insight_type == InsightType.SENTIMENT.value and not sentiment_enabled:
                logger.debug(
                    f"Filtering SENTIMENT insight for content type with sentiment_enabled=False: "
                    f"{item.get('insight_text', '')[:50]}..."
                )
                continue  # Skip this insight
            
            # Parse and validate continuation_of and correction_of
            continuation_of_raw = item.get("continuation_of")
            correction_of_raw = item.get("correction_of")
            
            continuation_of = self._parse_reference_id(continuation_of_raw, "continuation_of")
            correction_of = self._parse_reference_id(correction_of_raw, "correction_of")
            
            # Validate against prior insights if available
            if prior_insights is not None:
                continuation_of = self._validate_insight_reference(
                    continuation_of, prior_insights, "continuation_of"
                )
                correction_of = self._validate_insight_reference(
                    correction_of, prior_insights, "correction_of"
                )
            
            # Assign ID using system function
            insight_id = self._window_manager._get_next_insight_id()
            
            # Create WindowInsight with system-assigned ID
            insight = WindowInsight(
                insight_id=insight_id,
                insight_type=item.get("insight_type", "NOTES"),
                insight_text=item.get("insight_text", ""),
                confidence=item.get("confidence", 0.0),
                window_id=window_id,
                timestamp_start=window_start,
                timestamp_end=window_end,
                classification=item.get("classification", "~"),
                continuation_of=continuation_of,
                correction_of=correction_of,
            )
            
            # Add insight to window via WindowManager
            result = self._window_manager.add_insight_to_window(window_id, insight)
            if result == -1:
                logger.error(
                    f"Failed to add insight {insight_id} (type={insight.insight_type}) to window {window_id}. "
                    f"Insight text: {insight.insight_text[:100]}..."
                )
            
            # Use as_dict() for clean export to summary
            insights_for_summary.append(insight.as_dict())
        
        # Update parsed_data with assigned insight_ids
        if is_list_format:
            # Return list format
            return insights_for_summary
        else:
            # Return dict format
            parsed_data["insights"] = insights_for_summary
            return parsed_data
    
    def reset(self):
        """Reset all accumulated state for a new stream."""
        self._window_manager.clear()
        self.in_flight_windows.clear()
        self._skipped_segments_buffer = None
        # Clear temp buffer state for fresh stream
        self._temp_segment_buffer.clear()
        self._temp_buffer_timing.clear()
        self._temp_buffer_window_ids.clear()
        self._transcription_window_counter = 0
        # Reset content type state
        self._content_type_state = ContentTypeState()
        self._user_content_type_override = None
        # Reset auto-detection trigger for new stream
        self._auto_detect_content_type_detection = True
        # Reset in-flight detection flag for new stream
        self._content_type_detection_in_progress = False
        # Reset raw response storage
        self._last_summary_raw_response = None
        # Reset last processed timestamp for new stream
        self._last_processed_timestamp = 0.0
        logger.info("SummaryClient reset complete - all state cleared for new stream")
    
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
        
        # Get sentiment_enabled from rules
        sentiment_enabled = False
        if content_type in CONTENT_TYPE_RULE_MODIFIERS:
            sentiment_enabled = CONTENT_TYPE_RULE_MODIFIERS[content_type].get("sentiment_enabled", False)
        
        self._content_type_state = ContentTypeState(
            content_type=content_type,
            confidence=confidence,
            source=source,
            sentiment_enabled=sentiment_enabled
        )
        logger.info(f"Content type set to: {content_type} (source: {source}, confidence: {confidence:.2f}, sentiment_enabled={sentiment_enabled})")
    
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
    
    def is_sentiment_enabled(self) -> bool:
        """
        Check if sentiment tracking is enabled for the current content type.
        
        Returns:
            True if sentiment insights should be included, False otherwise
        """
        return self._content_type_state.sentiment_enabled
    
    async def _send_monitoring_event(self, event_data: Dict[str, Any], event_type: str):
        """Send a monitoring event if callback is configured."""
        if self._send_monitoring_event_callback:
            try:
                await self._send_monitoring_event_callback(event_data, event_type)
            except Exception as e:
                logger.warning(f"Failed to send monitoring event: {e}")
    
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
        transcription_window_id: int,
        window_start: float,
        window_end: float
    ):
        """
        Store skipped segments for merging with next window.
        Also accumulates the transcription_window_id for traceability.
        
        Args:
            segments: List of transcription segments
            transcription_window_id: Unique identifier for this window
            window_start: Start timestamp
            window_end: End timestamp
        """
        # Get non-overlapping text
        new_text = self.get_new_text_for_summary_window(segments)
        
        if not new_text:
            return
                
        # Store for merging with next window
        self._skipped_segments_buffer = {
            "segments": segments,
            "transcription_window_id": transcription_window_id,
            "window_start": window_start,
            "window_end": window_end,
            "text": new_text
        }
        
        logger.info(f"Stored skipped window {transcription_window_id} for merging ({len(new_text)} chars)")
    
    async def process_content_type_detection(
        self,
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
        # Set in-progress flag to prevent concurrent detection requests
        self._content_type_detection_in_progress = True
        logger.debug("Content type detection started - in_progress flag set to True")
        
        try:
            # Check for user override first
            if self._user_content_type_override:
                logger.info(f"Skipping content type detection - user override active: {self._user_content_type_override}")
                return None
            
            # Store previous content type for change detection
            previous_content_type = self._content_type_state.content_type
            
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
                        max_tokens=self.max_tokens,
                        temperature=0.1,
                        response_format={"type": "json_schema", "json_schema": {"name": "content_type_detection", "schema": self.content_type_response_json_schema}},
                    )
                    
                    # Extract input tokens from response usage
                    input_tokens = 0
                    if response.usage:
                        input_tokens = response.usage.prompt_tokens or 0
                        logger.info(f"Content type detection input tokens: {input_tokens}")
                    
                    # Send monitoring event with input token count
                    await self._send_monitoring_event({
                        "input_tokens": input_tokens,
                        "context_length": context_length,
                        "window_start": window_start,
                        "window_end": window_end,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()
                    }, "content_type_detection_tokens")
                    
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
                    
                    # Get sentiment_enabled from rules
                    sentiment_enabled = False
                    if result.content_type in CONTENT_TYPE_RULE_MODIFIERS:
                        sentiment_enabled = CONTENT_TYPE_RULE_MODIFIERS[result.content_type].get("sentiment_enabled", False)
                    
                    # Update state with new context length and sentiment_enabled
                    self._content_type_state = ContentTypeState(
                        content_type=result.content_type,
                        confidence=result.confidence,
                        source=ContentTypeSource.AUTO_DETECTED.value,
                        last_detection_text=context_to_use,
                        context_length=context_length,
                        sentiment_enabled=sentiment_enabled
                    )
                    
                    logger.info(f"Content type detected: {result.content_type} (confidence: {result.confidence:.2f}, sentiment_enabled={sentiment_enabled})")
                    
                    # Check for content type change and send monitoring event
                    if previous_content_type != result.content_type:
                        # Create event data dict
                        event_data = {
                            "previous_content_type": previous_content_type,
                            "new_content_type": result.content_type,
                            "confidence": result.confidence,
                            "reasoning": getattr(result, 'reasoning', 'No reasoning provided'),
                            "context_length": context_length,
                            "source": ContentTypeSource.AUTO_DETECTED.value,
                            "window_start": window_start,
                            "window_end": window_end,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat()
                        }
                        
                        await self._send_monitoring_event(event_data, "content_type_changed")
                        logger.info(f"Sent content_type_changed monitoring event: {previous_content_type} -> {result.content_type}")
                    
                    return self._content_type_state
                
                # Return current state if LLM returns empty response
                return self._content_type_state
                
            except Exception as e:
                logger.error(f"Content type detection error: {e}")
                return self._content_type_state
        finally:
            # Always clear the in-progress flag, regardless of success or failure
            self._content_type_detection_in_progress = False
            logger.debug("Content type detection completed - in_progress flag set to False")
    
    def _parse_content_type_response(self, content: str) -> ContentTypeDetectionSchema:
        """
        Parse the content type detection response from LLM.
        
        Args:
            content: The raw response content from the LLM
            
        Returns:
            ContentTypeDetectionSchema with parsed content type, confidence, and reasoning
        """
        try:
            data = json.loads(content)
            content_type_str = data.get("content_type", ContentType.UNKNOWN.value)
            confidence = data.get("confidence", 0.0)
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Validate content_type is a valid enum value
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                logger.warning(f"Invalid content type '{content_type_str}', defaulting to UNKNOWN")
                content_type = ContentType.UNKNOWN
            
            return ContentTypeDetectionSchema(
                content_type=content_type,
                confidence=confidence,
                reasoning=reasoning
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse content type response as JSON: {e}")
            return ContentTypeDetectionSchema(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"JSON parse error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error parsing content type response: {e}")
            return ContentTypeDetectionSchema(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Parse error: {str(e)}"
            )
    