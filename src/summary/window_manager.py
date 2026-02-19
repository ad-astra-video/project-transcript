"""
Window management for transcription summarization.

Contains WindowInsight, TranscriptionWindow, SummaryWindow, and WindowManager classes.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


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


@dataclass
class TranscriptionWindow:
    """A transcription window with new text (deduplicated)."""
    transcription_window_id: int
    new_text: str  # Text after removing overlap from previous window
    timestamp_start: float
    timestamp_end: float
    segments: List[Dict[str, Any]]  # Original segments for reference
    char_count: int  # Length of new_text for context limit calculation


@dataclass
class SummaryWindow:
    """A summary window with text and insights from multiple transcription windows."""
    window_id: int
    text: str  # Merged text from transcription windows
    insights: List[WindowInsight]
    timestamp_start: float
    timestamp_end: float
    char_count: int  # Length of text for context limit calculation
    processed: bool = False  # Track if window has been processed by LLM
    transcription_window_ids: List[int] = field(default_factory=list)  # List of transcription window IDs that created this summary


class WindowManager:
    """Manages summary windows and their text/insights."""
    
    def __init__(
        self,
        context_limit: int = 50000,  # Max characters for accumulated text
        raw_text_context_limit: int = 1500,  # Max characters for raw text in LLM context
        transcription_windows_per_summary_window: int = 8  # Number of transcription windows per summary window
    ):
        self._summary_windows: List[SummaryWindow] = []  # Ordered oldest -> newest (RENAMED from _windows)
        self._transcription_windows: Dict[int, TranscriptionWindow] = {}  # Dict of transcription windows keyed by ID
        self._pending_transcription_ids: List[int] = []  # Track pending transcription IDs for summary creation
        self._char_count: int = 0
        self._next_window_id: int = 0
        self.context_limit = context_limit  # Max characters for accumulated text
        self.raw_text_context_limit = raw_text_context_limit  # Max characters for raw text in LLM context
        self.transcription_windows_per_summary_window = transcription_windows_per_summary_window  # Number of transcription windows per summary window
        self._first_window_timestamp: Optional[float] = None  # Track first window timestamp for self-contained delay logic
        self._next_insight_id: int = 0  # Counter for unique insight IDs
        self._last_processed_timestamp_end: Optional[float] = None  # Track last processed timestamp end for deduplication
    
    def add_summary_window(
        self,
        text: str,
        timestamp_start: float,
        timestamp_end: float,
        transcription_window_ids: List[int],
        window_id: Optional[int] = None
    ) -> int:
        """
        Add a new summary window, dropping oldest if over char limit.
        Also tracks first window timestamp for self-contained delay logic.
        
        Args:
            text: Text content for this window
            timestamp_start: Start timestamp in seconds
            timestamp_end: End timestamp in seconds
            transcription_window_ids: List of transcription window IDs that created this summary
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
        while new_char_count > self.context_limit and self._summary_windows:
            oldest = self._summary_windows.pop(0)
            self._char_count -= oldest.char_count
            new_char_count = self._char_count + len(text)
        
        # Create and add window
        window = SummaryWindow(
            window_id=actual_window_id,
            text=text,
            insights=[],
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            char_count=len(text),
            transcription_window_ids=transcription_window_ids
        )
        self._summary_windows.append(window)
        self._char_count += len(text)

        logger.debug(f"Added summary window {actual_window_id}, char_count={self._char_count}, total_windows={len(self._summary_windows)}")

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
        if len(self._summary_windows) <= 1:
            logger.debug(f"Not enough windows for accumulation: {len(self._summary_windows)} <= 1")
            return "", [], 0, 0.0
        
        # All windows except the last one (current window being analyzed)
        accumulated_windows = self._summary_windows[:-1]
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
        if not self._summary_windows:
            return ""
        
        # Get text from all windows, newest first
        all_text_parts = [w.text for w in self._summary_windows]
        return " ".join(all_text_parts)
    
    def get_recent_windows_text(self, char_limit: int) -> str:
        """
        Get text from recent windows up to the specified character limit.
        
        Iterates through summary windows in reverse order (newest first),
        adding text until the character limit is exceeded. If a window exceeds
        the remaining limit, it is added in full and the method returns.
        
        Args:
            char_limit: Maximum number of characters to include
            
        Returns:
            Concatenated text from recent windows, newest to oldest, up to char_limit
        """
        if not self._summary_windows:
            return ""
        
        result_parts = []
        total_chars = 0
        
        # Iterate through windows in reverse order (newest first)
        for window in reversed(self._summary_windows):
            window_text = window.text
            window_chars = len(window_text)
            
            # Add the window text (even if it exceeds limit, we include it)
            result_parts.append(window_text)
            total_chars += window_chars
            
            # Stop if we've reached or exceeded the limit
            if total_chars >= char_limit:
                break
        
        # Return joined text (newest first order)
        return " ".join(result_parts)
    
    def get_text_and_window_ids_since_timestamp(self, timestamp: float) -> tuple[str, List[int]]:
        """Get text and window IDs from all windows with timestamp_start >= given timestamp.
        
        Args:
            timestamp: The timestamp to get text and IDs since
            
        Returns:
            Tuple of (concatenated_text, list_of_window_ids)
        """
        text_parts = []
        window_ids = []
        for window in self._summary_windows:
            if window.timestamp_start >= timestamp:
                if window.text:
                    text_parts.append(window.text)
                window_ids.append(window.window_id)
        return " ".join(text_parts), window_ids
    
    def get_window_insights(self, window_id: int) -> List[WindowInsight]:
        """Get insights for a specific window."""
        for window in self._summary_windows:
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
        for window in reversed(self._summary_windows):
            if window.window_id == window_id:
                window.insights.append(insight)
                logger.debug(f"Added insight {insight.insight_id} to window {window_id}")
                return insight.insight_id
        
        # Window not found - log error with diagnostic info
        available_ids = [w.window_id for w in self._summary_windows]
        logger.error(
            f"Failed to add insight {insight.insight_id} to window {window_id} - window not found. "
            f"Available window IDs: {available_ids}, Total windows: {len(self._summary_windows)}"
        )
        return -1  # Window not found
    
    def clear(self):
        """Clear all windows and reset internal counters for fresh stream."""
        self._summary_windows.clear()
        self._transcription_windows.clear()
        self._pending_transcription_ids.clear()
        self._char_count = 0
        # Reset all internal counters for fresh stream
        self._next_window_id = 0
        self._next_insight_id = 0
        self._first_window_timestamp = None
        logger.debug("WindowManager cleared - all counters reset")
    
    def __len__(self):
        return len(self._summary_windows)
    
    def get_unprocessed_text(self) -> str:
        """
        Get text from all unprocessed windows, concatenated.
        Used when waiting for initial delay - all windows are unprocessed.
        
        Returns:
            Concatenated text from all unprocessed windows
        """
        if not self._summary_windows:
            return ""
        
        # Get text from all unprocessed windows
        unprocessed_text_parts = []
        for window in self._summary_windows:
            if not window.processed:
                unprocessed_text_parts.append(window.text)
        
        return " ".join(unprocessed_text_parts)
    
    def mark_all_windows_processed(self):
        """Mark all current windows as processed. Called after first summary."""
        for window in self._summary_windows:
            window.processed = True
    
    # === Transcription Window Management Methods ===
    
    def _extract_text_from_segments(self, segments: List[Dict[str, Any]]) -> str:
        """Extract text from transcription segments."""
        return " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
    
    def _deduplicate_text(self, segments: List[Dict[str, Any]]) -> str:
        """Remove overlapping text from current transcription."""
        # First window - no deduplication needed
        if self._last_processed_timestamp_end is None:
            return self._extract_text_from_segments(segments)
        
        # Check if word timestamps available
        has_timestamps = any(
            isinstance(seg.get("words"), list) and len(seg.get("words", [])) > 0
            for seg in segments
        )
        
        if has_timestamps:
            return self._deduplicate_with_timestamps(segments)
        else:
            return self._deduplicate_with_word_matching(segments)
    
    def _deduplicate_with_timestamps(self, segments: List[Dict[str, Any]]) -> str:
        """Filter words where start_ms/1000 < _last_processed_timestamp_end."""
        ref_ts = self._last_processed_timestamp_end
        
        filtered_segments = []
        for seg in segments:
            if not isinstance(seg.get("words"), list):
                filtered_segments.append(seg)
                continue
            
            filtered_words = []
            for word in seg["words"]:
                word_start = word.get("start_ms", 0) / 1000
                if word_start >= ref_ts:
                    filtered_words.append(word)
            
            new_seg = dict(seg)
            new_seg["words"] = filtered_words
            filtered_segments.append(new_seg)
        
        return self._extract_text_from_segments(filtered_segments)
    
    def _deduplicate_with_word_matching(self, segments: List[Dict[str, Any]]) -> str:
        """Fallback word matching deduplication."""
        # Get last transcription window id
        pending_ids = self.get_pending_transcription_ids()
        last_id = None
        
        if pending_ids:
            last_id = pending_ids[-1]
        elif self._summary_windows:
            last_summary = self._summary_windows[-1]
            trans_ids = last_summary.transcription_window_ids
            if trans_ids:
                last_id = trans_ids[-1]
        
        if last_id is None:
            return self._extract_text_from_segments(segments)
        
        last_window = self._transcription_windows.get(last_id)
        if not last_window:
            return self._extract_text_from_segments(segments)
        
        previous_text = last_window.new_text
        current_text = self._extract_text_from_segments(segments)
        
        if not previous_text or not current_text:
            return current_text
        
        prev_words = previous_text.split()
        curr_words = current_text.split()
        
        # Find overlap
        overlap_count = 0
        for i in range(1, min(len(prev_words), len(curr_words)) + 1):
            if prev_words[-i].lower() == curr_words[i-1].lower():
                overlap_count = i
            else:
                break
        
        if overlap_count > 0:
            return " ".join(curr_words[overlap_count:])
        
        return current_text
    
    def add_transcription_window(
        self,
        transcription_window_id: int,
        segments: List[Dict[str, Any]],
        window_start_ts: float,
        window_end_ts: float
    ) -> None:
        """
        Add a transcription window to the dict and track ID for summary creation.
        
        Args:
            transcription_window_id: Unique ID for this transcription window
            segments: Original segments from transcription
            window_start_ts: Start timestamp in seconds
            window_end_ts: End timestamp in seconds
        """
        # Extract and deduplicate text
        new_text = self._deduplicate_text(segments)
        
        trans_window = TranscriptionWindow(
            transcription_window_id=transcription_window_id,
            new_text=new_text,
            timestamp_start=window_start_ts,
            timestamp_end=window_end_ts,
            segments=segments,
            char_count=len(new_text)
        )
        self._transcription_windows[transcription_window_id] = trans_window
        self._pending_transcription_ids.append(transcription_window_id)
        
        # Update last processed timestamp end
        self._last_processed_timestamp_end = window_end_ts
        
        logger.debug(f"Added transcription window {transcription_window_id}, pending count={len(self._pending_transcription_ids)}")
    
    def get_transcription_window(self, transcription_window_id: int) -> Optional[TranscriptionWindow]:
        """Get a transcription window by ID."""
        return self._transcription_windows.get(transcription_window_id)
    
    def get_pending_transcription_ids(self) -> List[int]:
        """Get list of pending transcription window IDs."""
        return self._pending_transcription_ids.copy()
    
    def clear_pending_transcription_ids(self) -> None:
        """Clear pending transcription IDs after summary window is created."""
        self._pending_transcription_ids.clear()
    
    def maybe_create_summary_window(self) -> Optional[int]:
        """Create summary window if threshold reached, clear pending IDs."""
        if len(self._pending_transcription_ids) >= self.transcription_windows_per_summary_window:
            transcription_ids = self._pending_transcription_ids.copy()
            
            merged_text = self.get_merged_text_for_transcription_ids(transcription_ids)
            timestamp_start, timestamp_end = self.get_merged_timing_for_transcription_ids(transcription_ids)
            
            window_id = self.add_summary_window(
                text=merged_text,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                transcription_window_ids=transcription_ids
            )
            
            self.clear_pending_transcription_ids()
            return window_id
        
        return None
    
    def get_transcription_windows_for_ids(self, transcription_window_ids: List[int]) -> List[TranscriptionWindow]:
        """Get transcription windows for a list of IDs."""
        return [
            self._transcription_windows[tid]
            for tid in transcription_window_ids
            if tid in self._transcription_windows
        ]
    
    def get_merged_text_for_transcription_ids(self, transcription_window_ids: List[int]) -> str:
        """Get merged text from multiple transcription windows."""
        windows = self.get_transcription_windows_for_ids(transcription_window_ids)
        # Sort by timestamp to maintain order
        windows.sort(key=lambda w: w.timestamp_start)
        return " ".join(w.new_text for w in windows if w.new_text)
    
    def get_merged_timing_for_transcription_ids(self, transcription_window_ids: List[int]) -> tuple[float, float]:
        """Get merged start/end timestamps from multiple transcription windows."""
        windows = self.get_transcription_windows_for_ids(transcription_window_ids)
        if not windows:
            return 0.0, 0.0
        windows.sort(key=lambda w: w.timestamp_start)
        return windows[0].timestamp_start, windows[-1].timestamp_end
    
    def get_window_transcription_ids(self, window_id: int) -> List[int]:
        """Get transcription window IDs for a summary window."""
        for window in self._summary_windows:
            if window.window_id == window_id:
                return window.transcription_window_ids
        return []
    
    def get_window_start(self, window_id: int) -> float:
        """Get start timestamp for a summary window."""
        for window in self._summary_windows:
            if window.window_id == window_id:
                return window.timestamp_start
        return 0.0
    
    def get_window_end(self, window_id: int) -> float:
        """Get end timestamp for a summary window."""
        for window in self._summary_windows:
            if window.window_id == window_id:
                return window.timestamp_end
        return 0.0
    
    def get_window_text(self, window_id: int) -> str:
        """Get text for a specific summary window."""
        for window in self._summary_windows:
            if window.window_id == window_id:
                return window.text
        return ""
    
    def get_transcription_text_since_timestamp(self, timestamp: float) -> str:
        """Get text from all transcription windows with timestamp_start >= given timestamp."""
        text_parts = []
        for tw in sorted(self._transcription_windows.values(), key=lambda w: w.timestamp_start):
            if tw.timestamp_start >= timestamp:
                if tw.new_text:
                    text_parts.append(tw.new_text)
        return " ".join(text_parts)