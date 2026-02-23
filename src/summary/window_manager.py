"""
Window management for transcription summarization.

Contains TranscriptionWindow, SummaryWindow, and WindowManager classes.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


# ==================== Constants for Plugin Result Size Limits ====================

MAX_PLUGIN_RESULTS_PER_WINDOW = 50
MAX_PLUGIN_RESULT_SIZE = 1024 * 1024 * 5  # 5MB
MAX_PLUGIN_NAME_LENGTH = 64


# ==================== Custom Exceptions ====================

class WindowManagerError(Exception):
    """Base exception for WindowManager errors."""
    pass


class WindowNotFoundError(WindowManagerError):
    """Raised when a window lookup fails."""
    pass


class PluginResultError(WindowManagerError):
    """Raised when plugin result operations fail."""
    pass


class FormatCallbackError(WindowManagerError):
    """Raised when a format callback fails."""
    pass


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
    """A summary window with text from multiple transcription windows."""
    window_id: int
    text: str  # Merged text from transcription windows
    timestamp_start: float
    timestamp_end: float
    char_count: int  # Length of text for context limit calculation
    processed: bool = False  # Track if window has been processed by LLM
    transcription_window_ids: List[int] = field(default_factory=list)  # List of transcription window IDs that created this summary
    
    # Plugin results storage
    plugin_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def store_result(
        self,
        plugin_name: str,
        result: Any,
        include_in_context: bool = True,
    ) -> None:
        """Store a plugin's result in this window."""
        # Validate plugin_name
        if not plugin_name or not isinstance(plugin_name, str):
            raise PluginResultError(
                f"Invalid plugin_name: {plugin_name!r}. Must be non-empty string."
            )
        
        # Validate result size to prevent memory issues
        result_size = len(str(result))
        if result_size > MAX_PLUGIN_RESULT_SIZE:
            raise PluginResultError(
                f"Plugin result too large: {result_size} bytes > {MAX_PLUGIN_RESULT_SIZE} bytes"
            )
        
        self.plugin_results[plugin_name] = {
            "data": result,
            "include_in_context": include_in_context,
        }
    
    def get_result(self, plugin_name: str) -> Optional[Any]:
        """Get a plugin's result from this window."""
        plugin_data = self.plugin_results.get(plugin_name)
        if plugin_data:
            return plugin_data.get("data")
        return None
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get all plugin results from this window."""
        return {
            name: data.get("data")
            for name, data in self.plugin_results.items()
        }
    
    def format_result_for_context(
        self,
        data: Any,
        format_callback: Optional[Callable[[Any], str]] = None,
    ) -> Optional[str]:
        """Format plugin data for context using provided callback."""
        if not data:
            return None
        
        if format_callback:
            try:
                return format_callback(data)
            except Exception as e:
                logger.error(
                    "Format callback failed for plugin data: %s",
                    str(e),
                    exc_info=True
                )
                return None
        
        # Default: JSON dump if no callback
        try:
            return json.dumps(data, indent=2)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to serialize plugin result to JSON: %s",
                str(e)
            )
            return str(data)  # Fallback to string conversion


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
        self._window_id_to_index: Dict[int, int] = {}  # Maps window_id -> list index for O(1) lookup
        self._pending_transcription_ids: List[int] = []  # Track pending transcription IDs for summary creation
        self._char_count: int = 0
        self._next_window_id: int = 0
        self.context_limit = context_limit  # Max characters for accumulated text
        self.raw_text_context_limit = raw_text_context_limit  # Max characters for raw text in LLM context
        self.transcription_windows_per_summary_window = transcription_windows_per_summary_window  # Number of transcription windows per summary window
        self._first_window_timestamp: Optional[float] = None  # Track first window timestamp for self-contained delay logic
        self._last_processed_timestamp_end: Optional[float] = None  # Track last processed timestamp end for deduplication
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Plugin format callbacks registry
        self._plugin_format_callbacks: Dict[str, Callable[[Any], str]] = {}
    
    # Register plugin format callback
    def register_plugin_format_callback(
        self,
        plugin_name: str,
        callback: Callable[[Any], str],
    ) -> None:
        """Register a format callback for a plugin."""
        with self._lock:
            self._plugin_format_callbacks[plugin_name] = callback
    
    # Get window by ID (O(1) lookup)
    def get_window(self, window_id: int) -> Optional[SummaryWindow]:
        """Get a summary window by ID using O(1) lookup."""
        index = self._window_id_to_index.get(window_id)
        
        if index is None:
            logger.warning(
                "Window ID %d not found in index. "
                "Window may never have existed or was dropped.",
                window_id
            )
            return None
        
        if not 0 <= index < len(self._summary_windows):
            logger.error(
                "Window index %d out of bounds for %d windows. "
                "Index may be stale.",
                index, len(self._summary_windows)
            )
            return None
        
        return self._summary_windows[index]
    
    def store_plugin_result(
        self,
        window_id: int,
        plugin_name: str,
        result: Any,
        include_in_context: bool = True,
    ) -> bool:
        """Store a plugin result in a window.
        
        Convenience method that combines get_window and store_result.
        
        Args:
            window_id: The ID of the summary window
            plugin_name: Name of the plugin storing the result
            result: The result data to store
            include_in_context: Whether to include in AI context
            
        Returns:
            True if successful, False if window not found
        """
        with self._lock:
            window = self.get_window(window_id)
            if window:
                window.store_result(
                    plugin_name=plugin_name,
                    result=result,
                    include_in_context=include_in_context,
                )
                return True
            return False
    
    # Get accumulated text and results with per-plugin token limits
    def get_accumulated_text_and_results(
        self,
        text_token_limit: int = 0,
        result_types: Optional[List[str]] = None,
        result_token_limit: Optional[Dict[str, int]] = None
    ) -> Tuple[str, Dict[str, List[str]], int, Dict[str, float]]:
        """
        Get accumulated text and plugin results from summary windows.
        
        Args:
            text_token_limit: Max tokens for accumulated text
            result_types: Optional list of plugin names to include
            result_token_limit: Optional dict of {plugin_name: token_limit}.
                               If None, all results are included.
        
        Returns:
            Tuple of (accumulated_text, plugin_results_dict, text_token_count, results_per_window_dict)

            plugin_results_dict format: {plugin_name: [formatted_result_text, ...], ...}
            results_per_window_dict format: {plugin_name: results_per_window_ratio, ...}
        """
        try:
            with self._lock:
                accumulated_text_parts = []
                text_token_count = 0
                plugin_results: Dict[str, List[str]] = {}  # plugin_name -> list of formatted texts
                
                # Track result counts per plugin for metrics
                plugin_result_counts: Dict[str, int] = {}
                
                # Track token usage per plugin (only used when result_token_limit is provided)
                plugin_token_counts: Dict[str, int] = {}
                
                for window in reversed(self._summary_windows):
                    if text_token_limit > 0:
                        # Estimate tokens for this window's text
                        window_tokens = window.char_count // 4
                        
                        # Check if adding this window would exceed text limit
                        if text_token_limit == 0 or text_token_count + window_tokens <= text_token_limit:
                            accumulated_text_parts.insert(0, window.text)
                            text_token_count += window_tokens
                        else:
                            # at limit of context, exit loop
                            break
                    
                    # if not requesting results from plugins then skip
                    if result_types is None:
                        continue

                    # Collect plugin results for this window (regardless of text limit)
                    if window.plugin_results:
                        for plugin_name, data in window.plugin_results.items():
                            if result_types and plugin_name not in result_types:
                                continue
                            
                            plugin_data = data.get("data")
                            include_flag = data.get("include_in_context", True)
                            
                            if not include_flag or not plugin_data:
                                continue
                            
                            callback = self._plugin_format_callbacks.get(plugin_name)
                            formatted = window.format_result_for_context(plugin_data, callback)
                            
                            if formatted:
                                result_tokens = len(formatted) // 4
                                
                                # Check per-plugin token limit if provided
                                if result_token_limit is not None:
                                    plugin_limit = result_token_limit.get(plugin_name)
                                    if plugin_limit is not None:
                                        current_count = plugin_token_counts.get(plugin_name, 0)
                                        if current_count + result_tokens > plugin_limit:
                                            continue
                                
                                # Append to list (not overwrite)
                                if plugin_name not in plugin_results:
                                    plugin_results[plugin_name] = []
                                    plugin_token_counts[plugin_name] = 0
                                    plugin_result_counts[plugin_name] = 0
                                plugin_results[plugin_name].append(formatted)
                                plugin_result_counts[plugin_name] += 1
                                if result_token_limit is not None:
                                    plugin_token_counts[plugin_name] += result_tokens
                
                accumulated_text = "\n".join(accumulated_text_parts)
                
                # Calculate results per window for each plugin
                window_count = len(self._summary_windows) if self._summary_windows else 1
                results_per_window = {
                    plugin_name: count / window_count
                    for plugin_name, count in plugin_result_counts.items()
                }
                
                return accumulated_text, plugin_results, text_token_count, results_per_window
        except RuntimeError as e:
            logger.error("Runtime error during accumulation: %s", str(e))
            return "", {}, 0, {}
        except Exception as e:
            logger.error("Unexpected error during accumulation: %s", str(e))
            return "", {}, 0, {}
    
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
        with self._lock:
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
            
            # Rebuild index after dropping windows (indices have shifted)
            if self._window_id_to_index:
                self._window_id_to_index = {
                    w.window_id: i
                    for i, w in enumerate(self._summary_windows)
                }
            
            # Create and add window
            window = SummaryWindow(
                window_id=actual_window_id,
                text=text,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                char_count=len(text),
                transcription_window_ids=transcription_window_ids
            )
            self._summary_windows.append(window)
            self._char_count += len(text)
            
            # Add new window to index dictionary
            self._window_id_to_index[actual_window_id] = len(self._summary_windows) - 1

            logger.debug(f"Added summary window {actual_window_id}, char_count={self._char_count}, total_windows={len(self._summary_windows)}")

            return actual_window_id
    
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
    
    def clear(self):
        """Clear all windows and reset internal counters for fresh stream."""
        self._summary_windows.clear()
        self._transcription_windows.clear()
        self._window_id_to_index.clear()
        self._pending_transcription_ids.clear()
        self._char_count = 0
        # Reset all internal counters for fresh stream
        self._next_window_id = 0
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
        idx = self._window_id_to_index.get(window_id)
        if idx is not None:
            return self._summary_windows[idx].transcription_window_ids
        return []
    
    def get_window_start(self, window_id: int) -> float:
        """Get start timestamp for a summary window."""
        idx = self._window_id_to_index.get(window_id)
        if idx is not None:
            return self._summary_windows[idx].timestamp_start
        return 0.0
    
    def get_window_end(self, window_id: int) -> float:
        """Get end timestamp for a summary window."""
        idx = self._window_id_to_index.get(window_id)
        if idx is not None:
            return self._summary_windows[idx].timestamp_end
        return 0.0
    
    def get_window_text(self, window_id: int) -> str:
        """Get text for a specific summary window."""
        idx = self._window_id_to_index.get(window_id)
        if idx is not None:
            return self._summary_windows[idx].text
        return ""
    
    def get_transcription_text_since_timestamp(self, timestamp: float) -> str:
        """Get text from all transcription windows with timestamp_start >= given timestamp."""
        text_parts = []
        for tw in sorted(self._transcription_windows.values(), key=lambda w: w.timestamp_start):
            if tw.timestamp_start >= timestamp:
                if tw.new_text:
                    text_parts.append(tw.new_text)
        return " ".join(text_parts)