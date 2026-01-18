"""
Summary client for LLM-based transcription cleaning and summarization.
"""

import asyncio
import json
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SummarySegment:
    """Represents a summarized/cleaned segment."""
    original_text: str
    cleaned_text: str
    summary: str
    timestamp_start: float
    timestamp_end: float
    speaker: str | None = None


@dataclass
class WindowInsight:
    """Insight extracted from a summary window."""
    insight_type: str  # ACTION, DECISION, QUESTION, FACT, RISK
    text: str
    window_id: int  # Which summary window this belongs to
    timestamp_start: float
    timestamp_end: float


@dataclass
class SummaryWindow:
    """A 5-second summary window with text and insights."""
    window_id: int
    text: str  # Non-overlapping text for this window
    insights: List[WindowInsight]
    timestamp_start: float
    timestamp_end: float
    char_count: int  # Length of text for context limit calculation


class WindowManager:
    """Manages summary windows and their text/insights."""
    
    def __init__(self, max_chars: int = 100000):
        self._windows: List[SummaryWindow] = []  # Ordered oldest -> newest
        self._char_count: int = 0
        self._next_window_id: int = 0
        self.max_chars = max_chars
    
    def add_window(self, text: str, timestamp_start: float, timestamp_end: float) -> int:
        """
        Add a new window, dropping oldest if over char limit.
        
        Args:
            text: Text content for this window
            timestamp_start: Start timestamp in seconds
            timestamp_end: End timestamp in seconds
        
        Returns:
            window_id of the added window
        """
        window_id = self._next_window_id
        
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
        self._char_count = len(text)  # Reset to current window only
        self._next_window_id += 1
        
        return window_id
    
    def get_accumulated_text(self) -> str:
        """Get all window text concatenated."""
        return " ".join(w.text for w in self._windows)
    
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


class SummaryClient:
    """Client for LLM-based transcription cleaning and summarization."""
    
    def __init__(
        self,
        base_url: str = "http://byoc-transcription-vllm:5000/v1",
        api_key: str = "",
        history_length: int = 0,
        model: Optional[str] = None,
        max_tokens: int = 3072,
        temperature: float = 0.0,
        system_prompt: str = """
You are a real-time conversation intelligence engine.

You receive an ongoing stream of imperfect speech-to-text transcription. The text may be partial, corrected later, out of order, or lack punctuation.

Your task is to continuously extract high-signal insights from the conversation, prioritizing:

- Action items and commitments
- Decisions and agreements
- Open questions and unresolved issues
- Key facts, numbers, dates, and names
- Shifts in topic, intent, or sentiment

Guidelines:
- Do not summarize everything. Surface only what materially changes understanding or next steps.
- Treat statements as tentative until reinforced or confirmed later.
- Prefer concise, atomic insights over long prose.
- Update or invalidate earlier insights if new information contradicts them.
- Ignore filler speech, false starts, and verbal noise unless it affects meaning.
- When unsure, mark confidence as low rather than guessing. Do not return RISK because of uncertainty.

Output Format:
- Clearly label insight type (e.g. ACTION, DECISION, QUESTION, FACT, RISK)
- JSON should be list of insights in format example [{"type": "ACTION", "text": "insight text", "confidence": 0.90}, {"type": "QUESTION", "text": "insight text", "confidence": 0.20}]
- Only output valid JSON and one json object per response after thinking briefly
- keys in the json object should always be lowercase

You are optimized for live understanding, not post-hoc summarization. Think briefly.
""".strip()
    ):
        """
        Initialize the summary client.
        
        Args:
            base_url: Base URL for the OpenAI-compatible API
            api_key: API key for authentication
            history_length: Number of previous segments to include in context (0 = all history)
            model: Model name to use for summarization (None to fetch from /models endpoint)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: System prompt for the LLM
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.history_length = history_length
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        # Window-based state management
        self._window_manager: WindowManager = WindowManager()
        
        # Track last processed timestamp per window
        self._window_last_timestamp: Dict[int, float] = {}
        
        # Track whether we've performed the first summary call
        self._has_performed_summary: bool = False
        
        # Lock for thread safety (though we're single-threaded in async)
        self._lock: Optional[asyncio.Lock] = None
        
        # Concurrency limiter for summary calls (set via env var)
        try:
            max_concurrent = int(os.getenv("MAX_CONCURRENT_SUMMARIES", "4"))
        except Exception:
            max_concurrent = 4
        self.max_concurrent_summaries: int = max(1, max_concurrent)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrent_summaries)
    
    async def initialize(self):
        """
        Initialize the HTTP client and fetch the model from the /models endpoint.
        
        This method creates an HTTP client, calls the /models endpoint to get available models,
        and sets the model to the first available model or the default one.
        """
        logger.info("SummaryClient.initialize called")
        
        try:
            # Create a fresh session for this request to avoid event loop issues
            connector = aiohttp.TCPConnector(
                limit=100, 
                limit_per_host=30, 
                enable_cleanup_closed=True,
                use_dns_cache=False
            )
            timeout = aiohttp.ClientTimeout(total=None)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as client:
                async with client.get(f"{self.base_url}/models") as response:
                    response.raise_for_status()
                    result = await response.json()
            logger.info(f"Received models from {self.base_url}/models")
            
            #setup lock
            if self._lock is None:
                self._lock = asyncio.Lock()
            
            # Extract model from response - typically a list of models
            if isinstance(result, list) and len(result) > 0:
                self.model = result[0]
            elif isinstance(result, dict):
                # Handle common response formats
                if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                    # OpenAI-compatible format: {"data": [{"id": "model_name", ...}]}
                    self.model = result["data"][0].get("id", result["data"][0].get("object", ""))
                elif "default" in result:
                    self.model = result["default"]
                elif "model" in result:
                    self.model = result["model"]
                else:
                    # Try to get first value from dict
                    values = list(result.values())
                    if values and isinstance(values[0], str):
                        self.model = values[0]
            
            logger.info(f"SummaryClient initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Error fetching model from /models endpoint: {e}")
            # Fallback to a default model if available
            if self.model is None:
                self.model = "Qwen/Qwen3-VL-4B-Instruct"
            logger.warning(f"Using fallback model: {self.model}")
    
    def update_params(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        history_length: Optional[int] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
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
            return f"\nRecent Transcript:\n{accumulated_text}"
        return ""
    
    async def summarize_text(self, text: str, context: str = "") -> str:
        """
        Send text to LLM for cleaning and summarization.
        
        Args:
            text: Text to summarize/clean
            context: Additional context from previous segments
            
        Returns:
            Cleaned and summarized text
        """
        logger.info(f"summarize_text called with text length={len(text)}, context length={len(context)}")
        
        async def _do_request() -> str:
            # Create a fresh session for this request to avoid event loop issues
            # This ensures the session is created in the current event loop context
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                enable_cleanup_closed=True,
                use_dns_cache=False
            )
            timeout = aiohttp.ClientTimeout(total=None)
            
            try:
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as client:
                    messages = []
                    
                    # System prompt
                    messages.append({
                        "role": "system",
                        "content": self.system_prompt
                    })
                    
                    # Context from history
                    if context:
                        messages.append({
                            "role": "system",
                            "content": f"Previous context:\n{context}"
                        })
                    
                    # User message with text to clean
                    messages.append({
                        "role": "user",
                        "content": f"Analyze the following transcript text and report only new or changed insights since the previous context.:\n\n{text}"
                    })
                    
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    
                    logger.info(f"Sending to LLM: {self.base_url}/chat/completions messages: {messages} ")
                    
                    # Make HTTP request without timeout - rely on outer timeout in main.py
                    async with client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                    
                    # Extract summary text from response
                    if "choices" in result and len(result["choices"]) > 0:
                        summary_text = result["choices"][0]["message"]["content"]
                        summary_text_results = summary_text.split("</think>")
                        if len(summary_text_results) > 1:
                            summary_text = summary_text_results[-1].strip()
                        summary_text = summary_text.replace("```json", "").replace("```", "").strip()

                        logger.info(f"summarize_text received response, length={len(summary_text)}")
                        return summary_text.strip()
                    
                    logger.info("summarize_text received empty response")
                    return ""
                    
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error from LLM API: {e.status} - {e.message}")
                raise
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
        
        # Add window to WindowManager (automatically drops oldest if over char limit)
        self._window_manager.add_window(new_text, window_start, window_end)
        
        # Get context from accumulated text
        context = self._build_context()
        
        # If this is the first summary, require at least 5 words
        accumulated_text = self._window_manager.get_accumulated_text()
        word_count = len(accumulated_text.split())
        if not self._has_performed_summary and word_count < 5:
            logger.info(f"Skipping first summary - only {word_count} accumulated words (need >=5)")
            return []
        
        # Send new text + accumulated context to LLM
        logger.info(f"Sending {len(new_text)} chars new text + {len(accumulated_text)} chars context to LLM")
        summary_text = await self.summarize_text(new_text, context)
        
        logger.info(f"Processed window {window_id}, summary length={len(summary_text)}")
        
        # Extract insights and add to window
        if summary_text:
            insights = self._extract_insights(summary_text, window_id, window_start, window_end)
            for insight in insights:
                self._window_manager.add_insight_to_window(window_id, insight)
        
        # Create summary segment
        summary_segment = SummarySegment(
            original_text=accumulated_text,
            cleaned_text=accumulated_text,
            summary=summary_text,
            timestamp_start=window_start,
            timestamp_end=window_end,
            speaker=segments[0].get("speaker") if segments else None
        )
        
        # Mark that we have performed at least one summary
        self._has_performed_summary = True
        
        logger.info(f"Returning summary segment for window {window_id}")
        
        return [summary_segment]
    
    def _extract_insights(
        self,
        summary_text: str,
        window_id: int,
        window_start: float,
        window_end: float
    ) -> List[WindowInsight]:
        """
        Extract insights from summary text and return as WindowInsight objects.
        
        Args:
            summary_text: Text returned from LLM
            window_id: The window these insights belong to
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
        
        Returns:
            List of WindowInsight objects
        """
        insights = []
        
        if not summary_text:
            return insights
        
        try:
            # Try to parse as JSON array
            data = json.loads(summary_text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        insight_type = item.get("type", "FACT")
                        insight_text = item.get("text", "") or item.get("insight", "")
                        if insight_text:
                            insights.append(WindowInsight(
                                insight_type=insight_type,
                                text=insight_text,
                                window_id=window_id,
                                timestamp_start=window_start,
                                timestamp_end=window_end
                            ))
            elif isinstance(data, dict):
                # Single object with insight types
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                insight_type = item.get("type", "FACT")
                                insight_text = item.get("text", "") or item.get("insight", "")
                                if insight_text:
                                    insights.append(WindowInsight(
                                        insight_type=insight_type,
                                        text=insight_text,
                                        window_id=window_id,
                                        timestamp_start=window_start,
                                        timestamp_end=window_end
                                    ))
        except json.JSONDecodeError:
            # Fallback: try to extract bullet points from text
            lines = summary_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                    text = line.lstrip('-*•').strip()
                    if text:
                        insights.append(WindowInsight(
                            insight_type="FACT",
                            text=text,
                            window_id=window_id,
                            timestamp_start=window_start,
                            timestamp_end=window_end
                        ))
        
        return insights
    
    def reset(self):
        """Reset all accumulated state."""
        self._window_manager.clear()
        self._window_last_timestamp.clear()
        self._has_performed_summary = False
        logger.info("SummaryClient state reset")