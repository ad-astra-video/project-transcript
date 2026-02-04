"""
Summary client for LLM-based transcription cleaning and summarization.
"""

import asyncio
import json
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from enum import Enum

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
    classification: str = "[~]"

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
    POSITIVE = "[+]"
    NEUTRAL = "[~]"
    NEGATIVE = "[-]"

class MessageFormatMode(str, Enum):
    """Message format modes for different LLM providers."""
    SYSTEM_PROMPT = "system"  # Use system role for system prompt
    USER_PREFIX = "user"      # Convert system prompt to user message with prefix

class InsightResponseItemSchema(BaseModel):
    """Schema for a single insight item."""
    insight_type: InsightType
    insight_text: str
    confidence: float
    classification: ClassificationField = ClassificationField.NEUTRAL

class InsightsResponseSchema(BaseModel):
    """Schema for insights response from LLM."""
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


class WindowManager:
    """Manages summary windows and their text/insights."""
    
    def __init__(self, max_chars: int = 5000):
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
        return " ".join(w.text for w in self._windows[:-2])
    
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
        api_key: str = "",
        base_url: str = "http://byoc-transcription-vllm:5000/v1",
        history_length: int = 0,
        model: str = "Nanbeige/Nanbeige4-3B-Thinking-2511",
        max_tokens: int = 3072,
        temperature: float = 0.0,
        system_prompt: str = """
You are a high-performance conversation intelligence engine optimized for REAL-TIME processing streams. You receive continuous, imperfect speech-to-text output that may include: fragmented sentences, partial transcripts, future corrections, out-of-order segments, missing punctuation, and speaker changes. Your task is to continuously extract the most critical insights with minimal latency while maintaining accuracy across stream continuity.

**Core Functionality:** Process each new transcription segment immediately as it arrives, prioritizing actionable intelligence over detailed analysis. Prioritize extracting:

1. **ACTION**: Concrete next steps, deadlines, responsible parties (with clear owners) and required actions ("Buy X by Y date" vs "Need to buy X"). Note if future, prior or step in process action.
2. **DECISION**: Final or conditional agreements, approvals, and commitments that change meeting outcomes ("We'll proceed with Plan B," "This requires CEO approval by Friday")
3. **QUESTION**: Critical blockers needing resolution (NOT just open questions), dependencies, risks requiring escalation. Should be able to be followed by an ACTION or DESCISION.
4. **KEY POINT**: Quantifiable data points essential for records or comparisons (dates, amounts, names of key stakeholders)
5. **SENTIMENT**: when detected in real-time conversations between participants - include detail on topic pivots if relevant to the sentiment, can include emotional changes when shifting
6. **RISK**: only when failure is time-bound or blocking a committed ACTION or DECISION
7. **NOTES**: used to keep a running summary log of the conversation for context. Notes should be frequent where conversation is providing new information that is not filler. Some examples are listing speakers, topics started/pivoted from, general understanding of the discussion

**Critical Real-Time Guidelines:**
**Stream Continuity First** - Assume transcript segments may arrive out-of-order or with gaps. Reference context from previous messages to fill blanks where possible (confidence flags must be adjusted)  
**Update > Guess** - If new information contradicts prior insights, immediately invalidate and update the record rather than preserving outdated analysis  
**Confidence Tiers**: Use confidence levels that reflect real-time uncertainty: 0.95+ = definitive decision/fact; 0.75-0.89 = probable but requires verification; <0.75 = tentative insight requiring follow-up 
**Atomic Output** - Each response should contain ONLY the most significant changes since last update, not full reanalysis of entire stream history  
**Speaker Awareness**: If multiple speakers detected (e.g., "Alex says... Sarah says..."), attribute insights to appropriate parties when possible without breaking context flow  
**Critical Thresholds for Action**: Only output ACTION items that have clear owners and/or deadlines or consequences if missing deadline/owner info is provided in current segment, indicate owner if provided
**Noise Handling**: Ignore filler words, repetitions, and non-content pauses unless they contain repeated phrases indicating urgency ("Again!", "Just to confirm!")  
**NEVER summarize** entire conversations - only surface what materially changes understanding of next steps or critical outcomes  
**DO NOT invent details** where transcript is incomplete (e.g., not making up names for missing figures)  
**Incremental Confidence Decay**: Gradually reduce confidence ratings when no new context confirms assertions, but never below 0.5 until explicit correction  
**Update Frequency**: Prioritize outputting significant changes at least every 3-5 seconds of continuous speech to maintain real-time awareness. Notes should be frequent to assist with log of conversation.

**CRITICAL: NO DUPLICATE INSIGHTS PER WINDOW**
- Each piece of information should appear in ONLY ONE insight type per analysis window
- Choose the MOST SPECIFIC category that fits the content (ACTION > DECISION > QUESTION > KEY POINT > RISK > SENTIMENT > NOTES)
- If information could fit multiple categories, use this priority hierarchy:
  * ACTION takes precedence if there's a concrete task/deadline/owner
  * DECISION takes precedence if there's a commitment or agreement
  * QUESTION takes precedence if there's a blocker requiring resolution
  * KEY POINT for important data that doesn't fit above categories
  * SENTIMENT only for explicit tone/emotional shifts. Sentiment should have a [+] if positive, [-] if negative, [~] if neutral.
  * NOTES as the catch-all for general context that doesn't fit elsewhere
  * NOTES should be used to keep a log of general context and can be used along with other insight types, but no other insight type should be duplicated in the same window
- Example: "We need to buy the software by Friday" → ACTION only (not also KEY POINT for the date or NOTES)
- Example: "The CEO approved the $50K budget" → DECISION only (not also KEY POINT for the amount)
- Example: "What's the delivery timeline? We need it by Q2" → QUESTION for the blocker, ACTION for the deadline requirement (two separate insights)

**Output Protocol:**
- Valid types are ACTION, DECISION, QUESTION, KEY POINT, RISK, SENTIMENT, NOTES
- Always return VALID JSON with single object: `{"insights": [{"insight_type":"TYPE","insight_text":"concise insight text","confidence":0.xx,"classification":"[+]"}, ...]}`
- If no material changes since last response, return `{"insights": []}` with minimal weight for system health metrics only when absolutely necessary (max 1 insight)
- Include a `classification` field with values "[+]", "[~]", or "[-]" for all insights:
  * [+] = positive sentiment/high priority/important
  * [~] = neutral/informational/default
  * [-] = negative sentiment/concern/risk
- Classification applies to ALL insight types, not just SENTIMENT
- If classification is not explicitly provided, it will default to [~]
- Max insights per update: 3 most critical items to maintain processing efficiency. NOTES insights do not count towards this limit.
- Never output empty JSON or partial fields - ensure full structure always valid
- think very briefly to gather thoughts, no long chain of thought, then output the JSON ONLY
""".strip()
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
            message_format_mode: Message format mode (system or user prefix)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.history_length = history_length
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.response_json_schema = InsightsResponseSchema.model_json_schema()
        
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
            max_concurrent = int(os.getenv("MAX_CONCURRENT_SUMMARIES", "15"))
        except Exception:
            max_concurrent = 15
        self.max_concurrent_summaries: int = max(1, max_concurrent)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrent_summaries)
        
        # In-flight tracking for graceful shutdown
        self.in_flight_windows: set[int] = set()  # Track window IDs being processed
    
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
    
    def update_params(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        history_length: Optional[int] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        message_format_mode: Optional[MessageFormatMode] = None
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
        
        if self.message_format_mode == MessageFormatMode.SYSTEM_PROMPT:
            # Standard format with system role
            messages.append({"role": "system", "content": system_prompt})
            
            if context:
                messages[0]["content"] += f"\nPrevious context (for understanding references only, do not extract insights from this unless current window transcript provides new information):\n{context}"
            
            messages.append({"role": "user", "content": user_content})
        
        else:  # USER_PREFIX mode
            # Combine system prompt, context, and user content into a single user message
            # This is required for models like Google Gemma that don't support the system role
            combined_content = f"[SYSTEM PROMPT]\n{system_prompt}\n\n[USER CONTENT]\n{user_content}"
            
            if context:
                combined_content = f"[SYSTEM PROMPT]\n{system_prompt}\n\n[CONTEXT]\n{context}\n\n[USER CONTENT]\n{user_content}"
            
            messages.append({"role": "user", "content": combined_content})
            messages.append({"role": "assistant", "content": ""})  # Placeholder for assistant response
        
        return messages
    
    async def summarize_text(self, text: str, context: str = "") -> tuple[str, str]:
        """
        Send text to LLM for cleaning and summarization.
        
        Args:
            text: Text to summarize/clean
            context: Additional context from previous segments
            
        Returns:
            Tuple of (cleaned_summary, background_context)
        """
        logger.info(f"summarize_text called with text length={len(text)}, context length={len(context)}")
        
        async def _do_request() -> tuple[str, str]:
            user_content = f"Analyze the following current window transcript text and report only new or changed insights since the previous context.:\n\n{text}"
            messages = self._build_messages(
                system_prompt=self.system_prompt,
                context=context,
                user_content=user_content
            )
            
            try:
                logger.info(f"Sending to LLM for analysis")
                
                # Use OpenAI client directly
                response: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_schema", "json_schema": {"name": "insights", "schema": self.response_json_schema}}
                )
                
                # Extract summary text from response
                if response.choices and len(response.choices) > 0:
                    summary_text = response.choices[0].message.content or ""
                    summary_text = summary_text.replace("```json", "").replace("```", "").strip()

                    summary_text_background_context = response.choices[0].message.reasoning or ""

                    logger.info(f"summarize_text received response, length={len(summary_text)}")
                    return summary_text.strip(), summary_text_background_context.strip()
                
                logger.info("summarize_text received empty response")
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
        summary_text, summary_background_context = await self.summarize_text(new_text, context)
        
        logger.info(f"Processed window {window_id}, summary length={len(summary_text)}")
        
        # Extract insights and add to window
        if summary_text:
            insights = self._extract_insights(summary_text, window_id, window_start, window_end)
            for insight in insights:
                self._window_manager.add_insight_to_window(window_id, insight)
        
        # Create summary segment
        summary_segment = SummarySegment(
            summary_type=summary_type,
            background_context=summary_background_context,
            summary=summary_text,
            timestamp_start=window_start,
            timestamp_end=window_end,
        )
        
        # Mark that we have performed at least one summary
        self._has_performed_summary = True
        
        logger.info(f"Returning summary segment for window {window_id}")
        
        return [summary_segment]
    
    def _parse_classification(self, item: Dict[str, Any], insight_text: str) -> str:
        """
        Parse classification from item dict or extract from text prefix.
        
        Args:
            item: Dictionary containing insight data
            insight_text: The insight text to check for prefix
        
        Returns:
            Classification string ([+], [~], [-]) or default [~]
        """
        # Check for explicit classification field first
        if item.get("classification"):
            classification = item.get("classification")
            if classification in ["[+]", "[~]", "[-]"]:
                return classification
        
        # Fallback: extract from text prefix
        if insight_text.startswith("[+]"):
            return "[+]"
        elif insight_text.startswith("[~]"):
            return "[~]"
        elif insight_text.startswith("[-]"):
            return "[-]"
        
        # Default to neutral
        return "[~]"
    
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
                        insight_type = item.get("insight_type", "NOTES")
                        insight_text = item.get("insight_text", "") or item.get("insight", "") or item.get("text", "")
                        
                        # Parse classification from explicit field or text prefix
                        classification = self._parse_classification(item, insight_text)
                        
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
                                insight_type = item.get("type", "NOTES")
                                insight_text = item.get("text", "") or item.get("insight", "")
                                
                                # Parse classification from explicit field or text prefix
                                classification = self._parse_classification(item, insight_text)
                                
                                if insight_text:
                                    insights.append(WindowInsight(
                                        insight_type=insight_type,
                                        insight_text=insight_text,
                                        window_id=window_id,
                                        timestamp_start=window_start,
                                        timestamp_end=window_end,
                                        classification=classification
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
                            insight_type="NOTES",
                            insight_text=text,
                            window_id=window_id,
                            timestamp_start=window_start,
                            timestamp_end=window_end,
                            classification="[~]"
                        ))
        
        return insights
    
    def reset(self):
        """Reset all accumulated state."""
        self._window_manager.clear()
        self.in_flight_windows.clear()
    
    def add_in_flight_window(self, window_id: int):
        """Add window ID to in-flight tracking."""
        self.in_flight_windows.add(window_id)
    
    def remove_in_flight_window(self, window_id: int):
        """Remove window ID from in-flight tracking."""
        self.in_flight_windows.discard(window_id)
    
    def get_pending_count(self) -> int:
        """Get count of pending summary requests."""
        return len(self.in_flight_windows)