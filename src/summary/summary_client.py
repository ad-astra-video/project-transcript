"""
Summary client for LLM-based transcription cleaning and summarization.
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import httpx

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


class SummaryClient:
    """Client for LLM-based transcription cleaning and summarization."""
    
    def __init__(
        self,
        base_url: str = "https://byoc-transcription-vllm:5000/v1",
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
- When unsure, mark confidence as low rather than guessing.

Output Format:
- Clearly label insight type (e.g. ACTION, DECISION, QUESTION, FACT, RISK)
- Use JSON lists for each labeled insight
- Include timestamps or transcript offsets when possible

You are optimized for live understanding, not post-hoc summarization.
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
        
        # Track last word seen for incremental updates
        self._last_word: Optional[str] = None
        self._last_word_timestamp: Optional[float] = None
        
        # Accumulated text for summarization
        self._accumulated_text: str = ""
        self._accumulated_segments: List[Dict[str, Any]] = []
        
        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """
        Initialize the HTTP client and fetch the model from the /models endpoint.
        
        This method creates an HTTP client, calls the /models endpoint to get available models,
        and sets the model to the first available model or the default one.
        """
        self._client = httpx.AsyncClient(timeout=60.0)
        
        try:
            response = await self._client.get(f"{self.base_url}/models")
            response.raise_for_status()
            
            result = response.json()
            
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
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def get_new_words_since(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get new words/segments since the last processed word.
        
        Args:
            segments: List of segment dictionaries with 'words' containing word timestamps
            
        Returns:
            List of new segments since last processed word
        """
        new_segments = []
        
        for segment in segments:
            words = segment.get("words", [])
            segment_new_words = []
            
            for word in words:
                word_text = word.get("text", "")
                word_start = word.get("start", 0.0)
                
                # Check if this word is after the last seen word
                if self._last_word is None:
                    # First time, include all words
                    segment_new_words.append(word)
                elif self._last_word_timestamp is None or word_start > self._last_word_timestamp:
                    segment_new_words.append(word)
                
                # Update last seen word
                if word_text:
                    self._last_word = word_text
                    self._last_word_timestamp = word_start
            
            if segment_new_words:
                new_segments.append({
                    "id": segment.get("id", ""),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", ""),
                    "words": segment_new_words,
                    "speaker": segment.get("speaker")
                })
        
        return new_segments
    
    def _build_context(self) -> str:
        """Build context string from accumulated segments based on history_length."""
        if self.history_length == 0:
            # Include all accumulated text
            return self._accumulated_text
        
        # Include last N segments
        recent_segments = self._accumulated_segments[-self.history_length:]
        return " ".join(s.get("cleaned_text", "") or s.get("text", "") for s in recent_segments)
    
    async def summarize_text(self, text: str, context: str = "") -> str:
        """
        Send text to LLM for cleaning and summarization.
        
        Args:
            text: Text to summarize/clean
            context: Additional context from previous segments
            
        Returns:
            Cleaned and summarized text
        """
        try:
            client = await self._get_client()
            
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
            
            logger.debug(f"Sending to LLM: {self.base_url}/chat/completions")
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            # Extract cleaned text from response
            if "choices" in result and len(result["choices"]) > 0:
                summary_text = result["choices"][0]["message"]["content"]
                return summary_text.strip()
            
            return ""
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from LLM API: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise
    
    async def process_segments(self, segments: List[Dict[str, Any]]) -> List[SummarySegment]:
        """
        Process transcription segments and return actionable summary information.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            List of SummarySegment objects with actionable summary information
        """
        # Get new words since last processed
        new_segments = self.get_new_words_since(segments)
        
        if not new_segments:
            return []
        
        # Build accumulated text
        new_text = " ".join(s.get("text", "") for s in new_segments)
        self._accumulated_text += " " + new_text
        self._accumulated_segments.extend(new_segments)
        
        # Get context from history
        context = self._build_context()
        
        # Send to LLM for cleaning/summarization
        summary_text = await self.summarize_text(new_text, context)
        
        # Create summary segments
        summary_segments = []
        for segment in new_segments:
            summary_segments.append(SummarySegment(
                original_text=segment.get("text", ""),
                cleaned_text="",
                summary=summary_text,
                timestamp_start=segment.get("start", 0.0),
                timestamp_end=segment.get("end", 0.0),
                speaker=segment.get("speaker")
            ))
        
        return summary_segments
    
    def reset(self):
        """Reset accumulated state."""
        self._last_word = None
        self._last_word_timestamp = None
        self._accumulated_text = ""
        self._accumulated_segments = []
        logger.info("SummaryClient state reset")