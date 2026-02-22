"""
Content type detection task implementation.

This module contains the core logic for detecting content types
from transcripts (e.g., GENERAL_MEETING, TECHNICAL_TALK, etc.).
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from enum import Enum

from openai.types.chat.chat_completion import ChatCompletion

from .prompts import CONTENT_TYPE_DETECTION_PROMPT
from ..llm_manager import LLMClient

logger = logging.getLogger(__name__)


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


class ContentTypeDetectionSchema(BaseModel):
    """Schema for content type detection response."""
    content_type: ContentType
    confidence: float
    reasoning: str


class ContentTypeState(BaseModel):
    """State for content type tracking."""
    content_type: str = ContentType.UNKNOWN.value
    previous_content_type: str = ""
    confidence: float = 0.0
    source: str = ContentTypeSource.INITIAL.value
    last_detection_text: str = ""
    context_length: int = 2000


class ContentTypeDetectionTask:
    """
    Task for detecting content types from transcripts.
    
    Analyzes transcript context to determine the content type
    (e.g., GENERAL_MEETING, TECHNICAL_TALK, PODCAST, etc.).
    """  
    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens: int = 350,
        temperature: float = 0.2,
    ):
        """Initialize the content type detection task.
        
        Args:
            llm_client: LLMClient for LLM calls (includes model and message building)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        """
        self._llm_client = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.content_type_response_json_schema = ContentTypeDetectionSchema.model_json_schema()
    
    def _build_messages(
        self,
        context_text: str,
        context_length: int
    ) -> List[Dict[str, str]]:
        """Build messages for content type detection.
        
        Uses LLMClient's build_messages to create properly formatted messages.
        
        Args:
            context_text: The transcript context text
            context_length: Length of context to use
            
        Returns:
            List of message dictionaries
        """
        user_content = f"""## TRANSCRIPT CONTEXT

Transcript Text (Last {context_length} characters):
{context_text}
"""
        
        return self._llm_client.build_messages(
            system_prompt=CONTENT_TYPE_DETECTION_PROMPT,
            user_content=user_content
        )
    
    async def detect_content_type(
        self,
        context_text: str,
        context_length: int = 2000
    ) -> ContentTypeDetectionSchema:
        """Detect content type from transcript context.
        
        Args:
            context_text: The transcript context text
            context_length: Length of context to use
            
        Returns:
            ContentTypeDetectionSchema with detected content type
        """
        # Don't exceed available text
        max_context_length = len(context_text)
        if context_length > max_context_length:
            context_length = max_context_length
        
        # Get context text (last N chars)
        context_to_use = context_text[-context_length:] if len(context_text) > context_length else context_text
        
        logger.info(f"Sending {len(context_to_use)} chars to content_type_detection")
        
        try:
            logger.info(f"Running content type detection (context_length={context_length})")
            
            # Build messages using _build_messages (avoids duplicating message formatting logic)
            messages = self._build_messages(
                context_text=context_to_use,
                context_length=context_length
            )
            
            # Use LLMClient's create_completion with pre-built messages
            reasoning, content, input_tokens, output_tokens = await self._llm_client.create_completion(
                system_prompt="",  # Messages already built by _build_messages
                user_content="",   # We pass messages directly
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_schema", "json_schema": {"name": "content_type_detection", "schema": self.content_type_response_json_schema}}
            )
            
            # Validate response before parsing
            if not content or not content.strip():
                logger.warning(f"Content type detection returned empty response. Messages: {messages}")
                return ContentTypeDetectionSchema(
                    content_type=ContentType.UNKNOWN,
                    confidence=0.0,
                    reasoning="Empty response from LLM"
                )
            
            # Check for partial/incomplete JSON
            content_stripped = content.strip()
            starts_with_brace = content_stripped.startswith('{')
            ends_with_brace = content_stripped.endswith('}')
            
            if not starts_with_brace or not ends_with_brace:
                logger.warning(f"Content type detection returned incomplete JSON. Content: {content_stripped}. Messages: {messages}")
                return ContentTypeDetectionSchema(
                    content_type=ContentType.UNKNOWN,
                    confidence=0.0,
                    reasoning="Incomplete JSON response from LLM"
                )
            
            result = self._parse_content_type_response(content)
            
            # Log when UNKNOWN is returned
            if result.content_type == ContentType.UNKNOWN:
                logger.warning(f"Content type detection returned UNKNOWN. Content: {content_stripped}. Messages: {messages}")
            
            return result
            
        except Exception as e:
            logger.error(f"Content type detection error: {e}. Messages: {messages}")
            return ContentTypeDetectionSchema(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Error: {str(e)}"
            )
    
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

__all__ = [
    "ContentTypeDetectionTask",
    "ContentTypeDetectionSchema",
    "ContentTypeState",
    "ContentType",
    "ContentTypeSource"
]