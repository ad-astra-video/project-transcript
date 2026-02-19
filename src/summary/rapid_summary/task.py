"""
Rapid summary task implementation.

This module contains the core logic for processing rapid summaries,
which provide quick, concise summaries of ongoing discussions.
"""

import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from ..llm_manager import LLMClient

from .prompts import RAPID_SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class RapidSummaryItemSchema(BaseModel):
    """Schema for a single rapid summary item."""
    item: str


class RapidSummaryResponseSchema(BaseModel):
    """Schema for rapid summary response from LLM."""
    summary: List[RapidSummaryItemSchema]


class RapidSummaryTask:
    """
    Task for processing rapid summaries.
    
    Provides quick, concise summaries of ongoing discussions
    using a separate, faster LLM endpoint.
    """
    
    # Class-level JSON schema - computed once at class definition time
    _rapid_summary_response_json_schema: Dict[str, Any] = RapidSummaryResponseSchema.model_json_schema()
    
    def __init__(
        self,
        llm_client: LLMClient,
        rapid_summary_response_json_schema: Dict[str, Any] = None,
    ):
        """Initialize the rapid summary task.
        
        Args:
            llm_client: LLMClient for rapid summary LLM calls (includes model and message building)
            rapid_summary_response_json_schema: JSON schema for response validation
        """
        self._llm_client = llm_client
        self.rapid_summary_response_json_schema = rapid_summary_response_json_schema if rapid_summary_response_json_schema is not None else self._rapid_summary_response_json_schema
    
    async def process_rapid_summary(self, text: str) -> str:
        """Process text through rapid summary LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary text string
        """
        if not self._llm_client:
            raise RuntimeError("Rapid summary client not initialized")
        
        user_content = f"Summarize this conversation:\n\n{text}"
        
        # Use LLMClient's create_completion which handles message building
        reasoning, content, input_tokens, output_tokens = await self._llm_client.create_completion(
            system_prompt=RAPID_SUMMARY_SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_schema", "json_schema": {"name": "rapid_summary", "schema": self.rapid_summary_response_json_schema}}
        )
        
        # Parse JSON response using Pydantic
        try:
            parsed = RapidSummaryResponseSchema.model_validate_json(content)
            if parsed.summary and len(parsed.summary) > 0:
                return parsed.summary[0].item
            return ""
        except Exception as e:
            logger.warning(f"Failed to parse rapid summary response: {e}")
            return content
    
    async def build_rapid_summary_payload(
        self,
        segments: List[Dict[str, Any]],
        transcription_window_id: int,
        window_start_ts: float,
        window_end_ts: float,
        context_since_last_summary: str,
        transcription_window_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Build rapid summary payload.
        
        Args:
            segments: List of transcription segments
            transcription_window_id: ID of the transcription window
            window_start_ts: Start timestamp of the window
            window_end_ts: End timestamp of the window
            context_since_last_summary: Context text since last summary
            transcription_window_ids: List of transcription window IDs
            
        Returns:
            Complete payload dictionary for rapid summary
        """
        from datetime import datetime, timezone
        import json
        
        # Get all text from segments
        text = self._get_text_from_segments(segments)
        
        # Combine with context from previous windows
        full_context = context_since_last_summary + "\n\n" + text if context_since_last_summary else text
        
        # Call rapid summary LLM
        summary_text = await self.process_rapid_summary(full_context)
        
        # Create payload with rapid_summary type and its own schema
        payload = {
            "type": "rapid_summary",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timing": {
                "transcription_window_ids": transcription_window_ids,
                "media_window_start_ms": int(window_start_ts * 1000),
                "media_window_end_ms": int(window_end_ts * 1000)
            },
            "summary": [
                {"item": summary_text}
            ]
        }
        
        return payload
    
    def _get_text_from_segments(self, segments: List[Dict[str, Any]]) -> str:
        """Extract text from segments."""
        text_parts = []
        for segment in segments:
            text = segment.get("text", "")
            if text:
                text_parts.append(text)
        return " ".join(text_parts)


__all__ = ["RapidSummaryTask", "RapidSummaryResponseSchema", "RapidSummaryItemSchema"]