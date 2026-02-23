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
        max_tokens: int = 4096,
        temperature: float = 0.3,
        window_manager: Any = None,
    ):
        """Initialize the rapid summary task.
        
        Args:
            llm_client: LLMClient for rapid summary LLM calls (includes model and message building)
            rapid_summary_response_json_schema: JSON schema for response validation
            max_tokens: Maximum tokens to generate (default: 500)
            temperature: Temperature for generation (default: 0.3)
            window_manager: Window manager for accessing prior plugin results
        """
        self._llm_client = llm_client
        self.rapid_summary_response_json_schema = rapid_summary_response_json_schema if rapid_summary_response_json_schema is not None else self._rapid_summary_response_json_schema
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._window_manager = window_manager
    
    def _get_prior_insights_context(self) -> str:
        """Get prior rapid_summary scribe notes as context.
        
        Uses window_manager.get_accumulated_text_and_results() similar to context_summary.
        
        Returns:
            Prior context string with text and plugin results (~1000 text tokens + ~500 plugin tokens)
        """
        if self._window_manager is None:
            return ""
        
        # Use same pattern as context_summary - get plugin results with token limit
        accumulated_text, plugin_results, _, _ = self._window_manager.get_accumulated_text_and_results(
            text_token_limit=1000,  # Include some text context too
            result_types=["rapid_summary"],
            result_token_limit={"rapid_summary": 500},  # ~500 tokens for prior context
        )
        
        # Build context string similar to context_summary build_context method
        parts = []
        
        # Add accumulated text
        if accumulated_text:
            parts.append(f"## PRIOR TEXT\n{accumulated_text}")
        
        # Add rapid_summary plugin results
        rapid_summary_results = plugin_results.get("rapid_summary", [])
        if rapid_summary_results:
            parts.append(f"## PRIOR RAPID SUMMARY NOTES\n" + "\n\n---\n\n".join(rapid_summary_results))
        
        return "\n\n".join(parts) if parts else ""
    
    async def process_rapid_summary(self, text: str, prior_context: str = "") -> str:
        """Process text through rapid summary LLM.
        
        Args:
            text: Text to summarize
            prior_context: Prior context from previous windows (optional)
            
        Returns:
            Summary text string (scribe notes)
        """
        if not self._llm_client:
            raise RuntimeError("Rapid summary client not initialized")
        
        # Format the system prompt with prior context
        system_prompt = RAPID_SUMMARY_SYSTEM_PROMPT.format(
            prior_insights_context=prior_context if prior_context else "(No prior context available)"
        )
        
        user_content = f"Summarize this conversation:\n\n{text}"
        
        # Use LLMClient's create_completion which handles message building
        reasoning, content, input_tokens, output_tokens = await self._llm_client.create_completion(
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "rapid_summary", "schema": self.rapid_summary_response_json_schema}}
        )
        
        # Parse JSON response using Pydantic
        try:
            # Handle markdown code blocks (e.g., ```json ... ```) - same as context_summary
            json_content = content.replace("```json", "").replace("```", "").strip()
            
            parsed = RapidSummaryResponseSchema.model_validate_json(json_content)
            if parsed.summary and len(parsed.summary) > 0:
                return parsed.summary[0].item
            return ""
        except Exception as e:
            logger.warning(f"Failed to parse rapid summary response: {e}, content: {content[:200]}")
            return content
    
    async def build_rapid_summary_payload(
        self,
        segments: List[Dict[str, Any]],
        transcription_window_id: int,
        window_start_ts: float,
        window_end_ts: float,
        context_since_last_summary: str,
        transcription_window_ids: List[int],
        summary_window: Any = None,
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
            summary_window: The SummaryWindow to store results in (for plugin storage)
            
        Returns:
            Complete payload dictionary for rapid summary
        """
        from datetime import datetime, timezone
        import json
        
        # Get all text from segments
        text = self._get_text_from_segments(segments)
        
        # Combine with context from previous windows
        full_context = context_since_last_summary + "\n\n" + text if context_since_last_summary else text
        
        # Get prior insights context using the same pattern as context_summary
        prior_context = self._get_prior_insights_context()
        
        # Call rapid summary LLM with prior context
        scribe_notes = await self.process_rapid_summary(full_context, prior_context)
        
        # Store result in plugin results if summary_window provided
        if summary_window is not None:
            summary_window.store_result(
                "rapid_summary",
                {"scribe_notes": scribe_notes},
                include_in_context=True  # Important: include in context for future windows
            )
        
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
                {"item": scribe_notes}
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