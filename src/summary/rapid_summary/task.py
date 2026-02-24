"""
Rapid summary task implementation.

This module contains the core logic for processing rapid summaries,
which provide quick, concise summaries of ongoing discussions.
"""

import json
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
            Prior context string with only plugin results (~200 tokens)
        """
        if self._window_manager is None:
            return ""
        
        # Use same pattern as context_summary - get plugin results with token limit
        # Note: We request text but don't include it in context (per user request)
        accumulated_text, plugin_results, _, _ = self._window_manager.get_accumulated_text_and_results(
            text_token_limit=1000,  # Keep at 1000 (0 means unlimited)
            result_types=["rapid_summary"],
            result_token_limit={"rapid_summary": 200},  # Reduced from 500
        )
        
        # Build context string - only include rapid_summary plugin results
        parts = []
        
        # Note: We intentionally do NOT include accumulated_text in the prior context
        # Only include the rapid_summary plugin results
        
        # Add rapid_summary plugin results - extract scribe_notes from JSON
        rapid_summary_results = plugin_results.get("rapid_summary", [])
        if rapid_summary_results:
            formatted_notes = []
            for result_json in rapid_summary_results:
                # Parse the JSON to extract scribe_notes
                try:
                    data = json.loads(result_json) if isinstance(result_json, str) else result_json
                    scribe_notes = data.get("scribe_notes", [])
                    if scribe_notes:
                        formatted_notes.extend(scribe_notes)
                except (json.JSONDecodeError, AttributeError):
                    # Fallback: treat as string
                    formatted_notes.append(str(result_json))
            
            if formatted_notes:
                # Format as bullet list for readability
                notes_text = "\n".join(f"- {note}" for note in formatted_notes)
                parts.append(f"## PRIOR RAPID SUMMARY NOTES\n{notes_text}")
        
        return "\n\n".join(parts) if parts else ""
    
    async def process_rapid_summary(self, text: str, prior_context: str = "") -> List[str]:
        """Process text through rapid summary LLM.
        
        Args:
            text: Text to summarize
            prior_context: Prior context from previous windows (optional)
            
        Returns:
            List of summary items (scribe notes)
        """
        if not self._llm_client:
            raise RuntimeError("Rapid summary client not initialized")
        
        # Format the system prompt with prior context using replace to avoid
        # conflicts with JSON braces in the prompt template
        prior_context_value = prior_context if prior_context else "(No prior context available)"
        logger.info(f"Rapid summary prior context length: {len(prior_context_value)} chars")
        system_prompt = RAPID_SUMMARY_SYSTEM_PROMPT.replace(
            "__PRIOR_INSIGHTS_CONTEXT__", prior_context_value
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
        
        # Log the raw response for debugging
        #logger.info(f"Rapid summary raw response: {content}")
        
        # Parse JSON response using Pydantic
        try:
            # Handle markdown code blocks (e.g., ```json ... ```) - same as context_summary
            json_content = content.replace("```json", "").replace("```", "").strip()
            
            parsed = RapidSummaryResponseSchema.model_validate_json(json_content)
            if parsed.summary and len(parsed.summary) > 0:
                # Return all items as a list for the frontend to process
                return [item.item for item in parsed.summary]
            return []
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
        
        # Call rapid summary LLM with prior context - returns a list of items
        scribe_notes_list = await self.process_rapid_summary(full_context, prior_context)
        
        # Ensure we have a list (handle empty case)
        if not scribe_notes_list:
            scribe_notes_list = []
        
        # Store result in plugin results if summary_window provided
        if summary_window is not None:
            summary_window.store_result(
                "rapid_summary",
                {"scribe_notes": scribe_notes_list},
                include_in_context=True  # Important: include in context for future windows
            )
        
        # Create payload with rapid_summary type - wrap each string in { item: string } format
        payload = {
            "type": "rapid_summary",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timing": {
                "transcription_window_ids": transcription_window_ids,
                "media_window_start_ms": int(window_start_ts * 1000),
                "media_window_end_ms": int(window_end_ts * 1000)
            },
            "summary": [{"item": item} for item in scribe_notes_list]
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