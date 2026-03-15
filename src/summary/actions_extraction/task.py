"""
Actions extraction task implementation.

Uses the fast LLM to extract concrete future actions and follow-up items
from rapid summary bullet points. Designed to be low-latency by operating
on already-distilled notes rather than raw transcript text.

The system prompt is dynamically adapted to the current content type so that
extraction criteria match the nature of the content (e.g., strict for meetings,
very conservative for lectures and broadcasts).
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel

from ..llm_manager import LLMClient
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)


class ActionsExtractionSchema(BaseModel):
    """Schema for the actions extraction LLM response."""

    actions: List[str]
    follow_ups: List[str]


class ActionsExtractionTask:
    """
    Uses the fast LLM to extract future actions and follow-up items from
    rapid-summary bullet points for a single summary window.

    The extraction criteria are adapted per content type via a dynamically
    built system prompt so that the model is appropriately strict for each
    class of content (e.g., virtually nothing for gameplay/news, strict
    explicit-commitment bar for meetings).
    """

    _json_schema: Dict[str, Any] = ActionsExtractionSchema.model_json_schema()

    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> None:
        """
        Args:
            llm_client: Reasoning LLM client (reasoning_llm_client).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        """
        self._llm_client = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def process(
        self,
        fast_summary_items: List[str],
        content_type: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """Extract actions and follow-ups from rapid-summary bullet points.

        Args:
            fast_summary_items: Bullet-point strings already produced by the
                rapid_summary plugin for one or more summary windows.
            content_type: Current content type (e.g. "GENERAL_MEETING").  Used
                to select the appropriate extraction criteria.

        Returns:
            dict with keys:
                - "actions": list[str]
                - "follow_ups": list[str]
                - "input_tokens": int
                - "output_tokens": int
        """
        if not self._llm_client:
            raise RuntimeError("ActionsExtractionTask: LLM client not initialized")

        if not fast_summary_items:
            return {"actions": [], "follow_ups": [], "input_tokens": 0, "output_tokens": 0}

        system_prompt = build_system_prompt(content_type)

        bullet_block = "\n".join(f"- {item}" for item in fast_summary_items)
        user_content = f"Extract actions and follow-ups from these notes:\n\n{bullet_block}"

        _, content, input_tokens, output_tokens, _ = await self._llm_client.create_completion(
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "actions_extraction",
                    "schema": self._json_schema,
                },
            },
        )

        json_content = content.replace("```json", "").replace("```", "").strip()
        if not json_content:
            return {
                "actions": [],
                "follow_ups": [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        try:
            parsed = ActionsExtractionSchema.model_validate_json(json_content)
        except Exception as exc:
            logger.warning(
                "actions_extraction: failed to parse LLM response: %s | raw: %r", exc, content
            )
            return {
                "actions": [],
                "follow_ups": [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        return {
            "actions": parsed.actions,
            "follow_ups": parsed.follow_ups,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


__all__ = ["ActionsExtractionTask", "ActionsExtractionSchema"]
