"""Insights distillation task implementation."""

import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError, field_validator

from ..llm_manager import LLMClient
from .prompts import INSIGHTS_DISTILLATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DistilledInsight(BaseModel):
    """Schema for one distilled insight item."""

    insight: str
    category: str
    confidence: float

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


class InsightsDistillationSchema(BaseModel):
    """Schema for insights distillation response."""

    insights: List[DistilledInsight]


class InsightsDistillationTask:
    """Run reasoning-model distillation over context + transcript summaries."""

    _json_schema: Dict[str, Any] = InsightsDistillationSchema.model_json_schema()

    def __init__(
        self,
        llm_client: LLMClient,
        max_tokens: int = 16384,
        temperature: float = 0.3,
    ) -> None:
        self._llm_client = llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _extract_context_insights(context_summary_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        summary_text = context_summary_result.get("summary_text", "{}")
        if not isinstance(summary_text, str):
            return []
        try:
            parsed = json.loads(summary_text)
            insights = parsed.get("insights", [])
            return insights if isinstance(insights, list) else []
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []

    async def process(
        self,
        context_summary_result: Dict[str, Any],
        transcript_summary_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        context_insights = self._extract_context_insights(context_summary_result)

        user_payload = {
            "context_summary": {
                "insights": context_insights,
            },
            "transcript_summary": {
                "sections": transcript_summary_result.get("sections", []),
                "key_points": transcript_summary_result.get("key_points", []),
                "topics": transcript_summary_result.get("topics", []),
                "summary": transcript_summary_result.get("summary", ""),
            },
        }

        _, content, input_tokens, output_tokens, _ = await self._llm_client.create_completion(
            system_prompt=INSIGHTS_DISTILLATION_SYSTEM_PROMPT,
            user_content=json.dumps(user_payload, ensure_ascii=False),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "insights_distillation",
                    "schema": self._json_schema,
                },
            },
        )

        json_content = content.replace("```json", "").replace("```", "").strip()
        
        # Handle empty LLM response - graceful degradation
        if not json_content:
            logger.warning("insights_distillation: LLM returned empty response")
            return {
                "insights": [],
                "detailed_insights": [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        
        try:
            parsed = InsightsDistillationSchema.model_validate_json(json_content)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning("insights_distillation: failed to parse LLM response: %s", e)
            return {
                "insights": [],
                "detailed_insights": [],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        insights = [item.insight for item in parsed.insights]
        detailed_insights = [item.model_dump() for item in parsed.insights]

        logger.info("insights_distillation: produced %d insights", len(insights))

        return {
            "insights": insights,
            "detailed_insights": detailed_insights,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
