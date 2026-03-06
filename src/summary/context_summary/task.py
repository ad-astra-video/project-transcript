"""
Context summary task implementation.

This module contains the core logic for processing context summaries,
including text summarization, insight extraction, and context building.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from enum import Enum

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from ..llm_manager import LLMClient
from dataclasses import dataclass, field
from .prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_OUTPUT_CONSTRAINTS,
    CONTENT_TYPE_RULE_MODIFIERS,
    INSIGHT_CONSOLIDATION_PROMPT,
    InsightConsolidationResponse
)

logger = logging.getLogger(__name__)


# ==================== WindowInsight (moved from window_manager.py) ====================

@dataclass
class WindowInsight:
    """Insight extracted from a summary window."""
    insight_id: int = 0  # Unique identifier assigned by system (not LLM)
    insight_type: str = ""
    insight_text: str = ""
    confidence: float = 0.0  # Confidence score from LLM (0.0-1.0)
    window_id: int = 0
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    classification: str = "~"
    continuation_of: Optional[int] = None  # Previous insight ID this continues
    correction_of: Optional[int] = None  # Previous insight ID this corrects
    
    # Excludes timestamp_start and timestamp_end
    #   this is used for sending over json data channel which includes timing for all insights sent
    def as_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "insight_text": self.insight_text,
            "confidence": self.confidence,
            "classification": self.classification,
            "continuation_of": self.continuation_of,
            "correction_of": self.correction_of,
        }


# ==================== End WindowInsight ====================


# ==================== Types and Schemas ====================

class InsightType(str, Enum):
    """Enumeration of possible insight types."""
    ACTION = "ACTION"
    DECISION = "DECISION"
    QUESTION = "QUESTION"
    KEY_POINT = "KEY POINT"
    RISK = "RISK"
    SENTIMENT = "SENTIMENT"
    PARTICIPANTS = "PARTICIPANTS"
    NOTES = "NOTES"


class ClassificationField(str, Enum):
    """Classification markers for insights - general and reusable across all insight types."""
    POSITIVE = "+"
    NEUTRAL = "~"
    NEGATIVE = "-"


class Topic(str, Enum):
    """Enumeration of valid topic categories for context summary.
    
    These values must exactly match the TOPICS defined in the system prompt.
    """
    MACHINE_LEARNING = "MACHINE_LEARNING"
    SOFTWARE_ENGINEERING = "SOFTWARE_ENGINEERING"
    DATA_ENGINEERING = "DATA_ENGINEERING"
    DEVOPS = "DEVOPS"
    CYBERSECURITY = "CYBERSECURITY"
    PRODUCT = "PRODUCT"
    BUSINESS = "BUSINESS"
    LEGAL = "LEGAL"
    RESEARCH = "RESEARCH"
    HEALTH = "HEALTH"
    EDUCATIONAL = "EDUCATIONAL"
    SPORTS = "SPORTS"
    ARTS = "ARTS"
    AGRICULTURE = "AGRICULTURE"
    HISTORY = "HISTORY"
    ENVIRONMENT = "ENVIRONMENT"
    FINANCE = "FINANCE"
    MARKETING = "MARKETING"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    GENERAL = "GENERAL"


class InsightResponseItemSchema(BaseModel):
    """Schema for a single insight item."""
    insight_type: InsightType
    insight_text: str
    confidence: float
    classification: ClassificationField
    continuation_of: Optional[int] = None
    correction_of: Optional[int] = None


class InsightsResponseSchema(BaseModel):
    """Schema for insights response from LLM.
    
    The topic field is required and must be one of the valid Topic enum values.
    """
    topic: Topic
    insights: List[InsightResponseItemSchema]


# ==================== End Types and Schemas ====================


class ContextSummaryTask:
    """
    Task for processing context summaries.
    
    Handles text summarization, insight extraction, and context building
    for the summary client.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rapid_llm_client: Optional[LLMClient] = None,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        system_prompt: str = SYSTEM_PROMPT,
        content_type_state: Any = None,
        window_manager: Any = None
    ):
        """Initialize the context summary task.
        
        Args:
            llm_client: Optional LLMClient for LLM calls (includes model and message building)
            rapid_llm_client: Optional LLMClient for rapid/small model calls (used for consolidation)
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: System prompt for the LLM
            content_type_state: Content type state object
            window_manager: Window manager for tracking windows
        """
        self._llm_client = llm_client
        self._rapid_llm_client = rapid_llm_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.insights_response_json_schema = InsightsResponseSchema.model_json_schema()
        self._consolidation_response_json_schema = InsightConsolidationResponse.model_json_schema()
        self._content_type_state = content_type_state
        self._window_manager = window_manager
        
        # Counter for unique insight IDs (used for continuation/correction references)
        self._next_insight_id: int = 0
        
        # Store last raw LLM response for debugging
        self._last_summary_raw_response: Optional[str] = None
    
    def _get_prior_insights_from_plugin_results(self) -> List[WindowInsight]:
        """
        Retrieve prior insights from plugin_results in all windows except the last one.
        
        The last window is the "current" window being analyzed - its insights are being
        created. All previous windows' insights form the prior context for continuation/correction.
        
        Returns:
            List of WindowInsight objects from prior windows
        """
        if self._window_manager is None:
            return []
        
        prior_insights = []
        
        # Get all summary windows
        windows = self._window_manager._summary_windows
        
        if len(windows) <= 1:
            return []
        
        # All windows except the last one (current window being analyzed)
        prior_windows = windows[:-1]
        
        for window in prior_windows:
            # Get context_summary plugin result from this window
            result = window.get_result("context_summary")
            if result:
                # Extract insights from the result
                summary_text = result.get("summary_text", "{}")
                try:
                    import json
                    summary_data = json.loads(summary_text)
                    insights_list = summary_data.get("insights", [])
                    
                    for insight_dict in insights_list:
                        # Convert dict back to WindowInsight for validation
                        insight = WindowInsight(
                            insight_id=insight_dict.get("insight_id", 0),
                            insight_type=insight_dict.get("insight_type", "NOTES"),
                            insight_text=insight_dict.get("insight_text", ""),
                            confidence=insight_dict.get("confidence", 0.0),
                            window_id=insight_dict.get("window_id", window.window_id),
                            timestamp_start=insight_dict.get("timestamp_start", window.timestamp_start),
                            timestamp_end=insight_dict.get("timestamp_end", window.timestamp_end),
                            classification=insight_dict.get("classification", "~"),
                            continuation_of=insight_dict.get("continuation_of"),
                            correction_of=insight_dict.get("correction_of"),
                        )
                        prior_insights.append(insight)
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.warning(f"Failed to parse prior insights from window {window.window_id}: {e}")
        
        return prior_insights
    
    def _get_next_insight_id(self) -> int:
        """
        Increment and return the next unique insight ID.
        First call returns 1, second returns 2, etc.
        
        Returns:
            Unique integer ID for the next insight
        """
        self._next_insight_id += 1
        return self._next_insight_id
    
    def reset(self):
        """Reset state for a new stream.
        
        Clears the insight ID counter to ensure fresh ID sequence
        for each new stream session.
        """
        self._next_insight_id = 0
        logger.debug("ContextSummaryTask reset - insight ID counter cleared")
    
    def _format_content_type_rules(self, content_type: str) -> str:
        """
        Format content type rules as prompt text for injection into system prompt.
        
        Args:
            content_type: The active content type string
            
        Returns:
            Formatted rules string for inclusion in system prompt
        """
        if content_type not in CONTENT_TYPE_RULE_MODIFIERS:
            return ""
        
        rules = CONTENT_TYPE_RULE_MODIFIERS[content_type]
        
        # Build RISK guidance section if present
        risk_guidance = rules.get("risk_guidance", "")
        risk_section = ""
        if risk_guidance:
            risk_section = f"""
### RISK Definition (Content-Type-Specific):
{risk_guidance}"""
        
        # Build KEY POINT guidance section if present
        key_point_guidance = rules.get("key_point_guidance", "")
        key_point_section = ""
        if key_point_guidance:
            key_point_section = f"""
### KEY POINT Guidance:
{key_point_guidance}"""
        
        # Build STORY guidance section if present
        story_guidance = rules.get("story_guidance", "")
        story_section = ""
        if story_guidance:
            story_section = f"""
### STORY Handling:
{story_guidance}"""
        
        # Format the rules into a clear, actionable prompt section
        formatted = f"""

## CONTENT TYPE RULES: {content_type}

### Priority Insight Types (Emphasize):
{chr(10).join(f'- {t}' for t in rules["emphasize"])}

### Suppressed Insight Types (Deemphasize):
{chr(10).join(f'- {t}' for t in rules["deemphasize"])}

{risk_section}

{key_point_section}

{story_section}

### Processing Guidelines:
- Sentiment Tracking: {"ENABLED" if rules["sentiment_enabled"] else "DISABLED"}
- Participant Tracking: {"ENABLED" if rules["participants_enabled"] else "DISABLED"}
- Action Strictness: {rules["action_strictness"].upper()}
- Notes Frequency: {rules["notes_frequency"].upper()}

### Application Rules:
- Apply these modifiers strictly to guide extraction focus
- Do NOT compensate by inventing other insight types
- Prefer omission over speculation when rules suppress insight types
"""
        return formatted
    
    def _get_effective_content_type(self) -> Tuple[str, float, str]:
        """
        Get effective content type considering user override.
        
        Returns:
            Tuple of (content_type, confidence, source)
        """
        if self._content_type_state:
            return (
                self._content_type_state.content_type,
                self._content_type_state.confidence,
                self._content_type_state.source
            )
        return ("UNKNOWN", 0.0, "INITIAL")
    
    def _build_messages(
        self,
        system_prompt: str,
        context: str = "",
        user_content: str = ""
    ) -> List[Dict[str, str]]:
        """
        Build system prompt and user content for LLMClient to format.
        
        The LLMClient handles message formatting based on its configured mode
        (SYSTEM_PROMPT or USER_PREFIX).
        
        Args:
            system_prompt: The system prompt text
            context: Optional context from previous segments
            user_content: The user message content
            
        Returns:
            List of message dictionaries in the correct format
        """
        # Get active content type for dynamic rule injection
        content_type, confidence, source = self._get_effective_content_type()
        logger.debug(f"Building messages with content_type={content_type}, confidence={confidence}, source={source}")
        
        # Format content type rules for this content type
        content_type_rules = self._format_content_type_rules(content_type)
        
        # Build system prompt with dynamic content type rules
        combined_system = system_prompt
        
        # Add content type rules after system prompt, before output constraints
        if content_type_rules:
            combined_system += content_type_rules
        
        # Add output constraints ALWAYS before PRIOR CONTEXT
        combined_system += f"\n{SYSTEM_PROMPT_OUTPUT_CONSTRAINTS}"
        
        if context:
            combined_system += f"\n\n## PRIOR CONTEXT\nThe following context is from previous transcript windows. Use it for understanding references only. Do not extract new insights from this context unless the current window transcript provides new information that builds on or contradicts it.\n\n{context}"
        
        combined_system += "\n\nAnalyze the following current window transcript text and report only new or changed insights since the previous context.:"
        
        # Use LLMClient's build_messages for final message construction
        # LLMClient handles the message format based on its configured mode
        if self._llm_client is not None:
            return self._llm_client.build_messages(
                system_prompt=combined_system,
                user_content=user_content
            )
        
        # Fallback if no LLMClient - use standard system/user format
        return [
            {"role": "system", "content": combined_system},
            {"role": "user", "content": user_content}
        ]
    
    async def process_context_summary(
        self,
        summary_window_id: int,
    ) -> Dict[str, Any]:
        """
        Process a summary window through context summary LLM for cleaning and summarization.
        
        Args:
            summary_window_id: The ID of the summary window to process
            
        Returns:
            Full payload dictionary with summary results
        """
        if self._window_manager is None:
            raise RuntimeError("WindowManager not initialized")
        
        # Build context using the new method (get text without plugin results for backward compatibility)
        context, context_text_length, results_per_window = self.build_context(
            text_token_limit=500,
            result_types=["context_summary"],
            result_token_limit={"context_summary": 1000}
        )
        
        # Get prior insights from plugin_results for continuation/correction validation
        prior_insights = self._get_prior_insights_from_plugin_results()
        
        # Calculate insights_per_window metric
        window_count = len(self._window_manager._summary_windows) - 1 if self._window_manager._summary_windows else 1
        insights_per_window = len(prior_insights) / window_count if window_count > 0 else 0.0
        
        # Get text to analyze based on whether this is the first summary
        # (This logic should be coordinated with the plugin's _has_performed_summary state)
        # For now, we'll get the text from the window manager
        content_to_analyze = self._window_manager.get_window_text(summary_window_id)
        
        if not content_to_analyze:
            logger.warning(f"No text found for summary_window_id={summary_window_id}")
            return {"summary_text": "{}", "reasoning_content": "", "input_tokens": 0, "output_tokens": 0}
        
        logger.debug(f"process_context_summary called with text length={len(content_to_analyze)}, context length={len(context)}")
        
        messages = self._build_messages(
            system_prompt=self.system_prompt,
            context=context,
            user_content=content_to_analyze
        )
        
        try:
            logger.debug(f"Sending to LLM for analysis")
            
            if self._llm_client is None:
                raise RuntimeError("LLMClient not initialized")
            
            reasoning, content, input_tokens, output_tokens, reasoning_tokens = await self._llm_client.create_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_schema", "json_schema": {"name": "insights", "schema": self.insights_response_json_schema}}
            )

            summary_text_raw = content
            summary_text = content.replace("```json", "").replace("```", "").strip()
            reasoning_content = reasoning or ""

            logger.info(f"process_context_summary received response, length={len(summary_text)}, input_tokens={input_tokens}, reasoning_tokens={reasoning_tokens}")
            self._last_summary_raw_response = summary_text_raw
            
            # Handle empty response from LLM
            if not summary_text:
                logger.warning("LLM returned empty response, returning empty insights")
                return {
                    "summary_text": "{}",
                    "reasoning_content": reasoning_content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "insights": [],
                }
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            raise

        # Get window timing
        window_start = self._window_manager.get_window_start(summary_window_id)
        window_end = self._window_manager.get_window_end(summary_window_id)
        
        # Get content type settings for filtering
        sentiment_enabled = True
        participants_enabled = True
        if self._content_type_state and self._content_type_state.content_type in CONTENT_TYPE_RULE_MODIFIERS:
            rules = CONTENT_TYPE_RULE_MODIFIERS[self._content_type_state.content_type]
            sentiment_enabled = rules.get("sentiment_enabled", True)
            participants_enabled = rules.get("participants_enabled", True)
        
        # Process the LLM result: parse JSON, extract insights, filter by content type
        processed = await self.process_result(
            summary_text=summary_text,
            summary_window_id=summary_window_id,
            window_start=window_start,
            window_end=window_end,
            prior_insights=prior_insights,
            sentiment_enabled=sentiment_enabled,
            participants_enabled=participants_enabled
        )
        
        payload = {
            "summary_text": processed.get("summary_text", summary_text),
            "reasoning_content": reasoning_content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "insights": processed.get("insights", []),
        }

        return payload
    
    def build_context(
        self,
        text_token_limit: int = 0,
        result_types: Optional[List[str]] = None,
        result_token_limit: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, int, Dict[str, float]]:
        """
        Build context string with text and plugin results from accumulated windows.
        
        Args:
            text_token_limit: Max tokens for accumulated text
            result_types: Optional list of plugin names to include
            result_token_limit: Optional dict of {plugin_name: token_limit}
        
        Returns:
            Tuple of (formatted_context_string, text_token_count, results_per_window_dict)
        """
        if self._window_manager is None:
            return "", 0, {}
        
        # Use get_accumulated_text_and_results
        accumulated_text, plugin_results, text_token_count, results_per_window = (
            self._window_manager.get_accumulated_text_and_results(
                text_token_limit=text_token_limit,
                result_types=result_types,
                result_token_limit=result_token_limit,
            )
        )
        
        parts = []
        
        # Add accumulated text
        if accumulated_text:
            parts.append(f"## PRIOR TEXT\n{accumulated_text}")
        
        # Add plugin results (dict of plugin_name -> list of formatted texts)
        if plugin_results:
            result_parts = []
            for plugin_name, formatted_list in plugin_results.items():
                # Join all results for this plugin
                result_parts.extend(formatted_list)
            
            if result_parts:
                parts.append(f"## OTHER PLUGIN RESULTS\n" + "\n\n---\n\n".join(result_parts))
        
        context_string = "\n\n".join(parts) if parts else ""
        return context_string, text_token_count, results_per_window
    
    def _format_insights_for_context(self, insights: List[Any]) -> str:
        """Format insights for inclusion in Prior Context with IDs and timing hints."""
        formatted = []
        for insight in insights:
            # Format with ID and timing hints
            # ID is included so LLM can reference it in continuation_of/correction_of
            id_hint = f"[#{insight.insight_id}]"
            timing_hint = f"[{insight.timestamp_start:.1f}s - {insight.timestamp_end:.1f}s]"
            
            # Add continuation/correction markers if present
            markers = []
            if insight.continuation_of:
                markers.append(f"CONTINUATION of insight #{insight.continuation_of}")
            if insight.correction_of:
                markers.append(f"CORRECTION of insight #{insight.correction_of}")
            
            marker_text = f" ({', '.join(markers)})" if markers else ""
            
            formatted.append(
                f"- **{insight.insight_type}** {id_hint} {timing_hint}: "
                f"{insight.insight_text}{marker_text}"
            )
        
        return "\n".join(formatted)
    
    # ==================== Insight Consolidation Methods ====================
    
    def _find_duplicate_insight_types(
        self,
        insights: List[WindowInsight]
    ) -> Dict[str, List[WindowInsight]]:
        """Find insight types that appear more than once, excluding continuations/corrections.
        
        Args:
            insights: List of extracted insights from the current window
            
        Returns:
            Dictionary mapping insight_type to list of insights of that type,
            only for types with more than 1 non-continuation/correction insight
        """
        from collections import defaultdict
        
        # Group insights by type, excluding continuations and corrections
        type_groups: Dict[str, List[WindowInsight]] = defaultdict(list)
        
        for insight in insights:
            # Skip insights that are continuations or corrections - they are explicitly linked
            if insight.continuation_of is not None or insight.correction_of is not None:
                continue
            
            insight_type = insight.insight_type.value if hasattr(insight.insight_type, 'value') else str(insight.insight_type)
            type_groups[insight_type].append(insight)
        
        # Return only types with more than 1 insight
        return {
            insight_type: insight_list
            for insight_type, insight_list in type_groups.items()
            if len(insight_list) > 1
        }
    
    async def _consolidate_similar_insights(
        self,
        insights: List[WindowInsight]
    ) -> List[WindowInsight]:
        """Check for duplicate insight types and consolidate if similar.
        
        This method:
        1. Groups insights by type (excluding continuations/corrections)
        2. For each type with multiple insights, calls rapid_summary LLM to check similarity
        3. If similar, consolidates into a single insight
        
        Args:
            insights: List of extracted insights from the current window
            
        Returns:
            List of insights with similar duplicate types consolidated
        """
        if not self._rapid_llm_client or len(insights) < 2:
            return insights
        
        # Find duplicate insight types
        duplicate_types = self._find_duplicate_insight_types(insights)
        
        if not duplicate_types:
            return insights
        
        # Build a set of insight IDs to exclude (those that get consolidated)
        consolidated_ids: set = set()
        new_insights: List[WindowInsight] = []
        
        for insight_type, type_insights in duplicate_types.items():
            if len(type_insights) < 2:
                new_insights.append(type_insights[0])
                continue
            
            logger.info(
                f"Found {len(type_insights)} insights of type {insight_type}, "
                f"checking for consolidation: {[i.insight_text[:30] for i in type_insights]}"
            )
            
            # Format insights for the consolidation prompt
            insights_text = "\n".join(
                f"- Insight {i+1}: {insight.insight_text}"
                for i, insight in enumerate(type_insights)
            )
            
            # Call rapid_summary LLM to check similarity
            user_content = f"## Insights to Compare\n{insights_text}"
            
            try:
                reasoning, content, input_tokens, output_tokens, reasoning_tokens = await self._rapid_llm_client.create_completion(
                    system_prompt=INSIGHT_CONSOLIDATION_PROMPT,
                    user_content=user_content,
                    temperature=0.3,
                    max_tokens=1000,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "insight_consolidation",
                            "schema": self._consolidation_response_json_schema
                        }
                    }
                )
                
                # Parse the response
                try:
                    parsed = InsightConsolidationResponse.model_validate_json(content)
                    
                    if parsed.are_similar and parsed.consolidated_text.strip():
                        # Consolidate: create a single insight with combined text
                        # Use the highest confidence from the group
                        max_confidence = max(insight.confidence for insight in type_insights)
                        
                        # Use the classification from the first insight
                        first_classification = type_insights[0].classification
                        
                        consolidated_insight = WindowInsight(
                            insight_id=self._get_next_insight_id(),
                            insight_type=type_insights[0].insight_type.value if hasattr(type_insights[0].insight_type, 'value') else type_insights[0].insight_type,
                            insight_text=parsed.consolidated_text.strip(),
                            confidence=max_confidence,
                            window_id=type_insights[0].window_id,
                            timestamp_start=type_insights[0].timestamp_start,
                            timestamp_end=type_insights[0].timestamp_end,
                            classification=type_insights[0].classification.value if hasattr(type_insights[0].classification, 'value') else type_insights[0].classification,
                            continuation_of=None,
                            correction_of=None
                        )
                        
                        new_insights.append(consolidated_insight)
                        
                        # Mark these IDs as consolidated (exclude from final list)
                        for insight in type_insights:
                            consolidated_ids.add(id(insight))
                        
                        logger.info(
                            f"Consolidated {len(type_insights)} insights of type {insight_type} "
                            f"into: {parsed.consolidated_text[:50]}..."
                        )
                    else:
                        # Not similar - keep all insights
                        new_insights.extend(type_insights)
                        # Mark as processed so they're not added again
                        for insight in type_insights:
                            consolidated_ids.add(id(insight))
                        
                except Exception as e:
                    logger.warning(f"Failed to parse consolidation response: {e}")
                    # Keep all insights on parse failure
                    new_insights.extend(type_insights)
                    # Mark as processed so they're not added again
                    for insight in type_insights:
                        consolidated_ids.add(id(insight))
                    
            except Exception as e:
                logger.warning(f"Error calling consolidation LLM: {e}")
                # Keep all insights on error
                new_insights.extend(type_insights)
                # Mark as processed so they're not added again
                for insight in type_insights:
                    consolidated_ids.add(id(insight))
        
        # Add insights that weren't part of any duplicate type
        for insight in insights:
            if id(insight) not in consolidated_ids:
                new_insights.append(insight)
        
        return new_insights
    
    # ==================== Insight Processing Methods ====================
    
    def _parse_reference_id(self, value: Any, field_name: str) -> Optional[int]:
        """
        Parse and validate a reference ID field from LLM response.
        
        Args:
            value: The raw value from LLM response
            field_name: Name of the field for logging purposes
        
        Returns:
            Valid integer ID or None if invalid/missing
        """
        if value is None:
            return None
        
        try:
            # Handle string representations (LLM may return "42" instead of 42)
            if isinstance(value, str):
                parsed = int(value)
                if parsed <= 0:
                    logger.warning(f"Invalid {field_name}: '{value}' (must be positive integer)")
                    return None
                return parsed
            
            # Handle numeric types
            if isinstance(value, (int, float)):
                if value <= 0 or not float(value).is_integer():
                    logger.warning(f"Invalid {field_name}: {value} (must be positive integer)")
                    return None
                return int(value)
            
            # Log unexpected types
            logger.warning(f"Unexpected {field_name} type: {type(value).__name__} (value: {value})")
            return None
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse {field_name}: {value} ({e})")
            return None
    
    def _validate_insight_reference(
        self,
        ref_id: Optional[int],
        prior_insights: List[WindowInsight],
        field_name: str
    ) -> Optional[int]:
        """
        Validate that a reference ID exists in prior insights and resolve to root.
        
        If the referenced insight is itself a continuation/correction of another
        insight, this traverses the chain to find the root (original) insight ID.
        
        Example:
            - Insight #5: "Fix the bug" (root - no continuation/correction)
            - Insight #1: "We should fix the bug" - continuation_of=5
            - Insight #2: "We decided to fix the bug" - continuation_of=1
            
            If current insight references #1, returns #5 (root), not #1.
        
        Args:
            ref_id: The reference ID to validate
            prior_insights: List of prior insights to check against
            field_name: Name of the field for logging purposes
        
        Returns:
            Root insight ID or None if invalid/not found
        """
        if ref_id is None:
            return None
        
        # Build lookup map for prior insights by ID
        insights_by_id = {insight.insight_id: insight for insight in prior_insights}
        
        # Check if the referenced insight exists
        if ref_id not in insights_by_id:
            valid_ids = set(insights_by_id.keys())
            logger.warning(
                f"Invalid {field_name}: {ref_id} not found in prior insights. "
                f"Valid IDs: {sorted(valid_ids)[:10]}"
            )
            return None
        
        # Traverse the chain to find the root
        return self._resolve_to_root(ref_id, insights_by_id)
    
    def _resolve_to_root(
        self,
        insight_id: int,
        insights_by_id: Dict[int, WindowInsight]
    ) -> int:
        """
        Resolve an insight ID to its root by following continuation/correction chain.
        
        Args:
            insight_id: The starting insight ID
            insights_by_id: Map of insight_id -> WindowInsight
        
        Returns:
            The root insight ID (the original insight that is not a continuation/correction)
        """
        visited = set()  # Prevent infinite loops
        current_id = insight_id
        
        while current_id is not None:
            # Prevent infinite loop
            if current_id in visited:
                logger.warning(
                    f"Circular reference detected at insight {current_id}, "
                    f"stopping chain resolution"
                )
                return current_id
            
            visited.add(current_id)
            
            # Get the current insight
            current_insight = insights_by_id.get(current_id)
            if current_insight is None:
                # Shouldn't happen if validation passed, but handle gracefully
                return current_id
            
            # Check if this insight continues or corrects another
            next_id = current_insight.continuation_of or current_insight.correction_of
            
            if next_id is None:
                # This is the root - no more continuation/correction
                return current_id
            
            # Continue traversing
            current_id = next_id
        
        return insight_id  # Fallback
    
    def _extract_insights(
        self,
        parsed_data: Dict,
        window_id: int,
        window_start: float,
        window_end: float,
        prior_insights: Optional[List[WindowInsight]] = None,
        sentiment_enabled: bool = True,
        participants_enabled: bool = True
    ) -> Dict:
        """
        Extract insights from parsed JSON data, assign IDs, and insert into parsed_data.
        Filters out SENTIMENT insights when sentiment_enabled is False for the content type.
        
        Args:
            parsed_data: Parsed JSON data from LLM response (dict with "insights" key or list)
            window_id: The window these insights belong to
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
            prior_insights: Optional list of prior insights for reference validation
            sentiment_enabled: Whether to include SENTIMENT insights
            participants_enabled: Whether to include PARTICIPANTS insights
        
        Returns:
            Parsed data with insights array updated to include assigned insight_ids
        """
        if not parsed_data:
            return parsed_data
        
        # Handle both dict with "insights" key and list format
        if isinstance(parsed_data, list):
            # List format: insights are the list items themselves
            insights_list = parsed_data
            is_list_format = True
        else:
            # Dict format: insights are in "insights" key
            insights_list = parsed_data.get("insights", [])
            is_list_format = False
        
        insights_for_summary = []
        
        for item in insights_list:
            if not isinstance(item, dict):
                continue
            
            # Check if this is a SENTIMENT insight that should be filtered
            insight_type = item.get("insight_type", "")
            if insight_type == InsightType.SENTIMENT.value and not sentiment_enabled:
                logger.debug(
                    f"Filtering SENTIMENT insight for content type with sentiment_enabled=False: "
                    f"{item.get('insight_text', '')[:50]}..."
                )
                continue  # Skip this insight
            
            # Check if this is a PARTICIPANTS insight that should be filtered
            if insight_type == InsightType.PARTICIPANTS.value and not participants_enabled:
                logger.debug(
                    f"Filtering PARTICIPANTS insight for content type with participants_enabled=False: "
                    f"{item.get('insight_text', '')[:50]}..."
                )
                continue  # Skip this insight
            
            # Parse and validate continuation_of and correction_of
            continuation_of_raw = item.get("continuation_of")
            correction_of_raw = item.get("correction_of")
            
            continuation_of = self._parse_reference_id(continuation_of_raw, "continuation_of")
            correction_of = self._parse_reference_id(correction_of_raw, "correction_of")
            
            # Validate against prior insights if available
            if prior_insights is not None:
                continuation_of = self._validate_insight_reference(
                    continuation_of, prior_insights, "continuation_of"
                )
                correction_of = self._validate_insight_reference(
                    correction_of, prior_insights, "correction_of"
                )
            
            # Assign ID using local counter (for continuation/correction references)
            insight_id = self._get_next_insight_id()
            
            # Create WindowInsight with system-assigned ID
            insight = WindowInsight(
                insight_id=insight_id,
                insight_type=item.get("insight_type", "NOTES"),
                insight_text=item.get("insight_text", ""),
                confidence=item.get("confidence", 0.0),
                window_id=window_id,
                timestamp_start=window_start,
                timestamp_end=window_end,
                classification=item.get("classification", "~"),
                continuation_of=continuation_of,
                correction_of=correction_of,
            )
            
            # Keep WindowInsight objects throughout the pipeline for proper attribute access
            insights_for_summary.append(insight)
        
        # Update parsed_data with assigned insight_ids
        if is_list_format:
            # Return list format
            return insights_for_summary
        else:
            # Return dict format
            parsed_data["insights"] = insights_for_summary
            return parsed_data
    
    async def process_result(
        self,
        summary_text: str,
        summary_window_id: int,
        window_start: float,
        window_end: float,
        prior_insights: Optional[List[WindowInsight]] = None,
        sentiment_enabled: bool = True,
        participants_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Process the LLM result: parse JSON, extract insights, and build final payload.
        
        Args:
            summary_text: Raw JSON string from LLM response
            summary_window_id: The window ID these insights belong to
            window_start: Start timestamp of the window
            window_end: End timestamp of the window
            prior_insights: Optional list of prior insights for reference validation
            sentiment_enabled: Whether to include SENTIMENT insights
            participants_enabled: Whether to include PARTICIPANTS insights
        
        Returns:
            Processed payload with summary_text and insights
        """
        # Parse the JSON from LLM response
        try:
            parsed_data = json.loads(summary_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse summary JSON: {e}")
            return {
                "summary_text": "{}",
                "insights": [],
                "error": str(e)
            }
        
        # Validate topic field - must be present and a valid Topic enum value
        if isinstance(parsed_data, dict):
            topic_value = parsed_data.get("topic")
            if topic_value is None:
                logger.error("Topic field is missing from LLM response - topic is now required")
                return {
                    "summary_text": "{}",
                    "insights": [],
                    "error": "topic field is required but was not provided by LLM"
                }
            
            # Validate topic is a valid enum value
            try:
                parsed_data["topic"] = Topic(topic_value).value
            except ValueError:
                logger.warning(f"Invalid topic value '{topic_value}' from LLM, defaulting to 'GENERAL'")
                parsed_data["topic"] = Topic.GENERAL.value
        
        # Extract and process insights
        processed_data = self._extract_insights(
            parsed_data=parsed_data,
            window_id=summary_window_id,
            window_start=window_start,
            window_end=window_end,
            prior_insights=prior_insights,
            sentiment_enabled=sentiment_enabled,
            participants_enabled=participants_enabled
        )
        
        # Get the processed insights list
        if isinstance(processed_data, dict):
            processed_insights = processed_data.get("insights", [])
        else:
            processed_insights = processed_data if isinstance(processed_data, list) else []
        
        # Consolidate similar insights of the same type
        if self._rapid_llm_client and len(processed_insights) > 1:
            processed_insights = await self._consolidate_similar_insights(processed_insights)
            #logger.info(f"Consolidated insights: {processed_insights}")
        
        # Build the final payload - convert WindowInsight objects to dicts for JSON serialization
        insights_as_dicts = [
            insight.as_dict() if hasattr(insight, 'as_dict') else insight
            for insight in processed_insights
        ]
        
        # Also convert processed_data if it contains WindowInsight objects
        if isinstance(processed_data, dict):
            if "insights" in processed_data:
                processed_data = processed_data.copy()
                # Use processed_insights (which contains consolidated insights) instead of original processed_data
                processed_data["insights"] = [
                    insight.as_dict() if hasattr(insight, 'as_dict') else insight
                    for insight in processed_insights
                ]
            summary_text_json = json.dumps(processed_data)
        else:
            summary_text_json = summary_text
        
        payload = {
            "summary_text": summary_text_json,
            "insights": insights_as_dicts,
        }
        
        return payload


__all__ = ["ContextSummaryTask", "InsightsResponseSchema"]