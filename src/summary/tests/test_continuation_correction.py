"""
Unit tests for continuation_of and correction_of parsing and validation.

In the refactored code, these are handled by ContextSummaryTask and the
InsightResponseItemSchema in context_summary/task.py.
"""

import pytest
from src.summary.summary_client import SummaryClient
from src.summary.context_summary.task import WindowInsight, InsightResponseItemSchema, InsightType, ClassificationField


class TestContinuationCorrectionParsing:
    """Tests for continuation_of and correction_of parsing and validation."""
    
    def test_schema_includes_continuation_of_field(self):
        """Test that InsightResponseItemSchema includes continuation_of field."""
        schema = InsightResponseItemSchema.model_json_schema()
        assert "continuation_of" in schema["properties"], \
            "InsightResponseItemSchema should include continuation_of field"
    
    def test_schema_includes_correction_of_field(self):
        """Test that InsightResponseItemSchema includes correction_of field."""
        schema = InsightResponseItemSchema.model_json_schema()
        assert "correction_of" in schema["properties"], \
            "InsightResponseItemSchema should include correction_of_field"
    
    def test_insight_with_continuation_of(self):
        """Test creating insight with continuation_of field."""
        insight = InsightResponseItemSchema(
            insight_type=InsightType.KEY_POINT,
            insight_text="This continues the previous point",
            confidence=0.9,
            classification=ClassificationField.POSITIVE,
            continuation_of=1
        )
        
        assert insight.continuation_of == 1
        assert insight.insight_text == "This continues the previous point"
    
    def test_insight_with_correction_of(self):
        """Test creating insight with correction_of field."""
        insight = InsightResponseItemSchema(
            insight_type=InsightType.KEY_POINT,
            insight_text="This corrects the previous point",
            confidence=0.9,
            classification=ClassificationField.NEGATIVE,
            correction_of=2
        )
        
        assert insight.correction_of == 2
        assert insight.insight_text == "This corrects the previous point"
    
    def test_insight_without_continuation_or_correction(self):
        """Test creating insight without continuation or correction."""
        insight = InsightResponseItemSchema(
            insight_type=InsightType.ACTION,
            insight_text="Complete this task",
            confidence=0.95,
            classification=ClassificationField.POSITIVE
        )
        
        assert insight.continuation_of is None
        assert insight.correction_of is None


class TestWindowInsightContinuation:
    """Tests for WindowInsight continuation and correction fields."""
    
    def test_window_insight_continuation_of(self):
        """Test WindowInsight with continuation_of."""
        insight = WindowInsight(
            insight_id=1,
            insight_type="KEY POINT",
            insight_text="Test insight",
            confidence=0.9,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=10.0,
            continuation_of=5
        )
        
        assert insight.continuation_of == 5
    
    def test_window_insight_correction_of(self):
        """Test WindowInsight with correction_of."""
        insight = WindowInsight(
            insight_id=2,
            insight_type="KEY POINT",
            insight_text="Test correction",
            confidence=0.8,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=10.0,
            correction_of=1
        )
        
        assert insight.correction_of == 1
    
    def test_window_insight_as_dict(self):
        """Test WindowInsight.as_dict() includes continuation_of."""
        insight = WindowInsight(
            insight_id=1,
            insight_type="KEY POINT",
            insight_text="Test insight",
            confidence=0.9,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=10.0,
            continuation_of=5,
            correction_of=None
        )
        
        result = insight.as_dict()
        
        assert "continuation_of" in result
        assert result["continuation_of"] == 5


class TestWindowInsightPipeline:
    """Tests for the WindowInsight object pipeline.
    
    The fix ensures WindowInsight objects are used throughout the pipeline
    and only converted to dicts at the final step when building the payload.
    """
    
    def test_extract_insights_returns_window_insight_objects(self):
        """Test that _extract_insights returns WindowInsight objects, not dicts."""
        from src.summary.context_summary.task import ContextSummaryTask, WindowInsight
        from unittest.mock import MagicMock
        
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            rapid_llm_client=MagicMock(),
            content_type_state=MagicMock(),
            window_manager=MagicMock(),
        )
        
        # Simulate parsed data from LLM
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "KEY POINT",
                    "insight_text": "Test insight",
                    "confidence": 0.9,
                    "classification": "+",
                    "continuation_of": None,
                    "correction_of": None,
                }
            ]
        }
        
        result = task._extract_insights(
            parsed_data=parsed_data,
            window_id=1,
            window_start=0.0,
            window_end=10.0,
            prior_insights=None,
            sentiment_enabled=True,
            participants_enabled=True
        )
        
        # Should return dict with insights key
        assert isinstance(result, dict)
        assert "insights" in result
        
        # Insights should be WindowInsight objects, not dicts
        insights = result["insights"]
        assert len(insights) == 1
        assert isinstance(insights[0], WindowInsight)
        assert insights[0].insight_text == "Test insight"
    
    @pytest.mark.asyncio
    async def test_process_result_converts_to_dict_at_end(self):
        """Test that process_result converts WindowInsight to dict in the final payload."""
        from src.summary.context_summary.task import ContextSummaryTask
        from unittest.mock import MagicMock
        
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            rapid_llm_client=MagicMock(),
            content_type_state=MagicMock(),
            window_manager=MagicMock(),
        )
        
        # Simulate valid JSON response from LLM
        summary_text = '{"analysis": "Test", "insights": [{"insight_type": "KEY POINT", "insight_text": "Test", "confidence": 0.9, "classification": "+"}]}'
        
        result = await task.process_result(
            summary_text=summary_text,
            summary_window_id=1,
            window_start=0.0,
            window_end=10.0,
            prior_insights=None,
            sentiment_enabled=True,
            participants_enabled=True
        )
        
        # Final payload should have dict insights
        assert "insights" in result
        assert isinstance(result["insights"], list)
        assert len(result["insights"]) == 1
        # Should be dict now, not WindowInsight
        assert isinstance(result["insights"][0], dict)
        assert result["insights"][0]["insight_text"] == "Test"
    
    def test_find_duplicate_insight_types_with_window_insights(self):
        """Test that _find_duplicate_insight_types works with WindowInsight objects."""
        from src.summary.context_summary.task import ContextSummaryTask, WindowInsight
        from unittest.mock import MagicMock
        
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            rapid_llm_client=MagicMock(),
            content_type_state=MagicMock(),
            window_manager=MagicMock(),
        )
        
        # These are WindowInsight objects (as returned by _extract_insights after the fix)
        insights = [
            WindowInsight(
                insight_id=1,
                insight_type="KEY POINT",
                insight_text="First point",
                confidence=0.9,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=10.0,
                classification="+",
                continuation_of=None,
                correction_of=None,
            ),
            WindowInsight(
                insight_id=2,
                insight_type="KEY POINT",
                insight_text="Second point",
                confidence=0.8,
                window_id=0,
                timestamp_start=10.0,
                timestamp_end=20.0,
                classification="+",
                continuation_of=None,
                correction_of=None,
            ),
            WindowInsight(
                insight_id=3,
                insight_type="KEY POINT",
                insight_text="Continuation",
                confidence=0.7,
                window_id=0,
                timestamp_start=20.0,
                timestamp_end=30.0,
                classification="+",
                continuation_of=1,  # This is a continuation
                correction_of=None,
            ),
        ]
        
        result = task._find_duplicate_insight_types(insights)
        
        # Should find KEY POINT type with 2 non-continuation insights
        assert "KEY POINT" in result
        assert len(result["KEY POINT"]) == 2