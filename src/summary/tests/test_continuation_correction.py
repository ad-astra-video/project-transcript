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
            insight_id=1,
            insight_type="KEY POINT",
            insight_text="Test insight",
            confidence=0.9,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=10.0,
            correction_of=3
        )
        
        assert insight.correction_of == 3
    
    def test_window_insight_as_dict_includes_continuation(self):
        """Test that as_dict includes continuation and correction fields."""
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
        assert "correction_of" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])