"""
Unit tests for continuation_of and correction_of parsing and validation.
"""

import pytest
from src.summary.summary_client import SummaryClient, WindowInsight


class TestContinuationCorrectionParsing:
    """Tests for continuation_of and correction_of parsing and validation."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_schema_includes_continuation_of_field(self):
        """Test that InsightResponseItemSchema includes continuation_of field."""
        from src.summary.summary_client import InsightResponseItemSchema
        
        schema = InsightResponseItemSchema.model_json_schema()
        assert "continuation_of" in schema["properties"], \
            "InsightResponseItemSchema should include continuation_of field"
    
    def test_schema_includes_correction_of_field(self):
        """Test that InsightResponseItemSchema includes correction_of field."""
        from src.summary.summary_client import InsightResponseItemSchema
        
        schema = InsightResponseItemSchema.model_json_schema()
        assert "correction_of" in schema["properties"], \
            "InsightResponseItemSchema should include correction_of field"
    
    def test_parse_reference_id_with_valid_integer(self):
        """Test parsing valid integer reference ID."""
        client = self.create_client()
        
        result = client._parse_reference_id(42, "continuation_of")
        assert result == 42
    
    def test_parse_reference_id_with_string_integer(self):
        """Test parsing string representation of integer."""
        client = self.create_client()
        
        result = client._parse_reference_id("42", "continuation_of")
        assert result == 42
    
    def test_parse_reference_id_with_float_integer(self):
        """Test parsing float that represents an integer."""
        client = self.create_client()
        
        result = client._parse_reference_id(42.0, "continuation_of")
        assert result == 42
    
    def test_parse_reference_id_with_none(self):
        """Test parsing None value returns None."""
        client = self.create_client()
        
        result = client._parse_reference_id(None, "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_zero(self):
        """Test parsing zero returns None (invalid)."""
        client = self.create_client()
        
        result = client._parse_reference_id(0, "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_negative(self):
        """Test parsing negative number returns None (invalid)."""
        client = self.create_client()
        
        result = client._parse_reference_id(-1, "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_string_negative(self):
        """Test parsing negative string returns None (invalid)."""
        client = self.create_client()
        
        result = client._parse_reference_id("-5", "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_non_integer_float(self):
        """Test parsing non-integer float returns None."""
        client = self.create_client()
        
        result = client._parse_reference_id(42.5, "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_invalid_string(self):
        """Test parsing invalid string returns None."""
        client = self.create_client()
        
        result = client._parse_reference_id("invalid", "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_array(self):
        """Test parsing array returns None."""
        client = self.create_client()
        
        result = client._parse_reference_id([42], "continuation_of")
        assert result is None
    
    def test_parse_reference_id_with_object(self):
        """Test parsing object returns None."""
        client = self.create_client()
        
        result = client._parse_reference_id({"id": 42}, "continuation_of")
        assert result is None
    
    def test_validate_insight_reference_with_valid_reference(self):
        """Test validation passes for valid reference to existing insight."""
        client = self.create_client()
        
        # Create prior insights
        prior_insights = [
            WindowInsight(
                insight_id=1,
                insight_type="ACTION",
                insight_text="First action",
                confidence=0.95,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=5.0,
                classification="+"
            ),
            WindowInsight(
                insight_id=2,
                insight_type="DECISION",
                insight_text="First decision",
                confidence=0.90,
                window_id=1,
                timestamp_start=5.0,
                timestamp_end=10.0,
                classification="~"
            )
        ]
        
        result = client._validate_insight_reference(1, prior_insights, "continuation_of")
        assert result == 1
    
    def test_validate_insight_reference_with_nonexistent_reference(self):
        """Test validation fails for reference to non-existent insight."""
        client = self.create_client()
        
        # Create prior insights
        prior_insights = [
            WindowInsight(
                insight_id=1,
                insight_type="ACTION",
                insight_text="First action",
                confidence=0.95,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=5.0,
                classification="+"
            )
        ]
        
        result = client._validate_insight_reference(99, prior_insights, "continuation_of")
        assert result is None
    
    def test_validate_insight_reference_with_none(self):
        """Test validation passes for None reference."""
        client = self.create_client()
        
        prior_insights = [
            WindowInsight(
                insight_id=1,
                insight_type="ACTION",
                insight_text="First action",
                confidence=0.95,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=5.0,
                classification="+"
            )
        ]
        
        result = client._validate_insight_reference(None, prior_insights, "continuation_of")
        assert result is None
    
    def test_extract_insights_with_continuation_of(self):
        """Test extracting insights with valid continuation_of field."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Additional detail on previous point",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": 1
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] == 1
    
    def test_extract_insights_with_correction_of(self):
        """Test extracting insights with valid correction_of field."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "DECISION",
                    "insight_text": "Actually going with Option B",
                    "confidence": 0.95,
                    "classification": "~",
                    "correction_of": 5
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["correction_of"] == 5
    
    def test_extract_insights_with_string_continuation_of(self):
        """Test extracting insights with string continuation_of (should be parsed)."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Additional detail",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": "42"  # String instead of int
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] == 42
    
    def test_extract_insights_with_both_fields(self):
        """Test extracting insights with both continuation_of and correction_of."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Complex insight with both references",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": 1,
                    "correction_of": 2
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] == 1
        assert result["insights"][0]["correction_of"] == 2
    
    def test_extract_insights_with_invalid_continuation_of(self):
        """Test extracting insights with invalid continuation_of (should be None)."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Insight with invalid reference",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": "invalid"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] is None
    
    def test_extract_insights_with_reference_validation(self):
        """Test extracting insights with reference validation against prior insights."""
        client = self.create_client()
        
        # Add prior insights to window manager
        prior_insight = WindowInsight(
            insight_id=1,
            insight_type="ACTION",
            insight_text="Prior action",
            confidence=0.95,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=5.0,
            classification="+"
        )
        client._window_manager.add_insight_to_window(0, prior_insight)
        
        # Create prior insights list for validation
        prior_insights = [prior_insight]
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Continuation of prior action",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": 1  # Valid reference
                }
            ]
        }
        
        result = client._extract_insights(
            parsed_data, 1, 10.0, 15.0, prior_insights
        )
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] == 1
    
    def test_extract_insights_with_nonexistent_reference_validation(self):
        """Test extracting insights with non-existent reference (should be None after validation)."""
        client = self.create_client()
        
        # Add prior insight to window manager
        prior_insight = WindowInsight(
            insight_id=1,
            insight_type="ACTION",
            insight_text="Prior action",
            confidence=0.95,
            window_id=0,
            timestamp_start=0.0,
            timestamp_end=5.0,
            classification="+"
        )
        client._window_manager.add_insight_to_window(0, prior_insight)
        
        # Create prior insights list for validation
        prior_insights = [prior_insight]
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Reference to non-existent insight",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": 99  # Invalid reference
                }
            ]
        }
        
        result = client._extract_insights(
            parsed_data, 1, 10.0, 15.0, prior_insights
        )
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] is None  # Should be None after validation
    
    def test_extract_insights_without_prior_insights(self):
        """Test extracting insights without prior insights (validation skipped)."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Insight without prior context",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": 99  # Would be invalid, but no validation without prior_insights
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        # Without prior_insights, reference validation is skipped
        # Type validation passes since 99 is a valid integer
        # This is expected behavior - existence validation only happens when prior_insights provided
        assert result["insights"][0]["continuation_of"] == 99
    
    def test_extract_insights_with_zero_continuation_of(self):
        """Test extracting insights with zero continuation_of (invalid)."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "NOTES",
                    "insight_text": "Insight with zero reference",
                    "confidence": 0.85,
                    "classification": "~",
                    "continuation_of": 0
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["continuation_of"] is None
    
    def test_extract_insights_with_negative_correction_of(self):
        """Test extracting insights with negative correction_of (invalid)."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "DECISION",
                    "insight_text": "Correction with negative reference",
                    "confidence": 0.85,
                    "classification": "~",
                    "correction_of": -5
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 1
        assert result["insights"][0]["correction_of"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])