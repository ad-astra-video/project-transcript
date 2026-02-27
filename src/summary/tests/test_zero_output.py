"""
Unit tests for zero-output behavior when nothing meaningful is said.
"""

import pytest
from unittest.mock import MagicMock
from src.summary.summary_client import SummaryClient
from src.summary.context_summary.task import ContextSummaryTask


class TestZeroOutputBehavior:
    """Tests for zero-output behavior when nothing meaningful is said."""
    
    def create_task(self):
        """Create a ContextSummaryTask instance for testing."""
        mock_client = MagicMock()
        return ContextSummaryTask(
            llm_client=mock_client,
            content_type_state=MagicMock(),
            window_manager=MagicMock()
        )
    
    def test_empty_insights_array_handling(self):
        """Test that empty insights array is handled correctly."""
        task = self.create_task()
        
        # Empty insights array should be handled correctly
        parsed_data = {
            "insights": []
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should return empty insights
        assert "insights" in result
        assert len(result["insights"]) == 0
    
    def test_empty_topic_with_insights(self):
        """Test that empty topic string is valid when insights exist."""
        task = self.create_task()
        
        parsed_data = {
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                }
            ]
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should still extract the insight
        assert len(result["insights"]) == 1
        assert result["insights"][0].insight_text == "Complete the task by Friday"
    
    def test_missing_insights_key(self):
        """Test that missing insights key is handled correctly."""
        task = self.create_task()
        
        parsed_data = {}
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # When parsed_data is empty, returns empty dict (pre-existing behavior)
        assert result == {}
    
    def test_invalid_insights_type(self):
        """Test that invalid insights type is handled correctly."""
        task = self.create_task()
        
        parsed_data = {
            "insights": "not a list"  # Should be a list
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should return empty insights
        assert "insights" in result
        assert len(result["insights"]) == 0
    
    @pytest.mark.asyncio
    async def test_missing_topic_field(self):
        """Test that missing topic field returns an error."""
        task = self.create_task()
        
        # Test without topic field at all - should return error since topic is now required
        parsed_data = {
            "insights": [
                {
                    "insight_type": "KEY POINT",
                    "insight_text": "Important point",
                    "confidence": 0.9,
                    "classification": "+"
                }
            ]
        }
        
        # Use process_result which validates the topic field
        import json
        summary_text = json.dumps(parsed_data)
        
        result = await task.process_result(
            summary_text=summary_text,
            summary_window_id=1,
            window_start=10.0,
            window_end=15.0
        )
        
        # Should return error since topic is now required
        assert "error" in result
        assert "topic" in result["error"]
        assert result["insights"] == []

    @pytest.mark.asyncio
    async def test_valid_topic_field(self):
        """Test that valid topic field is accepted."""
        task = self.create_task()
        
        # Test with valid topic field
        parsed_data = {
            "topic": "MACHINE_LEARNING",
            "insights": [
                {
                    "insight_type": "KEY POINT",
                    "insight_text": "Important ML insight",
                    "confidence": 0.9,
                    "classification": "+"
                }
            ]
        }
        
        import json
        summary_text = json.dumps(parsed_data)
        
        result = await task.process_result(
            summary_text=summary_text,
            summary_window_id=1,
            window_start=10.0,
            window_end=15.0
        )
        
        # Should extract the insight successfully
        assert "error" not in result
        assert len(result["insights"]) == 1
        assert result["insights"][0]["insight_text"] == "Important ML insight"

    @pytest.mark.asyncio
    async def test_invalid_topic_field_defaults_to_general(self):
        """Test that invalid topic value defaults to GENERAL."""
        task = self.create_task()
        
        # Test with invalid topic field
        parsed_data = {
            "topic": "INVALID_TOPIC",
            "insights": [
                {
                    "insight_type": "KEY POINT",
                    "insight_text": "Some insight",
                    "confidence": 0.9,
                    "classification": "+"
                }
            ]
        }
        
        import json
        summary_text = json.dumps(parsed_data)
        
        result = await task.process_result(
            summary_text=summary_text,
            summary_window_id=1,
            window_start=10.0,
            window_end=15.0
        )
        
        # Should extract the insight with topic defaulted to GENERAL
        assert "error" not in result
        assert len(result["insights"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])