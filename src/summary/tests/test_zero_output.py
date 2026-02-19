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
            "analysis": "",
            "insights": []
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should return empty insights
        assert "insights" in result
        assert len(result["insights"]) == 0
    
    def test_empty_analysis_with_insights(self):
        """Test that empty analysis string is valid when insights exist."""
        task = self.create_task()
        
        parsed_data = {
            "analysis": "",
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
        assert result["insights"][0]["insight_text"] == "Complete the task by Friday"
    
    def test_missing_insights_key(self):
        """Test that missing insights key is handled correctly."""
        task = self.create_task()
        
        parsed_data = {
            "analysis": "Some analysis"
            # insights key missing
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should return empty insights
        assert "insights" in result
        assert len(result["insights"]) == 0
    
    def test_invalid_insights_type(self):
        """Test that invalid insights type is handled correctly."""
        task = self.create_task()
        
        parsed_data = {
            "analysis": "Some analysis",
            "insights": "not a list"  # Should be a list
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should return empty insights
        assert "insights" in result
        assert len(result["insights"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])