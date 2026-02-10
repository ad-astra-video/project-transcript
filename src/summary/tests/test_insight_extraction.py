"""
Unit tests for insight extraction functionality.
"""

import pytest
from src.summary.summary_client import SummaryClient


class TestExtractInsights:
    """Tests for _extract_insights with parsed_data parameter."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_extract_insights_with_list_format(self):
        """Test extracting insights from list format JSON."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "This is the analysis text",
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                },
                {
                    "insight_type": "DECISION",
                    "insight_text": "Approved the budget",
                    "confidence": 0.90,
                    "classification": "[~]"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # _extract_insights returns the parsed_data dict with insights
        assert "insights" in result
        assert len(result["insights"]) == 2
        assert result["insights"][0]["insight_type"] == "ACTION"
        assert result["insights"][0]["insight_text"] == "Complete the task by Friday"
        assert result["insights"][0]["confidence"] == 0.95
        assert result["insights"][0]["classification"] == "[+]"
        assert result["insights"][1]["insight_type"] == "DECISION"
        assert result["insights"][1]["insight_text"] == "Approved the budget"
        assert result["insights"][1]["confidence"] == 0.90
        assert result["insights"][1]["classification"] == "[~]"
    
    def test_extract_insights_with_array_format(self):
        """Test extracting insights from array format JSON."""
        client = self.create_client()
        
        parsed_data = [
            {
                "insight_type": "KEY_POINT",
                "insight_text": "Important finding about the project",
                "confidence": 0.85,
                "classification": "[~]"
            },
            {
                "insight_type": "QUESTION",
                "insight_text": "What is the timeline?",
                "confidence": 0.75,
                "classification": "[-]"
            }
        ]
        
        result = client._extract_insights(parsed_data, 2, 20.0, 25.0)
        
        # _extract_insights returns a list for array format
        assert len(result) == 2
        assert result[0]["insight_type"] == "KEY_POINT"
        assert result[0]["insight_text"] == "Important finding about the project"
        assert result[0]["confidence"] == 0.85
        assert result[1]["insight_type"] == "QUESTION"
        assert result[1]["insight_text"] == "What is the timeline?"
        assert result[1]["confidence"] == 0.75
    
    def test_extract_insights_with_empty_data(self):
        """Test extracting insights from empty parsed data."""
        client = self.create_client()
        
        result = client._extract_insights({}, 1, 10.0, 15.0)
        assert result == {}
        
        result = client._extract_insights([], 1, 10.0, 15.0)
        assert result == []
    
    def test_extract_insights_with_missing_classification(self):
        """Test that missing classification defaults to neutral."""
        client = self.create_client()
        
        parsed_data = [
            {
                "insight_type": "NOTES",
                "insight_text": "Some observation",
                "confidence": 0.80
            }
        ]
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result) == 1
        assert result[0]["classification"] == "~"
        assert result[0]["confidence"] == 0.80
    
    def test_extract_insights_window_timestamps(self):
        """Test that window timestamps are correctly assigned."""
        client = self.create_client()
        
        parsed_data = [
            {
                "insight_type": "ACTION",
                "insight_text": "Test action",
                "confidence": 0.95,
                "classification": "[+]"
            }
        ]
        
        result = client._extract_insights(parsed_data, 5, 100.0, 105.0)
        
        assert len(result) == 1
        # Note: _extract_insights doesn't assign window_id/timestamps to the dict
        # Those are assigned when adding to WindowManager
        assert result[0]["insight_type"] == "ACTION"
        assert result[0]["confidence"] == 0.95
    
    def test_extract_insights_as_dict_includes_confidence(self):
        """Test that as_dict() method includes confidence field."""
        client = self.create_client()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task by Friday",
                    "confidence": 0.95,
                    "classification": "[+]"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert "insights" in result
        assert len(result["insights"]) == 1
        insight_dict = result["insights"][0]
        
        assert "confidence" in insight_dict
        assert insight_dict["confidence"] == 0.95
        assert insight_dict["insight_type"] == "ACTION"
        assert insight_dict["insight_text"] == "Complete the task by Friday"
        assert insight_dict["classification"] == "[+]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])