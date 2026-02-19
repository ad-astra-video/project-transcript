"""
Unit tests for insight extraction functionality.

In the refactored code, insight extraction is handled by ContextSummaryTask.
"""

import pytest
from unittest.mock import MagicMock
from src.summary.summary_client import SummaryClient
from src.summary.context_summary.task import ContextSummaryTask, InsightType, ClassificationField


class TestExtractInsights:
    """Tests for _extract_insights in ContextSummaryTask."""
    
    def create_task(self):
        """Create a ContextSummaryTask instance for testing."""
        from src.summary.window_manager import WindowManager
        mock_client = MagicMock()
        return ContextSummaryTask(
            llm_client=mock_client,
            content_type_state=MagicMock(),
            window_manager=WindowManager()
        )
    
    def test_extract_insights_with_list_format(self):
        """Test extracting insights from list format JSON."""
        task = self.create_task()
        
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
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
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
    
    def test_extract_insights_with_empty_list(self):
        """Test extracting insights with empty list."""
        task = self.create_task()
        
        parsed_data = {
            "analysis": "Analysis",
            "insights": []
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert "insights" in result
        assert len(result["insights"]) == 0
    
    def test_extract_insights_assigns_window_id(self):
        """Test that insights are assigned the correct window_id and stored in WindowManager."""
        task = self.create_task()
        
        # Add the summary window first (required for adding insights)
        task._window_manager.add_summary_window("Test content", 10.0, 15.0, [1])
        
        parsed_data = {
            "analysis": "Analysis",
            "insights": [
                {
                    "insight_type": "KEY POINT",
                    "insight_text": "Test insight",
                    "confidence": 0.9,
                    "classification": "~"
                }
            ]
        }
        
        result = task._extract_insights(parsed_data, 0, 10.0, 15.0)
        
        # In refactored code, insights are stored in WindowManager with window_id
        # Verify insight was added to window 0
        window_insights = task._window_manager.get_window_insights(0)
        assert len(window_insights) == 1
        assert window_insights[0].window_id == 0


class TestInsightTypeEnum:
    """Tests for InsightType enum."""
    
    def test_all_insight_types_exist(self):
        """Test that all expected insight types exist."""
        assert InsightType.ACTION == "ACTION"
        assert InsightType.DECISION == "DECISION"
        assert InsightType.QUESTION == "QUESTION"
        assert InsightType.KEY_POINT == "KEY POINT"
        assert InsightType.RISK == "RISK"
        assert InsightType.SENTIMENT == "SENTIMENT"
        assert InsightType.PARTICIPANTS == "PARTICIPANTS"
        assert InsightType.NOTES == "NOTES"


class TestClassificationField:
    """Tests for ClassificationField enum."""
    
    def test_all_classification_fields_exist(self):
        """Test that all expected classification fields exist."""
        assert ClassificationField.POSITIVE == "+"
        assert ClassificationField.NEUTRAL == "~"
        assert ClassificationField.NEGATIVE == "-"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])