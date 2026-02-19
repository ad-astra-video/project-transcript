"""
Unit tests for KEY POINT classification with breakthrough-level threshold.

In the refactored code, this is handled by ContextSummaryTask.
"""

import pytest
from unittest.mock import MagicMock
from src.summary.summary_client import SummaryClient
from src.summary.context_summary.task import ContextSummaryTask


class TestKeyPointClassification:
    """Tests for KEY POINT classification with breakthrough-level threshold."""
    
    def create_task(self):
        """Create a ContextSummaryTask instance for testing."""
        mock_client = MagicMock()
        return ContextSummaryTask(
            llm_client=mock_client,
            content_type_state=MagicMock(),
            window_manager=MagicMock()
        )
    
    def test_key_point_classification_for_critical_threshold(self):
        """Test that critical thresholds are classified as KEY POINT."""
        task = self.create_task()
        
        # These should be KEY POINT - critical thresholds
        critical_threshold_insights = [
            "The system fails above 10,000 concurrent connections",
            "At 500ms latency, user experience degrades significantly",
            "The error rate is below 1%",
            "Budget is $50,000",
        ]
        
        for insight_text in critical_threshold_insights:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "KEY POINT",
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected KEY POINT for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "KEY POINT"
    
    def test_notes_classification_for_explanations(self):
        """Test that explanations are classified as NOTES, not KEY POINT."""
        task = self.create_task()
        
        # These should be NOTES - explanations of how things work
        explanation_insights = [
            "The system uses a microservices architecture",
            "Data is stored in a PostgreSQL database",
            "The API is built using FastAPI",
        ]
        
        for insight_text in explanation_insights:
            parsed_data = {
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_type": "NOTES",
                        "insight_text": insight_text,
                        "confidence": 0.95,
                        "classification": "~"
                    }
                ]
            }
            
            result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
            assert len(result["insights"]) == 1, f"Expected NOTES for: {insight_text}"
            assert result["insights"][0]["insight_type"] == "NOTES"
    
    def test_action_classification_for_tasks(self):
        """Test that tasks are classified as ACTION."""
        task = self.create_task()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the project by Friday",
                    "confidence": 0.95,
                    "classification": "+"
                }
            ]
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        assert result["insights"][0]["insight_type"] == "ACTION"
    
    def test_decision_classification_for_decisions(self):
        """Test that decisions are classified as DECISION."""
        task = self.create_task()
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "DECISION",
                    "insight_text": "Approved the budget increase",
                    "confidence": 0.95,
                    "classification": "+"
                }
            ]
        }
        
        result = task._extract_insights(parsed_data, 1, 10.0, 15.0)
        assert result["insights"][0]["insight_type"] == "DECISION"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])