"""
Unit tests for sentiment filtering functionality.
"""

import pytest
from src.summary.summary_client import SummaryClient


class TestSentimentEnabledFiltering:
    """Tests for sentiment_enabled filtering in _extract_insights."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_is_sentiment_enabled_returns_true_for_general_meeting(self):
        """Test that is_sentiment_enabled returns True for GENERAL_MEETING."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is True
    
    def test_is_sentiment_enabled_returns_true_for_customer_support(self):
        """Test that is_sentiment_enabled returns True for CUSTOMER_SUPPORT."""
        client = self.create_client()
        client.set_content_type("CUSTOMER_SUPPORT", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is True
    
    def test_is_sentiment_enabled_returns_false_for_technical_talk(self):
        """Test that is_sentiment_enabled returns False for TECHNICAL_TALK."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_lecture_or_talk(self):
        """Test that is_sentiment_enabled returns False for LECTURE_OR_TALK."""
        client = self.create_client()
        client.set_content_type("LECTURE_OR_TALK", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_interview(self):
        """Test that is_sentiment_enabled returns False for INTERVIEW."""
        client = self.create_client()
        client.set_content_type("INTERVIEW", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_podcast(self):
        """Test that is_sentiment_enabled returns False for PODCAST."""
        client = self.create_client()
        client.set_content_type("PODCAST", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_news_update(self):
        """Test that is_sentiment_enabled returns False for NEWS_UPDATE."""
        client = self.create_client()
        client.set_content_type("NEWS_UPDATE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_gameplay_commentary(self):
        """Test that is_sentiment_enabled returns False for GAMEPLAY_COMMENTARY."""
        client = self.create_client()
        client.set_content_type("GAMEPLAY_COMMENTARY", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_debate(self):
        """Test that is_sentiment_enabled returns False for DEBATE."""
        client = self.create_client()
        client.set_content_type("DEBATE", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_is_sentiment_enabled_returns_false_for_unknown(self):
        """Test that is_sentiment_enabled returns False for UNKNOWN."""
        client = self.create_client()
        client.set_content_type("UNKNOWN", confidence=0.9, source="AUTO_DETECTED")
        assert client.is_sentiment_enabled() is False
    
    def test_extract_insights_includes_sentiment_when_enabled(self):
        """Test that SENTIMENT insights are included when sentiment_enabled=True."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Positive tone detected",
                    "confidence": 0.85,
                    "classification": "+"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task",
                    "confidence": 0.95,
                    "classification": "+"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include both insights (SENTIMENT + ACTION)
        assert len(result["insights"]) == 2
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert "SENTIMENT" in insight_types
        assert "ACTION" in insight_types
    
    def test_extract_insights_filters_sentiment_when_disabled(self):
        """Test that SENTIMENT insights are filtered when sentiment_enabled=False."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Positive tone detected",
                    "confidence": 0.85,
                    "classification": "+"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Complete the task",
                    "confidence": 0.95,
                    "classification": "+"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should only include ACTION insight (SENTIMENT filtered out)
        assert len(result["insights"]) == 1
        assert result["insights"][0]["insight_type"] == "ACTION"
    
    def test_extract_insights_non_sentiment_always_included(self):
        """Test that non-SENTIMENT insights are always included regardless of sentiment_enabled."""
        client = self.create_client()
        client.set_content_type("TECHNICAL_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "DECISION",
                    "insight_text": "Approved the budget",
                    "confidence": 0.90,
                    "classification": "~"
                },
                {
                    "insight_type": "QUESTION",
                    "insight_text": "What is the timeline?",
                    "confidence": 0.75,
                    "classification": "~"
                },
                {
                    "insight_type": "KEY_POINT",
                    "insight_text": "Important finding",
                    "confidence": 0.85,
                    "classification": "~"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include all non-SENTIMENT insights
        assert len(result["insights"]) == 3
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert "DECISION" in insight_types
        assert "QUESTION" in insight_types
        assert "KEY_POINT" in insight_types
    
    def test_extract_insights_mixed_insights_partially_filtered(self):
        """Test that mixed insights (SENTIMENT + others) are partially filtered correctly."""
        client = self.create_client()
        client.set_content_type("CUSTOMER_SUPPORT", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Customer frustration detected",
                    "confidence": 0.85,
                    "classification": "-"
                },
                {
                    "insight_type": "ACTION",
                    "insight_text": "Escalate to manager",
                    "confidence": 0.95,
                    "classification": "+"
                },
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Positive resolution",
                    "confidence": 0.80,
                    "classification": "+"
                },
                {
                    "insight_type": "DECISION",
                    "insight_text": "Issue resolved",
                    "confidence": 0.90,
                    "classification": "~"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # Should include all insights (sentiment_enabled=True for CUSTOMER_SUPPORT)
        assert len(result["insights"]) == 4
        insight_types = [i["insight_type"] for i in result["insights"]]
        assert insight_types.count("SENTIMENT") == 2
        assert "ACTION" in insight_types
        assert "DECISION" in insight_types
    
    def test_extract_insights_empty_insights_list(self):
        """Test that empty insights list is handled correctly."""
        client = self.create_client()
        client.set_content_type("GENERAL_MEETING", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": []
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        assert len(result["insights"]) == 0
    
    def test_extract_insights_only_sentiment_filtered(self):
        """Test that when only SENTIMENT insights exist and are filtered, result is empty."""
        client = self.create_client()
        client.set_content_type("LECTURE_OR_TALK", confidence=0.9, source="AUTO_DETECTED")
        
        parsed_data = {
            "analysis": "Test analysis",
            "insights": [
                {
                    "insight_type": "SENTIMENT",
                    "insight_text": "Engaged audience",
                    "confidence": 0.85,
                    "classification": "+"
                }
            ]
        }
        
        result = client._extract_insights(parsed_data, 1, 10.0, 15.0)
        
        # SENTIMENT should be filtered out
        assert len(result["insights"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])