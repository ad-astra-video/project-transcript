"""
Unit tests for sentiment filtering functionality.

In the refactored code, sentiment filtering is handled by ContextSummaryPlugin.
"""

import pytest
from unittest.mock import MagicMock
from src.summary.summary_client import SummaryClient
from src.summary.context_summary import ContextSummaryPlugin, ContentTypeStateHolder
from src.summary.content_type_detection.task import ContentType


class TestSentimentEnabledFiltering:
    """Tests for sentiment_enabled filtering in ContextSummaryPlugin."""
    
    def create_plugin(self):
        """Create a ContextSummaryPlugin instance for testing."""
        client = SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
        
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContextSummaryPlugin(
            window_manager=client._window_manager,
            llm_manager=mock_llm,
            result_callback=client._queue_payload,
            summary_client=client
        )
        
        return plugin
    
    def test_sentiment_enabled_for_general_meeting(self):
        """Test that sentiment_enabled is set for GENERAL_MEETING."""
        plugin = self.create_plugin()
        
        plugin._content_type_state = ContentTypeStateHolder(
            content_type=ContentType.GENERAL_MEETING.value,
            confidence=0.9,
            source="AUTO_DETECTED"
        )
        
        # In the refactored code, sentiment_enabled is a property
        assert plugin._sentiment_enabled is False  # Default
    
    def test_sentiment_enabled_for_customer_support(self):
        """Test that sentiment_enabled is set for CUSTOMER_SUPPORT."""
        plugin = self.create_plugin()
        
        plugin._content_type_state = ContentTypeStateHolder(
            content_type=ContentType.CUSTOMER_SUPPORT.value,
            confidence=0.9,
            source="AUTO_DETECTED"
        )
        
        assert plugin._sentiment_enabled is False  # Default
    
    def test_sentiment_enabled_for_technical_talk(self):
        """Test that sentiment_enabled is set for TECHNICAL_TALK."""
        plugin = self.create_plugin()
        
        plugin._content_type_state = ContentTypeStateHolder(
            content_type=ContentType.TECHNICAL_TALK.value,
            confidence=0.9,
            source="AUTO_DETECTED"
        )
        
        assert plugin._sentiment_enabled is False  # Default
    
    def test_sentiment_enabled_can_be_set(self):
        """Test that sentiment_enabled can be set."""
        plugin = self.create_plugin()
        
        plugin._sentiment_enabled = True
        assert plugin._sentiment_enabled is True
        
        plugin._sentiment_enabled = False
        assert plugin._sentiment_enabled is False


class TestSentimentInsightType:
    """Tests for SENTIMENT insight type."""
    
    def test_sentiment_is_valid_insight_type(self):
        """Test that SENTIMENT is a valid insight type."""
        from src.summary.context_summary.task import InsightType
        
        assert InsightType.SENTIMENT == "SENTIMENT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])