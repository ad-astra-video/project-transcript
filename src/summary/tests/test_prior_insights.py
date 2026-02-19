"""
Unit tests for prior insights accumulation functionality.

In the refactored code, prior insights are handled by WindowManager.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager, WindowInsight


class TestPriorInsightsAccumulation:
    """Tests for prior insights accumulation across windows in WindowManager."""
    
    def create_window_manager(self):
        """Create a WindowManager instance for testing."""
        return WindowManager()
    
    def test_first_window_has_zero_prior_insights(self):
        """Test that first summary window has no prior insights."""
        wm = self.create_window_manager()
        
        # Add first summary window
        wm.add_summary_window("First text", 0.0, 10.0, [1])
        
        # Get accumulated text and insights
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # First window should have no prior insights
        assert len(insights) == 0
    
    def test_second_window_has_prior_insights_from_first(self):
        """Test that second window has prior insights from first window."""
        wm = self.create_window_manager()
        
        # Add first window with insights
        wm.add_summary_window("First text", 0.0, 10.0, [1])
        wm._summary_windows[0].insights = [
            WindowInsight(
                insight_id=1,
                insight_type="KEY POINT",
                insight_text="First insight",
                confidence=0.9,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=10.0,
                classification="~"
            )
        ]
        
        # Add second window
        wm.add_summary_window("Second text", 10.0, 20.0, [2])
        
        # Get accumulated text and insights
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # Should have insights from first window
        assert len(insights) >= 1
    
    def test_insights_stored_per_window(self):
        """Test that insights are stored per window."""
        wm = self.create_window_manager()
        
        # Add first window with insights
        wm.add_summary_window("First text", 0.0, 10.0, [1])
        wm._summary_windows[0].insights = [
            WindowInsight(
                insight_id=1,
                insight_type="ACTION",
                insight_text="First action",
                confidence=0.9,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=10.0,
                classification="+"
            )
        ]
        
        # Add second window with insights
        wm.add_summary_window("Second text", 10.0, 20.0, [2])
        wm._summary_windows[1].insights = [
            WindowInsight(
                insight_id=2,
                insight_type="DECISION",
                insight_text="Second decision",
                confidence=0.85,
                window_id=1,
                timestamp_start=10.0,
                timestamp_end=20.0,
                classification="~"
            )
        ]
        
        # Verify each window has its own insights
        assert len(wm._summary_windows[0].insights) == 1
        assert len(wm._summary_windows[1].insights) == 1
        assert wm._summary_windows[0].insights[0].insight_type == "ACTION"
        assert wm._summary_windows[1].insights[0].insight_type == "DECISION"


class TestSummaryClientWithPriorInsights:
    """Tests for SummaryClient handling prior insights through WindowManager."""
    
    def test_summary_client_window_manager_stores_insights(self):
        """Test that SummaryClient's WindowManager stores insights."""
        client = SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
        
        # Add window with insights
        client._window_manager.add_summary_window("Test text", 0.0, 10.0, [1])
        client._window_manager._summary_windows[0].insights = [
            WindowInsight(
                insight_id=1,
                insight_type="KEY POINT",
                insight_text="Test insight",
                confidence=0.9,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=10.0,
                classification="~"
            )
        ]
        
        # Verify insights stored
        assert len(client._window_manager._summary_windows[0].insights) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])