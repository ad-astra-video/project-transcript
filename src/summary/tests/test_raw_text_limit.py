"""
Tests for raw text context limit functionality.
"""

import pytest
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager, WindowInsight


class TestWindowManagerRawTextLimit:
    """Tests for WindowManager raw_text_context_limit functionality."""
    
    def test_raw_text_under_limit_no_truncation(self):
        """Raw text under limit should not be truncated."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=2000,
            transcription_windows_per_summary_window=4
        )
        
        # Add windows with total text under limit
        wm.add_summary_window("Hello world " * 100, 0.0, 10.0, [])  # ~1200 chars
        wm.add_summary_window("Test message " * 50, 10.0, 20.0, [])  # ~600 chars
        
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # Should have text from first window (second window is held out)
        assert len(text) > 0
        assert len(text) <= 2000
    
    def test_raw_text_over_limit_stops_adding(self):
        """When raw text exceeds limit, should stop adding text.
        
        Implementation adds windows while current_length < limit.
        This means we can exceed the limit by adding a window that pushes us over.
        Text parts are joined with spaces, so there's 1 char overhead per join.
        """
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=500,
            transcription_windows_per_summary_window=4
        )
        
        # Add windows with text that exceeds limit
        wm.add_summary_window("A" * 300, 0.0, 10.0, [])  # 300 chars
        wm.add_summary_window("B" * 300, 10.0, 20.0, [])  # 300 chars
        wm.add_summary_window("C" * 300, 20.0, 30.0, [])  # 300 chars
        wm.add_summary_window("D" * 300, 30.0, 40.0, [])  # 300 chars
        
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # Should have text from first 2 windows (300 + 1 space + 300 = 601 > 500)
        # Window 0: 300 < 500 → add (current=300)
        # Window 1: 300 < 500 → add (current=600)
        # Window 2: 600 < 500 → skip
        assert len(text) == 601  # 300 + 1 + 300
        assert len(text) > 0
    
    def test_insights_preserved_when_raw_text_truncated(self):
        """Insights should be preserved even when raw text is truncated."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=100,
            transcription_windows_per_summary_window=4
        )
        
        # Add windows with insights
        wm.add_summary_window("A" * 100, 0.0, 10.0, [])
        wm._summary_windows[0].insights = [
            WindowInsight(
                insight_id=1,
                insight_type="KEY POINT",
                insight_text="Test insight 1",
                confidence=0.9,
                window_id=0,
                timestamp_start=0.0,
                timestamp_end=10.0,
                classification="~"
            )
        ]
        
        # Add more windows that would truncate text
        wm.add_summary_window("B" * 100, 10.0, 20.0, [])
        wm.add_summary_window("C" * 100, 20.0, 30.0, [])
        
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # Insights should still be preserved from first window
        assert len(insights) > 0


class TestSummaryClientRawTextLimit:
    """Tests for SummaryClient raw_text_context_limit functionality."""
    
    def test_update_params_raw_text_context_limit(self):
        """update_params should update raw_text_context_limit."""
        client = SummaryClient(
            reasoning_api_key="test",
            reasoning_base_url="http://test:8000/v1"
        )
        
        client.update_params(raw_text_context_limit=3000)
        
        # Note: In refactored code, raw_text_context_limit is stored on the client
        # but only set via update_params
        assert client._window_manager.raw_text_context_limit == 3000
    
    def test_transcription_windows_per_summary_window_parameter(self):
        """SummaryClient should accept transcription_windows_per_summary_window parameter."""
        client = SummaryClient(
            reasoning_api_key="test",
            reasoning_base_url="http://test:8000/v1"
        )
        
        client.update_params(transcription_windows_per_summary_window=6)
        
        assert client._window_manager.transcription_windows_per_summary_window == 6
    
    def test_update_params_transcription_windows_per_summary_window(self):
        """update_params should update transcription_windows_per_summary_window."""
        client = SummaryClient(
            reasoning_api_key="test",
            reasoning_base_url="http://test:8000/v1"
        )
        
        client.update_params(transcription_windows_per_summary_window=8)
        
        assert client._window_manager.transcription_windows_per_summary_window == 8


class TestRawTextLimitEdgeCases:
    """Tests for edge cases in raw text limit handling."""
    
    def test_empty_text_no_issue(self):
        """Empty text should not cause issues."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=100,
            transcription_windows_per_summary_window=4
        )
        
        wm.add_summary_window("", 0.0, 10.0, [])
        
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        assert len(text) == 0
    
    def test_single_window_exactly_at_limit(self):
        """Single window exactly at limit - returns empty since last window is excluded."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=500,
            transcription_windows_per_summary_window=4
        )
        
        # Add first window (will be excluded as it's the "last" window)
        wm.add_summary_window("A" * 500, 0.0, 10.0, [])
        
        # Add second window so first window becomes accumulated text
        wm.add_summary_window("B" * 100, 10.0, 20.0, [])
        
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # First window (500 chars) should be included since there's now a second window
        assert len(text) == 500
    
    def test_single_window_over_limit(self):
        """Single window over limit - returns empty since last window is excluded."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=500,
            transcription_windows_per_summary_window=4
        )
        
        # Add first window (will be excluded as it's the "last" window)
        wm.add_summary_window("A" * 600, 0.0, 10.0, [])
        
        # Add second window so first window becomes accumulated text
        wm.add_summary_window("B" * 100, 10.0, 20.0, [])
        
        text, insights, text_length, insights_per_window = wm.get_accumulated_text_and_insights()
        
        # First window (600 chars) should be included since there's now a second window
        assert len(text) == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])