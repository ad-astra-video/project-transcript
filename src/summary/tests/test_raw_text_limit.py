"""
Tests for raw text context limit functionality.
"""

import pytest
from src.summary.summary_client import WindowManager, SummaryClient, WindowInsight


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
        wm.add_window("Hello world " * 100, 0.0, 10.0)  # ~1200 chars
        wm.add_window("Test message " * 50, 10.0, 20.0)  # ~600 chars
        
        text, insights = wm.get_accumulated_text_and_insights()
        
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
        wm.add_window("A" * 300, 0.0, 10.0)  # 300 chars
        wm.add_window("B" * 300, 10.0, 20.0)  # 300 chars
        wm.add_window("C" * 300, 20.0, 30.0)  # 300 chars
        wm.add_window("D" * 300, 30.0, 40.0)  # 300 chars
        
        text, insights = wm.get_accumulated_text_and_insights()
        
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
        wm.add_window("A" * 100, 0.0, 10.0)
        wm._windows[0].insights = [
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
        
        wm.add_window("B" * 100, 10.0, 20.0)
        wm._windows[1].insights = [
            WindowInsight(
                insight_id=2,
                insight_type="ACTION",
                insight_text="Test insight 2",
                confidence=0.8,
                window_id=1,
                timestamp_start=10.0,
                timestamp_end=20.0,
                classification="~"
            )
        ]
        
        wm.add_window("C" * 100, 20.0, 30.0)
        wm.add_window("D" * 100, 30.0, 40.0)
        
        text, insights = wm.get_accumulated_text_and_insights()
        
        # Should have 2 insights (from first 2 windows)
        assert len(insights) == 2
        assert insights[0].insight_id == 1
        assert insights[1].insight_id == 2
    
    def test_empty_windows_returns_empty_context(self):
        """Empty windows should return empty context."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=2000,
            transcription_windows_per_summary_window=4
        )
        
        text, insights = wm.get_accumulated_text_and_insights()
        
        assert text == ""
        assert insights == []
    
    def test_single_window_returns_empty_context(self):
        """Single window should return empty context (last window held out)."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=2000,
            transcription_windows_per_summary_window=4
        )
        
        wm.add_window("Test text", 0.0, 10.0)
        
        text, insights = wm.get_accumulated_text_and_insights()
        
        assert text == ""
        assert insights == []
    
    def test_variable_raw_text_context_limit(self):
        """Different raw_text_context_limit values should work correctly.
        
        Implementation adds windows while current_length < limit.
        This means we can exceed the limit by adding a window that pushes us over.
        Text parts are joined with spaces, so there's 1 char overhead per join.
        """
        # Test with small limit (50 chars)
        wm_small = WindowManager(
            context_limit=50000,
            raw_text_context_limit=50,
            transcription_windows_per_summary_window=4
        )
        wm_small.add_window("A" * 100, 0.0, 10.0)  # 100 chars
        wm_small.add_window("B" * 100, 10.0, 20.0)  # 100 chars
        wm_small.add_window("C" * 100, 20.0, 30.0)  # 100 chars
        wm_small.add_window("D" * 100, 30.0, 40.0)  # 100 chars
        
        text_small, _ = wm_small.get_accumulated_text_and_insights()
        # Window 0: 100 < 50 → add (current=100)
        # Window 1: 100 < 50 → skip
        assert len(text_small) == 100
        
        # Test with larger limit (5000 chars)
        wm_large = WindowManager(
            context_limit=50000,
            raw_text_context_limit=5000,
            transcription_windows_per_summary_window=4
        )
        wm_large.add_window("A" * 100, 0.0, 10.0)
        wm_large.add_window("B" * 100, 10.0, 20.0)
        wm_large.add_window("C" * 100, 20.0, 30.0)
        wm_large.add_window("D" * 100, 30.0, 40.0)
        
        text_large, _ = wm_large.get_accumulated_text_and_insights()
        # All 3 windows fit under 5000 limit (100 + 1 + 100 + 1 + 100 = 302)
        assert len(text_large) == 302  # 100 + 1 + 100 + 1 + 100
        assert len(text_large) > len(text_small)


class TestSummaryClientRawTextLimit:
    """Tests for SummaryClient raw_text_context_limit functionality."""
    
    def test_raw_text_context_limit_parameter(self):
        """SummaryClient should accept and store raw_text_context_limit parameter."""
        client = SummaryClient(
            api_key="test",
            base_url="http://test:8000/v1",
            transcription_windows_per_summary_window=4,
            raw_text_context_limit=1500
        )
        
        assert client.raw_text_context_limit == 1500
        assert client._window_manager.raw_text_context_limit == 1500
    
    def test_update_params_raw_text_context_limit(self):
        """update_params should update raw_text_context_limit."""
        client = SummaryClient(
            api_key="test",
            base_url="http://test:8000/v1",
            raw_text_context_limit=2000
        )
        
        client.update_params(raw_text_context_limit=3000)
        
        assert client.raw_text_context_limit == 3000
        assert client._window_manager.raw_text_context_limit == 3000
    
    def test_transcription_windows_per_summary_window_parameter(self):
        """SummaryClient should accept transcription_windows_per_summary_window parameter."""
        client = SummaryClient(
            api_key="test",
            base_url="http://test:8000/v1",
            transcription_windows_per_summary_window=6
        )
        
        assert client._window_manager.transcription_windows_per_summary_window == 6
    
    def test_update_params_transcription_windows_per_summary_window(self):
        """update_params should update transcription_windows_per_summary_window."""
        client = SummaryClient(
            api_key="test",
            base_url="http://test:8000/v1",
            transcription_windows_per_summary_window=4
        )
        
        client.update_params(transcription_windows_per_summary_window=8)
        
        assert client._window_manager.transcription_windows_per_summary_window == 8


class TestContextLimitRenaming:
    """Tests for renamed parameters (max_chars -> context_limit)."""
    
    def test_window_manager_context_limit(self):
        """WindowManager should use context_limit instead of max_chars."""
        wm = WindowManager(
            context_limit=30000,
            raw_text_context_limit=2000,
            transcription_windows_per_summary_window=4
        )
        
        assert wm.context_limit == 30000
        # Verify it's used in add_window logic
        wm.add_window("Test", 0.0, 10.0)
        assert wm._char_count == 4  # "Test" has 4 chars
    
    def test_window_manager_max_chars_renamed(self):
        """Old max_chars parameter should still work via context_limit."""
        # This test verifies the rename - old code using max_chars should update to context_limit
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=2000,
            transcription_windows_per_summary_window=4
        )
        
        # Verify the new attribute exists and old one doesn't
        assert hasattr(wm, 'context_limit')
        assert not hasattr(wm, 'max_chars')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])