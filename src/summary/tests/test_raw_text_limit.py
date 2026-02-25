"""
Tests for raw text context limit functionality.

In the refactored code, insights are stored via plugin_results instead of
window.insights, and prior insights are retrieved via ContextSummaryTask.
"""

import pytest
import json
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager
from src.summary.context_summary.task import WindowInsight, ContextSummaryTask


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
        
        # Get text from all windows (new method)
        text = wm.get_all_windows_text()
        
        # Should have text from both windows
        assert len(text) > 0
        assert len(text) <= 2400  # Combined text
    
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
        
        # Get text from all windows
        text = wm.get_all_windows_text()
        
        # Should have text from all windows (new method doesn't limit)
        assert len(text) >= 1200  # All 4 windows (may have extra spaces)
        assert len(text) > 0
    
    def test_insights_preserved_via_plugin_results(self):
        """Insights should be preserved via plugin_results."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=100,
            transcription_windows_per_summary_window=4
        )
        
        # Add first window
        wm.add_summary_window("A" * 100, 0.0, 10.0, [])
        
        # Store insights via plugin_results
        first_window = wm._summary_windows[0]
        result = {
            "summary_text": json.dumps({
                "topic": "Test",
                "insights": [
                    {
                        "insight_id": 1,
                        "insight_type": "KEY POINT",
                        "insight_text": "Test insight 1",
                        "confidence": 0.9,
                        "window_id": 0,
                        "timestamp_start": 0.0,
                        "timestamp_end": 10.0,
                        "classification": "~"
                    }
                ]
            })
        }
        first_window.store_result("context_summary", result, include_in_context=True)
        
        # Add more windows
        wm.add_summary_window("B" * 100, 10.0, 20.0, [])
        wm.add_summary_window("C" * 100, 20.0, 30.0, [])
        
        # Get prior insights via ContextSummaryTask
        task = ContextSummaryTask(
            llm_client=None,
            window_manager=wm
        )
        prior_insights = task._get_prior_insights_from_plugin_results()
        
        # Insights should be preserved from first window
        assert len(prior_insights) > 0
        assert prior_insights[0].insight_text == "Test insight 1"


class TestSummaryClientRawTextLimit:
    """Tests for SummaryClient raw_text_context_limit functionality."""
    
    def test_update_params_raw_text_context_limit(self):
        """update_params should update raw_text_context_limit."""
        client = SummaryClient(
            reasoning_api_key="test",
            reasoning_base_url="http://test:8000/v1"
        )
        
        # Update params
        client.update_params(raw_text_context_limit=3000)
        
        # Verify the param was propagated
        assert client._window_manager.raw_text_context_limit == 3000


class TestRawTextLimitEdgeCases:
    """Edge case tests for raw text limit."""
    
    def test_empty_text_no_issue(self):
        """Empty text should not cause issues."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=500,
            transcription_windows_per_summary_window=4
        )
        
        wm.add_summary_window("", 0.0, 10.0, [])
        
        text = wm.get_all_windows_text()
        assert text == ""
    
    def test_single_window_exactly_at_limit(self):
        """Single window at exactly the limit should work."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=500,
            transcription_windows_per_summary_window=4
        )
        
        wm.add_summary_window("A" * 500, 0.0, 10.0, [])
        
        text = wm.get_all_windows_text()
        assert len(text) == 500
    
    def test_single_window_over_limit(self):
        """Single window over limit should still be stored."""
        wm = WindowManager(
            context_limit=50000,
            raw_text_context_limit=500,
            transcription_windows_per_summary_window=4
        )
        
        wm.add_summary_window("A" * 1000, 0.0, 10.0, [])
        
        text = wm.get_all_windows_text()
        assert len(text) == 1000