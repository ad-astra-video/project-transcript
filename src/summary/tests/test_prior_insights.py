"""
Unit tests for prior insights accumulation functionality.

In the refactored code, prior insights are retrieved from plugin_results
using ContextSummaryTask._get_prior_insights_from_plugin_results().
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager
from src.summary.context_summary.task import WindowInsight, ContextSummaryTask


class TestPriorInsightsAccumulation:
    """Tests for prior insights retrieval from plugin_results."""
    
    def create_window_manager(self):
        """Create a WindowManager instance for testing."""
        return WindowManager()
    
    def test_first_window_has_zero_prior_insights(self):
        """Test that first summary window has no prior insights."""
        wm = self.create_window_manager()
        
        # Add first summary window
        wm.add_summary_window("First text", 0.0, 10.0, [1])
        
        # Create task and get prior insights
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=wm
        )
        
        prior_insights = task._get_prior_insights_from_plugin_results()
        
        # First window should have no prior insights
        assert len(prior_insights) == 0
    
    def test_second_window_has_prior_insights_from_first(self):
        """Test that second window has prior insights from first window via plugin_results."""
        wm = self.create_window_manager()
        
        # Add first window
        wm.add_summary_window("First text", 0.0, 10.0, [1])
        
        # Store insights in first window via plugin_results
        first_window = wm._summary_windows[0]
        result = {
            "summary_text": json.dumps({
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_id": 1,
                        "insight_type": "KEY POINT",
                        "insight_text": "First insight",
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
        
        # Add second window
        wm.add_summary_window("Second text", 10.0, 20.0, [2])
        
        # Create task and get prior insights
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=wm
        )
        
        prior_insights = task._get_prior_insights_from_plugin_results()
        
        # Should have insights from first window
        assert len(prior_insights) >= 1
        assert prior_insights[0].insight_text == "First insight"
    
    def test_insights_stored_per_window_via_plugin_results(self):
        """Test that insights are stored per window via plugin_results."""
        wm = self.create_window_manager()
        
        # Add first window
        wm.add_summary_window("First text", 0.0, 10.0, [1])
        
        # Store insights in first window via plugin_results
        first_window = wm._summary_windows[0]
        result = {
            "summary_text": json.dumps({
                "analysis": "Test analysis",
                "insights": [
                    {
                        "insight_id": 1,
                        "insight_type": "ACTION",
                        "insight_text": "First action",
                        "confidence": 0.9,
                        "window_id": 0,
                        "timestamp_start": 0.0,
                        "timestamp_end": 10.0,
                        "classification": "+"
                    }
                ]
            })
        }
        first_window.store_result("context_summary", result, include_in_context=True)
        
        # Add second window
        wm.add_summary_window("Second text", 10.0, 20.0, [2])
        
        # Store insights in second window via plugin_results
        second_window = wm._summary_windows[1]
        result2 = {
            "summary_text": json.dumps({
                "analysis": "Test analysis 2",
                "insights": [
                    {
                        "insight_id": 2,
                        "insight_type": "DECISION",
                        "insight_text": "Second decision",
                        "confidence": 0.8,
                        "window_id": 1,
                        "timestamp_start": 10.0,
                        "timestamp_end": 20.0,
                        "classification": "~"
                    }
                ]
            })
        }
        second_window.store_result("context_summary", result2, include_in_context=True)
        
        # Create task and get prior insights (should only get from first window)
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=wm
        )
        
        prior_insights = task._get_prior_insights_from_plugin_results()
        
        # Should have insights from first window only (second window is current)
        assert len(prior_insights) == 1
        assert prior_insights[0].insight_type == "ACTION"


class TestContextSummaryTaskPriorInsights:
    """Tests for ContextSummaryTask prior insights retrieval."""
    
    def test_get_prior_insights_from_plugin_results_empty(self):
        """Test that prior insights returns empty list when no windows."""
        wm = WindowManager()
        
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=wm
        )
        
        prior_insights = task._get_prior_insights_from_plugin_results()
        assert prior_insights == []
    
    def test_get_prior_insights_from_plugin_results_single_window(self):
        """Test that prior insights returns empty when only one window exists."""
        wm = WindowManager()
        wm.add_summary_window("Text", 0.0, 10.0, [1])
        
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=wm
        )
        
        prior_insights = task._get_prior_insights_from_plugin_results()
        # With only one window, there are no prior insights
        assert prior_insights == []