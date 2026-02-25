"""
Tests for plugin result token limits behavior.

Verifies that:
1. Text limit reached but plugin limits not reached - should continue iterating
2. Text limit and plugin limits both reached - should break
3. Plugin limits larger than text limit - should collect more plugin results
4. No limits - should collect all (existing behavior)
"""

import pytest
from src.summary.window_manager import WindowManager


class TestPluginResultTokenLimits:
    """Test cases for plugin result token limit behavior."""
    
    def test_text_limit_reached_continues_for_plugin_results(self):
        """When text limit is reached but plugin limits are not, should continue iterating."""
        wm = WindowManager()
        
        # Add 3 windows with text and plugin results
        for i in range(3):
            wm.add_summary_window(
                text=f"Window {i} text content here",
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="test_plugin",
                result={"data": f"Result {i}"},
                include_in_context=True
            )
        
        # Set text limit low, plugin limit high
        text, results, text_count, _ = wm.get_accumulated_text_and_results(
            text_token_limit=50,  # Very low - only first window
            result_types=["test_plugin"],
            result_token_limit={"test_plugin": 1000}  # High - all windows
        )
        
        # Should have text from only 1 window (due to limit)
        assert text_count <= 50
        
        # Should have plugin results from ALL 3 windows (not limited)
        assert len(results.get("test_plugin", [])) == 3
    
    def test_both_limits_reached_breaks(self):
        """When both text and plugin limits are reached, should break."""
        wm = WindowManager()
        
        # Add 3 windows
        for i in range(3):
            wm.add_summary_window(
                text=f"Window {i} text",
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="test_plugin",
                result={"data": f"Result {i}"},
                include_in_context=True
            )
        
        # Set both limits low
        text, results, text_count, _ = wm.get_accumulated_text_and_results(
            text_token_limit=50,
            result_types=["test_plugin"],
            result_token_limit={"test_plugin": 10}  # Very low
        )
        
        # Should have limited results from fewer windows
        assert len(results.get("test_plugin", [])) < 3
    
    def test_no_limits_collects_all(self):
        """When no limits are set, should collect all (existing behavior)."""
        wm = WindowManager()
        
        for i in range(3):
            wm.add_summary_window(
                text=f"Window {i} text",
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="test_plugin",
                result={"data": f"Result {i}"},
                include_in_context=True
            )
        
        # No limits
        text, results, _, _ = wm.get_accumulated_text_and_results(
            result_types=["test_plugin"]
        )
        
        # Should have all
        assert len(results.get("test_plugin", [])) == 3
        assert "Window 0" in text
        assert "Window 1" in text
        assert "Window 2" in text
    
    def test_plugin_limit_larger_than_text_limit(self):
        """Plugin limits larger than text limit should collect more plugin results."""
        wm = WindowManager()
        
        # Add 5 windows with text and plugin results
        for i in range(5):
            wm.add_summary_window(
                text=f"Window {i} text content here " * 10,  # ~60 tokens
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="test_plugin",
                result={"data": f"Result {i}"},
                include_in_context=True
            )
        
        # Text limit: ~100 tokens (should fit ~1-2 windows)
        # Plugin limit: 1000 tokens (should fit all 5 windows)
        text, results, text_count, _ = wm.get_accumulated_text_and_results(
            text_token_limit=100,
            result_types=["test_plugin"],
            result_token_limit={"test_plugin": 1000}
        )
        
        # Text should be limited
        assert text_count <= 100
        
        # Plugin results should NOT be limited (all 5 windows)
        assert len(results.get("test_plugin", [])) == 5
    
    def test_multiple_plugins_different_limits(self):
        """Test with multiple plugins having different token limits."""
        wm = WindowManager()
        
        for i in range(3):
            wm.add_summary_window(
                text=f"Window {i} text",
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="plugin_a",
                result={"data": f"Result A {i}"},
                include_in_context=True
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="plugin_b",
                result={"data": f"Result B {i}"},
                include_in_context=True
            )
        
        # Text limit low, plugin_a limit high, plugin_b limit low
        text, results, text_count, _ = wm.get_accumulated_text_and_results(
            text_token_limit=50,
            result_types=["plugin_a", "plugin_b"],
            result_token_limit={
                "plugin_a": 1000,  # High - all windows
                "plugin_b": 10     # Low - fewer windows
            }
        )
        
        # plugin_a should have all 3 results
        assert len(results.get("plugin_a", [])) == 3
        
        # plugin_b should have fewer than 3 results
        assert len(results.get("plugin_b", [])) < 3
    
    def test_no_result_types_collects_all(self):
        """When result_types is None, should skip plugin collection entirely."""
        wm = WindowManager()
        
        for i in range(3):
            wm.add_summary_window(
                text=f"Window {i} text",
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
            wm.store_plugin_result(
                window_id=i,
                plugin_name="test_plugin",
                result={"data": f"Result {i}"},
                include_in_context=True
            )
        
        # result_types=None should skip plugin collection
        text, results, text_count, _ = wm.get_accumulated_text_and_results(
            text_token_limit=50,
            result_types=None
        )
        
        # Should have text (limited)
        assert text_count <= 50
        
        # Should have no plugin results
        assert results == {}
    
    def test_text_only_no_plugins(self):
        """When only text is needed (no result_types), should work correctly."""
        wm = WindowManager()
        
        for i in range(3):
            wm.add_summary_window(
                text=f"Window {i} text",
                timestamp_start=i * 10.0,
                timestamp_end=(i + 1) * 10.0,
                transcription_window_ids=[i]
            )
        
        text, results, text_count, _ = wm.get_accumulated_text_and_results(
            text_token_limit=50
        )
        
        # Should have limited text
        assert text_count <= 50
        
        # Should have no plugin results
        assert results == {}