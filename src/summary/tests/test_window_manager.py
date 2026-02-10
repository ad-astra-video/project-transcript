"""
Unit tests for WindowManager functionality.
"""

import pytest
from src.summary.summary_client import WindowManager


class TestWindowManager:
    """Tests for WindowManager with _first_window_timestamp tracking."""
    
    def test_first_window_timestamp_set_on_first_window(self):
        """Test that _first_window_timestamp is set when first window is added."""
        manager = WindowManager()
        assert manager._first_window_timestamp is None
        
        manager.add_window("test text 1", 10.0, 15.0)
        
        assert manager._first_window_timestamp == 10.0
    
    def test_first_window_timestamp_not_updated_on_subsequent_windows(self):
        """Test that _first_window_timestamp is not updated on subsequent windows."""
        manager = WindowManager()
        
        manager.add_window("test text 1", 10.0, 15.0)
        first_timestamp = manager._first_window_timestamp
        
        manager.add_window("test text 2", 20.0, 25.0)
        
        assert manager._first_window_timestamp == first_timestamp
    
    def test_first_window_timestamp_with_multiple_windows(self):
        """Test first window timestamp tracking with multiple windows."""
        manager = WindowManager()
        
        manager.add_window("first window", 5.0, 10.0)
        manager.add_window("second window", 15.0, 20.0)
        manager.add_window("third window", 25.0, 30.0)
        
        assert manager._first_window_timestamp == 5.0


class TestWindowManagerReset:
    """Tests for WindowManager reset behavior."""
    
    def test_window_manager_counters_reset_on_clear(self):
        """Test that all internal counters are reset when clear() is called."""
        manager = WindowManager()
        
        # Add some windows and insights
        manager.add_window("text1", 0.0, 5.0)
        manager.add_window("text2", 5.0, 10.0)
        manager._next_insight_id = 5  # Simulate some insights
        
        # Verify initial state
        assert manager._next_window_id == 2
        assert manager._next_insight_id == 5
        assert manager._first_window_timestamp == 0.0
        
        # Clear
        manager.clear()
        
        # Verify counters reset
        assert manager._next_window_id == 0
        assert manager._next_insight_id == 0
        assert manager._first_window_timestamp is None
    
    def test_window_manager_clear_with_empty_windows(self):
        """Test that clear() works correctly when called on empty manager."""
        manager = WindowManager()
        
        # Verify initial state
        assert manager._next_window_id == 0
        assert manager._next_insight_id == 0
        assert manager._first_window_timestamp is None
        assert len(manager._windows) == 0
        
        # Clear empty manager
        manager.clear()
        
        # Verify still zero
        assert manager._next_window_id == 0
        assert manager._next_insight_id == 0
        assert manager._first_window_timestamp is None
    
    def test_window_manager_first_timestamp_reset(self):
        """Test that _first_window_timestamp is reset to None on clear."""
        manager = WindowManager()
        
        # Add a window to set first timestamp
        manager.add_window("test text", 10.0, 15.0)
        assert manager._first_window_timestamp == 10.0
        
        # Clear
        manager.clear()
        
        # Verify reset to None
        assert manager._first_window_timestamp is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])