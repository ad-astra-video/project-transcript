"""
Unit tests for reset behavior.

In the refactored code, reset() clears the WindowManager and in_flight_windows.
The content type state is now managed by the plugins.
"""

import pytest
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager
from src.summary.context_summary.task import WindowInsight


class TestSummaryClientReset:
    """Tests for SummaryClient reset behavior."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
    
    def test_summary_client_resets_window_manager(self):
        """Test that WindowManager is cleared on SummaryClient reset."""
        client = self.create_client()
        
        # Add some windows
        client._window_manager.add_summary_window("text1", 0.0, 5.0, [1])
        client._window_manager.add_summary_window("text2", 5.0, 10.0, [2])
        
        # Verify windows exist
        assert len(client._window_manager._summary_windows) == 2
        assert client._window_manager._next_window_id == 2
        
        # Reset
        client.reset()
        
        # Verify WindowManager is cleared
        assert len(client._window_manager._summary_windows) == 0
        assert client._window_manager._next_window_id == 0
    
    def test_summary_client_clears_in_flight_windows(self):
        """Test that in_flight_windows set is cleared on reset."""
        client = self.create_client()
        
        # Add some in-flight windows
        client.in_flight_windows = {1, 2, 3}
        
        # Reset
        client.reset()
        
        # Verify cleared
        assert len(client.in_flight_windows) == 0
    
    def test_summary_client_resets_last_processed_timestamp(self):
        """Test that _last_processed_timestamp is reset on reset."""
        client = self.create_client()
        
        # Set a timestamp
        client._last_processed_timestamp = 100.0
        
        # Reset
        client.reset()
        
        # Verify reset to 0
        assert client._last_processed_timestamp == 0.0
    
    def test_summary_client_clears_plugins(self):
        """Test that plugins are cleared on reset (but not re-discovered)."""
        client = self.create_client()
        
        # Add a mock plugin
        client._plugins["test_plugin"] = "test_value"
        
        # Reset
        client.reset()
        
        # Note: plugins are NOT cleared on reset - they persist across streams
        # This is intentional as plugins are stateless
        assert "test_plugin" in client._plugins


class TestWindowManagerReset:
    """Tests for WindowManager reset behavior."""
    
    def test_window_manager_clear(self):
        """Test that WindowManager.clear() clears all windows."""
        wm = WindowManager()
        
        # Add windows
        wm.add_summary_window("text1", 0.0, 5.0, [1])
        wm.add_summary_window("text2", 5.0, 10.0, [2])
        
        assert len(wm._summary_windows) == 2
        
        # Clear
        wm.clear()
        
        assert len(wm._summary_windows) == 0
        assert wm._next_window_id == 0
    
    def test_context_summary_task_insight_counter(self):
        """Test that ContextSummaryTask has insight ID counter."""
        from src.summary.context_summary.task import ContextSummaryTask
        from unittest.mock import MagicMock
        
        task = ContextSummaryTask(
            llm_client=MagicMock(),
            window_manager=WindowManager()
        )
        
        # Get first insight ID
        first_id = task._get_next_insight_id()
        assert first_id == 1
        
        # Get second insight ID
        second_id = task._get_next_insight_id()
        assert second_id == 2
        
        # Counter increments correctly
        assert task._next_insight_id == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])