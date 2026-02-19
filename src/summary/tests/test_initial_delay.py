"""
Unit tests for initial delay logic.

In the refactored code, initial delay is handled by ContextSummaryPlugin._should_process().
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager


class TestInitialDelayLogic:
    """Tests for self-contained initial delay logic in ContextSummaryPlugin."""
    
    def create_client_with_plugin(self, delay_seconds: float = 30.0):
        """Create a SummaryClient with plugins initialized for testing."""
        client = SummaryClient(
            reasoning_api_key="test_key",
            reasoning_model="test_model",
            initial_summary_delay_seconds=delay_seconds
        )
        
        # Manually create and register the context_summary plugin for testing
        from src.summary.context_summary import ContextSummaryPlugin
        
        # Create a mock LLM manager
        mock_llm = MagicMock()
        mock_reasoning_client = MagicMock()
        mock_llm.reasoning_llm_client = mock_reasoning_client
        
        # Create the plugin with the specified delay
        plugin = ContextSummaryPlugin(
            window_manager=client._window_manager,
            llm_manager=mock_llm,
            result_callback=client._queue_payload,
            summary_client=client,
            initial_summary_delay_seconds=delay_seconds
        )
        
        # Register the plugin
        client._plugins["context_summary"] = plugin
        
        return client, plugin
    
    @pytest.mark.asyncio
    async def test_delay_applied_when_elapsed_less_than_delay(self):
        """Test that delay is applied when elapsed time is less than delay setting."""
        client, plugin = self.create_client_with_plugin(delay_seconds=30.0)
        
        # Add a window with timestamp 0
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("test text", 0.0, 5.0, [1])
        
        # Check if should process at timestamp 10 (only 10 seconds elapsed, need 30)
        should_process = plugin._should_process(summary_window_id=0)
        
        # Should NOT process (delay applied)
        assert should_process is False
    
    @pytest.mark.asyncio
    async def test_no_delay_when_elapsed_greater_than_delay(self):
        """Test that delay is not applied when elapsed time exceeds delay setting."""
        client, plugin = self.create_client_with_plugin(delay_seconds=30.0)
        
        # Add first window at timestamp 0
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("first text", 0.0, 5.0, [1])
        
        # Add second window at timestamp 40 (40 seconds elapsed, need 30)
        client._window_manager.add_summary_window("second text", 40.0, 45.0, [2])
        
        # Check if should process window at timestamp 40 (elapsed = 40, need 30)
        should_process = plugin._should_process(summary_window_id=1)
        
        # Should process (no delay)
        assert should_process is True
    
    @pytest.mark.asyncio
    async def test_no_delay_after_first_summary(self):
        """Test that delay is not applied after first summary has been performed."""
        client, plugin = self.create_client_with_plugin(delay_seconds=30.0)
        
        # Mark that a summary has already been performed
        plugin._has_performed_summary = True
        
        # Add a window with timestamp 0
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("test text", 0.0, 5.0, [1])
        
        # Check if should process - should be True since first summary already done
        should_process = plugin._should_process(summary_window_id=0)
        
        # Should process (no delay after first summary)
        assert should_process is True
    
    def test_update_params_changes_delay(self):
        """Test that update_params can change the delay setting."""
        client = SummaryClient(
            reasoning_api_key="test_key",
            reasoning_model="test_model",
            initial_summary_delay_seconds=30.0
        )
        
        assert client.initial_summary_delay_seconds == 30.0
        
        client.update_params(initial_summary_delay_seconds=60.0)
        
        assert client.initial_summary_delay_seconds == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])