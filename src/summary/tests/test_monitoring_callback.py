"""
Tests for monitoring event callback functionality.

In the refactored code, monitoring callbacks are invoked by plugins directly
or through the _send_monitoring_event method.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.summary.summary_client import SummaryClient


class TestMonitoringCallback:
    """Tests for monitoring event callback functionality."""
    
    @pytest.fixture
    def mock_callback(self):
        """Create a mock async monitoring callback."""
        callback = AsyncMock()
        callback.return_value = None
        return callback
    
    @pytest.fixture
    def client_with_callback(self, mock_callback):
        """Create a SummaryClient with monitoring callback."""
        with patch('src.summary.summary_client.AsyncOpenAI') as mock_openai:
            client = SummaryClient(
                reasoning_api_key="test-key",
                reasoning_base_url="http://test:8000/v1",
                reasoning_model="test-model",
                initial_summary_delay_seconds=10.0,
                send_monitoring_event_callback=mock_callback
            )
            return client
    
    @pytest.mark.asyncio
    async def test_callback_invoked_on_monitoring_event(self, client_with_callback, mock_callback):
        """Callback should be invoked when _send_monitoring_event is called."""
        # Create test event data
        event_data = {"test": "data", "value": 123}
        event_type = "test_event"
        
        # Call _send_monitoring_event
        await client_with_callback._send_monitoring_event(event_data, event_type)
        
        # Verify callback was invoked
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args
        assert call_args[0][0] == event_data
        assert call_args[0][1] == event_type
    
    @pytest.mark.asyncio
    async def test_callback_not_invoked_when_none(self):
        """Callback should not be invoked when not configured."""
        client = SummaryClient(
            reasoning_api_key="test-key",
            reasoning_base_url="http://test:8000/v1",
            reasoning_model="test-model"
        )
        
        # Should not raise an error
        await client._send_monitoring_event({"test": "data"}, "test_event")
    
    @pytest.mark.asyncio
    async def test_callback_error_handled_gracefully(self, client_with_callback, mock_callback):
        """Callback errors should be handled gracefully."""
        mock_callback.side_effect = Exception("Callback error")
        
        # Should not raise
        await client_with_callback._send_monitoring_event({"test": "data"}, "test_event")


class TestContentTypeDetectionPluginMonitoring:
    """Tests for content type detection plugin monitoring."""
    
    @pytest.fixture
    def mock_callback(self):
        """Create a mock async monitoring callback."""
        return AsyncMock()
    
    def create_plugin_with_callback(self, mock_callback):
        """Create a ContentTypeDetectionPlugin with monitoring callback."""
        from src.summary.content_type_detection import ContentTypeDetectionPlugin
        
        client = SummaryClient(
            reasoning_api_key="test-key",
            reasoning_base_url="http://test:8000/v1",
            reasoning_model="test-model",
            send_monitoring_event_callback=mock_callback
        )
        
        # Create mock LLM manager
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        # Create the plugin
        plugin = ContentTypeDetectionPlugin(
            window_manager=client._window_manager,
            llm_manager=mock_llm,
            result_callback=client._queue_payload,
            summary_client=client
        )
        
        return client, plugin
    
    @pytest.mark.asyncio
    async def test_content_type_detection_sends_result(self, mock_callback):
        """Content type detection should send result through callback."""
        client, plugin = self.create_plugin_with_callback(mock_callback)
        
        # Add some text to the window manager
        client._window_manager.add_summary_window("test text", 0.0, 10.0, [1])
        
        # Mock the task's detect_content_type method
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            from src.summary.content_type_detection.task import ContentTypeDetectionSchema
            mock_detect.return_value = ContentTypeDetectionSchema(
                content_type="TECHNICAL_TALK",
                confidence=0.85,
                reasoning="Test reasoning"
            )
            
            # Process content type detection
            result = await plugin.process(summary_window_id=0)
            
            # Verify result was sent
            assert result is not None
            assert result.get("type") == "content_type_detection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])