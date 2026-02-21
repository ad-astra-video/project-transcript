"""
Tests for content type detection timing behavior.

In the refactored code, content type detection is handled by ContentTypeDetectionPlugin.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager


class TestContentTypeDetectionPlugin:
    """Tests for ContentTypeDetectionPlugin behavior."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock SummaryClient for testing."""
        with patch('src.summary.summary_client.AsyncOpenAI') as mock_openai:
            client = SummaryClient(
                reasoning_api_key="test-key",
                reasoning_base_url="http://test:8000/v1",
                reasoning_model="test-model",
                initial_summary_delay_seconds=10.0
            )
            return client
    
    @pytest.fixture
    def plugin_with_mock_llm(self, mock_client):
        """Create a ContentTypeDetectionPlugin with mocked LLM."""
        from src.summary.content_type_detection import ContentTypeDetectionPlugin
        
        # Create mock LLM manager
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContentTypeDetectionPlugin(
            window_manager=mock_client._window_manager,
            llm_manager=mock_llm,
            result_callback=mock_client._queue_payload,
            summary_client=mock_client,
            detection_interval=1  # Run on every call for testing
        )
        
        return mock_client, plugin
    
    @pytest.mark.asyncio
    async def test_detection_runs_when_auto_detect_enabled(self, plugin_with_mock_llm):
        """Content type detection should run when auto_detect is enabled."""
        client, plugin = plugin_with_mock_llm
        
        # Add text to window manager
        client._window_manager.add_summary_window("Test technical content about coding", 0.0, 10.0, [1])
        
        # Mock the task's detect method
        from src.summary.content_type_detection.task import ContentTypeDetectionSchema
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = ContentTypeDetectionSchema(
                content_type="TECHNICAL_TALK",
                confidence=0.85,
                reasoning="Test reasoning"
            )
            
            # Process content type detection
            result = await plugin.process(summary_window_id=0)
            
            # Verify detection ran
            assert result is not None
            assert result.get("content_type") == "TECHNICAL_TALK"
    
    @pytest.mark.asyncio
    async def test_detection_skipped_when_interval_not_reached(self, plugin_with_mock_llm):
        """Content type detection should be skipped when interval not reached."""
        client, plugin = plugin_with_mock_llm
        
        # Set detection_interval to a high value
        plugin._detection_interval = 10
        plugin._detection_counter = 0
        
        # Add text to window manager
        client._window_manager.add_summary_window("Test content", 0.0, 10.0, [1])
        
        # Process content type detection - counter becomes 1, but interval is 10
        result = await plugin.process(summary_window_id=0)
        
        # Should return empty (skipped because interval not reached)
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_user_override_takes_precedence(self, plugin_with_mock_llm):
        """User override should take precedence over auto-detection."""
        client, plugin = plugin_with_mock_llm
        
        # Set user override
        plugin.set_content_type_override("INTERVIEW")
        
        # Add text to window manager
        client._window_manager.add_summary_window("Test content", 0.0, 10.0, [1])
        
        # Process content type detection
        result = await plugin.process(summary_window_id=0)
        
        # Should return user override
        assert result is not None
        assert result.get("content_type") == "INTERVIEW"
        assert result.get("source") == "USER_OVERRIDE"
    
    @pytest.mark.asyncio
    async def test_clear_user_override(self, plugin_with_mock_llm):
        """Clearing user override should allow auto-detection to work."""
        client, plugin = plugin_with_mock_llm
        
        # Set and then clear user override
        plugin.set_content_type_override("INTERVIEW")
        plugin.set_content_type_override(None)
        
        # User override should be cleared
        assert plugin._user_content_type_override is None
    
    @pytest.mark.asyncio
    async def test_detection_prevents_concurrent_runs(self, plugin_with_mock_llm):
        """Detection should prevent concurrent runs."""
        client, plugin = plugin_with_mock_llm
        
        # Mark detection as in progress
        plugin._in_progress = True
        
        # Add text to window manager
        client._window_manager.add_summary_window("Test content", 0.0, 10.0, [1])
        
        # Process content type detection - should be skipped
        result = await plugin.process(summary_window_id=0)
        
        # Should return empty (skipped due to in-progress)
        assert result == {}


class TestContentTypeStateManagement:
    """Tests for content type state management in plugins."""
    
    def test_content_type_state_holder(self):
        """Test ContentTypeStateHolder."""
        from src.summary.context_summary import ContentTypeStateHolder
        
        state = ContentTypeStateHolder(
            content_type="TECHNICAL_TALK",
            confidence=0.85,
            source="AUTO_DETECTED"
        )
        
        assert state.content_type == "TECHNICAL_TALK"
        assert state.confidence == 0.85
        assert state.source == "AUTO_DETECTED"
    
    def test_content_type_state_holder_defaults(self):
        """Test ContentTypeStateHolder default values."""
        from src.summary.context_summary import ContentTypeStateHolder
        
        state = ContentTypeStateHolder()
        
        assert state.content_type == "UNKNOWN"
        assert state.confidence == 0.0
        assert state.source == "INITIAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])