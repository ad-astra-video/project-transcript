"""
Tests for event processing in content_type_detection plugin.

This module tests that the content_type_detection plugin correctly processes
the events it subscribes to:
- summary_window_available: Process a summary window for content type detection
- on_update_params: Handle parameter update events
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any

from src.summary.content_type_detection import ContentTypeDetectionPlugin, init_plugin
from src.summary.window_manager import WindowManager


class TestContentTypeDetectionEventProcessing:
    """Test event processing in ContentTypeDetectionPlugin."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM manager."""
        mock = MagicMock()
        mock.rapid_llm_client = MagicMock()
        return mock

    @pytest.fixture
    def mock_window_manager(self):
        """Create a mock window manager with test data."""
        wm = MagicMock(spec=WindowManager)
        wm.get_recent_windows_text = MagicMock(return_value="Sample transcription text for content type detection")
        return wm

    @pytest.fixture
    def result_callback(self):
        """Create a mock result callback."""
        return AsyncMock()

    @pytest.fixture
    def mock_summary_client(self):
        """Create a mock summary client."""
        mock = MagicMock()
        mock._notify_plugins = AsyncMock()
        return mock

    @pytest.fixture
    def plugin(self, mock_llm, mock_window_manager, result_callback, mock_summary_client):
        """Create a ContentTypeDetectionPlugin instance for testing."""
        return ContentTypeDetectionPlugin(
            window_manager=mock_window_manager,
            llm_manager=mock_llm,
            result_callback=result_callback,
            summary_client=mock_summary_client,
            detection_interval=1  # Run on every call for testing
        )

    @pytest.mark.asyncio
    async def test_process_summary_window_available(self, plugin, mock_window_manager, result_callback):
        """Test that plugin processes summary_window_available event."""
        # Mock the task's detect_content_type method
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="MEETING",
                confidence=0.85,
                reasoning="Detected meeting content"
            )
            
            # Call the process method (handles summary_window_available event)
            result = await plugin.process(summary_window_id=1)
            
            # Verify the task was called
            mock_detect.assert_called_once()
            
            # Verify result callback was called with payload
            result_callback.assert_called_once()
            call_args = result_callback.call_args[0][0]
            assert call_args["type"] == "content_type_detection"
            assert call_args["content_type"] == "MEETING"
            assert call_args["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_process_emits_content_type_detected_event(self, plugin, mock_window_manager, result_callback, mock_summary_client):
        """Test that processing emits on_content_type_detected event."""
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="MEETING",
                confidence=0.85,
                reasoning="Detected meeting content"
            )
            
            # Call the process method
            await plugin.process(summary_window_id=1)
            
            # Verify on_content_type_detected event was emitted
            mock_summary_client._notify_plugins.assert_called_once_with(
                "on_content_type_detected",
                content_type="MEETING",
                confidence=0.85,
                source="AUTO_DETECTED",
                reasoning="Detected meeting content"
            )

    @pytest.mark.asyncio
    async def test_process_with_user_override(self, plugin, result_callback, mock_summary_client):
        """Test that user override bypasses detection."""
        # Set user override
        plugin.set_content_type_override("PRESENTATION")
        
        # Call process - should use override without calling LLM
        result = await plugin.process(summary_window_id=1)
        
        # Verify result uses override
        assert result["content_type"] == "PRESENTATION"
        assert result["confidence"] == 1.0
        assert result["source"] == "USER_OVERRIDE"
        
        # Verify event was emitted with override
        mock_summary_client._notify_plugins.assert_called_once_with(
            "on_content_type_detected",
            content_type="PRESENTATION",
            confidence=1.0,
            source="USER_OVERRIDE",
            reasoning="User override"
        )

    @pytest.mark.asyncio
    async def test_process_update_params_event(self, plugin):
        """Test that plugin handles on_update_params event."""
        # Call the handler for on_update_params event
        plugin.on_update_params(
            reasoning_max_tokens=512,
            reasoning_temperature=0.3,
            content_type_context_limit=3000
        )
        
        # Verify parameters were updated
        assert plugin._max_tokens == 512
        assert plugin._temperature == 0.3
        assert plugin._content_type_context_limit == 3000

    @pytest.mark.asyncio
    async def test_process_respects_detection_interval(self, plugin):
        """Test that detection runs only after counter reaches interval."""
        # Set detection_interval to a higher value
        plugin._detection_interval = 3
        plugin._detection_counter = 0
        
        # First call - counter becomes 1, but interval is 3, so no detection
        result1 = await plugin.process(summary_window_id=1)
        assert result1 == {}
        
        # Second call - counter becomes 2, but interval is 3, so no detection
        result2 = await plugin.process(summary_window_id=2)
        assert result2 == {}
        
        # Third call - counter becomes 3, which equals interval, so detection runs
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="MEETING",
                confidence=0.85,
                reasoning="Meeting content"
            )
            result3 = await plugin.process(summary_window_id=3)
            
            # Detection should have run
            assert result3["content_type"] == "MEETING"
            mock_detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_prevents_concurrent_detection(self, plugin):
        """Test that concurrent detection is prevented."""
        # Set in_progress flag to simulate ongoing detection
        plugin._in_progress = True
        
        result = await plugin.process(summary_window_id=1)
        
        # Should return empty when detection is in progress
        assert result == {}

    @pytest.mark.asyncio
    async def test_content_type_state_tracking(self, plugin, mock_window_manager):
        """Test that content type state is tracked across windows."""
        # First detection
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="MEETING",
                confidence=0.85,
                reasoning="Meeting content"
            )
            
            await plugin.process(summary_window_id=1)
            
            assert plugin._current_content_type == "MEETING"
            
            # Re-enable auto_detect for second detection
            plugin._auto_detect = True
            
            # Second detection - should track previous
            mock_detect.return_value = MagicMock(
                content_type="PRESENTATION",
                confidence=0.90,
                reasoning="Presentation content"
            )
            
            await plugin.process(summary_window_id=2)
            
            # Verify previous content type was tracked
            assert plugin._current_content_type == "PRESENTATION"


class TestContentTypeDetectionPluginRegistration:
    """Test plugin registration and event subscription."""

    @pytest.fixture
    def mock_window_manager(self):
        """Create a mock window manager."""
        return MagicMock(spec=WindowManager)

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM manager."""
        return MagicMock()

    @pytest.fixture
    def mock_result_callback(self):
        """Create a mock result callback."""
        return AsyncMock()

    @pytest.fixture
    def mock_summary_client(self):
        """Create a mock summary client with registration method."""
        mock = MagicMock()
        mock.register_plugin_event_sub = MagicMock()
        return mock

    def test_init_plugin_registers_events(self, mock_window_manager, mock_llm, mock_result_callback, mock_summary_client):
        """Test that init_plugin registers all expected event handlers."""
        # Call init_plugin
        init_plugin(
            plugin_name="content_type_detection",
            window_manager=mock_window_manager,
            llm_manager=mock_llm,
            result_callback=mock_result_callback,
            summary_client=mock_summary_client
        )
        
        # Verify plugin was registered with correct events
        mock_summary_client.register_plugin_event_sub.assert_called_once()
        call_kwargs = mock_summary_client.register_plugin_event_sub.call_args[1]
        
        # Verify all expected events are registered
        registered_events = call_kwargs["events"]
        assert "summary_window_available" in registered_events
        assert "on_update_params" in registered_events
        
        # Verify the plugin instance is stored
        assert call_kwargs["plugin_name"] == "content_type_detection"
        assert call_kwargs["plugin_instance"] is not None


class TestContentTypeOverride:
    """Test content type override functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM manager."""
        return MagicMock()

    @pytest.fixture
    def mock_window_manager(self):
        """Create a mock window manager."""
        return MagicMock(spec=WindowManager)

    @pytest.fixture
    def plugin(self, mock_llm, mock_window_manager):
        """Create a ContentTypeDetectionPlugin instance."""
        mock_summary_client = MagicMock()
        mock_summary_client._notify_plugins = AsyncMock()
        return ContentTypeDetectionPlugin(
            window_manager=mock_window_manager,
            llm_manager=mock_llm,
            result_callback=AsyncMock(),
            summary_client=mock_summary_client,
            detection_interval=1  # Run on every call for testing
        )

    def test_set_content_type_override(self, plugin):
        """Test setting content type override."""
        plugin.set_content_type_override("INTERVIEW")
        
        assert plugin._user_content_type_override == "INTERVIEW"

    def test_clear_content_type_override(self, plugin):
        """Test clearing content type override."""
        plugin.set_content_type_override("INTERVIEW")
        plugin.set_content_type_override(None)
        
        assert plugin._user_content_type_override is None

    @pytest.mark.asyncio
    async def test_override_prevents_auto_detect(self, plugin):
        """Test that setting override prevents auto-detect from running."""
        plugin.set_content_type_override("PRESENTATION")
        
        # Process should use override and skip auto-detect
        result = await plugin.process(summary_window_id=1)
        
        # Verify override is used
        assert result["content_type"] == "PRESENTATION"
        assert result["source"] == "USER_OVERRIDE"
        # Verify _user_content_type_override is still set (prevents future auto-detect)
        assert plugin._user_content_type_override == "PRESENTATION"


class TestContentTypeStateTracking:
    """Test content type state tracking across multiple windows."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM manager."""
        return MagicMock()

    @pytest.fixture
    def mock_window_manager(self):
        """Create a mock window manager."""
        wm = MagicMock(spec=WindowManager)
        wm.get_recent_windows_text = MagicMock(return_value="Sample text")
        return wm

    @pytest.fixture
    def plugin(self, mock_llm, mock_window_manager):
        """Create a ContentTypeDetectionPlugin instance."""
        mock_summary_client = MagicMock()
        mock_summary_client._notify_plugins = AsyncMock()
        return ContentTypeDetectionPlugin(
            window_manager=mock_window_manager,
            llm_manager=mock_llm,
            result_callback=AsyncMock(),
            summary_client=mock_summary_client,
            detection_interval=1  # Run on every call for testing
        )

    @pytest.mark.asyncio
    async def test_previous_content_type_in_payload(self, plugin, mock_window_manager):
        """Test that previous content type is included in payload."""
        # Set initial content type
        plugin._current_content_type = "MEETING"
        
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="PRESENTATION",
                confidence=0.90,
                reasoning="Presentation content"
            )
            
            result = await plugin.process(summary_window_id=1)
            
            # Verify previous content type is in payload
            assert result["previous_content_type"] == "MEETING"

    @pytest.mark.asyncio
    async def test_counter_resets_after_detection(self, plugin, mock_window_manager):
        """Test that detection counter resets after running detection."""
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="MEETING",
                confidence=0.85,
                reasoning="Meeting content"
            )
            
            # Initial counter should be 0
            assert plugin._detection_counter == 0
            
            await plugin.process(summary_window_id=1)
            
            # Counter should be reset to 0 after detection runs
            assert plugin._detection_counter == 0