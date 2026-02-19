"""
Tests for event processing in rapid_summary plugin.

This module tests that the rapid_summary plugin correctly processes
the events it subscribes to:
- summary_window_available: Process a summary window for rapid summarization
- on_context_summary_complete: Handle context summary completion events
- on_update_params: Handle parameter update events
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone
from typing import Dict, Any

from src.summary.rapid_summary import RapidSummaryPlugin, init_plugin
from src.summary.window_manager import WindowManager


class TestRapidSummaryEventProcessing:
    """Test event processing in RapidSummaryPlugin."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM manager."""
        mock = MagicMock()
        mock.rapid_llm_client = MagicMock()
        # Enable fast client to allow processing
        mock.fast_client = MagicMock()
        return mock

    @pytest.fixture
    def mock_window_manager(self):
        """Create a mock window manager with test data."""
        wm = MagicMock(spec=WindowManager)
        
        # Mock window data
        mock_window = MagicMock()
        mock_window.window_id = 1
        mock_window.timestamp_start = 100.0
        mock_window.timestamp_end = 110.0
        mock_window.text = "Test transcription text for rapid summary"
        
        wm._summary_windows = [mock_window]
        wm.get_window_transcription_ids = MagicMock(return_value=[1])
        wm.get_text_and_window_ids_since_timestamp = MagicMock(return_value=("", []))
        
        return wm

    @pytest.fixture
    def result_callback(self):
        """Create a mock result callback."""
        return AsyncMock()

    @pytest.fixture
    def mock_summary_client(self):
        """Create a mock summary client."""
        return MagicMock()

    @pytest.fixture
    def plugin(self, mock_llm, mock_window_manager, result_callback, mock_summary_client):
        """Create a RapidSummaryPlugin instance for testing."""
        return RapidSummaryPlugin(
            window_manager=mock_window_manager,
            llm_manager=mock_llm,
            result_callback=result_callback,
            summary_client=mock_summary_client
        )

    @pytest.mark.asyncio
    async def test_process_summary_window_available(self, plugin, mock_window_manager, result_callback):
        """Test that plugin processes summary_window_available event."""
        # Mock the task's build_rapid_summary_payload method
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "summary": [{"item": "Test rapid summary"}],
                "timing": {"summary_window_id": 1}
            }
            
            # Call the process method (handles summary_window_available event)
            result = await plugin.process(summary_window_id=1)
            
            # Verify the task was called
            mock_process.assert_called_once()
            
            # Verify result callback was called
            result_callback.assert_called_once()
            call_args = result_callback.call_args[0][0]
            assert call_args["type"] == "rapid_summary"

    @pytest.mark.asyncio
    async def test_process_context_summary_complete_event(self, plugin):
        """Test that plugin handles on_context_summary_complete event."""
        # Initial state - no timestamp tracked
        assert plugin._context_summary_timestamp == 0.0
        
        # Call the handler for on_context_summary_complete event
        await plugin.handle_context_summary_complete(
            summary_window_id=1,
            timestamp=150.0
        )
        
        # Verify timestamp was updated
        assert plugin._context_summary_timestamp == 150.0

    @pytest.mark.asyncio
    async def test_process_context_summary_complete_tracks_highest_timestamp(self, plugin):
        """Test that plugin tracks the highest context summary timestamp."""
        # Send events with out-of-order timestamps
        await plugin.handle_context_summary_complete(summary_window_id=1, timestamp=100.0)
        assert plugin._context_summary_timestamp == 100.0
        
        await plugin.handle_context_summary_complete(summary_window_id=2, timestamp=150.0)
        assert plugin._context_summary_timestamp == 150.0
        
        # Lower timestamp should not update
        await plugin.handle_context_summary_complete(summary_window_id=3, timestamp=120.0)
        assert plugin._context_summary_timestamp == 150.0

    @pytest.mark.asyncio
    async def test_process_update_params_event(self, plugin):
        """Test that plugin handles on_update_params event."""
        # Call the handler for on_update_params event
        plugin.on_update_params(
            fast_max_tokens=2048,
            fast_temperature=0.4
        )
        
        # Verify parameters were updated
        assert plugin._task.max_tokens == 2048
        assert plugin._task.temperature == 0.4

    @pytest.mark.asyncio
    async def test_process_summary_window_uses_context_timestamp(self, plugin, mock_window_manager):
        """Test that rapid summary uses context from tracked timestamp."""
        # Set context summary timestamp
        plugin._context_summary_timestamp = 100.0
        
        # Mock get_text_and_window_ids_since_timestamp to return context
        mock_window_manager.get_text_and_window_ids_since_timestamp.return_value = (
            "Previous context text",
            [1, 2]
        )
        
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"type": "rapid_summary", "summary": []}
            
            await plugin.process(summary_window_id=3)
            
            # Verify context was retrieved using tracked timestamp
            mock_window_manager.get_text_and_window_ids_since_timestamp.assert_called_once_with(100.0)

    @pytest.mark.asyncio
    async def test_process_without_fast_client_returns_empty(self, plugin, mock_llm):
        """Test that processing returns empty when fast client is not available."""
        # Disable fast client
        mock_llm.fast_client = None
        
        result = await plugin.process(summary_window_id=1)
        
        assert result == {}

    @pytest.mark.asyncio
    async def test_reset_clears_context_timestamp(self, plugin):
        """Test that reset clears the tracked context timestamp."""
        # Set a context timestamp
        plugin._context_summary_timestamp = 150.0
        
        # Call reset
        plugin.reset()
        
        # Verify timestamp was cleared
        assert plugin._context_summary_timestamp == 0.0


class TestRapidSummaryPluginRegistration:
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
            plugin_name="rapid_summary",
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
        assert "on_context_summary_complete" in registered_events
        assert "on_update_params" in registered_events
        
        # Verify the plugin instance is stored
        assert call_kwargs["plugin_name"] == "rapid_summary"
        assert call_kwargs["plugin_instance"] is not None


class TestRapidSummaryContextTracking:
    """Test context tracking behavior in RapidSummaryPlugin."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM manager."""
        mock = MagicMock()
        mock.rapid_llm_client = MagicMock()
        mock.fast_client = MagicMock()
        return mock

    @pytest.fixture
    def mock_window_manager(self):
        """Create a mock window manager."""
        wm = MagicMock(spec=WindowManager)
        
        # Mock window data
        mock_window = MagicMock()
        mock_window.window_id = 1
        mock_window.timestamp_start = 100.0
        mock_window.timestamp_end = 110.0
        mock_window.text = "Test text"
        
        wm._summary_windows = [mock_window]
        wm.get_text_and_window_ids_since_timestamp = MagicMock(return_value=("", []))
        return wm

    @pytest.fixture
    def plugin(self, mock_llm, mock_window_manager):
        """Create a RapidSummaryPlugin instance."""
        return RapidSummaryPlugin(
            window_manager=mock_window_manager,
            llm_manager=mock_llm,
            result_callback=AsyncMock(),
            summary_client=None
        )

    @pytest.mark.asyncio
    async def test_no_context_when_timestamp_is_zero(self, plugin, mock_window_manager):
        """Test that no context is retrieved when timestamp is 0."""
        # Ensure timestamp is 0 (no context summary completed yet)
        plugin._context_summary_timestamp = 0.0
        
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"type": "rapid_summary", "summary": []}
            
            await plugin.process(summary_window_id=1)
            
            # Verify empty context was passed
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["context_since_last_summary"] == ""

    @pytest.mark.asyncio
    async def test_context_retrieved_after_context_summary(self, plugin, mock_window_manager):
        """Test that context is retrieved after context summary completes."""
        # Set context timestamp
        plugin._context_summary_timestamp = 50.0
        mock_window_manager.get_text_and_window_ids_since_timestamp.return_value = (
            "Context from previous summary",
            [1, 2]
        )
        
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"type": "rapid_summary", "summary": []}
            
            await plugin.process(summary_window_id=3)
            
            # Verify context was retrieved
            mock_window_manager.get_text_and_window_ids_since_timestamp.assert_called_once_with(50.0)