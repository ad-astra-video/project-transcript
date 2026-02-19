"""
Tests for expected payloads returned from rapid_summary plugin.

This module tests that the rapid_summary plugin returns the expected
payload structure when processing summary windows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone
from typing import Dict, Any

from src.summary.rapid_summary import RapidSummaryPlugin
from src.summary.window_manager import WindowManager


class TestRapidSummaryPayload:
    """Test expected payload structure from RapidSummaryPlugin."""

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
    async def test_payload_structure(self, plugin, mock_window_manager, result_callback):
        """Test that plugin returns expected payload structure."""
        # Mock the task's build_rapid_summary_payload method
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {
                    "transcription_window_ids": [1],
                    "media_window_start_ms": 100000,
                    "media_window_end_ms": 110000
                },
                "summary": [{"item": "Test rapid summary content"}]
            }
            
            # Call the process method
            result = await plugin.process(summary_window_id=1)
            
            # Verify result callback was called
            result_callback.assert_called_once()
            call_args = result_callback.call_args[0][0]
            
            # Verify payload structure
            assert call_args["type"] == "rapid_summary"
            assert "timestamp_utc" in call_args
            assert "timing" in call_args
            assert "summary" in call_args
            
            # Verify timestamp_utc is valid ISO format
            datetime.fromisoformat(call_args["timestamp_utc"])

    @pytest.mark.asyncio
    async def test_payload_timing_fields(self, plugin, mock_window_manager, result_callback):
        """Test that timing fields are correctly populated."""
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {
                    "transcription_window_ids": [1, 2, 3],
                    "media_window_start_ms": 100000,
                    "media_window_end_ms": 110000
                },
                "summary": [{"item": "Test"}]
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            timing = call_args["timing"]
            
            # Verify timing fields
            assert "transcription_window_ids" in timing
            assert timing["transcription_window_ids"] == [1, 2, 3]
            assert "media_window_start_ms" in timing
            assert timing["media_window_start_ms"] == 100000
            assert "media_window_end_ms" in timing
            assert timing["media_window_end_ms"] == 110000

    @pytest.mark.asyncio
    async def test_payload_summary_structure(self, plugin, mock_window_manager, result_callback):
        """Test that summary field has correct structure."""
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {"transcription_window_ids": [1]},
                "summary": [
                    {"item": "First summary item"},
                    {"item": "Second summary item"}
                ]
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            summary = call_args["summary"]
            
            # Verify summary is a list
            assert isinstance(summary, list)
            assert len(summary) == 2
            
            # Verify summary items have correct structure
            for item in summary:
                assert "item" in item
                assert isinstance(item["item"], str)

    @pytest.mark.asyncio
    async def test_payload_fields_types(self, plugin, mock_window_manager, result_callback):
        """Test that payload fields have correct types."""
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {"transcription_window_ids": [1]},
                "summary": [{"item": "Test"}]
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            
            # Verify field types
            assert isinstance(call_args["type"], str)
            assert isinstance(call_args["timestamp_utc"], str)
            assert isinstance(call_args["timing"], dict)
            assert isinstance(call_args["summary"], list)

    @pytest.mark.asyncio
    async def test_payload_with_context_summary_timestamp(self, plugin, mock_window_manager, result_callback):
        """Test that payload is generated correctly when context_summary_timestamp is set."""
        # Simulate context_summary_complete event being received
        plugin._context_summary_timestamp = 100.0
        
        # Update mock to return context
        mock_window_manager.get_text_and_window_ids_since_timestamp = MagicMock(
            return_value=("Previous context text", [1, 2])
        )
        
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {"transcription_window_ids": [1]},
                "summary": [{"item": "Summary with context"}]
            }
            
            result = await plugin.process(summary_window_id=1)
            
            # Verify the task was called with context
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert "context_since_last_summary" in call_kwargs

    @pytest.mark.asyncio
    async def test_payload_empty_when_no_fast_client(self, plugin, mock_window_manager, result_callback):
        """Test that plugin returns empty when fast_client is not available."""
        # Disable fast client
        plugin._llm.fast_client = None
        
        # Call the process method
        result = await plugin.process(summary_window_id=1)
        
        # Should return empty dict when fast client not available
        assert result == {}
        result_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_payload_with_empty_summary(self, plugin, mock_window_manager, result_callback):
        """Test that plugin handles empty summary gracefully."""
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {"transcription_window_ids": [1]},
                "summary": []
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            # Empty summary is still a valid payload
            assert call_args["summary"] == []

    @pytest.mark.asyncio
    async def test_payload_timing_with_multiple_transcription_windows(self, plugin, mock_window_manager, result_callback):
        """Test that timing field correctly captures multiple transcription window IDs."""
        # Update mock to return multiple transcription window IDs
        mock_window_manager.get_window_transcription_ids = MagicMock(return_value=[1, 2, 3, 4, 5])
        
        with patch.object(plugin._task, 'build_rapid_summary_payload', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "type": "rapid_summary",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {
                    "transcription_window_ids": [1, 2, 3, 4, 5],
                    "media_window_start_ms": 50000,
                    "media_window_end_ms": 60000
                },
                "summary": [{"item": "Test"}]
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            timing = call_args["timing"]
            
            assert timing["transcription_window_ids"] == [1, 2, 3, 4, 5]
            assert timing["media_window_start_ms"] == 50000
            assert timing["media_window_end_ms"] == 60000