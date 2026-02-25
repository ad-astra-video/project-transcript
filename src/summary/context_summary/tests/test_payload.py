"""
Tests for expected payloads returned from context_summary plugin.

This module tests that the context_summary plugin returns the expected
payload structure when processing summary windows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone
from typing import Dict, Any

from src.summary.context_summary import ContextSummaryPlugin, ContentTypeStateHolder
from src.summary.window_manager import WindowManager


class TestContextSummaryPayload:
    """Test expected payload structure from ContextSummaryPlugin."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        mock = MagicMock()
        mock.reasoning_llm_client = MagicMock()
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
        mock_window.text = "Test transcription text for summary"
        
        wm._summary_windows = [mock_window]
        wm.get_window_start = MagicMock(return_value=100.0)
        wm.get_window_end = MagicMock(return_value=110.0)
        wm.get_window_transcription_ids = MagicMock(return_value=[1])
        wm._first_window_timestamp = 50.0  # Allow processing (elapsed > delay)
        
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
    def plugin(self, mock_llm_manager, mock_window_manager, result_callback, mock_summary_client):
        """Create a ContextSummaryPlugin instance for testing."""
        return ContextSummaryPlugin(
            window_manager=mock_window_manager,
            llm_manager=mock_llm_manager,
            result_callback=result_callback,
            summary_client=mock_summary_client,
            initial_summary_delay_seconds=5.0
        )

    @pytest.mark.asyncio
    async def test_payload_structure(self, plugin, mock_window_manager, result_callback):
        """Test that plugin returns expected payload structure."""
        # Mock the task's process_context_summary method
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Test reasoning content",
                "summary_text": '{"insights": []}',
                "input_tokens": 150,
                "output_tokens": 75,
                "insights": []
            }
            
            # Call the process method
            result = await plugin.process(summary_window_id=1)
            
            # Verify result callback was called
            result_callback.assert_called_once()
            call_args = result_callback.call_args[0][0]
            
            # Verify payload structure
            assert call_args["type"] == "context_summary"
            assert "timestamp_utc" in call_args
            assert "timing" in call_args
            assert "llm_usage" in call_args
            assert "segments" in call_args
            
            # Verify timestamp_utc is valid ISO format
            datetime.fromisoformat(call_args["timestamp_utc"])

    @pytest.mark.asyncio
    async def test_payload_timing_fields(self, plugin, mock_window_manager, result_callback):
        """Test that timing fields are correctly populated."""
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Test reasoning",
                "summary_text": "{}",
                "input_tokens": 100,
                "output_tokens": 50,
                "insights": []
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            timing = call_args["timing"]
            
            # Verify timing fields
            assert "summary_window_id" in timing
            assert timing["summary_window_id"] == 1
            assert "transcription_window_ids" in timing
            assert timing["transcription_window_ids"] == [1]
            assert "media_window_start_ms" in timing
            assert timing["media_window_start_ms"] == 100000  # 100.0 * 1000
            assert "media_window_end_ms" in timing
            assert timing["media_window_end_ms"] == 110000  # 110.0 * 1000

    @pytest.mark.asyncio
    async def test_payload_llm_usage_fields(self, plugin, mock_window_manager, result_callback):
        """Test that LLM usage fields are correctly populated."""
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Test reasoning",
                "summary_text": "{}",
                "input_tokens": 1234,
                "output_tokens": 567,
                "insights": []
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            llm_usage = call_args["llm_usage"]
            
            # Verify LLM usage fields
            assert "input_tokens" in llm_usage
            assert llm_usage["input_tokens"] == 1234
            assert "output_tokens" in llm_usage
            assert llm_usage["output_tokens"] == 567

    @pytest.mark.asyncio
    async def test_payload_segments_structure(self, plugin, mock_window_manager, result_callback):
        """Test that segments field has correct structure."""
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Background context from LLM",
                "summary_text": '{"insights": []}',
                "input_tokens": 100,
                "output_tokens": 50,
                "insights": []
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            segments = call_args["segments"]
            
            # Verify segments is a list
            assert isinstance(segments, list)
            assert len(segments) == 1
            
            # Verify segment structure
            segment = segments[0]
            assert "id" in segment
            assert "summary_type" in segment
            assert segment["summary_type"] == "context_summary"
            assert "background_context" in segment
            assert "summary" in segment

    @pytest.mark.asyncio
    async def test_payload_fields_types(self, plugin, mock_window_manager, result_callback):
        """Test that payload fields have correct types."""
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Test",
                "summary_text": "{}",
                "input_tokens": 100,
                "output_tokens": 50,
                "insights": []
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            
            # Verify field types
            assert isinstance(call_args["type"], str)
            assert isinstance(call_args["timestamp_utc"], str)
            assert isinstance(call_args["timing"], dict)
            assert isinstance(call_args["llm_usage"], dict)
            assert isinstance(call_args["segments"], list)

    @pytest.mark.asyncio
    async def test_payload_emits_context_summary_complete_event(self, plugin, mock_window_manager, result_callback, mock_summary_client):
        """Test that processing emits on_context_summary_complete event."""
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Test",
                "summary_text": "{}",
                "input_tokens": 100,
                "output_tokens": 50,
                "insights": []
            }
            
            result = await plugin.process(summary_window_id=1)
            
            # Verify on_context_summary_complete event was emitted
            mock_summary_client._notify_plugins.assert_called_once_with(
                "on_context_summary_complete",
                summary_window_id=1,
                timestamp=110.0  # window_end timestamp
            )

    @pytest.mark.asyncio
    async def test_payload_with_insights(self, plugin, mock_window_manager, result_callback):
        """Test that payload includes insights when returned from task."""
        mock_insight = MagicMock()
        mock_insight.insight_id = 1
        mock_insight.insight_type = "KEY_POINT"
        mock_insight.insight_text = "Test key point"
        mock_insight.timestamp_start = 100.0
        mock_insight.timestamp_end = 110.0
        
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "reasoning_content": "Test reasoning",
                "summary_text": "{}",
                "input_tokens": 100,
                "output_tokens": 50,
                "insights": [mock_insight]
            }
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            # The payload should include insights in the segments
            assert "segments" in call_args