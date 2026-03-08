"""
Tests for expected payloads returned from content_type_detection plugin.

This module tests that the content_type_detection plugin returns the expected
payload structure when processing summary windows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any

from src.summary.content_type_detection import ContentTypeDetectionPlugin
from src.summary.window_manager import WindowManager


class TestContentTypeDetectionPayload:
    """Test expected payload structure from ContentTypeDetectionPlugin."""

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
    async def test_payload_structure_auto_detected(self, plugin, mock_window_manager, result_callback):
        """Test that plugin returns expected payload structure for auto-detected content type."""
        # Mock the task's detect_content_type method
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="GENERAL_MEETING",
                confidence=0.85,
                reasoning="Detected meeting content based on discussion patterns"
            )
            
            # Call the process method
            result = await plugin.process(summary_window_id=1)
            
            # Verify result callback was called
            result_callback.assert_called_once()
            call_args = result_callback.call_args[0][0]
            
            # Verify payload structure
            assert call_args["type"] == "content_type_detection"
            assert "content_type" in call_args
            assert call_args["content_type"] == "GENERAL_MEETING"
            assert "confidence" in call_args
            assert call_args["confidence"] == 0.85
            assert "source" in call_args
            assert call_args["source"] == "AUTO_DETECTED"
            assert "previous_content_type" in call_args
            assert "timestamp_utc" in call_args
            
            # Verify timestamp_utc is valid ISO format
            datetime.fromisoformat(call_args["timestamp_utc"])

    @pytest.mark.asyncio
    async def test_payload_structure_user_override(self, plugin, mock_window_manager, result_callback):
        """Test that plugin returns expected payload structure for user override."""
        # Set user override before processing
        plugin.set_content_type_override("TECHNICAL_TALK")
        
        # Call the process method
        result = await plugin.process(summary_window_id=1)
        
        # Verify result callback was called
        result_callback.assert_called_once()
        call_args = result_callback.call_args[0][0]
        
        # Verify payload structure for user override
        assert call_args["type"] == "content_type_detection"
        assert call_args["content_type"] == "TECHNICAL_TALK"
        assert call_args["confidence"] == 1.0  # User override has full confidence
        assert call_args["source"] == "USER_OVERRIDE"
        assert "timestamp_utc" in call_args

    @pytest.mark.asyncio
    async def test_payload_fields_types(self, plugin, mock_window_manager, result_callback):
        """Test that payload fields have correct types."""
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = MagicMock(
                content_type="PODCAST",
                confidence=0.92,
                reasoning="Audio content with conversational patterns"
            )
            
            result = await plugin.process(summary_window_id=1)
            
            call_args = result_callback.call_args[0][0]
            
            # Verify field types
            assert isinstance(call_args["type"], str)
            assert isinstance(call_args["content_type"], str)
            assert isinstance(call_args["confidence"], float)
            assert isinstance(call_args["source"], str)
            assert isinstance(call_args["previous_content_type"], str)
            assert isinstance(call_args["timestamp_utc"], str)

    @pytest.mark.asyncio
    async def test_payload_previous_content_type_tracking(self, plugin, mock_window_manager, result_callback):
        """Test that previous_content_type is tracked correctly across calls."""
        with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
            # First call - should have "UNKNOWN" as previous_content_type (initial state)
            mock_detect.return_value = MagicMock(
                content_type="INTERVIEW",
                confidence=0.78,
                reasoning="Q&A format detected"
            )
            
            result1 = await plugin.process(summary_window_id=1)
            call_args1 = result_callback.call_args[0][0]
            assert call_args1["previous_content_type"] == "UNKNOWN"  # Initial state is "UNKNOWN"
            
            # Second call - should have previous_content_type set to first result
            # Need to reset auto_detect to allow processing again
            plugin._auto_detect = True
            result_callback.reset_mock()
            mock_detect.return_value = MagicMock(
                content_type="LECTURE_OR_TALK",
                confidence=0.88,
                reasoning="Educational content detected"
            )
            
            result2 = await plugin.process(summary_window_id=2)
            
            # Verify callback was called
            result_callback.assert_called_once()
            call_args2 = result_callback.call_args[0][0]
            assert call_args2["previous_content_type"] == "INTERVIEW"

    @pytest.mark.asyncio
    async def test_payload_all_content_types(self, plugin, mock_window_manager, result_callback):
        """Test that plugin handles all valid content types."""
        content_types = [
            "GENERAL_MEETING",
            "TECHNICAL_TALK",
            "LECTURE_OR_TALK",
            "INTERVIEW",
            "PODCAST",
            "STREAMER_MONOLOGUE",
            "NEWS_UPDATE",
            "GAMEPLAY_COMMENTARY",
            "CUSTOMER_SUPPORT",
            "DEBATE",
            "UNKNOWN"
        ]
        
        for i, content_type in enumerate(content_types):
            # Reset auto_detect for each iteration to allow processing
            plugin._auto_detect = True
            
            with patch.object(plugin._task, 'detect_content_type', new_callable=AsyncMock) as mock_detect:
                mock_detect.return_value = MagicMock(
                    content_type=content_type,
                    confidence=0.75,
                    reasoning="Test reasoning"
                )
                
                result_callback.reset_mock()
                result = await plugin.process(summary_window_id=i+1)
                
                # Verify callback was called
                result_callback.assert_called_once()
                call_args = result_callback.call_args[0][0]
                assert call_args["content_type"] == content_type