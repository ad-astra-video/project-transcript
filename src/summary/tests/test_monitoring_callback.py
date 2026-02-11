"""
Tests for monitoring event callback functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.summary.summary_client import SummaryClient, ContentType, ContentTypeState


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
                api_key="test-key",
                base_url="http://test:8000/v1",
                model="test-model",
                transcription_windows_per_summary_window=2,
                initial_summary_delay_seconds=10.0,
                send_monitoring_event_callback=mock_callback
            )
            client.client = AsyncMock()
            return client
    
    @pytest.mark.asyncio
    async def test_callback_invoked_on_content_type_change(self, client_with_callback, mock_callback):
        """Callback should be invoked when content type changes from UNKNOWN to known."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "TECHNICAL_TALK", "confidence": 0.85, "reasoning": "Test reasoning"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Simulate UNKNOWN state (initial state)
        assert client_with_callback._content_type_state.content_type == ContentType.UNKNOWN.value
        
        # Run detection
        result = await client_with_callback.process_content_type_detection(
            window_start=0.0,
            window_end=10.0
        )
        
        # Verify callback was invoked
        mock_callback.assert_called_once()
        
        # Verify callback was called with (dict, str) signature
        event_data, event_type = mock_callback.call_args[0]
        assert event_type == "content_type_changed"
        assert event_data["previous_content_type"] == ContentType.UNKNOWN.value
        assert event_data["new_content_type"] == "TECHNICAL_TALK"
        assert event_data["confidence"] == 0.85
        assert event_data["reasoning"] == "Test reasoning"
    
    @pytest.mark.asyncio
    async def test_callback_invoked_on_any_content_type_change(self, client_with_callback, mock_callback):
        """Callback should be invoked whenever content type changes, not just UNKNOWN -> known."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "INTERVIEW", "confidence": 0.90, "reasoning": "Question-answer format detected"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Set known content type state (different from what LLM will return)
        client_with_callback._content_type_state = ContentTypeState(
            content_type="TECHNICAL_TALK",
            confidence=0.85,
            source="AUTO_DETECTED"
        )
        
        # Run detection
        result = await client_with_callback.process_content_type_detection(
            window_start=0.0,
            window_end=10.0
        )
        
        # Verify callback WAS invoked (content type changed from TECHNICAL_TALK to INTERVIEW)
        mock_callback.assert_called_once()
        
        # Verify event data
        event_data, event_type = mock_callback.call_args[0]
        assert event_type == "content_type_changed"
        assert event_data["previous_content_type"] == "TECHNICAL_TALK"
        assert event_data["new_content_type"] == "INTERVIEW"
    
    @pytest.mark.asyncio
    async def test_callback_not_invoked_when_content_type_unchanged(self, client_with_callback, mock_callback):
        """Callback should NOT be invoked when content type remains the same."""
        # Set up mock LLM response returning same content type
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "TECHNICAL_TALK", "confidence": 0.90, "reasoning": "Re-check"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Set known content type state (same as what LLM will return)
        client_with_callback._content_type_state = ContentTypeState(
            content_type="TECHNICAL_TALK",
            confidence=0.85,
            source="AUTO_DETECTED"
        )
        
        # Run detection
        result = await client_with_callback.process_content_type_detection(
            window_start=0.0,
            window_end=10.0
        )
        
        # Verify callback was NOT invoked (content type unchanged)
        mock_callback.assert_not_called()
    
    def test_callback_not_set_when_none_provided(self):
        """Client should work normally when no callback is provided."""
        with patch('src.summary.summary_client.AsyncOpenAI') as mock_openai:
            client = SummaryClient(
                api_key="test-key",
                base_url="http://test:8000/v1",
                model="test-model"
            )
            assert client._send_monitoring_event_callback is None
    
    @pytest.mark.asyncio
    async def test_callback_exception_handled_gracefully(self, client_with_callback, mock_callback):
        """Callback exceptions should be caught and logged, not raised."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "LECTURE_OR_TALK", "confidence": 0.88, "reasoning": "Educational content"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Make callback raise an exception
        mock_callback.side_effect = Exception("Callback error")
        
        # Should not raise - exception should be caught
        result = await client_with_callback.process_content_type_detection(
            window_start=0.0,
            window_end=10.0
        )
        
        # Verify callback was still called
        mock_callback.assert_called_once()
        
        # Verify result was still returned
        assert result is not None
        assert result.content_type == "LECTURE_OR_TALK"
    
    @pytest.mark.asyncio
    async def test_callback_with_unknown_to_unknown(self, client_with_callback, mock_callback):
        """Callback should NOT be invoked when content type remains UNKNOWN."""
        # Set up mock LLM response returning UNKNOWN
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "UNKNOWN", "confidence": 0.5, "reasoning": "Not enough context"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run detection
        result = await client_with_callback.process_content_type_detection(
            window_start=0.0,
            window_end=10.0
        )
        
        # Verify callback was NOT invoked (still unknown, no change)
        mock_callback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_callback_includes_all_event_data(self, client_with_callback, mock_callback):
        """Callback should receive complete event data including timing and context."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "PODCAST", "confidence": 0.95, "reasoning": "Conversational format with multiple speakers"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Run detection with specific timing
        result = await client_with_callback.process_content_type_detection(
            window_start=5.5,
            window_end=15.0
        )
        
        # Verify callback was invoked
        mock_callback.assert_called_once()
        
        # Verify all event data fields
        event_data, event_type = mock_callback.call_args[0]
        assert event_type == "content_type_changed"
        assert "previous_content_type" in event_data
        assert "new_content_type" in event_data
        assert "confidence" in event_data
        assert "reasoning" in event_data
        assert "context_length" in event_data
        assert "source" in event_data
        assert "window_start" in event_data
        assert "window_end" in event_data
        assert "timestamp_utc" in event_data
        
        # Verify specific values
        assert event_data["new_content_type"] == "PODCAST"
        assert event_data["confidence"] == 0.95
        assert event_data["window_start"] == 5.5
        assert event_data["window_end"] == 15.0
        assert event_data["source"] == "AUTO_DETECTED"
    
    @pytest.mark.asyncio
    async def test_callback_exception_handled_gracefully(self, client_with_callback, mock_callback):
        """Callback exceptions should be caught and logged, not raised."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "LECTURE_OR_TALK", "confidence": 0.88, "reasoning": "Educational content"}'
        mock_response.choices[0].message.reasoning = ""
        client_with_callback.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Make callback raise an exception
        mock_callback.side_effect = Exception("Callback error")
        
        # Should not raise - exception should be caught
        result = await client_with_callback.process_content_type_detection(
            window_start=0.0,
            window_end=10.0
        )
        
        # Verify callback was still called
        mock_callback.assert_called_once()
        
        # Verify result was still returned
        assert result is not None
        assert result.content_type == "LECTURE_OR_TALK"