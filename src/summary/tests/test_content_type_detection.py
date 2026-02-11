"""
Tests for content type detection timing behavior.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.summary.summary_client import SummaryClient, WindowManager, ContentTypeState, ContentType


class TestContentTypeDetectionTiming:
    """Tests for content type detection timing behavior."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock SummaryClient for testing."""
        with patch('src.summary.summary_client.AsyncOpenAI') as mock_openai:
            client = SummaryClient(
                api_key="test-key",
                base_url="http://test:8000/v1",
                model="test-model",
                transcription_windows_per_summary_window=2,
                initial_summary_delay_seconds=10.0
            )
            # Mock the OpenAI client
            client.client = AsyncMock()
            return client
    
    @pytest.mark.asyncio
    async def test_detection_runs_at_75_percent_threshold_first_buffered_window(self, mock_client):
        """Content type detection should run when elapsed >= 75% of initial delay AND window is first buffered."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "TECHNICAL_TALK", "confidence": 0.85, "reasoning": "Test"}'
        mock_response.choices[0].message.reasoning = ""
        mock_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Simulate first window arriving
        mock_client._window_manager._first_window_timestamp = 0.0
        
        # Simulate buffering - first buffered window with elapsed >= 75% threshold
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = []  # Empty buffer
        
        # Create test segments
        segments = [{"id": "1", "start_ms": 0, "end_ms": 2500, "text": "Test transcription"}]
        
        # Call process_segments with elapsed time at 75% threshold (7.5s)
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=1,
            window_start=7.5,  # 75% of 10s initial delay
            window_end=10.0
        )
        
        # Verify detection ran and returned result
        assert result.get("type") == "content_type_detection"
        assert result.get("content_type") == "TECHNICAL_TALK"
        assert result.get("source") == "AUTO_DETECTED"
        assert mock_client._auto_detect_content_type_detection == False
    
    @pytest.mark.asyncio
    async def test_detection_skipped_before_75_percent_threshold(self, mock_client):
        """Content type detection should NOT run before elapsed >= 75% of initial delay."""
        # Simulate first window arriving
        mock_client._window_manager._first_window_timestamp = 0.0
        
        # Simulate buffering - first buffered window but elapsed < 75% threshold
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = []  # Empty buffer
        
        # Create test segments
        segments = [{"id": "1", "start_ms": 0, "end_ms": 2500, "text": "Test transcription"}]
        
        # Call process_segments with elapsed time below 75% threshold (5s)
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=1,
            window_start=5.0,  # 50% of 10s initial delay
            window_end=7.5
        )
        
        # Verify no detection ran - should return empty context_summary
        assert result.get("type") == "context_summary"
        assert result.get("segments") == []
        assert mock_client._auto_detect_content_type_detection == True  # Still pending
    
    @pytest.mark.asyncio
    async def test_detection_skipped_with_user_override(self, mock_client):
        """Detection should be skipped and content_type_detection sent when user override is set."""
        # Set user override
        mock_client._user_content_type_override = "INTERVIEW"
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = []  # Empty buffer
        
        # Create test segments
        segments = [{"id": "1", "start_ms": 0, "end_ms": 2500, "text": "Test transcription"}]
        
        # Call process_segments
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=1,
            window_start=7.5,
            window_end=10.0
        )
        
        # Verify detection was skipped and override was sent
        assert result.get("type") == "content_type_detection"
        assert result.get("content_type") == "INTERVIEW"
        assert result.get("source") == "USER_OVERRIDE"
        assert result.get("confidence") == 1.0
        assert mock_client._auto_detect_content_type_detection == False
    
    @pytest.mark.asyncio
    async def test_detection_not_runs_on_second_buffered_window(self, mock_client):
        """Content type detection should NOT run on second buffered window (buffer length > 1)."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "TECHNICAL_TALK", "confidence": 0.85, "reasoning": "Test"}'
        mock_response.choices[0].message.reasoning = ""
        mock_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Simulate first window arriving
        mock_client._window_manager._first_window_timestamp = 0.0
        
        # Simulate buffering - second buffered window (buffer already has items)
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = [{"id": "1"}]  # Already has one item
        
        # Create test segments
        segments = [{"id": "2", "start_ms": 2500, "end_ms": 5000, "text": "More test transcription"}]
        
        # Call process_segments
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=2,
            window_start=7.5,
            window_end=10.0
        )
        
        # Verify no detection ran - buffer already had items
        assert result.get("type") == "context_summary"
        assert result.get("segments") == []
        assert mock_client._auto_detect_content_type_detection == True  # Still pending
    
    def test_detection_state_resets_on_stream(self, mock_client):
        """Detection state should reset properly on new stream."""
        # Set up some state
        mock_client._auto_detect_content_type_detection = False
        mock_client._content_type_state = ContentTypeState(
            content_type="TECHNICAL_TALK",
            confidence=0.85,
            source="AUTO_DETECTED"
        )
        
        # Reset the client
        mock_client.reset()
        
        # Verify state was reset
        assert mock_client._auto_detect_content_type_detection == True
        assert mock_client._content_type_state.content_type == ContentType.UNKNOWN.value
        assert mock_client._content_type_state.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_detection_runs_after_summary_buffer_flush(self, mock_client):
        """Content type detection should run on first buffered window after summary (buffer flushes)."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "TECHNICAL_TALK", "confidence": 0.90, "reasoning": "Re-check"}'
        mock_response.choices[0].message.reasoning = ""
        mock_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Simulate first summary has been performed
        mock_client._has_performed_summary = True
        mock_client._window_manager._first_window_timestamp = 0.0
        
        # Simulate buffer was flushed (empty buffer)
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = []  # Empty buffer after flush
        
        # Create test segments
        segments = [{"id": "1", "start_ms": 0, "end_ms": 2500, "text": "Test transcription after summary"}]
        
        # Call process_segments
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=1,
            window_start=7.5,
            window_end=10.0
        )
        
        # Verify detection ran
        assert result.get("type") == "content_type_detection"
        assert result.get("content_type") == "TECHNICAL_TALK"
        assert result.get("source") == "AUTO_DETECTED"
    
    @pytest.mark.asyncio
    async def test_concurrent_detection_skipped_when_in_progress(self, mock_client):
        """Content type detection should be skipped when another detection is already in progress."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "TECHNICAL_TALK", "confidence": 0.85, "reasoning": "Test"}'
        mock_response.choices[0].message.reasoning = ""
        mock_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Simulate detection already in progress (concurrent worker scenario)
        mock_client._content_type_detection_in_progress = True
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = []  # Empty buffer (first window)
        mock_client._window_manager._first_window_timestamp = 0.0
        
        # Create test segments
        segments = [{"id": "1", "start_ms": 0, "end_ms": 2500, "text": "Test transcription"}]
        
        # Call process_segments - should skip detection because it's already in progress
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=1,
            window_start=7.5,  # 75% of 10s initial delay
            window_end=10.0
        )
        
        # Verify detection was skipped - no LLM call made
        assert result.get("type") == "context_summary"
        assert result.get("segments") == []
        assert mock_client.client.chat.completions.create.call_count == 0
        # Auto-detection flag should still be True (detection still pending)
        assert mock_client._auto_detect_content_type_detection == True
    
    @pytest.mark.asyncio
    async def test_detection_flag_cleared_on_completion(self, mock_client):
        """Content type detection in-progress flag should be cleared after detection completes."""
        # Set up mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"content_type": "PODCAST", "confidence": 0.92, "reasoning": "Test"}'
        mock_response.choices[0].message.reasoning = ""
        mock_client.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Verify flag is initially False
        assert mock_client._content_type_detection_in_progress == False
        
        # Simulate first window arriving with enough elapsed time
        mock_client._window_manager._first_window_timestamp = 0.0
        mock_client._auto_detect_content_type_detection = True
        mock_client._temp_segment_buffer = []
        
        segments = [{"id": "1", "start_ms": 0, "end_ms": 2500, "text": "Test transcription"}]
        
        # Call process_segments - should trigger detection
        result = await mock_client.process_segments(
            "context_summary",
            segments,
            transcription_window_id=1,
            window_start=7.5,
            window_end=10.0
        )
        
        # Verify detection ran and flag was cleared
        assert result.get("type") == "content_type_detection"
        assert mock_client._content_type_detection_in_progress == False
    
    def test_in_progress_flag_resets_on_stream(self, mock_client):
        """In-progress flag should reset properly on new stream."""
        # Set up in-progress state
        mock_client._content_type_detection_in_progress = True
        mock_client._auto_detect_content_type_detection = False
        
        # Reset the client
        mock_client.reset()
        
        # Verify in-progress flag was reset
        assert mock_client._content_type_detection_in_progress == False
        assert mock_client._auto_detect_content_type_detection == True


class TestWindowManagerBufferTracking:
    """Tests for WindowManager buffer tracking behavior."""
    
    def test_buffer_length_tracking(self):
        """Test that buffer length is correctly tracked."""
        manager = WindowManager(transcription_windows_per_summary_window=2)
        
        # Initially empty
        assert len(manager._windows) == 0
        
        # Add first window
        manager.add_window("First window text", 0.0, 2.5, window_id=1)
        assert len(manager._windows) == 1
        
        # Add second window
        manager.add_window("Second window text", 2.5, 5.0, window_id=2)
        assert len(manager._windows) == 2
        
        # Add third window (should trigger eviction due to char limit)
        manager.add_window("Third window text", 5.0, 7.5, window_id=3)
        # Depending on max_chars setting, oldest may be evicted
        assert len(manager._windows) >= 1  # At least one window should remain
    
    def test_first_window_timestamp_tracking(self):
        """Test that first window timestamp is correctly tracked."""
        manager = WindowManager(transcription_windows_per_summary_window=2)
        
        # Initially None
        assert manager._first_window_timestamp is None
        
        # Add first window
        manager.add_window("Test text", 5.0, 7.5, window_id=1)
        assert manager._first_window_timestamp == 5.0
        
        # Add second window - timestamp should not change
        manager.add_window("More text", 7.5, 10.0, window_id=2)
        assert manager._first_window_timestamp == 5.0
    
    def test_clear_resets_all_state(self):
        """Test that clear() resets all state including buffer tracking."""
        manager = WindowManager(transcription_windows_per_summary_window=2)
        
        # Add some windows
        manager.add_window("Test 1", 0.0, 2.5, window_id=1)
        manager.add_window("Test 2", 2.5, 5.0, window_id=2)
        
        # Verify state is set
        assert len(manager._windows) == 2
        assert manager._first_window_timestamp == 0.0
        assert manager._next_window_id == 2  # Counter increments by 1 for each explicit window_id
        
        # Clear
        manager.clear()
        
        # Verify all state is reset
        assert len(manager._windows) == 0
        assert manager._first_window_timestamp is None
        assert manager._next_window_id == 0
        assert manager._char_count == 0