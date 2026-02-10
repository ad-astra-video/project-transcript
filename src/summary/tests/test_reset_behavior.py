"""
Unit tests for reset behavior.
"""

import pytest
from src.summary.summary_client import (
    SummaryClient,
    WindowManager,
    WindowInsight,
    ContentTypeState,
    ContentType,
    ContentTypeSource,
)


class TestSummaryClientReset:
    """Tests for SummaryClient reset behavior."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_summary_client_clears_skipped_transcription_ids(self):
        """Test that _skipped_transcription_ids is cleared on reset."""
        client = self.create_client()
        
        # Add some skipped IDs
        client._skipped_transcription_ids = [1, 2, 3]
        
        # Reset
        client.reset()
        
        # Verify cleared
        assert len(client._skipped_transcription_ids) == 0
    
    def test_summary_client_resets_window_counters(self):
        """Test that window tracking counters are reset on reset()."""
        client = self.create_client()
        
        # Set some values
        client.summary_window_counter = 10
        client.summary_window_count = 5
        
        # Reset
        client.reset()
        
        # Verify reset
        assert client.summary_window_counter == 0
        assert client.summary_window_count == 0
    
    def test_summary_client_resets_window_manager(self):
        """Test that WindowManager is cleared on SummaryClient reset."""
        client = self.create_client()
        
        # Add some windows
        client._window_manager.add_window("text1", 0.0, 5.0)
        client._window_manager.add_window("text2", 5.0, 10.0)
        
        # Verify windows exist
        assert len(client._window_manager._windows) == 2
        assert client._window_manager._next_window_id == 2
        
        # Reset
        client.reset()
        
        # Verify WindowManager is cleared
        assert len(client._window_manager._windows) == 0
        assert client._window_manager._next_window_id == 0
    
    def test_summary_client_clears_in_flight_windows(self):
        """Test that in_flight_windows set is cleared on reset."""
        client = self.create_client()
        
        # Add some in-flight windows
        client.in_flight_windows = {1, 2, 3}
        
        # Reset
        client.reset()
        
        # Verify cleared
        assert len(client.in_flight_windows) == 0
    
    def test_summary_client_clears_skipped_segments_buffer(self):
        """Test that _skipped_segments_buffer is cleared on reset."""
        client = self.create_client()
        
        # Set skipped segments buffer
        client._skipped_segments_buffer = {
            "segments": [{"text": "test"}],
            "transcription_window_id": 1,
            "window_start": 0.0,
            "window_end": 5.0,
            "text": "test text"
        }
        
        # Reset
        client.reset()
        
        # Verify cleared
        assert client._skipped_segments_buffer is None
    
    def test_summary_client_resets_content_type_state(self):
        """Test that content type state is reset on reset."""
        client = self.create_client()
        
        # Set content type state
        client._content_type_state = ContentTypeState(
            content_type="GENERAL_MEETING",
            confidence=0.9,
            source="AUTO_DETECTED"
        )
        
        # Reset
        client.reset()
        
        # Verify reset to defaults
        assert client._content_type_state.content_type == ContentType.UNKNOWN.value
        assert client._content_type_state.confidence == 0.0
        assert client._content_type_state.source == ContentTypeSource.INITIAL.value


class TestWhisperClientReset:
    """Tests for WhisperClient reset behavior."""
    
    def test_whisper_client_resets_transcription_id(self):
        """Test that WhisperClient resets _next_transcription_id on reset."""
        # Create a mock WhisperClient (without actual model loading)
        from src.transcription.whisper_client import WhisperClient
        
        # We can't fully instantiate without faster_whisper, but we can test the reset logic
        # by checking the method exists and the counter logic
        
        # Create a minimal test by checking the method signature
        client = object.__new__(WhisperClient)
        client._next_transcription_id = 10
        client.logger = __import__('logging').getLogger(__name__)
        
        # Call reset
        client.reset()
        
        # Verify reset
        assert client._next_transcription_id == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])