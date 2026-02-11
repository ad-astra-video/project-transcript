"""
Unit tests for temp buffer accumulation with modulo-based processing.
"""

import pytest
from src.summary.summary_client import SummaryClient


class TestTempBufferForModuloProcessing:
    """Tests for temp buffer accumulation with modulo-based processing."""
    
    def create_client(self, transcription_windows_per_summary_window: int = 2):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(
            api_key="test_key",
            model="test_model",
            transcription_windows_per_summary_window=transcription_windows_per_summary_window
        )
    
    def test_add_to_temp_buffer_stores_segments(self):
        """Test that _add_to_temp_buffer stores segments in buffer."""
        client = self.create_client()
        
        assert len(client._temp_segment_buffer) == 0
        
        client._add_to_temp_buffer([{"id": "1", "text": "Hello"}], 1, 0.0, 2.5)
        
        # Counter is NOT incremented in _add_to_temp_buffer (it's done in process_segments)
        assert client._transcription_window_counter == 0
        assert len(client._temp_segment_buffer) == 1
        assert len(client._temp_buffer_window_ids) == 1
        assert client._temp_buffer_window_ids[0] == 1
    
    def test_add_to_temp_buffer_multiple_windows(self):
        """Test adding multiple windows to temp buffer."""
        client = self.create_client()
        
        client._add_to_temp_buffer([{"id": "1", "text": "Hello"}], 1, 0.0, 2.5)
        client._add_to_temp_buffer([{"id": "2", "text": "World"}], 2, 2.5, 5.0)
        client._add_to_temp_buffer([{"id": "3", "text": "Test"}], 3, 5.0, 7.5)
        
        # Counter is NOT incremented in _add_to_temp_buffer
        assert client._transcription_window_counter == 0
        assert len(client._temp_segment_buffer) == 3
        assert client._temp_buffer_window_ids == [1, 2, 3]
    
    def test_flush_temp_buffer_returns_merged_data(self):
        """Test that _flush_temp_buffer returns correctly merged data."""
        client = self.create_client()
        
        # Add segments to buffer
        client._add_to_temp_buffer([{"id": "1", "text": "Hello"}], 1, 0.0, 2.5)
        client._add_to_temp_buffer([{"id": "2", "text": "World"}], 2, 2.5, 5.0)
        
        # Flush buffer
        merged_segments, last_id, start, end, ids = client._flush_temp_buffer()
        
        assert len(merged_segments) == 2
        assert last_id == 2
        assert start == 0.0
        assert end == 5.0
        assert ids == [1, 2]
        
        # Buffer should be cleared
        assert len(client._temp_segment_buffer) == 0
        assert len(client._temp_buffer_timing) == 0
        assert len(client._temp_buffer_window_ids) == 0
    
    def test_flush_empty_buffer_returns_defaults(self):
        """Test that flushing empty buffer returns default values."""
        client = self.create_client()
        
        merged_segments, last_id, start, end, ids = client._flush_temp_buffer()
        
        assert merged_segments == []
        assert last_id == 0
        assert start == 0.0
        assert end == 0.0
        assert ids == []
    
    def test_buffer_accumulation_pattern_transcription_windows_per_summary_window_2(self):
        """Test buffer accumulation pattern with transcription_windows_per_summary_window=2.
        
        The counter is incremented in process_segments BEFORE calling _add_to_temp_buffer.
        When should_process is True, _flush_temp_buffer is called.
        """
        client = self.create_client(transcription_windows_per_summary_window=2)
        
        for i in range(1, 5):
            # Simulate what process_segments does: increment counter first
            client._transcription_window_counter += 1
            
            should_process = (client._transcription_window_counter % client._window_manager.transcription_windows_per_summary_window == 0)
            
            if not should_process:
                client._add_to_temp_buffer([{"id": str(i)}], i, 0.0, 2.5)
            else:
                # When should_process is True, flush the buffer (simulating process_segments behavior)
                client._flush_temp_buffer()
            
            # Verify buffer state based on modulo
            if client._transcription_window_counter % 2 == 1:
                assert len(client._temp_segment_buffer) == 1  # After window 1
            else:
                assert len(client._temp_segment_buffer) == 0  # After window 2 (flushed)
    
    def test_buffer_accumulation_pattern_transcription_windows_per_summary_window_3(self):
        """Test buffer accumulation pattern with transcription_windows_per_summary_window=3.
        
        The counter is incremented in process_segments BEFORE calling _add_to_temp_buffer.
        When should_process is True, _flush_temp_buffer is called.
        """
        client = self.create_client(transcription_windows_per_summary_window=3)
        
        for i in range(1, 7):
            # Simulate what process_segments does: increment counter first
            client._transcription_window_counter += 1
            
            should_process = (client._transcription_window_counter % client._window_manager.transcription_windows_per_summary_window == 0)
            
            if not should_process:
                client._add_to_temp_buffer([{"id": str(i)}], i, 0.0, 2.5)
            else:
                # When should_process is True, flush the buffer (simulating process_segments behavior)
                client._flush_temp_buffer()
            
            # Verify buffer state based on modulo
            mod = client._transcription_window_counter % 3
            if mod == 1:
                assert len(client._temp_segment_buffer) == 1  # After window 1
            elif mod == 2:
                assert len(client._temp_segment_buffer) == 2  # After window 2
            else:  # mod == 0
                assert len(client._temp_segment_buffer) == 0  # After window 3 (flushed)
    
    def test_reset_clears_temp_buffer(self):
        """Test that reset() clears all temp buffer state."""
        client = self.create_client()
        
        # Add to buffer
        client._add_to_temp_buffer([{"id": "1", "text": "Hello"}], 1, 0.0, 2.5)
        client._add_to_temp_buffer([{"id": "2", "text": "World"}], 2, 2.5, 5.0)
        
        assert len(client._temp_segment_buffer) == 2
        
        # Reset
        client.reset()
        
        # Verify cleared
        assert len(client._temp_segment_buffer) == 0
        assert len(client._temp_buffer_timing) == 0
        assert len(client._temp_buffer_window_ids) == 0
        assert client._transcription_window_counter == 0
    
    def test_update_transcription_windows_per_summary_window_with_pending_buffer(self):
        """Test update_transcription_windows_per_summary_window with pending buffer."""
        client = self.create_client(transcription_windows_per_summary_window=2)
        
        # Add one window to buffer
        client._add_to_temp_buffer([{"id": "1", "text": "Hello"}], 1, 0.0, 2.5)
        
        # Update transcription_windows_per_summary_window
        client.update_windows_to_accumulate(3)
        
        assert client._window_manager.transcription_windows_per_summary_window == 3
        # Buffer should still be there
        assert len(client._temp_segment_buffer) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])