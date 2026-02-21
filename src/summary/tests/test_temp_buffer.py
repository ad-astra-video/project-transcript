"""
Unit tests for temp buffer accumulation with modulo-based processing.

In the refactored code, temp buffer functionality is handled by WindowManager
which tracks transcription windows and manages accumulation.
"""

import pytest
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager, TranscriptionWindow


class TestWindowManagerTranscriptionTracking:
    """Tests for WindowManager transcription window tracking."""
    
    def create_window_manager(self, transcription_windows_per_summary_window: int = 2):
        """Create a WindowManager instance for testing."""
        return WindowManager(
            transcription_windows_per_summary_window=transcription_windows_per_summary_window
        )
    
    def test_add_transcription_window_stores_segments(self):
        """Test that add_transcription_window stores segments correctly."""
        wm = self.create_window_manager()
        
        # Add a transcription window
        # New signature: add_transcription_window(transcription_window_id, segments, window_start_ts, window_end_ts)
        wm.add_transcription_window(
            transcription_window_id=1,
            segments=[{"id": "1", "text": "Hello world"}],
            window_start_ts=0.0,
            window_end_ts=2.5
        )
        
        # Verify it was stored
        assert 1 in wm._transcription_windows
        assert wm._transcription_windows[1].new_text == "Hello world"
    
    def test_add_transcription_window_multiple_windows(self):
        """Test adding multiple transcription windows."""
        wm = self.create_window_manager()
        
        # New signature: add_transcription_window(transcription_window_id, segments, window_start_ts, window_end_ts)
        wm.add_transcription_window(1, [{"id": "1", "text": "Hello"}], 0.0, 2.5)
        wm.add_transcription_window(2, [{"id": "2", "text": "World"}], 2.5, 5.0)
        wm.add_transcription_window(3, [{"id": "3", "text": "Test"}], 5.0, 7.5)
        
        # Verify all were stored
        assert len(wm._transcription_windows) == 3
        assert 1 in wm._transcription_windows
        assert 2 in wm._transcription_windows
        assert 3 in wm._transcription_windows
    
    def test_pending_transcription_ids_tracking(self):
        """Test that pending transcription IDs are tracked correctly."""
        wm = self.create_window_manager()
        
        # Add transcription windows
        # New signature: add_transcription_window(transcription_window_id, segments, window_start_ts, window_end_ts)
        wm.add_transcription_window(1, [{"id": "1", "text": "Hello"}], 0.0, 2.5)
        wm.add_transcription_window(2, [{"id": "2", "text": "World"}], 2.5, 5.0)
        
        # Check pending IDs
        assert 1 in wm._pending_transcription_ids
        assert 2 in wm._pending_transcription_ids
    
    def test_transcription_windows_per_summary_window_config(self):
        """Test transcription_windows_per_summary_window configuration."""
        wm = self.create_window_manager(transcription_windows_per_summary_window=4)
        
        assert wm.transcription_windows_per_summary_window == 4


class TestSummaryClientWithWindowManager:
    """Tests for SummaryClient using WindowManager for buffer management."""
    
    def test_summary_client_uses_window_manager(self):
        """Test that SummaryClient uses WindowManager for buffer management."""
        client = SummaryClient(
            reasoning_api_key="test_key",
            reasoning_model="test_model"
        )
        client.update_params(transcription_windows_per_summary_window=2)
        
        assert client._window_manager is not None
        assert client._window_manager.transcription_windows_per_summary_window == 2
    
    def test_summary_client_adds_transcription_window(self):
        """Test that SummaryClient can add transcription windows through WindowManager."""
        client = SummaryClient(
            reasoning_api_key="test_key",
            reasoning_model="test_model"
        )
        
        # Add transcription window via window manager
        # New signature: add_transcription_window(transcription_window_id, segments, window_start_ts, window_end_ts)
        client._window_manager.add_transcription_window(
            transcription_window_id=1,
            segments=[{"id": "1", "text": "Test text"}],
            window_start_ts=0.0,
            window_end_ts=5.0
        )
        
        # Verify it was stored
        assert 1 in client._window_manager._transcription_windows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])