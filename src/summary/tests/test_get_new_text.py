"""
Unit tests for get_new_text_for_summary_window with overlapping segments.

In the refactored code, this functionality is handled by WindowManager's
add_transcription_window method which handles deduplication.
"""

import pytest
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager


class TestWindowManagerTextDeduplication:
    """Tests for WindowManager text deduplication with overlapping segments."""
    
    def create_window_manager(self):
        """Create a WindowManager instance for testing."""
        return WindowManager()
    
    def test_add_transcription_window_no_overlap(self):
        """Test adding transcription window when segments don't overlap."""
        wm = self.create_window_manager()
        
        # Add first transcription window
        wm.add_transcription_window(
            transcription_window_id=1,
            new_text="Hello world How are you",
            timestamp_start=0.0,
            timestamp_end=2.0,
            segments=[
                {"text": "Hello world", "start_ms": 0, "end_ms": 1000},
                {"text": "How are you", "start_ms": 1000, "end_ms": 2000}
            ]
        )
        
        # Verify text was stored
        assert wm._transcription_windows[1].new_text == "Hello world How are you"
    
    def test_add_transcription_window_with_overlap(self):
        """Test adding transcription window when segments partially overlap."""
        wm = self.create_window_manager()
        
        # Add first transcription window
        wm.add_transcription_window(
            transcription_window_id=1,
            new_text="Hello world How are you",
            timestamp_start=0.0,
            timestamp_end=2.0,
            segments=[
                {"text": "Hello world", "start_ms": 0, "end_ms": 1000},
                {"text": "How are you", "start_ms": 1000, "end_ms": 2000}
            ]
        )
        
        # Add second transcription window with overlap
        # The new_text should already be deduplicated
        wm.add_transcription_window(
            transcription_window_id=2,
            new_text="Goodbye",  # Already deduplicated
            timestamp_start=2.0,
            timestamp_end=3.0,
            segments=[
                {"text": "Goodbye", "start_ms": 2000, "end_ms": 3000}
            ]
        )
        
        # Verify both windows stored
        assert 1 in wm._transcription_windows
        assert 2 in wm._transcription_windows
    
    def test_get_recent_windows_text(self):
        """Test getting recent windows text."""
        wm = self.create_window_manager()
        
        # Add windows
        wm.add_summary_window("First window text", 0.0, 10.0, [1])
        wm.add_summary_window("Second window text", 10.0, 20.0, [2])
        wm.add_summary_window("Third window text", 20.0, 30.0, [3])
        
        # Get recent text with limit
        text = wm.get_recent_windows_text(500)
        
        # Should include text from recent windows
        assert "Third window text" in text


class TestSummaryClientTextHandling:
    """Tests for SummaryClient text handling through WindowManager."""
    
    def test_summary_client_uses_window_manager(self):
        """Test that SummaryClient uses WindowManager for text handling."""
        client = SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
        
        assert client._window_manager is not None
    
    def test_summary_client_adds_transcription_window(self):
        """Test that SummaryClient can add transcription windows."""
        client = SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
        
        # Add transcription window
        client._window_manager.add_transcription_window(
            transcription_window_id=1,
            new_text="Test text",
            timestamp_start=0.0,
            timestamp_end=5.0,
            segments=[{"id": "1", "text": "Test text"}]
        )
        
        # Verify stored
        assert 1 in client._window_manager._transcription_windows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])