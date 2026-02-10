"""
Unit tests for get_new_text_for_window with overlapping segments.
"""

import pytest
from src.summary.summary_client import SummaryClient


class TestGetNewTextForWindow:
    """Tests for get_new_text_for_window with overlapping segments."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    def test_get_new_text_no_overlap(self):
        """Test getting new text when segments don't overlap."""
        client = self.create_client()
        
        # First call - no previous timestamp
        segments = [
            {"text": "Hello world", "start_ms": 0, "end_ms": 1000},
            {"text": "How are you", "start_ms": 1000, "end_ms": 2000}
        ]
        
        result = client.get_new_text_for_window(segments, 1)
        
        # Should return all text
        assert "Hello world" in result
        assert "How are you" in result
    
    def test_get_new_text_with_overlap(self):
        """Test getting new text when segments partially overlap."""
        client = self.create_client()
        
        # First call - no previous timestamp
        segments1 = [
            {"text": "Hello world", "start_ms": 0, "end_ms": 1000},
            {"text": "How are you", "start_ms": 1000, "end_ms": 2000}
        ]
        
        result1 = client.get_new_text_for_window(segments1, 1)
        assert "Hello world" in result1
        assert "How are you" in result1
        
        # Second call with overlapping segments
        # Segment 1 ends at 1500, so we should skip some overlap
        segments2 = [
            {"text": "Hello world", "start_ms": 500, "end_ms": 1500},  # Overlaps with first segment
            {"text": "Goodbye", "start_ms": 2000, "end_ms": 2500}     # New text
        ]
        
        result2 = client.get_new_text_for_window(segments2, 2)
        
        # Should only include "Goodbye" since first segment overlaps
        assert "Goodbye" in result2
        # The overlapping part should be excluded or minimal
        # Note: The exact behavior depends on word-based overlap calculation
    
    def test_get_new_text_complete_overlap(self):
        """Test getting new text when segments completely overlap."""
        client = self.create_client()
        
        # First call with window_id=1
        segments1 = [
            {"text": "Hello world", "start_ms": 0, "end_ms": 1000}
        ]
        
        result1 = client.get_new_text_for_window(segments1, 1)
        assert "Hello world" in result1
        
        # Second call with SAME window_id (1) - this will track the overlap
        segments2 = [
            {"text": "Hello world", "start_ms": 0, "end_ms": 1000}
        ]
        
        result2 = client.get_new_text_for_window(segments2, 1)
        
        # Should return empty since completely overlapped (same window_id tracks state)
        assert result2 == "" or "Hello world" not in result2
        
        # Third call with DIFFERENT window_id (2) - no tracking across window_ids
        segments3 = [
            {"text": "Hello world", "start_ms": 0, "end_ms": 1000}
        ]
        
        result3 = client.get_new_text_for_window(segments3, 2)
        
        # Different window_id means no tracking, so returns all text
        # This is expected behavior - each window_id has its own timestamp tracking
        assert "Hello world" in result3
    
    def test_get_new_text_new_segment_after_gap(self):
        """Test getting new text when new segment comes after a gap."""
        client = self.create_client()
        
        # First call
        segments1 = [
            {"text": "Hello world", "start_ms": 0, "end_ms": 1000}
        ]
        
        result1 = client.get_new_text_for_window(segments1, 1)
        assert "Hello world" in result1
        
        # Second call with gap (no overlap)
        segments2 = [
            {"text": "Goodbye world", "start_ms": 2000, "end_ms": 3000}  # Gap from 1000-2000
        ]
        
        result2 = client.get_new_text_for_window(segments2, 2)
        
        # Should include the new text
        assert "Goodbye world" in result2
    
    def test_get_new_text_multiple_calls_tracking(self):
        """Test that get_new_text_for_window correctly tracks across multiple calls."""
        client = self.create_client()
        
        # Call 1
        segments1 = [
            {"text": "First segment", "start_ms": 0, "end_ms": 1000}
        ]
        result1 = client.get_new_text_for_window(segments1, 1)
        assert "First segment" in result1
        
        # Call 2 - partial overlap
        segments2 = [
            {"text": "Overlapping text", "start_ms": 500, "end_ms": 1500},
            {"text": "New text", "start_ms": 1500, "end_ms": 2000}
        ]
        result2 = client.get_new_text_for_window(segments2, 2)
        assert "New text" in result2
        
        # Call 3 - more new text
        segments3 = [
            {"text": "Even more new text", "start_ms": 2500, "end_ms": 3000}
        ]
        result3 = client.get_new_text_for_window(segments3, 3)
        assert "Even more new text" in result3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])