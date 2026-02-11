"""
Unit tests for initial delay logic.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient


class TestInitialDelayLogic:
    """Tests for self-contained initial delay logic."""
    
    def create_client(self, delay_seconds: float = 30.0):
        """Create a SummaryClient instance with specified delay."""
        return SummaryClient(
            api_key="test_key", 
            model="test_model",
            initial_summary_delay_seconds=delay_seconds
        )
    
    @pytest.mark.asyncio
    async def test_delay_applied_when_elapsed_less_than_delay(self):
        """Test that delay is applied when elapsed time is less than delay setting."""
        client = self.create_client(delay_seconds=30.0)
        
        # Add a window with timestamp 0
        client._window_manager.add_window("test text", 0.0, 5.0)
        
        # Try to process segments at timestamp 10 (only 10 seconds elapsed, need 30)
        segments = [{"text": "Test segment", "start": 0, "end": 5}]
        result = await client.process_segments(
            summary_type="test",
            segments=segments,
            transcription_window_id=1,
            window_start=10.0,  # Only 10 seconds elapsed
            window_end=15.0
        )
        
        # Should return empty segments list (delay applied, now returns dict)
        assert isinstance(result, dict)
        assert len(result.get("segments", [])) == 0
    
    @pytest.mark.asyncio
    async def test_no_delay_when_elapsed_greater_than_delay(self):
        """Test that delay is not applied when elapsed time exceeds delay setting."""
        # Use transcription_windows_per_summary_window=1 to ensure immediate processing
        client = self.create_client(delay_seconds=30.0)
        client._window_manager.transcription_windows_per_summary_window = 1
        
        # Add a window with timestamp 0
        client._window_manager.add_window("test text", 0.0, 5.0)
        
        # Mock the summarize_text method
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": "Analysis",
            "insights": []
        })
        mock_response.choices[0].message.reasoning = "Reasoning"
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Try to process segments at timestamp 40 (40 seconds elapsed, need 30)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=1,
                window_start=40.0,  # 40 seconds elapsed, exceeds 30s delay
                window_end=45.0
            )
            
            # Should return a result (no delay, now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
    
    def test_update_params_changes_delay(self):
        """Test that update_params can change the delay setting."""
        client = self.create_client(delay_seconds=30.0)
        
        assert client.initial_summary_delay_seconds == 30.0
        
        client.update_params(initial_summary_delay_seconds=60.0)
        
        assert client.initial_summary_delay_seconds == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])