"""
Unit tests for process_segments functionality and analysis override.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient


class TestProcessSegmentsAnalysisOverride:
    """Tests for analysis override logic in process_segments."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    @pytest.mark.asyncio
    async def test_analysis_field_used_as_background_context(self):
        """Test that analysis field is used as background_context when present."""
        # Use shorter delay and windows_to_accumulate=1 to ensure immediate processing
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0, windows_to_accumulate=1)
        
        # Mock the summarize_text method
        analysis_text = "This is the analysis from the LLM"
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": analysis_text,
            "insights": []
        })
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that analysis was used as background_context (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == analysis_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_analysis_missing(self):
        """Test that reasoning_content is used when analysis field is missing."""
        # Use shorter delay and windows_to_accumulate=1 to ensure immediate processing
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0, windows_to_accumulate=1)
        
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "insights": []
        })
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_analysis_empty(self):
        """Test that reasoning_content is used when analysis field is empty string."""
        # Use shorter delay and windows_to_accumulate=1 to ensure immediate processing
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0, windows_to_accumulate=1)
        
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "analysis": "",
            "insights": []
        })
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_json_parse_fails(self):
        """Test that reasoning_content is used when JSON parsing fails."""
        # Use shorter delay and windows_to_accumulate=1 to ensure immediate processing
        client = SummaryClient(api_key="test_key", model="test_model", initial_summary_delay_seconds=1.0, windows_to_accumulate=1)
        
        reasoning_text = "This is the reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_response.choices[0].message.reasoning = reasoning_text
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            # Add a window first
            client._window_manager.add_window("test text", 0.0, 5.0)
            
            # Process segments at timestamp 10 (exceeds 1 second delay)
            segments = [{"text": "Test segment", "start": 0, "end": 5}]
            result = await client.process_segments(
                summary_type="test",
                segments=segments,
                transcription_window_id=1,
                window_start=10.0,
                window_end=15.0
            )
            
            # Check that reasoning_content was used as fallback (now returns dict)
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])