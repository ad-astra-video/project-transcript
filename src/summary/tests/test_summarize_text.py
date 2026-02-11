"""
Unit tests for summarize_text functionality.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient


class TestSummarizeTextRawOutput:
    """Tests for summarize_text returning raw strings."""
    
    def create_client(self):
        """Create a SummaryClient instance for testing."""
        return SummaryClient(api_key="test_key", model="test_model")
    
    @pytest.mark.asyncio
    async def test_summarize_text_returns_raw_strings(self):
        """Test that summarize_text returns raw summary_text and reasoning_content."""
        client = self.create_client()
        
        raw_summary = '{"analysis": "Test analysis", "insights": []}'
        raw_reasoning = "Test reasoning content"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = raw_summary
        mock_response.choices[0].message.reasoning = raw_reasoning
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result_summary, result_reasoning, _ = await client.summarize_text(
                text="Test text",
                context="Test context"
            )
            
            # Should return raw strings (JSON will be parsed in process_segments)
            assert result_summary == raw_summary
            assert result_reasoning == raw_reasoning
    
    @pytest.mark.asyncio
    async def test_summarize_text_strips_code_blocks(self):
        """Test that summarize_text strips ```json code blocks."""
        client = self.create_client()
        
        raw_summary = '```json\n{"analysis": "Test", "insights": []}\n```'
        raw_reasoning = "Test reasoning"
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = raw_summary
        mock_response.choices[0].message.reasoning = raw_reasoning
        
        with patch.object(client.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result_summary, result_reasoning, _ = await client.summarize_text(
                text="Test text",
                context="Test context"
            )
            
            # Should strip code blocks
            assert result_summary == '{"analysis": "Test", "insights": []}'
            assert result_reasoning == raw_reasoning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])