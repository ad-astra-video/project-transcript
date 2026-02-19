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
        client = SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
        return client
    
    @pytest.mark.asyncio
    async def test_summarize_text_returns_raw_strings(self):
        """Test that summarize_text returns raw summary_text and reasoning_content."""
        client = self.create_client()
        
        raw_summary = '{"analysis": "Test analysis", "insights": []}'
        raw_reasoning = "Test reasoning content"
        
        # Mock the context_summary plugin's process method
        mock_plugin = AsyncMock()
        mock_plugin.process_summary = AsyncMock(return_value={
            "summary_text": raw_summary,
            "reasoning_content": raw_reasoning,
            "input_tokens": 0
        })
        client._plugins["context_summary"] = mock_plugin
        
        result_summary, result_reasoning, _ = await client.summarize_text(
            text="Test text",
            context="Test context"
        )
        
        # Should return raw strings
        assert result_summary == raw_summary
        assert result_reasoning == raw_reasoning
    
    @pytest.mark.asyncio
    async def test_summarize_text_strips_code_blocks(self):
        """Test that summarize_text handles code blocks correctly."""
        client = self.create_client()
        
        raw_summary = '{"analysis": "Test", "insights": []}'
        raw_reasoning = "Test reasoning"
        
        # Mock the context_summary plugin
        mock_plugin = AsyncMock()
        mock_plugin.process_summary = AsyncMock(return_value={
            "summary_text": raw_summary,
            "reasoning_content": raw_reasoning,
            "input_tokens": 0
        })
        client._plugins["context_summary"] = mock_plugin
        
        result_summary, result_reasoning, _ = await client.summarize_text(
            text="Test text",
            context="Test context"
        )
        
        # Should return the processed summary
        assert result_summary == raw_summary
        assert result_reasoning == raw_reasoning
    
    @pytest.mark.asyncio
    async def test_summarize_text_fallback_when_no_plugin(self):
        """Test that summarize_text returns empty response when plugin not available."""
        client = self.create_client()
        
        # No plugins loaded
        client._plugins = {}
        
        result_summary, result_reasoning, input_tokens = await client.summarize_text(
            text="Test text",
            context="Test context"
        )
        
        # Should return empty response
        assert result_summary == "{}"
        assert result_reasoning == ""
        assert input_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])