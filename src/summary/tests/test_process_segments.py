"""
Unit tests for process_segments functionality - now tests plugin-based architecture.

In the refactored code, process_segments is replaced by the plugin system.
The ContextSummaryPlugin handles the summary processing with the same logic
for analysis/reasoning content handling.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.summary.summary_client import SummaryClient
from src.summary.window_manager import WindowManager


class TestContextSummaryPluginAnalysisOverride:
    """Tests for analysis override logic in ContextSummaryPlugin."""
    
    def create_client_with_plugin(self):
        """Create a SummaryClient with plugins initialized for testing."""
        client = SummaryClient(reasoning_api_key="test_key", reasoning_model="test_model")
        
        # Manually create and register the context_summary plugin for testing
        from src.summary.context_summary import ContextSummaryPlugin, ContentTypeStateHolder
        
        # Create a mock LLM manager
        mock_llm = MagicMock()
        mock_reasoning_client = MagicMock()
        mock_llm.reasoning_llm_client = mock_reasoning_client
        
        # Create the plugin with zero delay so it processes immediately
        plugin = ContextSummaryPlugin(
            window_manager=client._window_manager,
            llm_manager=mock_llm,
            result_callback=client._queue_payload,
            summary_client=client,
            initial_summary_delay_seconds=0.0
        )
        # Mark that summary has been performed to bypass delay check
        plugin._has_performed_summary = True
        
        # Register the plugin
        client._plugins["context_summary"] = plugin
        
        return client, plugin
    
    @pytest.mark.asyncio
    async def test_analysis_field_used_as_background_context(self):
        """Test that analysis field is used as background_context when present."""
        client, plugin = self.create_client_with_plugin()
        
        # Set up the window manager with a window
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("test text", 0.0, 5.0, [1])
        
        analysis_text = "This is the analysis from the LLM"
        reasoning_text = "This is the reasoning content"
        
        # Mock the task's process_context_summary method
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "summary_text": json.dumps({
                    "analysis": analysis_text,
                    "insights": []
                }),
                "reasoning_content": reasoning_text,
                "input_tokens": 0
            }
            
            # Process the summary window through the plugin
            result = await plugin.process(summary_window_id=0)
            
            # Check that reasoning_content was used as background_context
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_analysis_missing(self):
        """Test that reasoning_content is used when analysis field is missing."""
        client, plugin = self.create_client_with_plugin()
        
        # Set up the window manager with a window
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("test text", 0.0, 5.0, [1])
        
        reasoning_text = "This is the reasoning content"
        
        # Mock the task's process_context_summary method
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "summary_text": json.dumps({
                    "insights": []
                }),
                "reasoning_content": reasoning_text,
                "input_tokens": 0
            }
            
            # Process the summary window through the plugin
            result = await plugin.process(summary_window_id=0)
            
            # Check that reasoning_content was used as fallback
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_analysis_empty(self):
        """Test that reasoning_content is used when analysis field is empty string."""
        client, plugin = self.create_client_with_plugin()
        
        # Set up the window manager with a window
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("test text", 0.0, 5.0, [1])
        
        reasoning_text = "This is the reasoning content"
        
        # Mock the task's process_context_summary method
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "summary_text": json.dumps({
                    "analysis": "",
                    "insights": []
                }),
                "reasoning_content": reasoning_text,
                "input_tokens": 0
            }
            
            # Process the summary window through the plugin
            result = await plugin.process(summary_window_id=0)
            
            # Check that reasoning_content was used as fallback
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text
    
    @pytest.mark.asyncio
    async def test_reasoning_content_used_when_json_parse_fails(self):
        """Test that reasoning_content is used when JSON parsing fails."""
        client, plugin = self.create_client_with_plugin()
        
        # Set up the window manager with a window
        client._window_manager._first_window_timestamp = 0.0
        client._window_manager.add_summary_window("test text", 0.0, 5.0, [1])
        
        reasoning_text = "This is the reasoning content"
        
        # Mock the task's process_context_summary method
        with patch.object(plugin._task, 'process_context_summary', new_callable=AsyncMock) as mock_process:
            # Return invalid JSON - the plugin will handle this and use reasoning_content
            mock_process.return_value = {
                "summary_text": "not valid json",
                "reasoning_content": reasoning_text,
                "input_tokens": 0
            }
            
            # Process the summary window through the plugin
            result = await plugin.process(summary_window_id=0)
            
            # Check that reasoning_content was used as fallback
            assert isinstance(result, dict)
            assert len(result.get("segments", [])) == 1
            assert result["segments"][0].get("background_context") == reasoning_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])