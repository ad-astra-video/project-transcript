"""
Tests for rapid summary functionality.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.summary.summary_client import SummaryClient, WindowManager
from src.summary.rapid_summary.task import RapidSummaryItemSchema, RapidSummaryResponseSchema


class TestRapidSummarySchemas:
    """Test Pydantic schemas for rapid summary."""

    def test_rapid_summary_item_schema_valid(self):
        """Test valid rapid summary item schema."""
        item = RapidSummaryItemSchema(item="This is a summary item")
        assert item.item == "This is a summary item"

    def test_rapid_summary_response_schema_valid(self):
        """Test valid rapid summary response schema."""
        response = RapidSummaryResponseSchema(
            summary=[
                RapidSummaryItemSchema(item="First summary item"),
                RapidSummaryItemSchema(item="Second summary item"),
            ]
        )
        assert len(response.summary) == 2
        assert response.summary[0].item == "First summary item"
        assert response.summary[1].item == "Second summary item"

    def test_rapid_summary_response_schema_empty(self):
        """Test rapid summary response schema with empty list."""
        response = RapidSummaryResponseSchema(summary=[])
        assert len(response.summary) == 0


class TestWindowManagerRapidSummary:
    """Test WindowManager methods for rapid summary context tracking."""

    @pytest.fixture
    def window_manager(self):
        """Create a WindowManager instance for testing."""
        return WindowManager(
            context_limit=50000,
            raw_text_context_limit=1000,
        )

    def test_get_text_and_window_ids_since_timestamp_empty(self, window_manager):
        """Test getting text since timestamp with no windows."""
        text, window_ids = window_manager.get_text_and_window_ids_since_timestamp(0.0)
        assert text == ""
        assert window_ids == []

    def test_get_text_and_window_ids_since_timestamp_with_windows(self, window_manager):
        """Test getting text since timestamp with multiple windows."""
        # Add windows with different timestamps
        window_manager.add_summary_window(
            text="First window text",
            timestamp_start=0.0,
            timestamp_end=10.0,
            transcription_window_ids=[]
        )
        window_manager.add_summary_window(
            text="Second window text",
            timestamp_start=10.0,
            timestamp_end=20.0,
            transcription_window_ids=[]
        )
        window_manager.add_summary_window(
            text="Third window text",
            timestamp_start=20.0,
            timestamp_end=30.0,
            transcription_window_ids=[]
        )

        # Get text since timestamp 10.0 (should include windows 2 and 3, since timestamp_start >= timestamp)
        text, window_ids = window_manager.get_text_and_window_ids_since_timestamp(10.0)
        
        assert "Second window text" in text
        assert "Third window text" in text
        assert "First window text" not in text
        assert len(window_ids) == 2
        
        # Also test with timestamp 0.0 to get all windows
        text_all, window_ids_all = window_manager.get_text_and_window_ids_since_timestamp(0.0)
        assert "First window text" in text_all
        assert "Second window text" in text_all
        assert "Third window text" in text_all
        assert len(window_ids_all) == 3

    def test_get_text_and_window_ids_since_timestamp_exact_match(self, window_manager):
        """Test getting text since timestamp with exact match on window boundary."""
        window_manager.add_summary_window(
            text="First window text",
            timestamp_start=0.0,
            timestamp_end=10.0,
            transcription_window_ids=[]
        )
        window_manager.add_summary_window(
            text="Second window text",
            timestamp_start=10.0,
            timestamp_end=20.0,
            transcription_window_ids=[]
        )

        # Get text since timestamp 10.0 (exact boundary)
        text, window_ids = window_manager.get_text_and_window_ids_since_timestamp(10.0)
        
        assert "Second window text" in text
        assert "First window text" not in text
        assert len(window_ids) == 1


class TestSummaryClientRapidSummary:
    """Test SummaryClient rapid summary methods."""

    @pytest.fixture
    def mock_summary_client(self):
        """Create a SummaryClient with mocked dependencies."""
        with patch('src.summary.summary_client.AsyncOpenAI'):
            client = SummaryClient(
                reasoning_base_url="http://test:5000/v1",
                reasoning_api_key="test-key",
                reasoning_model="test-model",
                rapid_base_url="http://test-rapid:5050/v1",
                rapid_api_key="test-rapid-key",
                rapid_model="test-rapid-model",
            )
            return client

    def test_rapid_client_initialized(self, mock_summary_client):
        """Test that rapid client is initialized."""
        # In refactored code, rapid config is passed to LLMManager
        # Check that llm has the rapid client configured
        assert hasattr(mock_summary_client.llm, '_rapid_llm_client')

    def test_rapid_client_default_values(self):
        """Test rapid client with default values."""
        with patch('src.summary.summary_client.AsyncOpenAI'):
            client = SummaryClient(
                reasoning_base_url="http://test:5000/v1",
                reasoning_api_key="test-key",
                reasoning_model="test-model",
            )
            # In refactored code, rapid config is passed to LLMManager
            # Check that llm has the rapid client configured
            assert hasattr(client.llm, '_rapid_llm_client')


class TestRapidSummaryPlugin:
    """Test RapidSummaryPlugin functionality."""
    
    @pytest.fixture
    def mock_summary_client(self):
        """Create a SummaryClient with mocked dependencies."""
        with patch('src.summary.summary_client.AsyncOpenAI'):
            client = SummaryClient(
                reasoning_base_url="http://test:5000/v1",
                reasoning_api_key="test-key",
                reasoning_model="test-model",
            )
            return client
    
    def test_rapid_summary_plugin_exists(self, mock_summary_client):
        """Test that rapid_summary plugin can be loaded."""
        # The plugin system should discover rapid_summary if it exists
        # This test verifies the plugin architecture works
        from src.summary.rapid_summary import RapidSummaryPlugin
        
        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.fast_llm_client = MagicMock()
        
        # Create plugin - note: RapidSummaryPlugin uses 'llm' not 'llm_manager'
        plugin = RapidSummaryPlugin(
            window_manager=mock_summary_client._window_manager,
            llm=mock_llm,
            result_callback=mock_summary_client._queue_payload,
            summary_client=mock_summary_client
        )
        
        assert plugin is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])