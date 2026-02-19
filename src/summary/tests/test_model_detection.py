"""
Tests for model detection functionality in SummaryClient.

Tests the fetch_loaded_model() method in LLMManager and initialize() method's
auto-detection behavior when model is empty.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from openai.types import Model as OpenAIModel

from src.summary.summary_client import SummaryClient
from src.summary.llm_manager import LLMManager


class TestFetchLoadedModel:
    """Tests for the fetch_loaded_model() method in LLMManager."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.models = MagicMock()
        client.models.list = AsyncMock()
        return client
    
    @pytest.fixture
    def llm_manager(self, mock_client):
        """Create an LLMManager with mocked client."""
        with patch('src.summary.llm_manager.AsyncOpenAI', return_value=mock_client):
            llm = LLMManager(
                fast_base_url="http://test:5050/v1",
                fast_api_key="test-key",
                reasoning_base_url="http://test:5000/v1",
                reasoning_api_key="test-key",
            )
            return llm, mock_client
    
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_success(self, llm_manager):
        """Test successful model detection from /models endpoint."""
        llm, mock_client = llm_manager
        
        # Mock the models.list() response - use data attribute like SyncPage
        mock_model = OpenAIModel(id="detected_model_name", created=0, object="model", owned_by="test")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response
        
        # Patch AsyncOpenAI in llm_manager module to return our mock client
        with patch('src.summary.llm_manager.AsyncOpenAI', return_value=mock_client):
            # Call fetch_loaded_model on LLMManager
            result = await llm.fetch_loaded_model()
        
        # Verify result
        assert result == "detected_model_name"
        
        # Verify the API was called correctly
        mock_client.models.list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_empty_response(self, llm_manager):
        """Test model detection with empty response."""
        llm, mock_client = llm_manager
        
        # Mock empty response
        mock_response = MagicMock()
        mock_response.data = []
        mock_client.models.list.return_value = mock_response
        
        # Patch AsyncOpenAI in llm_manager module to return our mock client
        with patch('src.summary.llm_manager.AsyncOpenAI', return_value=mock_client):
            # Call fetch_loaded_model
            result = await llm.fetch_loaded_model()
        
        # Should return None when no models
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_api_error(self, llm_manager):
        """Test model detection with API error."""
        llm, mock_client = llm_manager
        
        # Mock API error
        mock_client.models.list.side_effect = Exception("API Error")
        
        # Patch AsyncOpenAI in llm_manager module to return our mock client
        with patch('src.summary.llm_manager.AsyncOpenAI', return_value=mock_client):
            # Call fetch_loaded_model - should handle gracefully
            result = await llm.fetch_loaded_model()
        
        # Should return None on error
        assert result is None


class TestSummaryClientInitialize:
    """Tests for SummaryClient.initialize() method."""
    
    @pytest.fixture
    def mock_summary_client(self):
        """Create a SummaryClient with mocked dependencies."""
        with patch('src.summary.summary_client.AsyncOpenAI'):
            client = SummaryClient(
                reasoning_base_url="http://test:8000/v1",
                reasoning_api_key="test_key",
                reasoning_model="test_model"
            )
            return client
    
    @pytest.mark.asyncio
    async def test_initialize_delegates_to_llm_manager(self, mock_summary_client):
        """Test that initialize() delegates to LLMManager."""
        # Mock the LLMManager's initialize method
        with patch.object(mock_summary_client.llm, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = "detected_model"
            
            result = await mock_summary_client.initialize()
            
            # Verify LLMManager.initialize was called
            mock_init.assert_called_once()
            
            # Verify result is passed through
            assert result == "detected_model"
    
    @pytest.mark.asyncio
    async def test_initialize_returns_none_when_no_auto_detection(self, mock_summary_client):
        """Test that initialize() returns None when no auto-detection needed."""
        # Mock the LLMManager's initialize method
        with patch.object(mock_summary_client.llm, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = None
            
            result = await mock_summary_client.initialize()
            
            # Verify result is None
            assert result is None


class TestLLMManagerModelProperties:
    """Tests for LLMManager model properties."""
    
    def test_llm_manager_has_reasoning_model(self):
        """Test that LLMManager stores reasoning model."""
        with patch('src.summary.llm_manager.AsyncOpenAI'):
            llm = LLMManager(
                fast_base_url="http://test:5050/v1",
                fast_api_key="test-key",
                reasoning_base_url="http://test:5000/v1",
                reasoning_api_key="test-key",
                reasoning_model="test-reasoning-model"
            )
            
            assert llm.reasoning_model == "test-reasoning-model"
    
    def test_llm_manager_has_fast_model(self):
        """Test that LLMManager stores fast model."""
        with patch('src.summary.llm_manager.AsyncOpenAI'):
            llm = LLMManager(
                fast_base_url="http://test:5050/v1",
                fast_api_key="test-key",
                reasoning_base_url="http://test:5000/v1",
                reasoning_api_key="test-key",
                rapid_model="test-rapid-model"
            )
            
            assert llm.rapid_model == "test-rapid-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])