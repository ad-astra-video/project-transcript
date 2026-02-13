"""
Tests for model detection functionality in SummaryClient.

Tests the fetch_loaded_model() method and initialize() method's
auto-detection behavior when model is empty.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types import Model as OpenAIModel

from src.summary.summary_client import SummaryClient


class TestFetchLoadedModel:
    """Tests for the fetch_loaded_model() method."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.models = MagicMock()
        client.models.list = AsyncMock()
        return client
    
    @pytest.fixture
    def summary_client(self, mock_client):
        """Create a SummaryClient with mocked client."""
        with patch('src.summary.summary_client.AsyncOpenAI', return_value=mock_client):
            client = SummaryClient(
                base_url="http://test:8000/v1",
                api_key="test_key",
                model="test_model"
            )
            client.client = mock_client
            return client
    
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_success(self, summary_client, mock_client):
        """Test successful model detection from /models endpoint."""
        # Mock the models.list() response - use data attribute like SyncPage
        mock_model = OpenAIModel(id="detected_model_name", created=0, object="model", owned_by="test")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response
        
        # Call fetch_loaded_model
        result = await summary_client.fetch_loaded_model()
        
        # Verify result
        assert result == "detected_model_name"
        
        # Verify the API was called correctly
        mock_client.models.list.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_multiple_models(self, summary_client, mock_client):
        """Test that first model is returned when multiple models available."""
        # Mock multiple models in response
        mock_models = [
            OpenAIModel(id="primary_model", created=0, object="model", owned_by="test"),
            OpenAIModel(id="secondary_model", created=1, object="model", owned_by="test"),
        ]
        mock_response = MagicMock()
        mock_response.data = mock_models
        mock_client.models.list.return_value = mock_response
        
        # Call fetch_loaded_model
        result = await summary_client.fetch_loaded_model()
        
        # Should return first model
        assert result == "primary_model"
        
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_empty_response(self, summary_client, mock_client):
        """Test RuntimeError is raised when no models available."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.data = []
        mock_client.models.list.return_value = mock_response
        
        # Call fetch_loaded_model and expect RuntimeError
        with pytest.raises(RuntimeError, match="No models available from OpenAI API"):
            await summary_client.fetch_loaded_model()
            
    @pytest.mark.asyncio
    async def test_fetch_loaded_model_api_error(self, summary_client, mock_client):
        """Test RuntimeError is raised when API call fails."""
        # Mock API error
        mock_client.models.list.side_effect = Exception("Connection failed")
        
        # Call fetch_loaded_model and expect RuntimeError
        with pytest.raises(RuntimeError, match="Model detection failed"):
            await summary_client.fetch_loaded_model()


class TestInitializeModelDetection:
    """Tests for the initialize() method's model detection behavior."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.models = MagicMock()
        client.models.list = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_initialize_with_empty_model_detects_model(self, mock_client):
        """Test that initialize() detects model when model is empty."""
        # Mock the models.list() response
        mock_model = OpenAIModel(id="auto_detected_model", created=0, object="model", owned_by="test")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response
        
        with patch('src.summary.summary_client.AsyncOpenAI', return_value=mock_client):
            client = SummaryClient(
                base_url="http://test:8000/v1",
                api_key="test_key",
                model=""  # Empty model should trigger detection
            )
            client.client = mock_client
            
            # Call initialize
            detected = await client.initialize()
            
            # Verify model was detected and stored
            assert detected == "auto_detected_model"
            assert client.model == "auto_detected_model"
            
    @pytest.mark.asyncio
    async def test_initialize_with_none_model_detects_model(self, mock_client):
        """Test that initialize() detects model when model is None."""
        # Mock the models.list() response
        mock_model = OpenAIModel(id="auto_detected_model", created=0, object="model", owned_by="test")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response
        
        with patch('src.summary.summary_client.AsyncOpenAI', return_value=mock_client):
            client = SummaryClient(
                base_url="http://test:8000/v1",
                api_key="test_key",
                model=None  # None model should trigger detection
            )
            client.client = mock_client
            
            # Call initialize
            detected = await client.initialize()
            
            # Verify model was detected and stored
            assert detected == "auto_detected_model"
            assert client.model == "auto_detected_model"
            
    @pytest.mark.asyncio
    async def test_initialize_with_existing_model_skips_detection(self, mock_client):
        """Test that initialize() skips detection when model is already set."""
        with patch('src.summary.summary_client.AsyncOpenAI', return_value=mock_client):
            client = SummaryClient(
                base_url="http://test:8000/v1",
                api_key="test_key",
                model="existing_model"  # Model already set
            )
            client.client = mock_client
            
            # Call initialize
            detected = await client.initialize()
            
            # Verify no detection was attempted
            mock_client.models.list.assert_not_called()
            
            # Verify None is returned (no auto-detection)
            assert detected is None
            
            # Verify existing model is preserved
            assert client.model == "existing_model"
            
    @pytest.mark.asyncio
    async def test_initialize_detection_failure_raises_error(self, mock_client):
        """Test that initialize() raises RuntimeError when detection fails."""
        # Mock API error
        mock_client.models.list.side_effect = Exception("API unavailable")
        
        with patch('src.summary.summary_client.AsyncOpenAI', return_value=mock_client):
            client = SummaryClient(
                base_url="http://test:8000/v1",
                api_key="test_key",
                model=""  # Empty model should trigger detection
            )
            client.client = mock_client
            
            # Call initialize and expect RuntimeError
            with pytest.raises(RuntimeError, match="Model detection failed"):
                await client.initialize()


class TestModelDetectionIntegration:
    """Integration tests for model detection in load_model flow."""
    
    @pytest.mark.asyncio
    async def test_full_detection_flow(self):
        """Test the complete model detection flow."""
        # This would test the integration between load_model and SummaryClient
        # For now, just verify the components work together
        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_client.models.list = AsyncMock()
        mock_model = OpenAIModel(id="integrated_model", created=0, object="model", owned_by="test")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response
        
        with patch('src.summary.summary_client.AsyncOpenAI', return_value=mock_client):
            client = SummaryClient(
                base_url="http://test:8000/v1",
                api_key="test_key",
                model=""
            )
            client.client = mock_client
            
            # Full initialization flow
            detected = await client.initialize()
            
            assert detected == "integrated_model"
            assert client.model == "integrated_model"
            
    @pytest.mark.asyncio
    async def test_model_detection_with_custom_base_url(self):
        """Test model detection uses the correct base URL."""
        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_client.models.list = AsyncMock()
        mock_model = OpenAIModel(id="custom_url_model", created=0, object="model", owned_by="test")
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_client.models.list.return_value = mock_response
        
        with patch('src.summary.summary_client.AsyncOpenAI') as mock_openai:
            mock_openai.return_value = mock_client
            
            client = SummaryClient(
                base_url="http://custom-server:9000/v1",
                api_key="test_key",
                model=""
            )
            client.client = mock_client
            
            await client.initialize()
            
            # Verify OpenAI client was created with correct base_url
            mock_openai.assert_called_once_with(
                api_key="test_key",
                base_url="http://custom-server:9000/v1"
            )