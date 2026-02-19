"""
Tests for update_params propagation to LLMManager and plugins.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.summary.summary_client import SummaryClient
from src.summary.llm_manager import LLMManager
from src.summary.window_manager import WindowManager


class TestLLMManagerUpdateParams:
    """Tests for LLMManager.update_params method."""
    
    def test_update_fast_base_url(self):
        """update_params should update fast base URL."""
        llm = LLMManager(
            fast_base_url="http://old-fast:5050/v1",
            fast_api_key="old-key",
            reasoning_base_url="http://old-reasoning:5000/v1",
            reasoning_api_key="old-key",
            rapid_model="fast-model",
            reasoning_model="reasoning-model",
        )
        
        llm.update_params(
            fast_base_url="http://new-fast:5050/v1"
        )
        
        assert llm._fast_base_url == "http://new-fast:5050/v1"
    
    def test_update_fast_api_key(self):
        """update_params should update fast API key."""
        llm = LLMManager(
            fast_base_url="http://fast:5050/v1",
            fast_api_key="old-key",
            reasoning_base_url="http://reasoning:5000/v1",
            reasoning_api_key="old-key",
            rapid_model="fast-model",
            reasoning_model="reasoning-model",
        )
        
        llm.update_params(
            fast_api_key="new-key"
        )
        
        assert llm._fast_api_key == "new-key"
    
    def test_update_fast_model(self):
        """update_params should update fast model."""
        llm = LLMManager(
            fast_base_url="http://fast:5050/v1",
            fast_api_key="key",
            reasoning_base_url="http://reasoning:5000/v1",
            reasoning_api_key="key",
            rapid_model="old-model",
            reasoning_model="reasoning-model",
        )
        
        llm.update_params(
            fast_model="new-model"
        )
        
        assert llm._rapid_model == "new-model"
    
    def test_update_reasoning_base_url(self):
        """update_params should update reasoning base URL."""
        llm = LLMManager(
            fast_base_url="http://rapid:5050/v1",
            fast_api_key="key",
            reasoning_base_url="http://old-reasoning:5000/v1",
            reasoning_api_key="old-key",
            rapid_model="rapid-model",
            reasoning_model="reasoning-model",
        )
        
        llm.update_params(
            reasoning_base_url="http://new-reasoning:5000/v1"
        )
        
        assert llm._reasoning_base_url == "http://new-reasoning:5000/v1"
    
    def test_update_reasoning_api_key(self):
        """update_params should update reasoning API key."""
        llm = LLMManager(
            fast_base_url="http://rapid:5050/v1",
            fast_api_key="key",
            reasoning_base_url="http://reasoning:5000/v1",
            reasoning_api_key="old-key",
            rapid_model="rapid-model",
            reasoning_model="reasoning-model",
        )
        
        llm.update_params(
            reasoning_api_key="new-key"
        )
        
        assert llm._reasoning_api_key == "new-key"
    
    def test_update_reasoning_model(self):
        """update_params should update reasoning model."""
        llm = LLMManager(
            fast_base_url="http://rapid:5050/v1",
            fast_api_key="key",
            reasoning_base_url="http://reasoning:5000/v1",
            reasoning_api_key="key",
            rapid_model="rapid-model",
            reasoning_model="old-model",
        )
        
        llm.update_params(
            reasoning_model="new-model"
        )
        
        assert llm._reasoning_model == "new-model"


class TestPluginUpdateParams:
    """Tests for plugin on_update_params handlers."""
    
    def test_context_summary_plugin_update_delay(self):
        """ContextSummaryPlugin should update initial_summary_delay_seconds."""
        from src.summary.context_summary import ContextSummaryPlugin
        
        plugin = ContextSummaryPlugin(
            window_manager=WindowManager(),
            llm_manager=MagicMock(),
            result_callback=MagicMock(),
            summary_client=None,
            initial_summary_delay_seconds=15.0
        )
        
        plugin.on_update_params(initial_summary_delay_seconds=30.0)
        
        assert plugin._initial_summary_delay_seconds == 30.0
    
    def test_context_summary_plugin_update_max_tokens(self):
        """ContextSummaryPlugin should update reasoning_max_tokens."""
        from src.summary.context_summary import ContextSummaryPlugin
        from src.summary.context_summary.task import ContextSummaryTask
        
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContextSummaryPlugin(
            window_manager=WindowManager(),
            llm_manager=mock_llm,
            result_callback=MagicMock(),
            summary_client=None,
            initial_summary_delay_seconds=15.0
        )
        
        plugin.on_update_params(reasoning_max_tokens=4096)
        
        assert plugin._task.max_tokens == 4096
    
    def test_context_summary_plugin_update_temperature(self):
        """ContextSummaryPlugin should update reasoning_temperature."""
        from src.summary.context_summary import ContextSummaryPlugin
        
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContextSummaryPlugin(
            window_manager=WindowManager(),
            llm_manager=mock_llm,
            result_callback=MagicMock(),
            summary_client=None,
            initial_summary_delay_seconds=15.0
        )
        
        plugin.on_update_params(reasoning_temperature=0.7)
        
        assert plugin._task.temperature == 0.7
    
    def test_rapid_summary_plugin_update_max_tokens(self):
        """RapidSummaryPlugin should update fast_max_tokens."""
        from src.summary.rapid_summary import RapidSummaryPlugin
        from src.summary.rapid_summary.task import RapidSummaryTask
        
        mock_llm = MagicMock()
        mock_llm.rapid_llm_client = MagicMock()
        
        plugin = RapidSummaryPlugin(
            window_manager=WindowManager(),
            llm=mock_llm,
            result_callback=MagicMock(),
            summary_client=None
        )
        
        plugin.on_update_params(fast_max_tokens=1000)
        
        assert plugin._task.max_tokens == 1000
    
    def test_rapid_summary_plugin_update_temperature(self):
        """RapidSummaryPlugin should update fast_temperature."""
        from src.summary.rapid_summary import RapidSummaryPlugin
        
        mock_llm = MagicMock()
        mock_llm.rapid_llm_client = MagicMock()
        
        plugin = RapidSummaryPlugin(
            window_manager=WindowManager(),
            llm=mock_llm,
            result_callback=MagicMock(),
            summary_client=None
        )
        
        plugin.on_update_params(fast_temperature=0.5)
        
        assert plugin._task.temperature == 0.5
    
    def test_content_type_detection_plugin_update_max_tokens(self):
        """ContentTypeDetectionPlugin should update reasoning_max_tokens."""
        from src.summary.content_type_detection import ContentTypeDetectionPlugin
        
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContentTypeDetectionPlugin(
            window_manager=WindowManager(),
            llm_manager=mock_llm,
            result_callback=MagicMock(),
            summary_client=None
        )
        
        plugin.on_update_params(reasoning_max_tokens=2048)
        
        assert plugin._max_tokens == 2048
        assert plugin._task.max_tokens == 2048
    
    def test_content_type_detection_plugin_update_temperature(self):
        """ContentTypeDetectionPlugin should update reasoning_temperature."""
        from src.summary.content_type_detection import ContentTypeDetectionPlugin
        
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContentTypeDetectionPlugin(
            window_manager=WindowManager(),
            llm_manager=mock_llm,
            result_callback=MagicMock(),
            summary_client=None
        )
        
        plugin.on_update_params(reasoning_temperature=0.9)
        
        assert plugin._temperature == 0.9
        assert plugin._task.temperature == 0.9
    
    def test_content_type_detection_plugin_update_context_limit(self):
        """ContentTypeDetectionPlugin should update content_type_context_limit."""
        from src.summary.content_type_detection import ContentTypeDetectionPlugin
        
        mock_llm = MagicMock()
        mock_llm.reasoning_llm_client = MagicMock()
        
        plugin = ContentTypeDetectionPlugin(
            window_manager=WindowManager(),
            llm_manager=mock_llm,
            result_callback=MagicMock(),
            summary_client=None
        )
        
        plugin.on_update_params(content_type_context_limit=5000)
        
        assert plugin._content_type_context_limit == 5000


class TestSummaryClientUpdateParamsPropagation:
    """Tests for SummaryClient.update_params propagation to managers and plugins."""
    
    def test_propagates_to_llm_manager(self):
        """update_params should propagate to LLMManager."""
        client = SummaryClient(
            reasoning_base_url="http://old-reasoning:5000/v1",
            reasoning_api_key="old-key",
            reasoning_model="old-model",
            rapid_base_url="http://old-fast:5050/v1",
            rapid_api_key="old-key",
            rapid_model="old-model",
        )
        
        client.update_params(
            reasoning_base_url="http://new-reasoning:5000/v1",
            fast_base_url="http://new-fast:5050/v1",
            reasoning_model="new-model",
            fast_model="new-model",
        )
        
        assert client.llm._reasoning_base_url == "http://new-reasoning:5000/v1"
        assert client.llm._fast_base_url == "http://new-fast:5050/v1"
        assert client.llm._reasoning_model == "new-model"
        assert client.llm._rapid_model == "new-model"
    
    def test_propagates_to_context_summary_plugin(self):
        """update_params should propagate to context_summary plugin."""
        client = SummaryClient(
            reasoning_base_url="http://reasoning:5000/v1",
            rapid_base_url="http://fast:5050/v1",
        )
        
        # Manually register a mock plugin since plugins are auto-discovered
        mock_plugin = MagicMock()
        client._plugins["context_summary"] = mock_plugin
        
        client.update_params(
            initial_summary_delay_seconds=45.0,
            reasoning_max_tokens=4096,
            reasoning_temperature=0.7,
        )
        
        # Verify the plugin's on_update_params was called
        mock_plugin.on_update_params.assert_called_once()
        call_kwargs = mock_plugin.on_update_params.call_args.kwargs
        assert call_kwargs.get("initial_summary_delay_seconds") == 45.0
        assert call_kwargs.get("reasoning_max_tokens") == 4096
        assert call_kwargs.get("reasoning_temperature") == 0.7
    
    def test_propagates_to_rapid_summary_plugin(self):
        """update_params should propagate to rapid_summary plugin."""
        client = SummaryClient(
            reasoning_base_url="http://reasoning:5000/v1",
            rapid_base_url="http://fast:5050/v1",
        )
        
        mock_plugin = MagicMock()
        client._plugins["rapid_summary"] = mock_plugin
        
        client.update_params(
            fast_max_tokens=1000,
            fast_temperature=0.5,
        )
        
        mock_plugin.on_update_params.assert_called_once()
        call_kwargs = mock_plugin.on_update_params.call_args.kwargs
        assert call_kwargs.get("fast_max_tokens") == 1000
        assert call_kwargs.get("fast_temperature") == 0.5
    
    def test_propagates_to_content_type_detection_plugin(self):
        """update_params should propagate to content_type_detection plugin."""
        client = SummaryClient(
            reasoning_base_url="http://reasoning:5000/v1",
            rapid_base_url="http://rapid:5050/v1",
        )
        
        mock_plugin = MagicMock()
        client._plugins["content_type_detection"] = mock_plugin
        
        client.update_params(
            reasoning_max_tokens=2048,
            reasoning_temperature=0.9,
            content_type_context_limit=5000,
        )
        
        mock_plugin.on_update_params.assert_called_once()
        call_kwargs = mock_plugin.on_update_params.call_args.kwargs
        assert call_kwargs.get("reasoning_max_tokens") == 2048
        assert call_kwargs.get("reasoning_temperature") == 0.9
        assert call_kwargs.get("content_type_context_limit") == 5000