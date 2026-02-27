"""Tests for HealthMetrics class in llm_manager.py"""

import asyncio
import pytest
import threading
import time
from unittest.mock import AsyncMock, MagicMock


class TestHealthMetrics:
    """Test suite for HealthMetrics class."""
    
    @pytest.fixture
    def health_metrics(self):
        """Create a HealthMetrics instance for testing."""
        from src.summary.llm_manager import HealthMetrics
        return HealthMetrics(timeout_seconds=240.0)
    
    @pytest.fixture
    def health_metrics_with_callbacks(self):
        """Create a HealthMetrics instance with callbacks."""
        from src.summary.llm_manager import HealthMetrics
        result_callback = AsyncMock()
        monitoring_callback = AsyncMock()
        return HealthMetrics(
            timeout_seconds=240.0,
            result_callback=result_callback,
            monitoring_callback=monitoring_callback
        ), result_callback, monitoring_callback
    
    def test_initial_state(self, health_metrics):
        """Test HealthMetrics initial state."""
        metrics = health_metrics.get_metrics()
        
        # Access window metrics
        window = metrics["window"]
        assert window["total_requests"] == 0
        assert window["successful_requests"] == 0
        assert window["failed_requests"]["total"] == 0
        assert window["failed_requests"]["parse_failure"] == 0
        assert window["failed_requests"]["timeout_failure"] == 0
        assert window["success_rate_percentage"] == 0.0
        assert window["plugin_breakdown"] == {}
        assert health_metrics.stream_id is None
    
    def test_record_success(self, health_metrics):
        """Test recording a successful request."""
        health_metrics.record(
            plugin_name="rapid_summary",
            success=True,
            response_text="test response",
            elapsed_time=1.5
        )
        
        metrics = health_metrics.get_metrics()
        
        # Access window metrics
        window = metrics["window"]
        assert window["total_requests"] == 1
        assert window["successful_requests"] == 1
        assert window["failed_requests"]["total"] == 0
        assert window["success_rate_percentage"] == 100.0
        assert window["plugin_breakdown"]["rapid_summary"]["total"] == 1
        assert window["plugin_breakdown"]["rapid_summary"]["success"] == 1
    
    def test_record_parse_failure(self, health_metrics):
        """Test recording a parse failure."""
        health_metrics.record(
            plugin_name="context_summary",
            success=False,
            failure_type="parse_failure",
            response_text="invalid json",
            elapsed_time=0.5
        )
        
        metrics = health_metrics.get_metrics()
        
        assert window["total_requests"] == 1
        assert window["successful_requests"] == 0
        assert window["failed_requests"]["total"] == 1
        assert window["failed_requests"]["parse_failure"] == 1
        assert window["failed_requests"]["timeout_failure"] == 0
        assert window["success_rate_percentage"] == 0.0
        assert window["plugin_breakdown"]["context_summary"]["parse_failure"] == 1
    
    def test_record_timeout_failure(self, health_metrics):
        """Test recording a timeout failure."""
        health_metrics.record(
            plugin_name="rapid_summary",
            success=False,
            failure_type="timeout_failure",
            elapsed_time=240.0
        )
        
        metrics = health_metrics.get_metrics()
        
        assert window["total_requests"] == 1
        assert window["successful_requests"] == 0
        assert window["failed_requests"]["total"] == 1
        assert window["failed_requests"]["parse_failure"] == 0
        assert window["failed_requests"]["timeout_failure"] == 1
        assert window["success_rate_percentage"] == 0.0
        assert window["plugin_breakdown"]["rapid_summary"]["timeout_failure"] == 1
    
    def test_record_multiple_requests(self, health_metrics):
        """Test recording multiple requests."""
        # Successful requests
        health_metrics.record(plugin_name="rapid_summary", success=True)
        health_metrics.record(plugin_name="rapid_summary", success=True)
        
        # Failed requests
        health_metrics.record(plugin_name="context_summary", success=False, failure_type="parse_failure")
        health_metrics.record(plugin_name="rapid_summary", success=False, failure_type="timeout_failure")
        
        metrics = health_metrics.get_metrics()
        
        assert window["total_requests"] == 4
        assert window["successful_requests"] == 2
        assert window["failed_requests"]["total"] == 2
        assert window["success_rate_percentage"] == 50.0
    
    def test_model_type_breakdown(self, health_metrics):
        """Test that model_type breakdown tracks reasoning vs fast models."""
        # rapid_summary uses fast model
        health_metrics.record(plugin_name="rapid_summary", success=True)
        health_metrics.record(plugin_name="rapid_summary", success=True)
        health_metrics.record(plugin_name="rapid_summary", success=False, failure_type="parse_failure")
        
        # context_summary uses reasoning model
        health_metrics.record(plugin_name="context_summary", success=True)
        health_metrics.record(plugin_name="context_summary", success=False, failure_type="timeout_failure")
        
        metrics = health_metrics.get_metrics()
        
        # Check fast model breakdown
        assert "fast" in metrics["model_type_breakdown"]
        assert window["model_type_breakdown"]["fast"]["total"] == 3
        assert window["model_type_breakdown"]["fast"]["successful"] == 2
        assert window["model_type_breakdown"]["fast"]["failed"]["parse_failure"] == 1
        assert window["model_type_breakdown"]["fast"]["success_rate_percentage"] == 66.67
        
        # Check reasoning model breakdown
        assert "reasoning" in metrics["model_type_breakdown"]
        assert window["model_type_breakdown"]["reasoning"]["total"] == 2
        assert window["model_type_breakdown"]["reasoning"]["successful"] == 1
        assert window["model_type_breakdown"]["reasoning"]["failed"]["timeout_failure"] == 1
        assert window["model_type_breakdown"]["reasoning"]["success_rate_percentage"] == 50.0
    
    def test_model_type_explicit_override(self, health_metrics):
        """Test explicit model_type override in record()."""
        # Explicitly specify model_type
        health_metrics.record(plugin_name="custom_plugin", success=True, model_type="reasoning")
        health_metrics.record(plugin_name="custom_plugin", success=True, model_type="fast")
        
        metrics = health_metrics.get_metrics()
        
        assert window["model_type_breakdown"]["reasoning"]["total"] == 1
        assert window["model_type_breakdown"]["fast"]["total"] == 1
    
    def test_zero_requests_success_rate(self, health_metrics):
        """Test success rate calculation with zero requests."""
        metrics = health_metrics.get_metrics()
        
        assert window["success_rate_percentage"] == 0.0
    
    def test_set_stream_id(self, health_metrics):
        """Test setting stream_id."""
        assert health_metrics.stream_id is None
        
        health_metrics.set_stream_id("test-stream-123")
        
        assert health_metrics.stream_id == "test-stream-123"
        
        health_metrics.set_stream_id(None)
        
        assert health_metrics.stream_id is None
    
    @pytest.mark.asyncio
    async def test_publish_with_callbacks(self, health_metrics_with_callbacks):
        """Test publishing metrics via callbacks."""
        health_metrics, result_callback, monitoring_callback = health_metrics_with_callbacks
        
        # Record some requests
        health_metrics.record(plugin_name="rapid_summary", success=True)
        health_metrics.record(plugin_name="rapid_summary", success=True)
        health_metrics.record(plugin_name="context_summary", success=False, failure_type="parse_failure")
        
        # Publish
        result = await health_metrics.publish()
        
        # Verify result_callback was called (NO stream_id)
        result_callback.assert_called_once()
        call_args = result_callback.call_args[0][0]
        assert call_args["type"] == "summary_processing_health"
        assert call_args["total_requests"] == 3
        assert call_args["successful_requests"] == 2
        assert "stream_id" not in call_args  # stream_id should NOT be in result_callback
        
        # Verify monitoring_callback was called (WITH stream_id)
        monitoring_callback.assert_called_once()
        monitoring_call_args = monitoring_callback.call_args[0][0]
        assert monitoring_call_args["stream_id"] is None  # stream_id was not set
        
        # Verify counters were reset after publish
        metrics = health_metrics.get_metrics()
        assert window["total_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_publish_with_stream_id(self, health_metrics_with_callbacks):
        """Test publishing metrics with stream_id set."""
        health_metrics, result_callback, monitoring_callback = health_metrics_with_callbacks
        
        # Set stream_id before recording
        health_metrics.set_stream_id("stream-abc-123")
        health_metrics.record(plugin_name="rapid_summary", success=True)
        
        # Publish
        await health_metrics.publish()
        
        # Verify monitoring_callback received stream_id
        monitoring_call_args = monitoring_callback.call_args[0][0]
        assert monitoring_call_args["stream_id"] == "stream-abc-123"
        
        # Verify result_callback did NOT receive stream_id
        result_call_args = result_callback.call_args[0][0]
        assert "stream_id" not in result_call_args
    
    @pytest.mark.asyncio
    async def test_publish_without_callbacks(self, health_metrics):
        """Test publishing without callbacks (should not raise)."""
        health_metrics.record(plugin_name="rapid_summary", success=True)
        
        # Should not raise even without callbacks
        result = await health_metrics.publish()
        
        assert result["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_scheduler_publishes_periodically(self):
        """Test that scheduler publishes metrics every 60 seconds."""
        from src.summary.llm_manager import HealthMetrics
        
        result_callback = AsyncMock()
        monitoring_callback = AsyncMock()
        
        health_metrics = HealthMetrics(
            timeout_seconds=240.0,
            result_callback=result_callback,
            monitoring_callback=monitoring_callback
        )
        
        # Record a request
        health_metrics.record(plugin_name="rapid_summary", success=True)
        
        # Start scheduler
        await health_metrics.start_scheduler()
        
        # Wait a bit more than 60 seconds
        await asyncio.sleep(61)
        
        # Stop scheduler
        await health_metrics.stop_scheduler()
        
        # Verify publish was called
        assert result_callback.call_count >= 1
        assert monitoring_callback.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_scheduler_stops_cleanly(self):
        """Test that scheduler stops cleanly."""
        from src.summary.llm_manager import HealthMetrics
        
        health_metrics = HealthMetrics(timeout_seconds=240.0)
        
        await health_metrics.start_scheduler()
        
        # Stop immediately
        await health_metrics.stop_scheduler()
        
        # Should complete without error
    
    def test_thread_safety(self, health_metrics):
        """Test that recording is thread-safe."""
        def record_requests(plugin_name, count):
            for i in range(count):
                if i % 3 == 0:
                    health_metrics.record(plugin_name, success=False, failure_type="parse_failure")
                elif i % 3 == 1:
                    health_metrics.record(plugin_name, success=False, failure_type="timeout_failure")
                else:
                    health_metrics.record(plugin_name, success=True)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=record_requests, args=("rapid_summary", 100))
            threads.append(t)
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify total count
        metrics = health_metrics.get_metrics()
        assert window["total_requests"] == 500  # 5 threads * 100 requests
        # Due to thread timing, exact counts may vary slightly
        assert 160 <= metrics["successful_requests"] <= 170  # approximately 1/3
        assert 160 <= metrics["failed_requests"]["parse_failure"] <= 170  # approximately 1/3
        assert 160 <= metrics["failed_requests"]["timeout_failure"] <= 170  # approximately 1/3


class TestLLMClientHealthMetrics:
    """Test suite for LLMClient health metrics integration."""
    
    @pytest.mark.asyncio
    async def test_llm_client_records_success(self):
        """Test that LLMClient records successful requests."""
        from src.summary.llm_manager import HealthMetrics, LLMClient
        from openai import AsyncOpenAI
        from unittest.mock import AsyncMock, MagicMock, patch
        
        # Create health metrics
        health_metrics = HealthMetrics(timeout_seconds=240.0)
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.choices[0].message.reasoning = ""
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.completion_tokens_details = None
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create LLMClient with health metrics
        llm_client = LLMClient(
            client=mock_client,
            model="test-model",
            build_messages_callback=lambda system, user: [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            health_metrics=health_metrics,
            plugin_name="rapid_summary"
        )
        
        # Make a request
        result = await llm_client.create_completion(
            system_prompt="test",
            user_content="test"
        )
        
        # Verify health metrics were recorded
        metrics = health_metrics.get_metrics()
        assert window["total_requests"] == 1
        assert window["successful_requests"] == 1
        assert window["plugin_breakdown"]["rapid_summary"]["success"] == 1
    
    @pytest.mark.asyncio
    async def test_llm_client_records_timeout(self):
        """Test that LLMClient records timeout failures."""
        from src.summary.llm_manager import HealthMetrics, LLMClient
        from openai import AsyncOpenAI
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio
        
        # Create health metrics
        health_metrics = HealthMetrics(timeout_seconds=1.0)  # Short timeout for testing
        
        # Create mock client that times out
        mock_client = MagicMock(spec=AsyncOpenAI)
        
        async def slow_create(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return MagicMock()
        
        mock_client.chat.completions.create = slow_create
        
        # Create LLMClient with health metrics
        llm_client = LLMClient(
            client=mock_client,
            model="test-model",
            build_messages_callback=lambda system, user: [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            health_metrics=health_metrics,
            plugin_name="rapid_summary"
        )
        
        # Make a request that will timeout
        with pytest.raises(asyncio.TimeoutError):
            await llm_client.create_completion(
                system_prompt="test",
                user_content="test"
            )
        
        # Verify timeout was recorded
        metrics = health_metrics.get_metrics()
        assert window["total_requests"] == 1
        assert window["failed_requests"]["timeout_failure"] == 1
    
    @pytest.mark.asyncio
    async def test_llm_client_without_health_metrics(self):
        """Test that LLMClient works without health metrics."""
        from src.summary.llm_manager import LLMClient
        from openai import AsyncOpenAI
        from unittest.mock import AsyncMock, MagicMock
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.choices[0].message.reasoning = ""
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.completion_tokens_details = None
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Create LLMClient WITHOUT health metrics
        llm_client = LLMClient(
            client=mock_client,
            model="test-model",
            build_messages_callback=lambda system, user: [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            health_metrics=None,
            plugin_name="rapid_summary"
        )
        
        # Make a request - should work fine
        result = await llm_client.create_completion(
            system_prompt="test",
            user_content="test"
        )
        
        assert result[1] == "test response"


class TestLLMManagerHealthMetrics:
    """Test suite for LLMManager health metrics integration."""
    
    def test_llm_manager_creates_health_metrics(self):
        """Test that LLMManager creates HealthMetrics."""
        from src.summary.llm_manager import LLMManager, HealthMetrics
        
        llm_manager = LLMManager(
            fast_base_url="http://localhost:8000",
            fast_api_key="test-key",
            reasoning_base_url="http://localhost:8001",
            reasoning_api_key="test-key",
            rapid_model="test-rapid",
            reasoning_model="test-reasoning",
            request_timeout_seconds=240.0
        )
        
        # Verify health metrics was created
        assert hasattr(llm_manager, '_health_metrics')
        assert isinstance(llm_manager._health_metrics, HealthMetrics)
    
    def test_llm_manager_set_stream_id(self):
        """Test that LLMManager.set_stream_id propagates to HealthMetrics."""
        from src.summary.llm_manager import LLMManager
        
        llm_manager = LLMManager(
            fast_base_url="http://localhost:8000",
            fast_api_key="test-key",
            reasoning_base_url="http://localhost:8001",
            reasoning_api_key="test-key",
        )
        
        llm_manager.set_stream_id("test-stream-456")
        
        assert llm_manager._health_metrics.stream_id == "test-stream-456"
    
    @pytest.mark.asyncio
    async def test_llm_manager_start_stop_scheduler(self):
        """Test that LLMManager can start and stop scheduler."""
        from src.summary.llm_manager import LLMManager
        
        llm_manager = LLMManager(
            fast_base_url="http://localhost:8000",
            fast_api_key="test-key",
            reasoning_base_url="http://localhost:8001",
            reasoning_api_key="test-key",
        )
        
        # Start scheduler
        await llm_manager.start_scheduler()
        
        # Stop scheduler
        await llm_manager.stop_scheduler()
        
        # Should complete without error
