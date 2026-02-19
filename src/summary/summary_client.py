"""
Summary client for LLM-based transcription cleaning and summarization.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from .context_summary.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_OUTPUT_CONSTRAINTS
from .llm_manager import LLMManager, MessageFormatMode
from .window_manager import WindowManager, WindowInsight

logger = logging.getLogger(__name__)

# Callback type for monitoring events: takes event data dict and event type string (async)
MonitoringCallback = Callable[[Dict[str, Any], str], Awaitable[None]]


class SummaryClient:
    """Client for LLM-based transcription cleaning and summarization."""
    
    def __init__(
        self,
        reasoning_base_url: str = "http://byoc-transcription-vllm:5000/v1",
        reasoning_api_key: str = "",
        reasoning_model: str = "",
        rapid_base_url: str = "http://byoc-transcription-vllm-rapid-summary:5050/v1",
        rapid_api_key: str = "",
        rapid_model: str = "",
        initial_summary_delay_seconds: float = 15,
        send_monitoring_event_callback: Optional[MonitoringCallback] = None,
        send_data_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """
        Initialize the summary client.
        
        Args:
            reasoning_base_url: Base URL for the reasoning API
            reasoning_api_key: API key for the reasoning API
            reasoning_model: Model name to use for reasoning
            rapid_base_url: Base URL for the rapid summary API
            rapid_api_key: API key for the rapid summary API
            rapid_model: Model name to use for rapid summarization
            initial_summary_delay_seconds: Seconds to wait before first summary (default: 10.0)
            send_monitoring_event_callback: Optional callback for sending monitoring events
            send_data_callback: Optional callback for sending data to client (async function that takes JSON string)
        """
        # Window-based state management with configurable accumulation
        self._window_manager: WindowManager = WindowManager()
        # In-flight tracking for graceful shutdown
        self.in_flight_windows: set[int] = set()  # Track window IDs being processed

        # Concurrency limiter for summary calls (set via env var)
        try:
            max_concurrent = int(os.getenv("MAX_CONCURRENT_SUMMARIES", "15"))
        except Exception:
            max_concurrent = 15
        self.max_concurrent_summaries: int = max(1, max_concurrent)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrent_summaries)

        # Track last processed timestamp (global, not per-window)
        self._last_processed_timestamp: float = 0.0
                
        # LLM Manager for plugins - handles all LLM client creation
        self.llm = LLMManager(
            fast_base_url=rapid_base_url,
            fast_api_key=rapid_api_key,
            reasoning_base_url=reasoning_base_url,
            reasoning_api_key=reasoning_api_key,
            rapid_model=rapid_model,
            reasoning_model=reasoning_model,
        )
        
        # Initial summary delay configuration
        self.initial_summary_delay_seconds: float = initial_summary_delay_seconds
        
        # Monitoring event callback
        self._send_monitoring_event_callback: Optional[MonitoringCallback] = send_monitoring_event_callback
        
        # Send data callback for sending results to client
        self._send_data_callback: Optional[Callable[[str], Awaitable[None]]] = send_data_callback
        # Backward compatibility alias
                
        # Summary worker queue and results (internal)
        from collections import deque
        self._summary_queue: asyncio.Queue = asyncio.Queue()
        self._summary_results: deque = deque(maxlen=100)
        
        # Worker and sender tasks
        self._worker_tasks: list[asyncio.Task] = []
        self._sender_task: Optional[asyncio.Task] = None
        
        # Shutdown flags
        self._stop_requested: bool = False
        self._shutdown_requested: bool = False
        
        # Plugin system - dynamically discover and initialize plugins
        self._plugins: Dict[str, Any] = {}  # plugin_name -> plugin instance
        self._event_callbacks: Dict[str, Dict[str, Callable]] = {}  # plugin_name -> {event_name: callback}
        
        # Discover and initialize plugins
        self._discover_plugins()
    
    # ==================== Plugin System Methods ====================
    
    def _discover_plugins(self):
        """Dynamically discover plugins by scanning subdirectories."""
        import os
        import importlib
        
        # Get the directory containing this module
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get the parent package name (e.g., 'src.summary' from 'src.summary.summary_client')
        parent_package = __name__.rsplit('.', 1)[0]
        
        # Scan subdirectories for plugins
        for entry in os.listdir(plugin_dir):
            plugin_path = os.path.join(plugin_dir, entry)
            if os.path.isdir(plugin_path) and not entry.startswith('_'):
                # Try to import the plugin module and call init_plugin
                try:
                    module = importlib.import_module(f".{entry}", package=parent_package)
                    if hasattr(module, 'init_plugin'):
                        module.init_plugin(
                            plugin_name=entry,
                            window_manager=self._window_manager,
                            llm=self.llm,
                            result_callback=self._queue_payload,
                            summary_client=self,
                        )
                        logger.info(f"Loaded plugin: {entry}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin {entry}: {e}")
    
    def register_plugin_event_sub(self, plugin_name: str, plugin_instance: Any, events: Dict[str, Callable]):
        """Plugins call this to subscribe to events."""
        self._plugins[plugin_name] = plugin_instance
        self._event_callbacks[plugin_name] = events
        logger.info(f"Registered plugin: {plugin_name} with events: {list(events.keys())}")
    
    async def _notify_plugins(self, event_name: str, **kwargs):
        """Notify all plugins for an event and WAIT for completion."""
        tasks = []
        for plugin_name, events in self._event_callbacks.items():
            if event_name in events:
                callback = events[event_name]
                tasks.append(asyncio.create_task(callback(**kwargs)))
        
        # Wait for ALL plugins to complete - no separate semaphore needed
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _queue_payload(self, payload: Dict[str, Any]):
        """Queue a payload for sending."""
        self._summary_results.append(payload)
    
    # ==================== End Plugin System ====================
    
    async def summarize_text(
        self,
        text: str,
        context: str = ""
    ) -> Tuple[str, str, int]:
        """
        Summarize text using the context summary task.
        
        This method provides backward compatibility for tests and external code
        that expects a summarize_text method on the client.
        
        Args:
            text: Text to summarize
            context: Additional context from previous segments
            
        Returns:
            Tuple of (summary_json_string, reasoning_content, input_tokens)
        """
        # Use the context_summary plugin for summarization
        if "context_summary" in self._plugins:
            plugin = self._plugins["context_summary"]
            result = await plugin.process_summary(text=text, context=context)
            return result.get("summary_text", "{}"), result.get("reasoning_content", ""), result.get("input_tokens", 0)
        
        # Fallback: return empty response if plugin not available
        return "{}", "", 0
    
    async def initialize(self) -> Optional[str]:
        """
        Initialize the lock for async operations and detect model if needed.
        
        Delegates model detection to LLMManager which handles both reasoning
        and rapid model auto-detection.
        
        Returns:
            The detected model ID if model was auto-detected, None otherwise
        """
        logger.info("SummaryClient.initialize called")
                
        # Delegate model detection to LLMManager
        detected = await self.llm.initialize()
        
        # Note: warm_up() is now called inside llm.initialize() - failures will propagate
        logger.info("SummaryClient initialized")
        
        return detected
    
    def update_params(
        self,
        reasoning_base_url: Optional[str] = None,
        reasoning_api_key: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        rapid_base_url: Optional[str] = None,
        rapid_api_key: Optional[str] = None,
        rapid_model: Optional[str] = None,
        reasoning_max_tokens: Optional[int] = None,
        rapid_max_tokens: Optional[int] = None,
        reasoning_temperature: Optional[float] = None,
        rapid_temperature: Optional[float] = None,
        reasoning_system_prompt: Optional[str] = None,
        rapid_system_prompt: Optional[str] = None,
        transcription_windows_per_summary_window: Optional[int] = None,
        raw_text_context_limit: Optional[int] = None,
        initial_summary_delay_seconds: Optional[float] = None,
        content_type_context_limit: Optional[int] = None
    ):
        """
        Update client parameters dynamically.
        
        Args:
            reasoning_base_url: New base URL for the reasoning API
            reasoning_api_key: New API key for the reasoning API
            reasoning_model: New model name for the reasoning API
            rapid_base_url: New base URL for the rapid API
            rapid_api_key: New API key for the rapid API
            rapid_model: New model name for the rapid API
            reasoning_max_tokens: New max tokens for the reasoning API
            rapid_max_tokens: New max tokens for the rapid API
            reasoning_temperature: New temperature for the reasoning API
            rapid_temperature: New temperature for the rapid API
            reasoning_system_prompt: New system prompt for the reasoning API
            rapid_system_prompt: New system prompt for the rapid API
            transcription_windows_per_summary_window: New number of transcription windows per summary window
            raw_text_context_limit: New max characters for raw text in LLM context
            initial_summary_delay_seconds: New delay before first summary (default: 10.0)
            content_type_context_limit: New character limit for content type detection
        
        Note:
            message_format_mode is managed by LLMManager and cannot be updated here.
            Use LLMManager's message_format_mode property instead.
        """
        # Values SummaryClient uses - update directly
        if transcription_windows_per_summary_window is not None:
            self._window_manager.transcription_windows_per_summary_window = transcription_windows_per_summary_window
        if raw_text_context_limit is not None:
            self._window_manager.raw_text_context_limit = raw_text_context_limit
            logger.info(f"Updated raw_text_context_limit to {raw_text_context_limit}")
        if initial_summary_delay_seconds is not None:
            self.initial_summary_delay_seconds = initial_summary_delay_seconds
            logger.info(f"Updated initial_summary_delay_seconds to {initial_summary_delay_seconds}")
        
        # Pass to LLMManager (sync call) - values SummaryClient doesn't store
        self.llm.update_params(
            reasoning_base_url=reasoning_base_url,
            reasoning_api_key=reasoning_api_key,
            reasoning_model=reasoning_model,
            rapid_base_url=rapid_base_url,
            rapid_api_key=rapid_api_key,
            rapid_model=rapid_model,
        )
        
        # Notify plugins directly (sync call) - call on_update_params if it exists
        for plugin_name, plugin_instance in self._plugins.items():
            if hasattr(plugin_instance, 'on_update_params'):
                try:
                    # Call the sync version - plugins handle both sync and async internally
                    plugin_instance.on_update_params(
                        reasoning_max_tokens=reasoning_max_tokens,
                        rapid_max_tokens=rapid_max_tokens,
                        reasoning_temperature=reasoning_temperature,
                        rapid_temperature=rapid_temperature,
                        reasoning_system_prompt=reasoning_system_prompt,
                        initial_summary_delay_seconds=initial_summary_delay_seconds,
                        content_type_context_limit=content_type_context_limit,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update params for plugin {plugin_name}: {e}")
        
        logger.info(f"SummaryClient params updated")
    
    def reset(self):
        """Reset all accumulated state for a new stream."""
        self._window_manager.clear()
        self.in_flight_windows.clear()
        # Reset last processed timestamp for new stream
        self._last_processed_timestamp = 0.0
        logger.info("SummaryClient reset complete - all state cleared for new stream")
    
    async def _send_monitoring_event(self, event_data: Dict[str, Any], event_type: str):
        """Send a monitoring event if callback is configured."""
        if self._send_monitoring_event_callback:
            try:
                await self._send_monitoring_event_callback(event_data, event_type)
            except Exception as e:
                logger.warning(f"Failed to send monitoring event: {e}")
    
    def add_in_flight_window(self, window_id: int):
        """Add window ID to in-flight tracking."""
        self.in_flight_windows.add(window_id)
    
    def remove_in_flight_window(self, window_id: int):
        """Remove window ID from in-flight tracking."""
        self.in_flight_windows.discard(window_id)
    
    def get_pending_count(self) -> int:
        """Get count of pending summary requests."""
        return len(self.in_flight_windows)
        
    # =========================================================================
    # Summary Worker and Sender Methods (moved from pipeline/main.py)
    # =========================================================================
    
    async def start(self, num_workers: int = 4):
        """
        Start summary worker tasks and sender task.
        
        Args:
            num_workers: Number of worker tasks to create (default: 4)
        """
        import os
        
        # Reset stop flag for new stream
        self._stop_requested = False
        
        num_workers = int(os.getenv("MAX_CONCURRENT_SUMMARIES", str(num_workers)))
        
        if not self._worker_tasks or all(t.done() for t in self._worker_tasks):
            self._worker_tasks = [
                asyncio.create_task(self._summary_worker_task())
                for _ in range(num_workers)
            ]
            logger.info(f"Started {num_workers} summary worker tasks")
        
        # Start sender task if not already running
        if self._sender_task is None or self._sender_task.done():
            self._sender_task = asyncio.create_task(self._summary_sender_task())
            logger.info("Started summary sender task")
    
    async def stop(self, timeout: float = 5.0):
        """
        Stop all worker and sender tasks gracefully.
        
        Args:
            timeout: Maximum seconds to wait for workers to complete
        """
        # Signal workers to stop
        self._shutdown_requested = True
        
        # Send shutdown signals (None per worker)
        num_workers = len(self._worker_tasks)
        for _ in range(num_workers):
            self._summary_queue.put_nowait(None)
        logger.info(f"Sent {num_workers} shutdown signals to summary workers")
        
        # Wait for workers to complete
        if self._worker_tasks:
            done, pending = await asyncio.wait(
                self._worker_tasks,
                timeout=timeout
            )
            for task in pending:
                task.cancel()
        
        # Cancel sender if running
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
        
        # Clear task lists
        self._worker_tasks = []
        self._sender_task = None
        self._shutdown_requested = False
        logger.info("Summary worker tasks stopped")
    
    def clear(self):
        """Clear queues for fresh stream."""
        self._summary_queue._queue.clear()
        self._summary_results.clear()
        logger.info("Summary queues cleared")
    
    def queue_segments(
        self,
        segments: List[Any],
        transcription_window_id: int,
        window_start_ts: float,
        window_end_ts: float
    ):
        """
        Add transcription segments to the work queue.
        
        Args:
            segments: List of transcription segments
            transcription_window_id: ID of the transcription window
            window_start_ts: Start timestamp of the window
            window_end_ts: End timestamp of the window
        """
        self._summary_queue.put_nowait((
            segments,
            transcription_window_id,
            window_start_ts,
            window_end_ts
        ))
    
    def add_content_type_detection(
        self,
        content_type: str,
        confidence: float,
        source: str,
        previous_content_type: str = None
    ):
        """
        Add content type detection result to results deque.
        
        Args:
            content_type: Detected content type (e.g., "GENERAL_MEETING", "TECHNICAL_TALK")
            confidence: Confidence level (0.0-1.0)
            source: Source of detection ("AUTO_DETECTED", "USER_OVERRIDE")
            previous_content_type: Previous content type if it changed
        """
        payload = {
            "type": "content_type_detection",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "content_type": content_type,
            "confidence": confidence,
            "source": source,
            "previous_content_type": previous_content_type
        }
        self._summary_results.append(payload)
        logger.info(f"ADDED to summary_results: content_type_detection - type='{content_type}', confidence={confidence:.2f}, previous='{previous_content_type}'")
    
    async def _summary_worker_task(self):
        """Background task to process summary requests from the queue.
        
        This is a thin pass-through worker. The SummaryClient handles all
        buffering, merging, and payload building. The worker simply:
        1. Receives transcription windows from the queue
        2. Calls process_segments() on the client
        3. Sends the result directly to the results deque
        """
        import time
        
        while True:
            try:
                # Wait for work with timeout to allow checking shutdown state
                try:
                    work_item = await asyncio.wait_for(self._summary_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Timeout is expected - check if we should shutdown
                    if self._shutdown_requested and self._summary_queue.empty():
                        logger.info("Summary worker stopping - shutdown requested and queue empty")
                        break
                    continue
                except asyncio.CancelledError:
                    # Task was cancelled externally - exit gracefully
                    break
                
                # Check for None shutdown signal
                if work_item is None:
                    logger.info("Summary worker received shutdown signal - exiting")
                    break
                
                segments, transcription_window_id, window_start_ts, window_end_ts = work_item
                
                # Check if stop is requested before processing
                if self._stop_requested:
                    logger.info(f"Summary worker skipping window {transcription_window_id} - stop requested")
                    continue
                
                # Process through client - it handles all buffering, merging, and payload building
                try:
                    # Add window to in-flight tracking
                    self.add_in_flight_window(transcription_window_id)
                    
                    start = time.perf_counter()
                    result_payload = await asyncio.wait_for(
                        self.process_segments(
                            "context_summary",
                            segments,
                            transcription_window_id,
                            window_start_ts,
                            window_end_ts
                        ),
                        timeout=240.0
                    )
                    end = time.perf_counter()
                    
                    # Log monitoring event
                    if self._send_monitoring_event_callback is not None:
                        await self._send_monitoring_event_callback(
                            {
                                "duration_seconds": end - start,
                                "window_id": result_payload.get("summary_window_id", transcription_window_id),
                                "window_start_ms": window_start_ts * 1000,
                                "window_end_ms": window_end_ts * 1000
                            },
                            "llm_summary_request_stats"
                        )
                    
                    # Send result directly - client already built complete payload
                    if result_payload and result_payload.get("segments"):
                        self._summary_results.append(result_payload)
                        logger.debug(
                            f"Staged summary data for window {result_payload.get('summary_window_id', transcription_window_id)} "
                            f"with transcription_window_ids {result_payload.get('transcription_window_ids', [transcription_window_id])}"
                        )
                    elif result_payload.get("type") == "content_type_detection":
                        self._summary_results.append(result_payload)
                        logger.info(
                            f"Staged content_type_detection: {result_payload.get('content_type')} "
                            f"(source: {result_payload.get('source')})"
                        )
                    else:
                        logger.debug(
                            f"No summary segments for window {transcription_window_id} "
                            f"(client is buffering/merging windows)"
                        )
                        
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Summarization timed out for window {transcription_window_id} "
                        f"[{window_start_ts:.3f}s - {window_end_ts:.3f}s]"
                    )
                    if self._send_monitoring_event_callback is not None:
                        await self._send_monitoring_event_callback(
                            {
                                "window_id": transcription_window_id,
                                "window_start_ms": window_start_ts * 1000,
                                "window_end_ms": window_end_ts * 1000
                            },
                            "llm_summary_request_timeout"
                        )
                except Exception as e:
                    logger.error(f"Summary processing error: {e}")
                finally:
                    # Remove from in-flight tracking regardless of success or failure
                    self.remove_in_flight_window(transcription_window_id)
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Summary worker error: {e}")
        
        logger.info("Summary worker task stopped")
    
    async def _summary_sender_task(self):
        """Background task that monitors deque and sends results to client."""
        import json
        
        logger.info("Starting summary sender")
        while True:
            try:
                if len(self._summary_results) == 0:
                    await asyncio.sleep(0.1)  # Sleep if empty
                    continue
                
                # Drain all results and send
                while len(self._summary_results) > 0:
                    result = self._summary_results.popleft()
                    result_type = result.get("type", "unknown")
                    result_msg = f"SENDING from summary_results: type='{result_type}'"
                    if result_type == "content_type_detection":
                        result_msg += f"  content_type='{result.get('content_type')}', confidence={result.get('confidence')}, previous='{result.get('previous_content_type')}'"
                    logger.info(result_msg)
                    try:
                        if self._send_data_callback is not None:
                            await self._send_data_callback(json.dumps(result))
                            logger.debug(f"Sent summary result")
                    except Exception as e:
                        logger.error(f"Error sending summary result: {e}")
                
                await asyncio.sleep(0.05)  # Small pause between batches
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Summary sender error: {e}")
        logger.info("Summary sender task stopped")
    