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
from .context_summary.task import WindowInsight
from .llm_manager import LLMManager, MessageFormatMode
from .agent_manager import AgentManager
from .window_manager import WindowManager

logger = logging.getLogger(__name__)

# Callback type for monitoring events: takes event data dict and event type string (async)
MonitoringCallback = Callable[[Dict[str, Any], str], Awaitable[None]]


class SummaryClient:
    """Client for LLM-based transcription cleaning and summarization.
    
    Components:
    - LLMManager (self.llm): Provides fast and reasoning LLM clients for plugins.
    - AgentManager (self.agent): Provides autonomous agent + knowledge store.
    
    Usage:
        client = SummaryClient(reasoning_base_url=..., rapid_base_url=...)
        await client.initialize()  # Initializes both llm and agent
        # Plugins use client.llm for LLM calls
        # Agent uses client.llm.reasoning_client for autonomous queries
    """
    
    def __init__(
        self,
        reasoning_base_url: str = "http://byoc-transcription-vllm-insights:5000/v1",
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
        
        # LLM Manager: provides fast and reasoning LLM clients for plugins
        self.llm = LLMManager(
            fast_base_url=rapid_base_url,
            fast_api_key=rapid_api_key,
            reasoning_base_url=reasoning_base_url,
            reasoning_api_key=reasoning_api_key,
            rapid_model=rapid_model,
            reasoning_model=reasoning_model,
            health_result_callback=self._queue_payload,
            health_monitoring_callback=self._send_monitoring_event,
        )
        
        # Agent Manager: provides autonomous agent + knowledge store
        self.agent = AgentManager(result_callback=self._queue_payload)
        self.agent.set_window_manager(self._window_manager)
        
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

        # Last language detected by Whisper
        self.detected_language: Optional[str] = None
        self.language_confidence: float = 0.0
        
        # Plugin system - dynamically discover and initialize plugins
        self._plugins: Dict[str, Any] = {}  # plugin_name -> plugin instance
        self._event_callbacks: Dict[str, Dict[str, Callable]] = {}  # plugin_name -> {event_name: callback}
        
        # Note: Plugins are loaded in initialize() after LLMManager is ready
    
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
                            llm_manager=self.llm,
                            result_callback=self._queue_payload,
                            summary_client=self,
                            send_monitoring_event_callback=self._send_monitoring_event_callback,
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
        
        # Initialize the agent (builds PydanticAI agent with reasoning client)
        await self.agent.initialize(self.llm)
        
        # Note: warm_up() is now called inside llm.initialize() - failures will propagate
        
        # Load plugins after LLMManager is initialized (LLMClient instances are ready)
        self._discover_plugins()

        # Register agent as an event subscriber so it indexes fast summaries automatically
        self.register_plugin_event_sub(
            plugin_name="agent_manager",
            plugin_instance=self.agent,
            events={"fast_summary_available": self.agent.handle_fast_summary_available},
        )

        logger.info("SummaryClient initialized")
        
        return detected
    
    def on_language_detected(self, language: str, confidence: float) -> None:
        """Called per-transcription-window when Whisper detects a (new) language.

        Stores the latest language/confidence and propagates the information to
        the LLM layer (so the system-prompt language suffix and logit-bias are
        updated) as well as to any plugin that implements ``on_language_detected``.

        Args:
            language: ISO-639-1 code returned by faster-whisper (e.g. ``"en"``, ``"zh"``)
            confidence: Whisper's ``language_probability`` in [0, 1]
        """
        self.detected_language = language
        self.language_confidence = confidence
        logger.info(f"Language detected: {language} (confidence={confidence:.3f})")

        # Update LLM suffix + logit-bias for both fast and reasoning clients
        self.llm.update_language(language)

        # Fan out to plugins that want to know about language changes
        for plugin_name, plugin_instance in self._plugins.items():
            if hasattr(plugin_instance, 'on_language_detected'):
                try:
                    plugin_instance.on_language_detected(language, confidence)
                except Exception as e:
                    logger.warning(f"Plugin {plugin_name} on_language_detected error: {e}")

    def update_params(
        self,
        reasoning_base_url: Optional[str] = None,
        reasoning_api_key: Optional[str] = None,
        reasoning_model: Optional[str] = None,
        fast_base_url: Optional[str] = None,
        fast_api_key: Optional[str] = None,
        fast_model: Optional[str] = None,
        reasoning_max_tokens: Optional[int] = None,
        fast_max_tokens: Optional[int] = None,
        reasoning_temperature: Optional[float] = None,
        fast_temperature: Optional[float] = None,
        reasoning_system_prompt: Optional[str] = None,
        fast_system_prompt: Optional[str] = None,
        transcription_windows_per_summary_window: Optional[int] = None,
        raw_text_context_limit: Optional[int] = None,
        initial_summary_delay_seconds: Optional[float] = None,
        content_type_context_limit: Optional[int] = None,
        agent_chat_query: Optional[str] = None,
        agent_model_type: Optional[str] = None,
    ):
        """
        Update client parameters dynamically.
        
        Args:
            reasoning_base_url: New base URL for the reasoning API
            reasoning_api_key: New API key for the reasoning API
            reasoning_model: New model name for the reasoning API
            fast_base_url: New base URL for the fast API
            fast_api_key: New API key for the fast API
            fast_model: New model name for the fast API
            reasoning_max_tokens: New max tokens for the reasoning API
            fast_max_tokens: New max tokens for the fast API
            reasoning_temperature: New temperature for the reasoning API
            fast_temperature: New temperature for the fast API
            reasoning_system_prompt: New system prompt for the reasoning API
            fast_system_prompt: New system prompt for the fast API
            transcription_windows_per_summary_window: New number of transcription windows per summary window
            raw_text_context_limit: New max characters for raw text in LLM context
            initial_summary_delay_seconds: New delay before first summary (default: 10.0)
            content_type_context_limit: New character limit for content type detection
            agent_chat_query: Custom system prompt for the agent chat query
            agent_model_type: Which agent LLM to use — ``"reasoning"`` (default) or ``"fast"``.
        
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
            fast_base_url=fast_base_url,
            fast_api_key=fast_api_key,
            fast_model=fast_model,
        )
        
        # Notify plugins directly (sync call) - call on_update_params if it exists
        for plugin_name, plugin_instance in self._plugins.items():
            if hasattr(plugin_instance, 'on_update_params'):
                try:
                    # Call the sync version - plugins handle both sync and async internally
                    plugin_instance.on_update_params(
                        reasoning_max_tokens=reasoning_max_tokens,
                        fast_max_tokens=fast_max_tokens,
                        reasoning_temperature=reasoning_temperature,
                        fast_temperature=fast_temperature,
                        reasoning_system_prompt=reasoning_system_prompt,
                        initial_summary_delay_seconds=initial_summary_delay_seconds,
                        content_type_context_limit=content_type_context_limit,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update params for plugin {plugin_name}: {e}")
        
        # If a query was supplied, fire it against the agent and return the
        # response over the data connection as an "agent_response" payload.
        if agent_chat_query is not None:
            try:
                asyncio.create_task(self._run_ask_agent(agent_chat_query, model_type=agent_model_type or "reasoning"))
            except Exception as e:
                logger.warning(f"Failed to schedule ask_agent task: {e}")
        
        logger.info(f"SummaryClient params updated")
    
    def reset(self):
        """Reset all accumulated state for a new stream."""
        self._window_manager.clear()
        self.in_flight_windows.clear()
        # Reset last processed timestamp for new stream
        self._last_processed_timestamp = 0.0

        # Clear the in-memory transcript knowledge store
        self.agent.reset_knowledge_store()
        
        # Reset all plugins that have a reset method
        for plugin_name, plugin_instance in self._plugins.items():
            if hasattr(plugin_instance, 'reset'):
                try:
                    plugin_instance.reset()
                    logger.debug(f"Reset plugin: {plugin_name}")
                except Exception as e:
                    logger.warning(f"Failed to reset plugin {plugin_name}: {e}")
        
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

    async def ask_agent(self, query: str, model_type: str = "reasoning") -> Dict[str, Any]:
        """
        Submit a natural-language query to the autonomous agent.

        The agent will autonomously use the transcript knowledge store
        (semantic search over indexed video content) and web search
        (Tavily > Exa > DuckDuckGo) to compose a grounded answer.

        Args:
            query: The user's question.
            model_type: Which agent LLM to use — ``"reasoning"`` (default, thinking/slow)
                or ``"fast"`` (rapid/lightweight).

        Returns:
            Dict with ``response`` (str) and ``error`` (str | None) fields.
        """
        return await self.agent.ask_agent(query, model_type=model_type)

    async def _run_ask_agent(self, query: str, model_type: str = "reasoning") -> None:
        """Run ask_agent and push the response as an agent_response payload.

        Intended to be fired as a background task from update_params so the
        caller is not blocked while the agent runs.

        Args:
            query: The user's question forwarded from update_params.
            model_type: Which agent LLM to use — ``"reasoning"`` (default) or ``"fast"``.
        """
        try:
            result = await self.ask_agent(query, model_type=model_type)
            await self._queue_payload({
                "type": "agent_response",
                "query": query,
                "response": result.get("response"),
                "error": result.get("error"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"agent_response queued for query: {query[:60]}")
        except Exception as e:
            logger.error(f"ask_agent failed for query '{query[:60]}': {e}")
            await self._queue_payload({
                "type": "agent_response",
                "query": query,
                "response": None,
                "error": str(e),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

    async def handle_speakers_merged(self, merged_speakers: List[Dict[str, str]]) -> None:
        """Update knowledge store when speakers are merged.

        This should be called when the diarization system emits a speakers_merged
        event to keep the indexed transcript chunks in sync with the merged speakers.

        Args:
            merged_speakers: List of {"source": "speaker_1", "target": "speaker_0"} dicts.
        """
        if not merged_speakers:
            return
        if hasattr(self.agent, 'remap_speakers'):
            count = self.agent.remap_speakers(merged_speakers)
            logger.info(f"Updated {count} chunks after speaker merge: {merged_speakers}")

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
        
        # Start health metrics scheduler
        await self.llm.start_scheduler()
        
        # Start sender task if not already running
        if self._sender_task is None or self._sender_task.done():
            self._sender_task = asyncio.create_task(self._summary_sender_task())
            logger.info("Started summary sender task")
    
    async def stop(self, timeout: float = 240.0):
        """
        Stop all worker and sender tasks gracefully.
        
        Args:
            timeout: Maximum seconds to wait for workers to complete (default: 240s / 4 minutes)
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
        
        # Stop health metrics scheduler
        await self.llm.stop_scheduler()
        
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
    
    async def queue_segments(
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
        # Calculate word count from segments
        word_count = sum(len(seg.get("text", "").split()) for seg in segments)
        
        self._summary_queue.put_nowait((
            segments,
            transcription_window_id,
            window_start_ts,
            window_end_ts
        ))
        
        # Emit summary_window_queued monitoring event
        if self._send_monitoring_event_callback is not None:
            await self._send_monitoring_event_callback(
                {
                    "transcription_window_id": transcription_window_id,
                    "word_count": word_count,
                    "window_start_ms": int(window_start_ts * 1000),
                    "window_end_ms": int(window_end_ts * 1000),
                    "queue_size": self._summary_queue.qsize(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()
                },
                "summary_window_queued"
            )
    
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
    
    async def process_transcription(
        self,
        segments: List[Any],
        transcription_window_id: int,
        window_start_ts: float,
        window_end_ts: float
    ) -> Dict[str, Any]:
        """Process transcription window with deduplication and plugin notification."""
        # Add transcription window to manager
        self._window_manager.add_transcription_window(
            transcription_window_id=transcription_window_id,
            segments=segments,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts
        )
        
        # Check if summary window should be created
        summary_window_id = self._window_manager.maybe_create_summary_window()
        
        # Notify plugins of transcription window available
        await self._notify_plugins(
            "transcription_window_available",
            transcription_window_id=transcription_window_id
        )
        
        # If summary window created, notify plugins
        if summary_window_id is not None:
            await self._notify_plugins(
                "summary_window_available",
                summary_window_id=summary_window_id
            )

        # Index the deduplicated text into the knowledge store for semantic search
        deduplicated_text = self._window_manager.get_deduplicated_text(
            transcription_window_id
        ) if hasattr(self._window_manager, 'get_deduplicated_text') else ""
        if not deduplicated_text:
            # Fallback: join segments text directly
            deduplicated_text = " ".join(
                seg.get("text", "") for seg in segments if seg.get("text")
            )
        asyncio.create_task(
            self.agent.index_transcript_segment(
                text=deduplicated_text,
                timestamp=window_start_ts,
                duration=window_end_ts - window_start_ts,
                window_id=transcription_window_id,
            )
        )

        return {
            "transcription_window_id": transcription_window_id,
            "summary_window_id": summary_window_id
        }
    
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
                    
                    # Emit worker in-progress monitoring event
                    if self._send_monitoring_event_callback is not None:
                        await self._send_monitoring_event_callback(
                            {
                                "transcription_window_id": transcription_window_id,
                                "worker_id": id(asyncio.current_task()),
                                "queue_size": self._summary_queue.qsize(),
                                "in_flight_count": len(self.in_flight_windows),
                                "timestamp_utc": datetime.now(timezone.utc).isoformat()
                            },
                            "summary_worker_in_progress"
                        )
                    
                    start = time.perf_counter()
                    result_payload = await asyncio.wait_for(
                        self.process_transcription(
                            segments,
                            transcription_window_id,
                            window_start_ts,
                            window_end_ts
                        ),
                        timeout=360.0
                    )
                    end = time.perf_counter()
                    
                    # Log monitoring event
                    if self._send_monitoring_event_callback is not None:
                        await self._send_monitoring_event_callback(
                            {
                                "duration_seconds": end - start,
                                "transcription_window_id": result_payload.get("transcription_window_id", ""),
                                "summary_window_id": result_payload.get("summary_window_id", ""),
                                "window_start_ms": window_start_ts * 1000,
                                "window_end_ms": window_end_ts * 1000
                            },
                            "process_transcription_request_stats"
                        )                       
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Summarization timed out for window {transcription_window_id} "
                        f"[{window_start_ts:.3f}s - {window_end_ts:.3f}s]"
                    )
                    if self._send_monitoring_event_callback is not None:
                        await self._send_monitoring_event_callback(
                            {
                                "transcription_window_id": result_payload.get("transcription_window_id", ""),
                                "summary_window_id": result_payload.get("summary_window_id", ""),
                                "window_start_ms": window_start_ts * 1000,
                                "window_end_ms": window_end_ts * 1000
                            },
                            "process_transcription_request_stats"
                        )
                except Exception as e:
                    logger.exception("Summary processing error")
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
    