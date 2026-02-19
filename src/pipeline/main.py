"""
StreamProcessor-based pipeline for transcription and subtitle overlay.

This module implements model lifecycle and per-frame processing, while
pytrickle's StreamProcessor manages I/O and HTTP endpoints.
"""

import asyncio
import logging
import os
import sys
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Set
import time
from aiohttp import web

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from diarization.diarization_client import DiarizationClient, DiarizationResult, SpeakerSegment
from transcription.whisper_client import WhisperClient, TranscriptionSegment, WordTimestamp
from summary.summary_client import SummaryClient
from pytrickle import StreamProcessor
from pytrickle import AudioFrame
import numpy as np
import tempfile
import wave
import uuid

logger = logging.getLogger(__name__)


class TranscriberState:
    """Holds model, buffers, and runtime params for transcription and overlay."""

    def __init__(self):
        # Model and generators
        self.whisper_client: Optional[WhisperClient] = None
        self.summary_client: Optional[SummaryClient] = None
        self.diarization_client: Optional[DiarizationClient] = None

        # Whisper config
        self.whisper_model: str = "turbo"
        self.whisper_language: Optional[str] = None
        self.whisper_device: str = "cuda"
        self.compute_type: str = "float16"

        # Audio processing
        self.audio_sample_rate: int = 16000
        self.window_seconds: float = 2.5
        self.overlap_seconds: float = 0.5
        self.audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)  # mono float32 [-1,1]
        self.buffer_start_ts: Optional[float] = None  # seconds
        self.buffer_rate: Optional[int] = None

        # Transcript state
        self.current_segments: list[TranscriptionSegment] = []
        self.current_srt: str = ""
        self.last_sent_data_time: float = 0.0
        
        # Stream media time origin (reset on /stream/start)
        self.stream_start_media_ts: Optional[float] = None

        # Diarization process
        self.diarization_poll_task: Optional[asyncio.Task] = None
        self.diarization_counter: int = 0  # Track transcribe calls for diarization scheduling
        self.diarization_temp_files: list[dict] = []  # Track temp files with timestamps for combining
        self.diarization_window_timestamps: Dict[str, tuple[float, float]] = {}  # request_id -> (start_ts, end_ts)
        self.diarization_audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)  # Separate buffer for diarization audio
        self.diarization_buffer_start_ts: Optional[float] = None  # Start timestamp for diarization buffer
        
        # Temp file tracking for cleanup (when both transcribe and diarize are done)
        self.pending_temp_files: Dict[str, Dict] = {}

        # Stop request flag for graceful shutdown (used by diarization polling)
        self.stop_requested: bool = False  # Flag to signal workers to stop

        # Shutdown tracking for graceful shutdown via update_params
        self.shutdown_requested: bool = False  # Signal to stop accepting new work
        self.shutdown_completed: bool = False  # All in-flight work done


# Global state and processor reference
STATE: Optional[TranscriberState] = None
PROCESSOR: Optional[StreamProcessor] = None


def _frame_time_seconds(timestamp: int, time_base) -> float:
    try:
        return float(timestamp) * float(time_base)
    except Exception:
        return float(timestamp)


def _has_any_meaningful_segments(segments: list[TranscriptionSegment]) -> bool:
    """Check if any segment contains meaningful text (non-empty, non-whitespace)."""
    return any(bool(seg.text.strip()) for seg in segments)


async def load_model(**kwargs):
    """Initialize Whisper, SRT generator, fonts, and buffers from params."""
    global STATE
    STATE = TranscriberState()

    params = dict(kwargs or {})

    # Whisper params
    STATE.whisper_model = str(params.get("whisper_model", STATE.whisper_model))
    STATE.whisper_language = params.get("whisper_language", STATE.whisper_language)
    STATE.whisper_device = str(params.get("whisper_device", STATE.whisper_device))
    STATE.compute_type = str(params.get("compute_type", STATE.compute_type))

    # Audio params
    STATE.audio_sample_rate = int(params.get("audio_sample_rate", STATE.audio_sample_rate))
    STATE.window_seconds = float(params.get("chunk_window", STATE.window_seconds))
    STATE.overlap_seconds = float(params.get("chunk_overlap", STATE.overlap_seconds))

    # Summary params
    summary_base_url = params.get("summary_base_url", "http://byoc-transcription-vllm:5000/v1")
    summary_api_key = params.get("summary_api_key", "")
    summary_model = params.get("summary_model", os.environ.get("LOCAL_REASONING_MODEL", ""))
    
    # Rapid summary params
    rapid_summary_base_url = params.get("rapid_summary_base_url", "http://byoc-transcription-vllm-rapid-summary:5050/v1")
    rapid_summary_api_key = params.get("rapid_summary_api_key", "")
    rapid_summary_model = params.get("rapid_summary_model", os.environ.get("LOCAL_RAPID_SUMMARY_MODEL", ""))

    # Init model
    STATE.whisper_client = WhisperClient(
        model_size=STATE.whisper_model,
        device=STATE.whisper_device,
        compute_type=STATE.compute_type,
        language=STATE.whisper_language,
    )
    await STATE.whisper_client.initialize()

    # Initialize summary client
    STATE.summary_client = SummaryClient(
        reasoning_base_url=summary_base_url,
        reasoning_api_key=summary_api_key,
        reasoning_model=summary_model,
        send_monitoring_event_callback=PROCESSOR.send_monitoring_event if PROCESSOR else None,
        send_data_callback=PROCESSOR.send_data if PROCESSOR else None,
        rapid_base_url=rapid_summary_base_url,
        rapid_api_key=rapid_summary_api_key,
        rapid_model=rapid_summary_model
    )
    
    # Initialize and detect model if needed
    detected_model = await STATE.summary_client.initialize()
    
    if detected_model:
        # Model was auto-detected, update our tracking
        summary_model = detected_model
        logger.info(f"Using auto-detected model: {summary_model}")
    elif summary_model:
        logger.info(f"Using configured model: {summary_model}")
    else:
        raise RuntimeError("No model specified and model detection failed")
    
    logger.info("Summary client initialized")
    
    # Send startup warm-up request to verify model availability
    startup_success = await STATE.summary_client.startup_summary()
    if startup_success:
        logger.info("Startup warm-up request successful - model is responsive")
    else:
        logger.warning("Startup warm-up request failed - model may not be available")

    # Initialize diarization client
    STATE.diarization_client = DiarizationClient(
        hf_token=os.getenv("HF_TOKEN")
    )
    await STATE.diarization_client.initialize()
    await STATE.diarization_client.start()
    logger.info("Diarization client started")
    # Note: Polling task is started in on_stream_start(), not here


def _append_audio(frame: AudioFrame):
    """Append audio samples from frame into state's rolling mono buffer."""
    assert STATE is not None
    samples = frame.samples
    if samples.ndim == 2:
        # [channels, samples] or [samples, channels]
        mono = samples.mean(axis=0) if samples.shape[0] < samples.shape[1] else samples.mean(axis=1)
    else:
        mono = samples
    if mono.dtype == np.int16:
        mono = mono.astype(np.float32) / 32768.0
    elif mono.dtype == np.int32:
        mono = mono.astype(np.float32) / 2147483648.0
    else:
        mono = mono.astype(np.float32)

    start_ts = _frame_time_seconds(frame.timestamp, frame.time_base)
    if STATE.buffer_start_ts is None:
        STATE.buffer_start_ts = start_ts
        STATE.buffer_rate = frame.rate
    # Set stream media time origin on first frame (for stream-relative timestamps)
    if STATE.stream_start_media_ts is None:
        STATE.stream_start_media_ts = start_ts
        logger.info(f"Stream media time origin set to {start_ts:.3f}s")
    STATE.audio_buffer = np.concatenate([STATE.audio_buffer, mono])
    
    # Also append to diarization buffer for 6-second chunks
    STATE.diarization_audio_buffer = np.concatenate([STATE.diarization_audio_buffer, mono])
    if STATE.diarization_buffer_start_ts is None:
        STATE.diarization_buffer_start_ts = start_ts


def _buffer_duration_seconds() -> float:
    assert STATE is not None
    if STATE.buffer_rate is None or STATE.buffer_start_ts is None:
        return 0.0
    return len(STATE.audio_buffer) / float(STATE.buffer_rate)


def _diarization_buffer_duration_seconds() -> float:
    """Get duration of diarization audio buffer in seconds."""
    assert STATE is not None
    if STATE.buffer_rate is None or STATE.diarization_buffer_start_ts is None:
        return 0.0
    return len(STATE.diarization_audio_buffer) / float(STATE.buffer_rate)


def _write_wav(samples: np.ndarray, sample_rate: int) -> str:
    """Write audio samples to a temporary WAV file."""
    path = tempfile.mktemp(suffix=".wav")
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


def _mark_transcribed(audio_path: str):
    """Mark temp file as transcribed, check if can be deleted."""
    if STATE is None or audio_path not in STATE.pending_temp_files:
        return
    STATE.pending_temp_files[audio_path]['transcribed'] = True
    _check_cleanup(audio_path)


def _mark_diarized(audio_path: str):
    """Mark temp file as diarized, check if can be deleted."""
    if STATE is None or audio_path not in STATE.pending_temp_files:
        return
    STATE.pending_temp_files[audio_path]['diarized'] = True
    _check_cleanup(audio_path)
    
    # Also clean up individual temp files that were combined
    if STATE is not None and hasattr(STATE, 'diarization_temp_files'):
        for temp_path in STATE.diarization_temp_files:
            if temp_path in STATE.pending_temp_files:
                STATE.pending_temp_files[temp_path]['diarized'] = True
                _check_cleanup(temp_path)


def _check_cleanup(audio_path: str):
    """Delete temp file when both transcribing and diarization are complete."""
    if STATE is None or audio_path not in STATE.pending_temp_files:
        return
    state = STATE.pending_temp_files[audio_path]
    if state['transcribed'] and state['diarized']:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        del STATE.pending_temp_files[audio_path]




async def _send_speakers_message(segments: list[SpeakerSegment], window_start_ts: float, window_end_ts: float):
    """Send speakers message to client (client saves to database)."""
    if PROCESSOR is None or not segments:
        return
    
    # Convert timestamps to milliseconds and add window start offset
    adjusted_segments = []
    for seg in segments:
        adjusted_segments.append({
            "start_ms": int((window_start_ts + seg.start) * 1000),
            "end_ms": int((window_start_ts + seg.end) * 1000),
            "speaker": seg.speaker,
            "confidence": seg.confidence,
            "alt_speakers": seg.alt_speakers
        })
    
    speakers_payload = {
        "type": "speakers",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "timing": {
            "media_window_start_ms": int(window_start_ts * 1000),
            "media_window_end_ms": int(window_end_ts * 1000)
        },
        "segments": adjusted_segments
    }
    
    try:
        await PROCESSOR.send_data(json.dumps(speakers_payload))
        logger.debug(f"Sent speakers message with {len(segments)} speaker segments")
    except Exception as e:
        logger.warning(f"Failed to send speakers message: {e}")


async def _handle_diarization_result(result: DiarizationResult):
    """Handle diarization result and send to client."""
    logger.debug(f"Handling diarization result: {result.request_id}")
    
    # Mark temp file as diarized
    _mark_diarized(result.audio_path)
    
    # Remove from in-flight tracking in diarization client
    if not STATE is None:
        STATE.diarization_client.remove_in_flight_request(result.request_id)

    if result.error:
        logger.error(f"Diarization error: {result.error}")
        return
    
    # Send speakers message to client
    if result.segments and STATE is not None:
        # Use stored timestamps for the 6-second window
        if result.request_id in STATE.diarization_window_timestamps:
            window_start_ts, window_end_ts = STATE.diarization_window_timestamps[result.request_id]
            del STATE.diarization_window_timestamps[result.request_id]
        else:
            # Fallback to buffer-based calculation
            sr = STATE.buffer_rate
            total_len = len(STATE.audio_buffer)
            win_len = int(max(1, STATE.window_seconds * sr))
            start_idx = max(0, total_len - win_len)
            window_start_ts = STATE.buffer_start_ts + (start_idx / float(sr)) if STATE.buffer_start_ts else 0
            window_end_ts = window_start_ts + (win_len / float(sr))
        
        await _send_speakers_message(result.segments, window_start_ts, window_end_ts)


async def _poll_diarization_results():
    """Background task to poll diarization results from the separate process."""
    logger.info("Starting diarization result polling")
    poll_count = 0
    while STATE is not None and STATE.diarization_client is not None and not STATE.stop_requested:
        is_running = STATE.diarization_client.is_running
        if not is_running:
            logger.info(f"Diarization process not running, skipping poll cycle {poll_count}")
            await asyncio.sleep(0.1)
            continue
        poll_count += 1
        if poll_count % 50 == 0:
            logger.info(f"Diarization polling cycle {poll_count}")
        result = await STATE.diarization_client.get_result(timeout=0.05)
        if result is not None:
            logger.debug(f"Got diarization result: {result.request_id}")
            await _handle_diarization_result(result)
        else:
            logger.debug(f"No diarization result in poll cycle {poll_count}")
        await asyncio.sleep(0.1)
    logger.info("Stopping diarization result polling")


def _pull_transcription_samples(now_ts: float) -> Optional[tuple]:
    """Pull samples from audio buffer for transcription. Thread-safe synchronous operation.
    
    Returns:
        Tuple of (window_samples, window_start_ts, window_end_ts) or None if buffer too short.
    """
    assert STATE is not None and STATE.whisper_client is not None
    if STATE.buffer_rate is None or STATE.buffer_start_ts is None:
        return None
    
    sr = STATE.buffer_rate
    total_len = len(STATE.audio_buffer)
    win_len = int(max(1, STATE.window_seconds * sr))
    if total_len < win_len:
        return None
    
    start_idx = total_len - win_len
    window_samples = STATE.audio_buffer[start_idx:]
    window_start_ts = STATE.buffer_start_ts + (start_idx / float(sr))
    window_end_ts = window_start_ts + STATE.window_seconds
    
    # Update buffer - remove processed samples BEFORE spawning async task
    keep_len = int(max(0, STATE.overlap_seconds * sr))
    STATE.audio_buffer = STATE.audio_buffer[-keep_len:] if keep_len > 0 else np.zeros((0,), dtype=np.float32)
    STATE.buffer_start_ts = window_end_ts - (keep_len / float(sr))
    
    return (window_samples, window_start_ts, window_end_ts)


async def _process_transcription_async(window_samples: np.ndarray, window_start_ts: float, window_end_ts: float):
    """Process transcription asynchronously. Receives samples already pulled from buffer."""
    assert STATE is not None and STATE.whisper_client is not None
    
    sr = STATE.buffer_rate
    win_len = int(STATE.window_seconds * sr)
    
    temp_path = _write_wav(window_samples, sr)
    
    try:
        # Track temp file for cleanup
        STATE.pending_temp_files[temp_path] = {
            'transcribed': False,
            'diarized': False
        }
        
        # Transcribe audio window - get transcription_window_id from whisper's internal counter
        start = time.perf_counter()
        transcription_window_id, segments = await STATE.whisper_client.transcribe_audio(temp_path)
        logger.info(f"Transcription of window [{window_start_ts:.3f}s - {window_end_ts:.3f}s] took {time.perf_counter() - start:.2f}s, got {len(segments)} segments (transcription_window_id={transcription_window_id})")
        
        # Mark as transcribed
        _mark_transcribed(temp_path)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Cleanup on error
        _mark_transcribed(temp_path)
        _mark_diarized(temp_path)
        return

    STATE.current_segments = segments

    # Calculate end timestamp before queuing summary work
    end_ts = window_start_ts + (win_len / float(sr))

    # Queue summary work for async processing (non-blocking)
    # Skip logic is handled by the summary worker, not here
    if STATE.summary_client is not None:
        # Skip if all segments are blank
        if not _has_any_meaningful_segments(segments):
            logger.debug(f"Skipping summary work - all segments blank for window [{int(window_start_ts*1000)}ms - {int(end_ts*1000)}ms]")
        else:
            try:
                # Build segments payload
                segments_payload = _build_segments_payload(segments, window_start_ts)
                
                # Put work on queue for background processing
                # Worker will handle skip logic and content type detection
                # transcription_window_id comes from whisper's internal counter
                STATE.summary_client.queue_segments(segments_payload, transcription_window_id, window_start_ts, end_ts)
            except Exception as e:
                logger.error(f"Failed to queue summary work: {e}")

    if PROCESSOR is not None:
        # Skip sending if all segments are blank
        if not _has_any_meaningful_segments(segments):
            logger.debug(f"Skipping transcript - all segments blank for window [{int(window_start_ts*1000)}ms - {int(end_ts*1000)}ms]")
            return

        try:
            # Build structured segments payload for Streamplaces
            segments_payload = _build_segments_payload(segments, window_start_ts)
            
            payload = {
                "type": "transcript",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "transcription_window_id": transcription_window_id,
                "timing": {
                    "media_window_start_ms": int(window_start_ts * 1000),
                    "media_window_end_ms": int(end_ts * 1000)
                },
                "segments": segments_payload,
                "stats": {
                    "audio_duration_ms": int((end_ts - window_start_ts) * 1000)
                }
            }
            await PROCESSOR.send_data(json.dumps(payload))
        except Exception as e:
            logger.warning(f"Failed to send data payload: {e}")


def _pull_diarization_samples(allow_partial: bool = False) -> Optional[tuple]:
    """Pull samples from diarization audio buffer. Thread-safe synchronous operation.
    
    Args:
        allow_partial: If True, process partial windows (< 6 seconds). Used during shutdown.
    
    Returns:
        Tuple of (window_samples, window_start_ts, window_end_ts) or None if buffer too short.
    """
    assert STATE is not None
    if STATE.buffer_rate is None or STATE.diarization_buffer_start_ts is None:
        return None
    
    sr = STATE.buffer_rate
    diarization_window_seconds = 6.0
    win_len = int(diarization_window_seconds * sr)
    total_len = len(STATE.diarization_audio_buffer)
    
    if total_len < win_len:
        if allow_partial and total_len > 0:
            logger.info(f"Processing partial diarization window: {total_len} samples ({total_len/sr:.2f}s)")
        else:
            return None
    
    # Take the first 6 seconds from the buffer
    window_samples = STATE.diarization_audio_buffer[:win_len]
    window_start_ts = STATE.diarization_buffer_start_ts
    window_end_ts = window_start_ts + diarization_window_seconds
    
    # Update buffer - remove processed audio BEFORE spawning async task
    STATE.diarization_audio_buffer = STATE.diarization_audio_buffer[win_len:]
    STATE.diarization_buffer_start_ts = window_end_ts
    
    return (window_samples, window_start_ts, window_end_ts)


async def _process_diarization_async(window_samples: np.ndarray, window_start_ts: float, window_end_ts: float, allow_partial: bool = False):
    """Process diarization asynchronously. Receives samples already pulled from buffer."""
    assert STATE is not None and STATE.diarization_client is not None
    
    sr = STATE.buffer_rate
    
    temp_path = _write_wav(window_samples, sr)
    diarization_request_id = str(uuid.uuid4())
    
    logger.info(f"Diarization: preparing window [{window_start_ts:.3f}s - {window_end_ts:.3f}s], temp_path={temp_path}")
    
    try:
        # Track temp file for cleanup
        STATE.pending_temp_files[temp_path] = {
            'transcribed': False,
            'diarized': False
        }
        
        # Store timestamps for this request
        STATE.diarization_window_timestamps[diarization_request_id] = (window_start_ts, window_end_ts)
        
        # Add request to in-flight tracking in diarization client
        STATE.diarization_client.add_in_flight_request(diarization_request_id)
        
        # Send to diarization process
        logger.debug(f"Diarization: calling process_audio with request_id={diarization_request_id}")
        await STATE.diarization_client.process_audio(temp_path, diarization_request_id)
        logger.info(f"Diarization: successfully sent window [{window_start_ts:.3f}s - {window_end_ts:.3f}s] for processing")
        
    except Exception as e:
        logger.error(f"Diarization error in _process_diarization_async: {type(e).__name__}: {e}")
        logger.error(f"Diarization error details - temp_path={temp_path}, request_id={diarization_request_id}")
        _mark_diarized(temp_path)


def _build_segments_payload(segments: list[TranscriptionSegment], window_start_ts: float) -> list[dict]:
    """Build structured segments payload for Streamplace compatibility.
    
    Args:
        segments: List of TranscriptionSegment from Whisper
        window_start_ts: Window start timestamp in seconds (stream-relative)
    
    Returns:
        List of segment dicts with start_ms/end_ms in stream-relative time
    """
    segments_payload = []
    for seg in segments:
        # Convert to absolute stream-relative milliseconds
        seg_ms_start = int((window_start_ts + seg.start) * 1000)
        seg_ms_end = int((window_start_ts + seg.end) * 1000)
        
        # Build stable ID from timing and text hash
        seg_id = f"{seg_ms_start}-{hash(seg.text) & 0xFFFFFFFF:08x}"
        
        # Build word-level timestamps if available
        words_payload = []
        if seg.words:
            for w in seg.words:
                words_payload.append({
                    "start_ms": int((window_start_ts + w.start) * 1000),
                    "end_ms": int((window_start_ts + w.end) * 1000),
                    "text": w.text
                })
        
        segment_dict: dict = {
            "id": seg_id,
            "start_ms": seg_ms_start,
            "end_ms": seg_ms_end,
            "text": seg.text,
            "words": words_payload
        }
        if getattr(seg, "speaker", None) is not None:
            segment_dict["speaker"] = seg.speaker
        segments_payload.append(segment_dict)
    
    return segments_payload


async def process_audio_async(frame: AudioFrame):
    """Process audio frame and run transcription/diarization."""
    if STATE is None:
        logger.warning("process_audio_async called but STATE is None")
        return None
    
    logger.debug(f"process_audio_async: timestamp={frame.timestamp}, time_base={frame.time_base}, samples_shape={frame.samples.shape}")
    
    if STATE.shutdown_requested:
        return None  # Ignore new audio during shutdown

    _append_audio(frame)
    
    buffer_dur = _buffer_duration_seconds()
    logger.debug(f"process_audio_async: buffer_dur={buffer_dur:.3f}s, window_seconds={STATE.window_seconds}")
    
    if buffer_dur >= STATE.window_seconds:
        now_ts = _frame_time_seconds(frame.timestamp, frame.time_base)
        logger.debug(f"process_audio_async: pulling transcription samples")
        # Pull samples synchronously (awaited for thread-safe buffer access)
        pulled = _pull_transcription_samples(now_ts)
        if pulled is not None:
            window_samples, window_start_ts, window_end_ts = pulled
            # Process asynchronously in background
            asyncio.create_task(_process_transcription_async(window_samples, window_start_ts, window_end_ts))
    
    # Run diarization on 6-second chunks (non-blocking)
    diarization_dur = _diarization_buffer_duration_seconds()
    logger.debug(f"process_audio_async: diarization_dur={diarization_dur:.3f}s")
    
    if diarization_dur >= 6.0:
        logger.debug(f"process_audio_async: pulling diarization samples")
        # Pull samples synchronously (awaited for thread-safe buffer access)
        pulled = _pull_diarization_samples()
        if pulled is not None:
            window_samples, window_start_ts, window_end_ts = pulled
            # Process asynchronously in background
            asyncio.create_task(_process_diarization_async(window_samples, window_start_ts, window_end_ts))
    
    return None


async def _handle_graceful_shutdown():
    """
    Handle graceful shutdown via update_params.

    This is run in a async task so should run each step and wait for completion.
    1. Signal stop - no new audio to buffers
    2. Flush remaining buffered audio
    3. Send shutdown signals to workers
    4. Wait for completion with timeout
    5. Send shutdown complete or timeout signal to client

    """
    logger.info("Starting graceful shutdown process")
    
    # Phase 1: Signal stop - no new audio to buffers
    STATE.shutdown_requested = True
    
    # Phase 2: Flush remaining buffered audio
    await _flush_audio_buffers()
    
    # Phase 3: Stop summary workers gracefully
    if STATE.summary_client is not None:
        await STATE.summary_client.stop()
    
    STATE.diarization_client.send_shutdown_signal()
    
    # Phase 4: Wait for completion with timeout
    start_time = time.time()
    timeout = 180.0
    
    while time.time() - start_time < timeout:
        # Summary workers are already stopped in Phase 3 via SummaryClient.stop()
        all_workers_done = True
        
        # Check diarization
        diarization_idle = STATE.diarization_client.is_idle()
        
        if all_workers_done and diarization_idle:
            STATE.shutdown_completed = True
            await PROCESSOR.send_data(json.dumps({
                "type": "shutdown_complete",
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }))
            logger.info("Graceful shutdown completed successfully")
            return
        
        await asyncio.sleep(0.1)
    
    # Timeout - send timeout signal
    await PROCESSOR.send_data(json.dumps({
        "type": "shutdown_timeout",
        "pending_summaries": STATE.summary_client.get_pending_count(),
        "pending_diarizations": STATE.diarization_client.get_pending_count()
    }))
    logger.warning("Graceful shutdown timed out")


async def _flush_audio_buffers():
    """Process any remaining audio in buffers as final requests."""
    logger.info("Flushing remaining audio buffers")
    
    # Flush transcription buffer if has data
    if len(STATE.audio_buffer) > 0 and STATE.buffer_rate is not None:
        now_ts = STATE.buffer_start_ts + _buffer_duration_seconds()
        pulled = _pull_transcription_samples(now_ts)
        if pulled is not None:
            window_samples, window_start_ts, window_end_ts = pulled
            await _process_transcription_async(window_samples, window_start_ts, window_end_ts)
    
    # Flush diarization buffer if has data - allow partial windows during shutdown
    if len(STATE.diarization_audio_buffer) > 0:
        pulled = _pull_diarization_samples(allow_partial=True)
        if pulled is not None:
            window_samples, window_start_ts, window_end_ts = pulled
            await _process_diarization_async(window_samples, window_start_ts, window_end_ts, allow_partial=True)


async def update_params(params: dict):
    """Update runtime parameters for the transcription pipeline."""
    if STATE is None:
        return
    
    # Handle graceful shutdown request - fire and forget to avoid blocking HTTP handler
    if params.get("shutdown"):
        asyncio.create_task(_handle_graceful_shutdown())
        return
    
    # Whisper model parameters
    if "whisper_model" in params:
        STATE.whisper_model = str(params["whisper_model"])
    if "whisper_language" in params:
        STATE.whisper_language = params["whisper_language"]
    if "whisper_device" in params:
        STATE.whisper_device = str(params["whisper_device"])
    if "compute_type" in params:
        STATE.compute_type = str(params["compute_type"])
    # If Whisper model parameters changed, reinitialize the client
    whisper_params_changed = any(param in params for param in
                                ["whisper_model", "whisper_language", "whisper_device", "compute_type"])
    
    if whisper_params_changed and STATE.whisper_client is not None:
        
        # Reinitialize WhisperClient with new parameters
        STATE.whisper_client = WhisperClient(
            model_size=STATE.whisper_model,
            device=STATE.whisper_device,
            compute_type=STATE.compute_type,
            language=STATE.whisper_language,
        )
        await STATE.whisper_client.initialize()

    # Audio processing parameters
    if "audio_sample_rate" in params:
        STATE.audio_sample_rate = int(params["audio_sample_rate"])
    if "chunk_window" in params:
        STATE.window_seconds = float(params["chunk_window"])
    if "chunk_overlap" in params:
        STATE.overlap_seconds = float(params["chunk_overlap"])
    
    # Summary parameters
    if "summary_base_url" in params or "summary_api_key" in params or "summary_model" in params:
        if STATE.summary_client is not None:
            STATE.summary_client.update_params(
                reasoning_base_url=params.get("summary_base_url"),
                reasoning_api_key=params.get("summary_api_key"),
                reasoning_model=params.get("summary_model"),
            )
    if "rapid_base_url" in params or "rapid_api_key" in params or "rapid_model" in params:
        if STATE.summary_client is not None:
            STATE.summary_client.update_params(
                rapid_base_url=params.get("rapid_base_url"),
                rapid_api_key=params.get("rapid_api_key"),
                rapid_model=params.get("rapid_model"),
            )
    # Initial summary delay parameter
    if "initial_summary_delay_seconds" in params:
        delay = float(params["initial_summary_delay_seconds"])
        if STATE.summary_client is not None:
            STATE.summary_client.initial_summary_delay_seconds = delay
            logger.info(f"Updated initial_summary_delay_seconds to {delay}s")
    
    # Content type override parameter (user control)
    if "content_type_override" in params:
        override = params["content_type_override"]
        if STATE.summary_client is not None:
            if override is None or override == "":
                STATE.summary_client.set_content_type_override(None)
                logger.info("Content type user override cleared")
            else:
                STATE.summary_client.set_content_type_override(override)
                logger.info(f"Content type user override set to: {override}")
    
    # Content type auto-detection parameter (for initial detection)
    if "content_type" in params and "content_type_confidence" in params:
        ct = params["content_type"]
        conf = params.get("content_type_confidence", 0.0)
        if STATE.summary_client is not None:
            STATE.summary_client.set_content_type(ct, conf, "AUTO_DETECTED")
            logger.info(f"Content type auto-detected: {ct} (confidence: {conf:.2f})")

async def on_stream_start(params: dict):
    global STATE
    try:
        logger.info(f"Stream start request received, resetting media time origin")
        # Reset stream state for new stream
        if STATE is not None:
            STATE.stream_start_media_ts = None  # Will be set on first audio frame
            STATE.audio_buffer = np.zeros((0,), dtype=np.float32)
            STATE.buffer_start_ts = None
            STATE.buffer_rate = None
            STATE.current_segments = []
            STATE.current_srt = ""
            STATE.pending_temp_files = {}
            STATE.diarization_counter = 0  # Reset diarization counter for new stream
            STATE.diarization_temp_files = []  # Reset temp files list for new stream
            STATE.diarization_window_timestamps = {}  # Reset timestamps dict for new stream
            STATE.diarization_audio_buffer = np.zeros((0,), dtype=np.float32)  # Reset diarization buffer
            STATE.diarization_buffer_start_ts = None
            
            # Reset summary client state (includes window tracking state)
            if STATE.summary_client is not None:
                STATE.summary_client.reset()
            
            # Reset whisper client state for new stream (clear transcription ID counter)
            if STATE.whisper_client is not None:
                STATE.whisper_client.reset()
            
            # Reset diarization client state for new stream (clear speaker memory)
            if STATE.diarization_client is not None:
                STATE.diarization_client.reset()
            
            # Reset shutdown state for new stream
            STATE.shutdown_requested = False
            STATE.shutdown_completed = False
            
            # Clear summary queues to prevent stale data from previous stream
            if STATE.summary_client is not None:
                STATE.summary_client.clear()
            
            # Start summary worker for this stream
            if STATE.summary_client is not None:
                await STATE.summary_client.start()
            
            # Start diarization polling task for this stream
            if STATE.diarization_client is not None:
                STATE.diarization_poll_task = asyncio.create_task(_poll_diarization_results())
                logger.info("Diarization polling task started for stream")
        # Forward to the pytrickle server's actual handler
    except Exception as e:
        logger.error(f"Error in stream start: {e}")
        return web.Response(text=str(e), status=500)


async def on_stream_stop():
    global STATE
    if STATE is None:
        return
    
    # Record stop request time
    stop_requested_at = datetime.now(timezone.utc).isoformat()
    logger.info(f"Stop request received at {stop_requested_at}")
    
    # Signal workers to stop accepting new work via SummaryClient
    if STATE.summary_client is not None:
        STATE.summary_client._stop_requested = True
        logger.info("Summary client stop requested")
        
        # Stop the workers gracefully
        await STATE.summary_client.stop(timeout=5.0)
        workers_completed_at = datetime.now(timezone.utc).isoformat()
        wait_duration_seconds = (datetime.fromisoformat(workers_completed_at.replace('+00:00', '')) -
                               datetime.fromisoformat(stop_requested_at.replace('+00:00', ''))).total_seconds()
        logger.info(f"All summary workers completed in {wait_duration_seconds:.2f}s")
    else:
        workers_completed_at = stop_requested_at
        wait_duration_seconds = 0.0
    
    # Send worker stop timing payload before closing
    if PROCESSOR is not None:
        try:
            timing_payload = {
                "type": "summary_workers_stopped",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "timing": {
                    "stop_requested_at": stop_requested_at,
                    "workers_completed_at": workers_completed_at or stop_requested_at,
                    "wait_duration_seconds": round(wait_duration_seconds, 2)
                }
            }
            await PROCESSOR.send_data(json.dumps(timing_payload))
            logger.info(f"Sent summary workers stop timing: wait_duration={wait_duration_seconds:.2f}s")
        except Exception as e:
            logger.error(f"Error sending worker stop timing: {e}")
    
    # Optionally send final transcript packet with remaining segments
    if PROCESSOR is not None and STATE.current_segments:
        try:
            segments_payload = _build_segments_payload(STATE.current_segments, 0.0)
            payload = {
                "type": "transcript_final",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "segments": segments_payload,
            }
            await PROCESSOR.send_data(json.dumps(payload))
        except Exception:
            pass
    
    # Reset buffers
    STATE.audio_buffer = np.zeros((0,), dtype=np.float32)
    STATE.buffer_start_ts = None
    STATE.buffer_rate = None
    STATE.stream_start_media_ts = None
    STATE.pending_temp_files = {}
    
    
    # Reset summary client state (clear windows, insights, timestamps)
    if STATE.summary_client is not None:
        STATE.summary_client.reset()
        logger.info("Summary client state reset")
    
    # Reset diarization client state (keep process running)
    if STATE.diarization_client is not None:
        STATE.diarization_client.reset()
    
    # Cancel the polling task
    if STATE.diarization_poll_task is not None:
        STATE.diarization_poll_task.cancel()
        STATE.diarization_poll_task = None
        logger.info("Diarization polling task cancelled")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_model())

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting StreamProcessor on port %d", port)
    # Instantiate global processor so helper funcs can send data
    PROCESSOR = StreamProcessor(
        audio_processor=process_audio_async,
        param_updater=update_params,
        on_stream_start=on_stream_start,
        on_stream_stop=on_stream_stop,
        name="transcriber",
        port=port,
    )


    PROCESSOR.run()
