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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict
import time

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from transcription.diarization import DiarizationProcess, DiarizationResult, SpeakerSegment
from transcription.whisper_client import WhisperClient, TranscriptionSegment, WordTimestamp
from transcription.srt_generator import SRTGenerator
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
        self.srt_generator: SRTGenerator = SRTGenerator()

        # Whisper config
        self.whisper_model: str = "large"
        self.whisper_language: Optional[str] = None
        self.whisper_device: str = "cuda"
        self.compute_type: str = "float16"

        # Audio processing
        self.audio_sample_rate: int = 16000
        self.window_seconds: float = 3.0
        self.overlap_seconds: float = 2.0
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
        self.diarization_process: Optional[DiarizationProcess] = None
        self.diarization_poll_task: Optional[asyncio.Task] = None
        self.diarization_counter: int = 0  # Track transcribe calls for diarization scheduling
        self.diarization_temp_files: list[dict] = []  # Track temp files with timestamps for combining
        self.diarization_window_timestamps: Dict[str, tuple[float, float]] = {}  # request_id -> (start_ts, end_ts)
        self.diarization_audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)  # Separate buffer for diarization audio
        self.diarization_buffer_start_ts: Optional[float] = None  # Start timestamp for diarization buffer
        
        # Temp file tracking for cleanup (when both transcribe and diarize are done)
        self.pending_temp_files: Dict[str, Dict] = {}


# Global state and processor reference
STATE: Optional[TranscriberState] = None
PROCESSOR: Optional[StreamProcessor] = None


def _frame_time_seconds(timestamp: int, time_base) -> float:
    try:
        return float(timestamp) * float(time_base)
    except Exception:
        return float(timestamp)


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

    # Init model
    STATE.whisper_client = WhisperClient(
        model_size=STATE.whisper_model,
        device=STATE.whisper_device,
        compute_type=STATE.compute_type,
        language=STATE.whisper_language,
    )
    await STATE.whisper_client.initialize()

    # Initialize diarization in separate process
    STATE.diarization_process = DiarizationProcess(
        hf_token=os.getenv("HF_TOKEN")
    )
    await STATE.diarization_process.start()
    logger.info("Diarization process started")
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


async def _run_diarization_on_window():
    """Run diarization on accumulated 6-second audio window."""
    assert STATE is not None and STATE.diarization_process is not None
    if STATE.buffer_rate is None or STATE.diarization_buffer_start_ts is None:
        return
    
    sr = STATE.buffer_rate
    diarization_window_seconds = 6.0
    win_len = int(diarization_window_seconds * sr)
    total_len = len(STATE.diarization_audio_buffer)
    
    if total_len < win_len:
        return
    
    # Take the first 6 seconds from the buffer
    window_samples = STATE.diarization_audio_buffer[:win_len]
    window_start_ts = STATE.diarization_buffer_start_ts
    window_end_ts = window_start_ts + diarization_window_seconds
    
    temp_path = _write_wav(window_samples, sr)
    diarization_request_id = str(uuid.uuid4())
    
    try:
        # Track temp file for cleanup
        STATE.pending_temp_files[temp_path] = {
            'transcribed': False,
            'diarized': False
        }
        
        # Store timestamps for this request
        STATE.diarization_window_timestamps[diarization_request_id] = (window_start_ts, window_end_ts)
        
        # Send to diarization process
        await STATE.diarization_process.process_audio(temp_path, diarization_request_id)
        logger.info(f"Diarization on window [{window_start_ts:.3f}s - {window_end_ts:.3f}s]")
        
        # Remove processed audio from buffer
        STATE.diarization_audio_buffer = STATE.diarization_audio_buffer[win_len:]
        STATE.diarization_buffer_start_ts = window_end_ts
        
    except Exception as e:
        logger.error(f"Diarization error: {e}")
        _mark_diarized(temp_path)


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
    while STATE is not None and STATE.diarization_process is not None:
        is_running = STATE.diarization_process.is_running
        if not is_running:
            logger.info(f"Diarization process not running, skipping poll cycle {poll_count}")
            await asyncio.sleep(0.1)
            continue
        poll_count += 1
        if poll_count % 50 == 0:
            logger.info(f"Diarization polling cycle {poll_count}")
        result = await STATE.diarization_process.get_result(timeout=0.05)
        if result is not None:
            logger.debug(f"Got diarization result: {result.request_id}")
            await _handle_diarization_result(result)
        else:
            logger.debug(f"No diarization result in poll cycle {poll_count}")
        await asyncio.sleep(0.1)
    logger.info("Stopping diarization result polling")


async def _transcribe_current_window(now_ts: float):
    """Transcribe the current audio window."""
    assert STATE is not None and STATE.whisper_client is not None
    if STATE.buffer_rate is None or STATE.buffer_start_ts is None:
        return

    sr = STATE.buffer_rate
    total_len = len(STATE.audio_buffer)
    win_len = int(max(1, STATE.window_seconds * sr))
    if total_len < win_len:
        return

    start_idx = total_len - win_len
    window_samples = STATE.audio_buffer[start_idx:]
    window_start_ts = STATE.buffer_start_ts + (start_idx / float(sr))
    window_end_ts = window_start_ts + STATE.window_seconds

    temp_path = _write_wav(window_samples, sr)
    
    try:
        # Track temp file for cleanup
        STATE.pending_temp_files[temp_path] = {
            'transcribed': False,
            'diarized': False
        }
        
        # Transcribe audio window
        start = time.perf_counter()
        segments = await STATE.whisper_client.transcribe_audio(temp_path, int(window_start_ts * 1000))
        logger.info(f"Transcription of window [{window_start_ts:.3f}s - {window_end_ts:.3f}s] took {time.perf_counter() - start:.2f}s, got {len(segments)} segments")
        
        # Mark as transcribed
        _mark_transcribed(temp_path)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # Cleanup on error
        _mark_transcribed(temp_path)
        _mark_diarized(temp_path)
        return

    STATE.current_segments = segments

    keep_len = int(max(0, STATE.overlap_seconds * sr))
    STATE.audio_buffer = STATE.audio_buffer[-keep_len:] if keep_len > 0 else np.zeros((0,), dtype=np.float32)
    end_ts = window_start_ts + (win_len / float(sr))
    STATE.buffer_start_ts = end_ts - (keep_len / float(sr))

    if PROCESSOR is not None:
        try:
            # Build structured segments payload for Streamplaces
            segments_payload = _build_segments_payload(segments, window_start_ts)
            
            payload = {
                "type": "transcript",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
            logger.info(f"Sent transcript with {len(segments_payload)} segments for window [{int(window_start_ts*1000)}ms - {int(end_ts*1000)}ms]")
        except Exception as e:
            logger.warning(f"Failed to send data payload: {e}")


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
    if STATE is None:
        return None
    _append_audio(frame)
    if _buffer_duration_seconds() >= STATE.window_seconds:
        now_ts = _frame_time_seconds(frame.timestamp, frame.time_base)
        await _transcribe_current_window(now_ts)
    # Run diarization on 6-second chunks
    if _diarization_buffer_duration_seconds() >= 6.0:
        await _run_diarization_on_window()
    return None


async def update_params(params: dict):
    """Update runtime parameters for the transcription pipeline."""
    if STATE is None:
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
    
    # Audio processing parameters
    if "audio_sample_rate" in params:
        STATE.audio_sample_rate = int(params["audio_sample_rate"])
    if "chunk_window" in params:
        STATE.window_seconds = float(params["chunk_window"])
    if "chunk_overlap" in params:
        STATE.overlap_seconds = float(params["chunk_overlap"])
    
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


from aiohttp import web        

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
            
            # Start diarization polling task for this stream
            if STATE.diarization_process is not None:
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
    
    # Reset diarization process state (keep process running)
    if STATE.diarization_process is not None:
        await STATE.diarization_process.reset()
    
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
