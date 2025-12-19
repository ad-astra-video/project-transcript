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
from typing import Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from transcription.whisper_client import WhisperClient, TranscriptionSegment, WordTimestamp
from transcription.srt_generator import SRTGenerator
from pytrickle import StreamProcessor
from pytrickle import AudioFrame
import numpy as np
import tempfile
import wave

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
        self.overlap_seconds: float = 1.0
        self.audio_buffer: np.ndarray = np.zeros((0,), dtype=np.float32)  # mono float32 [-1,1]
        self.buffer_start_ts: Optional[float] = None  # seconds
        self.buffer_rate: Optional[int] = None

        # Transcript state
        self.current_segments: list[TranscriptionSegment] = []
        self.current_srt: str = ""
        self.last_sent_data_time: float = 0.0
        
        # Stream media time origin (reset on /stream/start)
        self.stream_start_media_ts: Optional[float] = None


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


def _buffer_duration_seconds() -> float:
    assert STATE is not None
    if STATE.buffer_rate is None or STATE.buffer_start_ts is None:
        return 0.0
    return len(STATE.audio_buffer) / float(STATE.buffer_rate)


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
        
        segments_payload.append({
            "id": seg_id,
            "start_ms": seg_ms_start,
            "end_ms": seg_ms_end,
            "text": seg.text,
            "words": words_payload
        })
    
    return segments_payload


def _write_wav(samples: np.ndarray, sample_rate: int) -> str:
    path = tempfile.mktemp(suffix=".wav")
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return path


async def _transcribe_current_window(now_ts: float):
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

    temp_path = _write_wav(window_samples, sr)
    try:
        segments = await STATE.whisper_client.transcribe_audio(temp_path, int(window_start_ts * 1000))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    STATE.current_segments = segments
    # Use absolute/global timestamps in SRT by offsetting with window_start_ts
    STATE.current_srt = STATE.srt_generator.generate_srt(
        segments,
        segment_idx=0,
        base_offset_seconds=window_start_ts,
    )

    keep_len = int(max(0, STATE.overlap_seconds * sr))
    STATE.audio_buffer = STATE.audio_buffer[-keep_len:] if keep_len > 0 else np.zeros((0,), dtype=np.float32)
    end_ts = window_start_ts + (win_len / float(sr))
    STATE.buffer_start_ts = end_ts - (keep_len / float(sr))

    if PROCESSOR is not None:
        try:
            # Build structured segments payload for Streamplace
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


async def process_audio_async(frame: AudioFrame):
    if STATE is None:
        return None
    _append_audio(frame)
    if _buffer_duration_seconds() >= STATE.window_seconds:
        now_ts = _frame_time_seconds(frame.timestamp, frame.time_base)
        await _transcribe_current_window(now_ts)
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

async def handle_stream_start(request: web.Request) -> web.Response:
    """Handle /stream/start endpoint - resets stream media time origin and forwards to pytrickle."""
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
        # Forward to the pytrickle server's actual handler
        return await PROCESSOR.server._handle_start_stream(request)
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
        on_stream_stop=on_stream_stop,
        name="transcriber",
        port=port,
    )

    # Add custom routes for orchestrator compatibility
    PROCESSOR.server.app.router.add_post('/stream/start', handle_stream_start)
    PROCESSOR.server.app.router.add_post('/stream/stop', on_stream_stop)
    logger.info(f"Added /stream/start and /stream/stop endpoints")

    PROCESSOR.run()
