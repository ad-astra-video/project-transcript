"""
Faster-whisper client for audio transcription.
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment with timing information."""
    start: float  # Start time in seconds (segment-relative)
    end: float    # End time in seconds (segment-relative)
    text: str     # Transcribed text
    
    
class WhisperClient:
    """Client for faster-whisper transcription."""
    
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str ="float16", language: Optional[str] = None):
        """
        Initialize the whisper client.
        
        Args:
            model: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            language: Language code for transcription (None for auto-detect)
        """
        if WhisperModel is None:
            raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper")
            
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.model = None
        self.language = language
        self.download_root = os.path.join(os.path.expanduser("~"), "models")
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the whisper model asynchronously."""
        async with self._lock:
            if self.model is None:
                logger.info(f"Loading whisper model: {self.model_size} on {self.device}")
                
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: WhisperModel(self.model_size, device=self.device)  # type: ignore
                )
                logger.info("Whisper model loaded successfully")
    
    async def transcribe_audio(self, audio_file_path: str, segment_idx: int) -> List[TranscriptionSegment]:
        """
        Transcribe audio file to text with timing information.
        
        Args:
            audio_file_path: Path to the audio file
            segment_idx: Segment index for logging
            
        Returns:
            List of transcription segments with timing
        """
        try:
            if self.model is None:
                await self.initialize()
            
            logger.debug(f"Transcribing audio for segment {segment_idx}")
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe( # type: ignore
                    audio_file_path,
                    language=self.language,
                    word_timestamps=True,
                    vad_filter=False
                )
            )
            
            # Convert to our format
            transcription_segments = []
            for segment in segments:
                transcription_segments.append(TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip()
                ))
            
            logger.debug(f"Transcribed {len(transcription_segments)} segments for segment {segment_idx}")
            return transcription_segments
            
        except Exception as e:
            logger.error(f"Error transcribing segment {segment_idx}: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"model": self.model_size, "device": self.device, "loaded": False}
        
        return {
            "model": self.model_size,
            "device": self.device,
            "loaded": True,
            "language": self.language
        }
