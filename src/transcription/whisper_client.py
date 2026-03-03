"""
Faster-whisper client for audio transcription.
"""

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """Represents a single word with timing information."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Word text


@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment with timing information."""
    id: str       # Stable ID for deduplication
    start: float  # Start time in seconds (segment-relative)
    end: float    # End time in seconds (segment-relative)
    text: str     # Transcribed text
    words: list   # List of WordTimestamp objects
    speaker: str | None = None  # Optional speaker label
    
    
class WhisperClient:
    """Client for faster-whisper transcription."""
    
    def __init__(self, model_size: str = "turbo", device: str = "cpu", compute_type: str ="float32", language: str | None = None, language_confidence_threshold: float = 0.5):
        """
        Initialize the whisper client.
        
        Args:
            model: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cpu, cuda)
            language: Language code for transcription (None for auto-detect)
            language_confidence_threshold: Minimum confidence for language detection (0.5 = 50%)
        """
        if WhisperModel is None:
            raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper")
            
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.model = None
        self.language = language
        self.language_confidence_threshold = language_confidence_threshold
        self.download_root = Path(os.environ.get("MODEL_DIR", "/models"))
        Path(self.download_root).mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._next_transcription_id: int = 0  # Internal counter for transcription IDs
        
    async def initialize(self):
        """Initialize the whisper model asynchronously."""
        async with self._lock:
            if self.model is None:
                logger.info(f"Loading whisper model: {self.model_size} on {self.device}")
                
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type, 
                        download_root=str(self.download_root)  # type: ignore
                ),
                )
                logger.info("Whisper model loaded successfully")
    
    async def transcribe_audio(self, audio_file_path: str, segment_idx: int = 0) -> tuple[int, List[TranscriptionSegment]]:
        """
        Transcribe audio file to text with timing information.
        
        Args:
            audio_file_path: Path to the audio file
            segment_idx: Optional segment index. If 0, uses internal counter.
            
        Returns:
            Tuple of (transcription_window_id, List of transcription segments with timing)
        """
        # Use provided segment_idx if non-zero, otherwise use internal counter
        if segment_idx != 0:
            transcription_window_id = segment_idx
        else:
            transcription_window_id = self._next_transcription_id
            self._next_transcription_id += 1
        
        try:
            if self.model is None:
                await self.initialize()
            
            logger.debug(f"Transcribing audio for transcription_window_id {transcription_window_id}")
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe( # type: ignore
                    audio_file_path,
                    language=self.language,
                    word_timestamps=True,
                    vad_filter=False,
                    language_detection_threshold=self.language_confidence_threshold
                )
            )
            
            # Check language confidence
            language_confidence = getattr(info, 'language_probability', 1.0)
            logger.debug(f"Language detection confidence: {language_confidence:.3f}")
            
            # Convert to our format
            transcription_segments = []
            for segment in segments:
                # Extract word-level timestamps if available
                words = []
                if hasattr(segment, 'words') and segment.words:
                    #logger.info(f"Processing {len(segment.words)} words: {segment.words}")
                    for word in segment.words:
                        word_text = word.word.strip() if hasattr(word, 'word') else str(word).strip()
                        if word.probability < 0.60:
                            word_text += "[?]"
                            continue
                        words.append(WordTimestamp(
                            start=word.start,
                            end=word.end,
                            text=word_text
                        ))
                
                # Generate stable ID from timing
                seg_id = f"{segment.start:.3f}-{segment.end:.3f}"
                
                # If language confidence is below threshold, return empty transcript
                if language_confidence < self.language_confidence_threshold:
                    transcription_segments.append(TranscriptionSegment(
                        id=seg_id,
                        start=segment.start,
                        end=segment.end,
                        text="",  # Empty text for low confidence
                        words=[]
                    ))
                else:
                    transcription_segments.append(TranscriptionSegment(
                        id=seg_id,
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                        words=words
                    ))
            
            logger.debug(f"Transcribed {len(transcription_segments)} segments for transcription_window_id {transcription_window_id} (confidence: {language_confidence:.3f})")
            
            # Return transcription_window_id along with segments
            return transcription_window_id, transcription_segments
            
        except Exception as e:
            logger.error(f"Error transcribing for transcription_window_id {transcription_window_id}: {e}")
            return transcription_window_id, []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {
                "model": self.model_size,
                "device": self.device,
                "loaded": False,
                "language_confidence_threshold": self.language_confidence_threshold
            }
        
        return {
            "model": self.model_size,
            "device": self.device,
            "loaded": True,
            "language": self.language,
            "language_confidence_threshold": self.language_confidence_threshold
        }
    
    def reset(self):
        """Reset internal state for a new stream.
        
        Clears the transcription ID counter to ensure fresh ID sequence
        for each new stream session.
        """
        self._next_transcription_id = 0
        logger.info("WhisperClient reset - transcription ID counter cleared")
