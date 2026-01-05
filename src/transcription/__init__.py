"""
Transcription module using faster-whisper for audio-to-text conversion.
"""

__version__ = "1.0.0"

from .diarization import DiarizationProcess, SpeakerSegment, DiarizationRequest, DiarizationResult
from .whisper_client import WhisperClient, TranscriptionSegment, WordTimestamp
from .srt_generator import SRTGenerator

__all__ = [
    "DiarizationProcess",
    "SpeakerSegment",
    "DiarizationRequest",
    "DiarizationResult",
    "WhisperClient",
    "TranscriptionSegment",
    "WordTimestamp",
    "SRTGenerator",
]
