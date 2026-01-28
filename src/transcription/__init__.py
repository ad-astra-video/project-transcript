"""
Transcription module using faster-whisper for audio-to-text conversion.
"""

__version__ = "1.0.0"

from .whisper_client import WhisperClient, TranscriptionSegment, WordTimestamp

__all__ = [
    "WhisperClient",
    "TranscriptionSegment",
    "WordTimestamp",
]
