"""
Diarization client for speaker diarization using pyannote.audio.
"""

from .diarization_client import DiarizationClient, SpeakerMemory, SpeakerSegment, DiarizationRequest, DiarizationResult

__all__ = [
    "DiarizationClient",
    "SpeakerMemory",
    "SpeakerSegment",
    "DiarizationRequest",
    "DiarizationResult",
]