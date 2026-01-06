"""
Summary module for LLM-based transcription cleaning and summarization.
"""

__version__ = "1.0.0"

from .summary_client import SummaryClient, SummarySegment

__all__ = [
    "SummaryClient",
    "SummarySegment",
]