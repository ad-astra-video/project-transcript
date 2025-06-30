"""
Video Transcription Pipeline Module

This module orchestrates the video transcription pipeline that:
1. Receives video segments via trickle subscriber
2. Decodes video and extracts audio
3. Transcribes audio using faster-whisper
4. Generates SRT subtitles
5. Integrates subtitles (hard/soft coding)
6. Re-encodes and publishes via trickle publisher
"""

__version__ = "1.0.0"
