"""
Configuration management for the video transcription pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the video transcription pipeline."""
    
    # Trickle server URLs
    subscribe_url: str = "http://0.0.0.0:3389/sample" ## localhost for testing
    publish_url: str = "http://127.0.0.1:3389/publish" ## localhost for testing
    text_url: Optional[str] = "http://127.0.0.1:3389/subtitles"
    events_url: Optional[str] = "http://127.0.0.1:3389/events"
    # For Docker, use the host's IP address
    # subscribe_url: str = "http://172.17.0.1:3389/sample"
    # publish_url: str = "http://172.17.0.1:3389/publish"
    
    # Transcription settings
    whisper_model: str = "large"  # base, small, medium, large
    whisper_language: Optional[str] = None  # Auto-detect if None
    whisper_device: str = "cuda"  # cpu, cuda
    compute_type: str = "float16"  # float16, float32, int8, etc. (depends on whisper model support)
    
    # Subtitle settings
    subtitle_style: str = "default"
    subtitle_font: str = "Arial"
    subtitle_font_size: int = 16
    subtitle_color: str = "white"
    subtitle_background: str = "black"
    subtitle_position: str = "bottom"  # bottom, top, center
    
    # Processing options
    hard_code_subtitles: str | bool = True  # True for hard, False for soft

    enable_text_url: str | bool = True  # Toggle for subtitle file posting
    enable_events_url: str | bool = True
    pipeline_uuid: Optional[str] = None
    
    # Performance settings
    audio_sample_rate: int = 16000  # Hz, required for whisper
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls(
            subscribe_url=os.getenv("SUBSCRIBE_URL", cls.subscribe_url),
            publish_url=os.getenv("PUBLISH_URL", cls.publish_url),
            text_url=os.getenv("TEXT_URL", cls.text_url),
            whisper_model=os.getenv("WHISPER_MODEL", cls.whisper_model),
            whisper_language=os.getenv("WHISPER_LANGUAGE", cls.whisper_language),
            whisper_device=os.getenv("WHISPER_DEVICE", cls.whisper_device),
            compute_type=os.getenv("COMPUTE_TYPE", cls.compute_type),
            subtitle_style=os.getenv("SUBTITLE_STYLE", cls.subtitle_style),
            subtitle_font=os.getenv("SUBTITLE_FONT", cls.subtitle_font),
            subtitle_font_size=int(os.getenv("SUBTITLE_FONT_SIZE", cls.subtitle_font_size)),
            subtitle_color=os.getenv("SUBTITLE_COLOR", cls.subtitle_color),
            subtitle_background=os.getenv("SUBTITLE_BACKGROUND", cls.subtitle_background),
            subtitle_position=os.getenv("SUBTITLE_POSITION", cls.subtitle_position),
            hard_code_subtitles=os.getenv("HARD_CODE_SUBTITLES", (cls.hard_code_subtitles)),
            enable_text_url=os.getenv("ENABLE_TEXT_URL", str(cls.enable_text_url)),
            events_url=os.getenv("EVENTS_URL", cls.events_url),
            enable_events_url=os.getenv("ENABLE_EVENTS_URL", str(cls.enable_events_url)),
            pipeline_uuid=os.getenv("PIPELINE_UUID", cls.pipeline_uuid),
            audio_sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", cls.audio_sample_rate)),
        )