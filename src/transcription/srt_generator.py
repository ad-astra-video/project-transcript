"""
SRT subtitle file generation from transcription segments.
"""

import logging
from datetime import timedelta
from typing import List
from .whisper_client import TranscriptionSegment

logger = logging.getLogger(__name__)


class SRTGenerator:
    """Generator for SRT subtitle files."""
    
    def __init__(self):
        pass
    
    def generate_srt(self, 
                    transcription_segments: List[TranscriptionSegment], 
                    segment_idx: int) -> str:
        """
        Generate SRT subtitle content from transcription segments.
        
        Uses segment-relative timing (Option A) - each segment starts from 00:00:00.
        
        Args:
            transcription_segments: List of transcription segments with timing
            segment_idx: Current segment index for logging
            
        Returns:
            SRT formatted subtitle string
        """
        if not transcription_segments:
            logger.debug(f"No transcription segments for segment {segment_idx}")
            return ""
        
        srt_lines = []
        subtitle_counter = 1
        
        for segment in transcription_segments:
            if not segment.text.strip():
                continue
                
            # Format timing (segment-relative)
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            
            # Create SRT entry (standard format: number, timecode, text, blank line)
            srt_lines.append(str(subtitle_counter))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(segment.text.strip())
            srt_lines.append("")  # Blank line between entries
            
            subtitle_counter += 1
        
        # Join with newlines to create proper SRT format
        result = "\n".join(srt_lines)
        logger.debug(f"Generated SRT for segment {segment_idx} with {len(transcription_segments)} subtitles")
        return result
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT formatted time string
        """
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int((seconds - total_seconds) * 1000)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def save_srt_file(self, srt_content: str, output_path: str) -> bool:
        """
        Save SRT content to file.
        
        Args:
            srt_content: SRT formatted content
            output_path: Path to save the SRT file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            logger.debug(f"Saved SRT file to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save SRT file to {output_path}: {e}")
            return False