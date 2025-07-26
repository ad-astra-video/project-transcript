"""
FFmpeg-based video decoder for trickle video segments.
"""

import asyncio
import logging
import tempfile
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FFmpegDecoder:
    """Decoder for video segments using FFmpeg."""
    
    def __init__(self, audio_sample_rate: int = 16000):
        self.audio_sample_rate = audio_sample_rate
        
    async def decode_segment(self, segment_data: bytes, segment_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Decode a video segment and extract audio.
        
        Args:
            segment_data: Raw video segment data (MP2T format)
            segment_idx: Segment index for tracking
            
        Returns:
            Tuple of (video_file_path, audio_file_path) or (None, None) on error
        """
        try:
            # Create temporary files for processing
            temp_dir = tempfile.mkdtemp(prefix=f"segment_{segment_idx}_")
            input_file = os.path.join(temp_dir, f"input_{segment_idx}.ts")
            video_file = input_file  
            audio_file = os.path.join(temp_dir, f"audio_{segment_idx}.flac")
            
            # Write segment data to temp file
            with open(input_file, 'wb') as f:
                f.write(segment_data)
            
            # Only extract audio – no video re-muxing required
            audio_success = await self._extract_audio(input_file, audio_file)
            
            if not audio_success:
                logger.error(f"Failed to extract audio for segment {segment_idx}")
                self._cleanup_files([input_file, audio_file])
                return None, None 

            logger.debug(f"Successfully decoded segment {segment_idx}")
            return video_file, audio_file
            
        except Exception as e:
            logger.error(f"Error decoding segment {segment_idx}: {e}")
            return None, None
    
    async def _extract_audio(self, input_file: str, output_file: str) -> bool:
        """Extract audio stream from segment with fallback strategies."""
        try:
            # Strategy 1: Try direct conversion with increased analysis
            if await self._extract_audio_direct(input_file, output_file):
                return True
                
            logger.error(f"Audio extraction strategy failed for {input_file}")
            return False
            
        except Exception as e:
            logger.error(f"Error in audio extraction: {e}")
            return False
    
    async def _extract_audio_direct(self, input_file: str, output_file: str) -> bool:
        """Direct audio extraction with enhanced analysis parameters."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-analyzeduration', '10M',  # Increase analysis duration
                '-probesize', '10M',        # Increase probe size
                '-i', input_file,
                '-vn',  # No video
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',  # Mono audio
                '-c:a', 'flac',  # FLAC encoder
                '-map', '0:a?',  # Map audio stream if available, ignore if missing
                output_file
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.debug(f"Direct audio extraction failed: {stderr.decode()}")
                return False
                
            return os.path.exists(output_file) and os.path.getsize(output_file) > 0
            
        except Exception as e:
            logger.debug(f"Error in direct audio extraction: {e}")
            return False
    
    def _cleanup_files(self, file_paths: list):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {e}")
    
    def cleanup_segment_files(self, video_file: str, audio_file: str):
        """Clean up processed segment files."""
        self._cleanup_files([video_file, audio_file])
        
        # Also cleanup the temp directory if empty
        try:
            temp_dir = os.path.dirname(video_file)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
