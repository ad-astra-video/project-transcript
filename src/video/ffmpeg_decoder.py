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
                
            # Find the actual audio file created (could be .ac3, .flac, or .wav)
            actual_audio_file = None
            for ext in ['.ac3', '.flac', '.wav']:
                potential_file = audio_file.replace('.flac', ext)
                if os.path.exists(potential_file):
                    actual_audio_file = potential_file
                    break    

            logger.debug(f"Successfully decoded segment {segment_idx}")
            return video_file, audio_file
            
        except Exception as e:
            logger.error(f"Error decoding segment {segment_idx}: {e}")
            return None, None
    
    async def _extract_audio(self, input_file: str, output_file: str) -> bool:
        """Extract audio stream from segment with fallback strategies."""
        try:
            # Strategy 2: Try copying audio stream as-is (no conversion)
            if await self._extract_audio_copy(input_file, output_file):
                return True
                
            # Strategy 3: Try WAV conversion (more basic format)
            wav_file = output_file.replace('.flac', '.wav')
            if await self._extract_audio_wav(input_file, wav_file):
                # Rename WAV to expected FLAC name for compatibility
                os.rename(wav_file, output_file)
                return True
                
            logger.error(f"All audio extraction strategies failed for {input_file}")
            return False
            
        except Exception as e:
            logger.error(f"Error in audio extraction: {e}")
            return False
    
    async def _extract_audio_copy(self, input_file: str, output_file: str) -> bool:
        """Extract audio by copying stream (no conversion)."""
        try:
            # Copy audio stream as-is, then convert to FLAC in a second step
            temp_audio = output_file.replace('.flac', '_temp.ac3')
            
            # Step 1: Copy audio stream
            cmd1 = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-vn',  # No video
                '-c:a', 'copy',  # Copy audio without re-encoding
                temp_audio
            ]
            
            process1 = await asyncio.create_subprocess_exec(
                *cmd1,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout1, stderr1 = await process1.communicate()
            
            if process1.returncode != 0 or not os.path.exists(temp_audio):
                logger.debug(f"Audio copy failed: {stderr1.decode()}")
                return False
            
            # Step 2: Convert to required format using external tools if available
            cmd2 = [
                'ffmpeg', '-y',
                '-i', temp_audio,
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',
                '-f', 'flac',  # Force FLAC format
                output_file
            ]
            
            process2 = await asyncio.create_subprocess_exec(
                *cmd2,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout2, stderr2 = await process2.communicate()
            
            # Cleanup temp file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            if process2.returncode != 0:
                logger.debug(f"Audio conversion failed: {stderr2.decode()}")
                return False
                
            return os.path.exists(output_file)
            
        except Exception as e:
            logger.debug(f"Error in audio copy strategy: {e}")
            return False
    
    async def _extract_audio_wav(self, input_file: str, output_file: str) -> bool:
        """Extract audio as WAV (most basic format)."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', input_file,
                '-vn',  # No video
                '-ar', str(self.audio_sample_rate),
                '-ac', '1',  # Mono audio
                '-c:a', 'pcm_s16le',  # PCM 16-bit little-endian (basic format)
                '-f', 'wav',
                output_file
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.debug(f"WAV extraction failed: {stderr.decode()}")
                return False
                
            return os.path.exists(output_file)
            
        except Exception as e:
            logger.debug(f"Error in WAV extraction: {e}")
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
