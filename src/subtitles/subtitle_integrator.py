"""
Subtitle integration for hard and soft coding subtitles into video using FFmpeg.
"""

import asyncio
import logging
import tempfile
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SubtitleIntegrator:
    """Integrator for adding subtitles to video segments using FFmpeg."""
    
    def __init__(self, 
                 subtitle_font: str = "Arial",
                 subtitle_font_size: int = 16,
                 subtitle_color: str = "white",
                 subtitle_background: str = "black",
                 subtitle_position: str = "bottom"):
        """
        Initialize subtitle integrator.
        
        Args:
            subtitle_font: Font family for subtitles
            subtitle_font_size: Font size for subtitles
            subtitle_color: Text color for subtitles
            subtitle_background: Background color for subtitles
            subtitle_position: Position of subtitles (bottom, top, center)
        """
        self.subtitle_font = subtitle_font
        self.subtitle_font_size = subtitle_font_size
        self.subtitle_color = subtitle_color
        self.subtitle_background = subtitle_background
        self.subtitle_position = subtitle_position
    
    # ------------------------------------------------------------------
    # Methods for subtitle integration (TS and MKV containers)
    # ------------------------------------------------------------------
    async def add_subtitles_ts(self, video_file_path: str, srt_content: str, segment_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """Add hard-coded (burned-in) subtitles to an MPEG-TS segment.

        Args:
            video_file_path: input video file path
            srt_content: SRT text
            segment_idx: index for logging and temp naming
        Returns: Tuple of (output_file_path, error_message) or (None, error_message) on error
        """
        try:
            if not srt_content.strip():
                logger.debug(f"No subtitle text for segment {segment_idx}")
                return video_file_path, None

            # prepare temp files
            temp_dir = tempfile.mkdtemp(prefix=f"ts_subs_{segment_idx}_")
            srt_file = os.path.join(temp_dir, f"subtitles_{segment_idx}.srt")
            output_file = os.path.join(temp_dir, f"output_{segment_idx}.ts")

            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)

            # Burn subtitles (overlay) – requires re-encode
            # Build ASS style string from configuration
            style_parts = []
            # Font family and size
            if self.subtitle_font:
                style_parts.append(f"FontName={self.subtitle_font}")
            style_parts.append(f"Fontsize={self.subtitle_font_size}")

            # Map simple color names → ASS BGR hex (without alpha) e.g. &H00BBGGRR&
            color_map = {
                "white": "FFFFFF",
                "black": "000000",
                "red": "0000FF",
                "green": "00FF00",
                "blue": "FF0000",
                "yellow": "00FFFF",
                "cyan": "FFFF00",
                "magenta": "FF00FF",
            }
            if self.subtitle_color:
                hex_color = color_map.get(self.subtitle_color.lower(), None)
                if hex_color:
                    style_parts.append(f"PrimaryColour=&H00{hex_color}&")
            if self.subtitle_background and self.subtitle_background.lower() != "none":
                bg_hex = color_map.get(self.subtitle_background.lower(), None)
                if bg_hex:
                    # ASS BackColour controls box background when enabled
                    style_parts.append(f"BackColour=&H00{bg_hex}&")

            # Alignment – bottom(2), center(5), top(8)
            alignment_map = {
                "bottom": 2,
                "center": 5,
                "top": 8,
            }
            align_val = alignment_map.get(self.subtitle_position.lower(), 2)
            style_parts.append(f"Alignment={align_val}")

            force_style = ",".join(style_parts)
            vf_arg = f"subtitles={srt_file}:force_style='{force_style}'"
            cmd = [
                "ffmpeg", "-y",
                "-i", video_file_path,
                "-vf", vf_arg,
                "-c:v", "mpeg2video", "-qscale:v", "2",
                "-c:a", "copy",
                "-f", "mpegts",
                output_file,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = f"FFmpeg TS subtitle integration failed for segment {segment_idx}: {stderr.decode()}"
                logger.error(error_msg)
                return video_file_path, error_msg
            if not os.path.exists(output_file):
                error_msg = f"Output TS not created for segment {segment_idx}"
                logger.error(error_msg)
                return video_file_path, error_msg
            return output_file, None
        except Exception as e:
            logger.error(f"Error adding TS subtitles to segment {segment_idx}: {e}")
            return video_file_path, str(e)
            
    async def add_subtitles_mkv(self, video_file_path: str, srt_content: str, segment_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """Add soft subtitles to a video segment using Matroska container.

        Args:
            video_file_path: input video file path
            srt_content: SRT text
            segment_idx: index for logging and temp naming
        Returns: Tuple of (output_file_path, error_message) or (None, error_message) on error
        """
        try:
            if not srt_content.strip():
                logger.debug(f"No subtitle text for segment {segment_idx}")
                return video_file_path, None

            # prepare temp files
            temp_dir = tempfile.mkdtemp(prefix=f"mkv_subs_{segment_idx}_")
            srt_file = os.path.join(temp_dir, f"subtitles_{segment_idx}.srt")
            output_file = os.path.join(temp_dir, f"output_{segment_idx}.mkv")

            with open(srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)

            # Embed SRT as a subtitle track in MKV container without re-encoding
            # Add timestamp correction to ensure proper playback
            cmd = [
                "ffmpeg", "-y",
                "-i", video_file_path,
                "-i", srt_file,
                "-c:v", "copy",   
                "-c:a", "copy",   
                "-c:s", "srt",    
                "-disposition:s:0", "default",
                "-metadata:s:s:0", "language=eng",
                "-f", "matroska",
                output_file,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = f"FFmpeg MKV subtitle integration failed for segment {segment_idx}: {stderr.decode()}"
                logger.error(error_msg)
                return video_file_path, error_msg
            if not os.path.exists(output_file):
                error_msg = f"Output MKV not created for segment {segment_idx}"
                logger.error(error_msg)
                return video_file_path, error_msg
            return output_file, None
        except Exception as e:
            logger.error(f"Error adding MKV subtitles to segment {segment_idx}: {e}")
            return video_file_path, str(e)

    async def integrate_subtitles(self, segment_idx: int, video_file_path: str, srt_content: str, hard: bool) -> Tuple[Optional[bytes], Optional[str]]:
        """Integrate subtitles into the video segment.
        
        Args:
            segment_idx: The index of the segment.
            video_file_path: The path to the video file.
            srt_file_path: The path to the SRT subtitle file.
            hard: Whether to hard-code subtitles into the video.
        Returns: 
            A tuple containing the processed video data as bytes and an optional error message string.
        """
        if hard:
            output_file, error = await self.add_subtitles_ts(video_file_path, srt_content, segment_idx)
        else:
            output_file, error = await self.add_subtitles_mkv(video_file_path, srt_content, segment_idx)

        if not output_file:
            logger.error(f"Subtitle integration failed for segment {segment_idx}: {error}")
            logger.debug(f"Returning original video file path: {video_file_path}")
            file_to_read = video_file_path
        else:
            file_to_read = output_file

        with open(file_to_read, "rb") as f:
            processed_data = f.read()
        
        return processed_data, error if not output_file else None
