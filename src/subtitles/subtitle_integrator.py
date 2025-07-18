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
    async def add_subtitles_ts(self, video_file_path: str, srt_content: str, segment_idx: int) -> Optional[str]:
        """Add hard-coded (burned-in) subtitles to an MPEG-TS segment.

        Args:
            video_file_path: input video file path
            srt_content: SRT text
            segment_idx: index for logging and temp naming
        Returns: path to new ts file with burned-in subtitles or None on error
        """
        try:
            if not srt_content.strip():
                logger.debug(f"No subtitle text for segment {segment_idx}")
                return video_file_path

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
                logger.error(
                    f"FFmpeg TS subtitle integration failed for segment {segment_idx}: {stderr.decode()}"
                )
                return None
            if not os.path.exists(output_file):
                logger.error(f"Output TS not created for segment {segment_idx}")
                return None
            return output_file
        except Exception as e:
            logger.error(f"Error adding TS subtitles to segment {segment_idx}: {e}")
            return None
            
    async def add_subtitles_mkv(self, video_file_path: str, srt_content: str, segment_idx: int) -> Optional[str]:
        """Add soft subtitles to a video segment using Matroska container.

        Args:
            video_file_path: input video file path
            srt_content: SRT text
            segment_idx: index for logging and temp naming
        Returns: path to new mkv file or None on error
        """
        try:
            if not srt_content.strip():
                logger.debug(f"No subtitle text for segment {segment_idx}")
                return video_file_path

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
                logger.error(
                    f"FFmpeg MKV subtitle integration failed for segment {segment_idx}: {stderr.decode()}"
                )
                return None
            if not os.path.exists(output_file):
                logger.error(f"Output MKV not created for segment {segment_idx}")
                return None
            return output_file
        except Exception as e:
            logger.error(f"Error adding MKV subtitles to segment {segment_idx}: {e}")
            return None
            
    async def prepare_subtitles(self, video_file_path: str, srt_content: str, segment_idx: int, hard: bool) -> Tuple[Optional[str], str]:
        """Prepare subtitles using the appropriate container format based on hard/soft setting.
        
        Args:
            video_file_path: input video file path
            srt_content: SRT text
            segment_idx: index for logging and temp naming
            hard: True → burn subtitles (TS); False → embed as separate stream (MKV)
        Returns: 
            Tuple of (output_file_path, mime_type) or (None, default_mime_type) on error
        """
        if hard:
            # Hard subtitles - use TS container
            output_file = await self.add_subtitles_ts(video_file_path, srt_content, segment_idx)
            return output_file, "video/mp2t"
        else:
            # Soft subtitles - use MKV container
            output_file = await self.add_subtitles_mkv(video_file_path, srt_content, segment_idx)
            return output_file, "video/x-matroska"
