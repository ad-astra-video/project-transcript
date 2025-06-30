"""
Main pipeline orchestration for video transcription and subtitle integration.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import aiohttp

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from pipeline.config import PipelineConfig
from video.ffmpeg_decoder import FFmpegDecoder
from transcription.whisper_client import WhisperClient
from transcription.srt_generator import SRTGenerator
from subtitles.subtitle_integrator import SubtitleIntegrator
from trickle.trickle_subscriber import TrickleSubscriber
from trickle.trickle_publisher import TricklePublisher

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Main pipeline for video transcription and subtitle integration."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the video pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.decoder = FFmpegDecoder()
        self.whisper_client = WhisperClient(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.compute_type, 
            language=config.whisper_language
        )
        self.srt_generator = SRTGenerator()
        self.subtitle_integrator = SubtitleIntegrator(
            subtitle_font=config.subtitle_font,
            subtitle_font_size=config.subtitle_font_size,
            subtitle_color=config.subtitle_color,
            subtitle_background=config.subtitle_background,
            subtitle_position=config.subtitle_position,
        )
        
        # Processing queues for robust async flow
        self.input_segment_queue = asyncio.Queue(maxsize=10)  # Buffer input segments
        self.output_segment_queue = asyncio.Queue(maxsize=10)  # Buffer output segments
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent_segments)
        self.running = False
        
        # Flow metrics for monitoring
        self.flow_metrics = {
            'input_segments': 0,
            'processed_segments': 0,
            'published_segments': 0,
            'input_queue_depth': 0,
            'output_queue_depth': 0,
            'errors': 0
        }
        
    async def initialize(self):
        """Initialize the pipeline components."""
        try:
            logger.info("Initializing video processing pipeline...")
            
            # Initialize whisper client
            await self.whisper_client.initialize()
            
            logger.info("Pipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def run(self):
        """Run the main pipeline processing loop."""
        self.running = True
        logger.info("Starting video processing pipeline...")
        
        try:
            # Start concurrent tasks
            tasks = await asyncio.gather(
                self._trickle_subscriber_task(),
                self._segment_processor_task(),
                self._trickle_publisher_task(),
                return_exceptions=True
            )
            
            # Log any task errors
            for i, result in enumerate(tasks):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.running = False
            await self.cleanup()
    
    async def _trickle_subscriber_task(self):
        """Continuously fetch and buffer trickle segments for smooth flow."""
        try:
            logger.info("Starting async trickle subscriber with enhanced buffering")
            async with TrickleSubscriber(self.config.subscribe_url) as subscriber:
                segment_count = 0
                consecutive_empty = 0
                max_consecutive_empty = 50  # Allow more retries
                # TODO: Make this wait for streams insetad of crashing on empty
                
                while self.running:
                    try:
                        # Fetch segment with backoff on empty
                        current_segment = await subscriber.next()
                        if current_segment is None:
                            consecutive_empty += 1
                            if consecutive_empty >= max_consecutive_empty:
                                logger.info("No more segments available, ending subscriber")
                                break
                            # Progressive backoff for empty segments
                            sleep_time = min(0.1 * consecutive_empty, 2.0)
                            await asyncio.sleep(sleep_time)
                            continue
                        
                        consecutive_empty = 0  # Reset counter on successful fetch
                        segment_count += 1
                        
                        # Read segment data
                        segment_data = await self._read_complete_segment(current_segment)
                        if segment_data:
                            # Queue segment with priority handling
                            try:
                                await asyncio.wait_for(
                                    self.input_segment_queue.put((segment_count, segment_data)), 
                                    timeout=0.5  # Longer timeout for buffering
                                )
                                self.flow_metrics['input_segments'] += 1
                                self.flow_metrics['input_queue_depth'] = self.input_segment_queue.qsize()
                                
                                if segment_count % 10 == 0:
                                    logger.debug(f"Buffered segment {segment_count} (queue depth: {self.flow_metrics['input_queue_depth']})")
                                    
                            except asyncio.TimeoutError:
                                # If queue is full, wait longer rather than dropping
                                logger.warning(f"Input queue full, waiting for space...")
                                await self.input_segment_queue.put((segment_count, segment_data))
                        
                        # Close segment
                        if hasattr(current_segment, 'close'):
                            try:
                                await current_segment.close()
                            except:
                                pass
                                
                    except Exception as e:
                        logger.error(f"Error in subscriber: {e}")
                        self.flow_metrics['errors'] += 1
                        await asyncio.sleep(0.2)
                        
            logger.info(f"Trickle subscriber finished: {segment_count} segments buffered")
                        
        except Exception as e:
            logger.error(f"Trickle subscriber task error: {e}")
            self.flow_metrics['errors'] += 1
        finally:
            # Signal end of input
            await self.input_segment_queue.put(None)
    
    async def _read_complete_segment(self, segment_reader) -> Optional[bytes]:
        """Read complete segment data from segment reader."""
        try:
            data_chunks = []
            while True:
                chunk = await segment_reader.read(8192)
                if not chunk:
                    break
                data_chunks.append(chunk)
            
            if data_chunks:
                return b''.join(data_chunks)
            return None
            
        except Exception as e:
            logger.error(f"Error reading segment data: {e}")
            return None
    
    async def _segment_processor_task(self):
        """Process segments from input queue and add to output queue."""
        processed_count = 0
        
        try:
            while self.running:
                try:
                    # Get segment from input queue
                    segment_item = await asyncio.wait_for(
                        self.input_segment_queue.get(), 
                        timeout=1.0
                    )
                    
                    if segment_item is None:  # End signal
                        logger.info("Received end signal for segment processor")
                        break
                    
                    segment_idx, segment_data = segment_item
                    
                    # Process segment concurrently
                    async with self.semaphore:
                        processed_data = await self._process_single_segment(segment_idx, segment_data)
                        if processed_data:
                            # Add to output queue
                            await self.output_segment_queue.put((segment_idx, processed_data))
                            processed_count += 1
                            self.flow_metrics['processed_segments'] = processed_count
                            self.flow_metrics['output_queue_depth'] = self.output_segment_queue.qsize()
                    
                    # Mark task done
                    self.input_segment_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue  # Check if still running
                except Exception as e:
                    logger.error(f"Error in segment processor: {e}")
                    self.flow_metrics['errors'] += 1
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Segment processor task error: {e}")
        finally:
            # Signal end of processing
            await self.output_segment_queue.put(None)
    
    async def _trickle_publisher_task(self):
        """Publish processed segments via trickle publisher."""
        published_count = 0
        
        # Determine container type based on subtitle mode
        hard_flag = bool(self.config.hard_code_subtitles) and str(self.config.hard_code_subtitles).lower() != "false"
        mime_type = "video/mp2t" if hard_flag else "video/x-matroska"
        logger.info(f"Using container format: {mime_type} ({'hard' if hard_flag else 'soft'} subtitles)")
        
        try:
            async with TricklePublisher(self.config.publish_url, mime_type=mime_type) as publisher:
                while self.running:
                    try:
                        # Get processed segment from output queue
                        segment_item = await asyncio.wait_for(
                            self.output_segment_queue.get(),
                            timeout=1.0
                        )
                        
                        if segment_item is None:  # End signal
                            logger.info("Received end signal for publisher")
                            break
                        
                        segment_idx, segment_data = segment_item
                        
                        # Publish segment
                        await self._publish_segment_data(publisher, segment_data, segment_idx)
                        published_count += 1
                        self.flow_metrics['published_segments'] = published_count
                        
                        # Mark task done
                        self.output_segment_queue.task_done()
                        
                        if published_count % 10 == 0:
                            logger.info(f"Published {published_count} segments")
                    
                    except asyncio.TimeoutError:
                        continue  # Check if still running
                    except Exception as e:
                        logger.error(f"Error in publisher: {e}")
                        self.flow_metrics['errors'] += 1
                        await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Trickle publisher task error: {e}")
    
    async def _process_single_segment(self, segment_idx: int, segment_data: bytes) -> Optional[bytes]:
        """
        Process a single video segment.
        
        Args:
            segment_idx: Segment index
            segment_data: Raw segment data
            
        Returns:
            Processed segment data or None on error
        """
        temp_files = []
        
        try:
            logger.debug(f"Processing segment {segment_idx}")
            
            # Step 1: extract audio from segment data
            video_file, audio_file = await self.decoder.decode_segment(segment_data, segment_idx)
            if not video_file or not audio_file:
                logger.error(f"Failed to decode segment {segment_idx}")
                # Return original segment data as fallback
                return segment_data
            temp_files.extend([video_file, audio_file])
            
            # Step 2: Transcribe audio
            transcription = await self.whisper_client.transcribe_audio(audio_file, segment_idx)
            if not transcription:
                logger.debug(f"No transcription found for segment {segment_idx}")
                # Return original segment data without subtitles
                return segment_data
            
            # Step 3: Generate SRT content
            srt_content = self.srt_generator.generate_srt(
                transcription, segment_idx
            )
            
            # Step 4: Integrate subtitles using appropriate container format
            hard_flag = bool(self.config.hard_code_subtitles) and str(self.config.hard_code_subtitles).lower() != "false"
            final_video, _ = await self.subtitle_integrator.prepare_subtitles(
                video_file,
                srt_content,
                segment_idx,
                hard=hard_flag,
            )

            if not final_video:
                logger.error(f"Failed to add subtitles for segment {segment_idx}")
                return segment_data
            else:
                temp_files.append(final_video)
            
            # Step 5: Read processed video data
            with open(final_video, 'rb') as f:
                processed_data = f.read()
            
            # Step 6: Optionally send subtitle file to data_url
            if self.config.enable_data_url and self.config.data_url and srt_content:
                await self._send_subtitle_to_data_url(srt_content, segment_idx)
            
            logger.debug(f"Successfully processed segment {segment_idx}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing segment {segment_idx}: {e}")
            self.flow_metrics['errors'] += 1
            return segment_data  # Return original as fallback
        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        # Try to remove parent directory if empty
                        parent_dir = os.path.dirname(temp_file)
                        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    
    async def _publish_segment_data(self, publisher, segment_data: bytes, segment_idx: int):
        """
        Publish processed video segment data via trickle publisher.
        
        Args:
            publisher: TricklePublisher instance
            segment_data: Processed video data
            segment_idx: Segment index
        """
        try:
            # Get segment writer from publisher
            segment_writer = await publisher.next()
            if not segment_writer:
                logger.error(f"Failed to get segment writer for segment {segment_idx}")
                return
            
            # Write video data
            await segment_writer.write(segment_data)
            await segment_writer.close()
            
            logger.debug(f"Published segment {segment_idx} ({len(segment_data)} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to publish segment {segment_idx}: {e}")
            self.flow_metrics['errors'] += 1
    
    async def _send_subtitle_to_data_url(self, srt_content: str, segment_idx: int):
        """
        Send subtitle content to data URL.
        
        Args:
            srt_content: SRT subtitle content
            segment_idx: Segment index
        """
        if not self.config.data_url:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.data_url,
                    data=srt_content.encode('utf-8'),
                    headers={
                        'Content-Type': 'text/plain; charset=utf-8',
                        'X-Segment-Id': str(segment_idx)
                    }
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Successfully sent subtitle for segment {segment_idx} to data URL")
                    else:
                        logger.warning(f"Failed to send subtitle for segment {segment_idx}: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending subtitle for segment {segment_idx} to data URL: {e}")
    
    async def stop(self):
        """Stop the pipeline."""
        self.running = False
        logger.info("Pipeline stop requested")
    
    async def cleanup(self):
        """Cleanup pipeline resources."""
        try:
            logger.info(f"Pipeline cleanup complete. Final metrics: {self.flow_metrics}")
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")


async def main():
    """Main entry point for the video processing pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration
        config = PipelineConfig.from_env()
        logger.info(f"Starting pipeline with config:")
        logger.info(f"  Subscribe URL: {config.subscribe_url}")
        logger.info(f"  Publish URL: {config.publish_url}")
        logger.info(f"  Whisper Model: {config.whisper_model}")
        logger.info(f"  Hard Code Subtitles: {config.hard_code_subtitles}")
        
        # Create and run pipeline
        pipeline = VideoPipeline(config)
        await pipeline.initialize()
        await pipeline.run()
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
