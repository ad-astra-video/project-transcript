"""
Main pipeline orchestration for video transcription and subtitle integration.
"""

import asyncio
import logging
import os
import sys
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from pipeline.config import PipelineConfig
from video.ffmpeg_decoder import FFmpegDecoder
from transcription.whisper_client import WhisperClient
from transcription.srt_generator import SRTGenerator
from subtitles.subtitle_integrator import SubtitleIntegrator
from pytrickle.subscriber import TrickleSubscriber
from pytrickle.publisher import TricklePublisher

logger = logging.getLogger(__name__)

@dataclass
class StreamStats:
    pipeline_uuid: Optional[str]
    segment_idx: int
    segments_processed_total: int
    timestamp_utc: str
    status: str
    error_msg: Optional[str] = None
    processing_ms: Optional[float] = None
    ingest_to_process_ms: Optional[float] = None



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
        
        # Text publisher (created lazily if data_url provided)
        self.text_publisher: Optional[TricklePublisher] = None
        self.events_publisher: Optional[TricklePublisher] = None
        self.stats_queue: asyncio.Queue[StreamStats] = asyncio.Queue()

        self.running = False
        
        # Flow metrics for monitoring
        self.flow_metrics = {
            "segments_received": 0,
            "segments_processed": 0,
            "segments_dropped": 0,
            "errors": 0,
        }
        self.segments_processed_count = 0
        
    async def initialize(self):
        """Initialize the pipeline components."""
        try:
            logger.info("Initializing video processing pipeline...")
            
            # Initialize whisper client
            await self.whisper_client.initialize()

            # Initialize text publisher if configured
            if self.config.enable_data_url and self.config.data_url:
                self.text_publisher = TricklePublisher(self.config.data_url, mime_type="application/json")
                await self.text_publisher.start()

            if self.config.enable_events_url and self.config.events_url:
                self.events_publisher = TricklePublisher(self.config.events_url, mime_type="application/json")
                await self.events_publisher.start()
            
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
                self._send_events_from_stats_queue(),
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
                consecutive_empty = 0  # Tracks consecutive empty polls for backoff
                max_empty_retries = 10  # Max retries before giving up

                while self.running:
                    try:
                        # Fetch segment with backoff on empty
                        current_segment = await subscriber.next()
                        if current_segment is None:
                            # No segment available yet – wait and retry with progressive backoff
                            consecutive_empty += 1
                            if consecutive_empty >= max_empty_retries:
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
                                received_time = asyncio.get_event_loop().time()
                                await asyncio.wait_for(
                                    self.input_segment_queue.put((segment_count, segment_data, received_time)),
                                    timeout=0.5  # Longer timeout for buffering
                                )
                                self.flow_metrics['segments_received'] += 1
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
    
    async def _segment_worker(self, segment_idx: int, segment_data: bytes, received_time: float):
        """Process a single segment and publish result to output queue."""
        try:
            result = await self._process_single_segment(segment_idx, segment_data, received_time)
        except Exception as e:
            logger.error(f"Error in segment worker for segment {segment_idx}: {e}")
            self.flow_metrics['errors'] += 1
            result = None
        finally:
            # Mark the input queue task as done regardless of outcome
            self.input_segment_queue.task_done()

        if result:
            if isinstance(result, tuple):
                processed_data, srt_content = result
                await self.output_segment_queue.put((segment_idx, processed_data, srt_content))
            else:
                await self.output_segment_queue.put((segment_idx, result))
            self.flow_metrics['segments_processed'] += 1
            self.flow_metrics['output_queue_depth'] = self.output_segment_queue.qsize()

    async def _segment_processor_task(self):
        """Spawn worker tasks for each incoming segment and wait for their completion."""
        pending_tasks = set()
        try:
            while self.running:
                try:
                    segment_item = await asyncio.wait_for(
                        self.input_segment_queue.get(),
                        timeout=1.0
                    )

                    if segment_item is None:  # End signal
                        logger.info("Received end signal for segment processor")
                        self.running = False
                        break

                    segment_idx, segment_data, received_time = segment_item

                    # Spawn a worker task
                    task = asyncio.create_task(self._segment_worker(segment_idx, segment_data, received_time))
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)

                    # Update queue depth metric
                    self.flow_metrics['input_queue_depth'] = self.input_segment_queue.qsize()

                except asyncio.TimeoutError:
                    continue  # Check if still running
                except Exception as e:
                    logger.error(f"Error spawning segment worker: {e}")
                    self.flow_metrics['errors'] += 1
                    await asyncio.sleep(0.1)
        finally:
            # Wait for any remaining workers to finish
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
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
                            self.running = False
                            break
                        
                        if len(segment_item) == 3:
                            segment_idx, segment_data, srt_content = segment_item
                        else:
                            segment_idx, segment_data = segment_item
                            srt_content = None
                        
                        # Publish segment
                        await self._publish_segment_data(publisher, segment_data, segment_idx)
                        
                        # Send subtitle after video segment is published
                        if srt_content:
                            await self._send_subtitle_to_data_url(srt_content, segment_idx)
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

    async def _process_single_segment(
        self, segment_idx: int, segment_data: bytes, received_time: float
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Process a single video segment.
        
        Args:
            segment_idx: Segment index
            segment_data: Raw segment data
            received_time: Time the segment was received
            
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
                return segment_data, None
            temp_files.extend([video_file, audio_file])
            
            # Step 2: Transcribe audio
            transcription = await self.whisper_client.transcribe_audio(audio_file, segment_idx)
            if not transcription:
                logger.debug(f"No transcription found for segment {segment_idx}")
                # Return original segment data without subtitles
                return segment_data, None
            
            # Step 3: Generate SRT content
            srt_content = self.srt_generator.generate_srt(
                transcription, segment_idx
            )
            
            # Step 4: Integrate subtitles using appropriate container format
            hard_flag = bool(self.config.hard_code_subtitles) and str(self.config.hard_code_subtitles).lower() != "false"
            processing_start_time = asyncio.get_event_loop().time()
            processing_ms = (asyncio.get_event_loop().time() - processing_start_time) * 1000
            ingest_to_process_ms = (processing_start_time - received_time) * 1000

            processed_data, error_msg = await self.subtitle_integrator.integrate_subtitles(
                segment_idx, video_file, srt_content, hard=hard_flag
            )
            self.segments_processed_count += 1

            if not processed_data:
                logger.error(f"Error processing segment {segment_idx}, skipping: {error_msg}")
                self.flow_metrics["segments_dropped"] += 1
                status = "error"

            else:
                status = "ok"

            stats = StreamStats(
                pipeline_uuid=self.config.pipeline_uuid,
                segment_idx=segment_idx,
                segments_processed_total=self.segments_processed_count,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                status=status,
                error_msg=error_msg,
                processing_ms=processing_ms,
                ingest_to_process_ms=ingest_to_process_ms,
            )
            if self.config.enable_events_url:
                await self.stats_queue.put(stats)

            if not processed_data:
                return (None, None)

            return (processed_data, srt_content if (self.config.enable_data_url and self.config.data_url and srt_content) else None)
            
        except Exception as e:
            logger.error(f"Error processing segment {segment_idx}: {e}")
            self.flow_metrics['errors'] += 1
            # Return original data with None for SRT to maintain tuple structure
            return segment_data, None
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
        # Use Trickle text publisher if available
        if self.running and self.text_publisher:
            try:
                writer = await self.text_publisher.next()
                # Create JSON structure for subtitle data
                subtitle_data = {
                    "segment_idx": segment_idx,
                    "srt_content": srt_content,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()
                }
                await writer.write(json.dumps(subtitle_data).encode("utf-8"))
                await writer.close()
                logger.debug(f"Sent subtitle for segment {segment_idx} via Trickle text channel")
                return
            except Exception as e:
                logger.error(f"Error sending subtitle for segment {segment_idx} to data URL: {e}")

    async def _send_events_from_stats_queue(self):
        """
        Continuously send events from the stats_queue to the events URL.
        """
        if not self.events_publisher:
            logger.warning("Events publisher is not configured.")
            return

        while self.running:
            try:
                stats: StreamStats = await asyncio.wait_for(self.stats_queue.get(), timeout=10.0)
                writer = await self.events_publisher.next()
                if stats:
                    await writer.write(json.dumps(asdict(stats)).encode("utf-8"))
                    await writer.close()
                logger.debug(f"Sent event for segment {stats.segment_idx} via Trickle events channel")
                self.stats_queue.task_done()
            except Exception as e:
                logger.error(f"Error sending event to events URL: {e}")
                await asyncio.sleep(0.1) 
                break 
                 
                
    async def stop(self):
        """Stop the pipeline."""
        self.running = False
        logger.info("Pipeline stop requested")

    async def cleanup(self):
        """Cleanup pipeline resources."""
        try:
            # Close text publisher if open
            if self.text_publisher:
                await self.text_publisher.close()
            if self.events_publisher:
                await self.events_publisher.close()

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
