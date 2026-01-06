"""
Diarization process management for speaker diarization.

This module provides a separate process for running pyannote speaker diarization,
isolating it from the main async event loop.
"""

import asyncio
import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import deque
import tempfile
import wave
import numpy as np
import math

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing information."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str   # Speaker label
    confidence: float  # Confidence score for the speaker assignment
    alt_speakers: Optional[Dict[str, float]] = None  # Alternative speakers with scores

@dataclass
class DiarizationRequest:
    """Request for diarization processing."""
    audio_path: str
    request_id: str


@dataclass
class DiarizationResult:
    """Result from diarization processing."""
    request_id: str
    audio_path: str
    segments: List[SpeakerSegment]
    error: Optional[str] = None


class SpeakerMemory:
    """
    Track speakers across audio segments using embedding similarity.
    
    Features:
    - Running average of speaker embeddings (more stable than single sample)
    - Temporal context (recent speaker more likely to continue)
    - Configurable similarity threshold
    - Speaker history for debugging
    """
    
    def __init__(
        self, 
        threshold: float = 0.91,
        recency_boost: float = 0.00,
        history_size: int = 20,
        min_samples_for_match: int = 1
    ):
        """
        Args:
            threshold: Cosine similarity threshold for speaker match (0.65-0.75 typical)
            recency_boost: Bonus added to most recent speaker's similarity
            history_size: Number of recent speaker IDs to track
            min_samples_for_match: Minimum observations before matching against a speaker
        """
        self.threshold = threshold
        self.recency_boost = recency_boost
        self.min_samples_for_match = min_samples_for_match
        
        # Speaker profiles
        self.centroids: Dict[str, np.ndarray] = {}  # speaker_id -> normalized embedding
        self.counts: Dict[int, int] = {}             # speaker_id -> sample count
        self.speaker_counter = 0
        
        # Temporal tracking
        self.history: deque = deque(maxlen=history_size)
        self.last_speaker: Optional[str] = None
        
        # Debugging
        self.match_log: List[dict] = []
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between normalized embeddings.
        Both inputs should already be normalized.
        """
        return float(np.dot(a, b))
    
    def identify(
        self, 
        embedding: np.ndarray,
        segment_info: Optional[dict] = None
    ) -> Tuple[str, float]:
        """
        Match embedding to existing speaker or create new one.
        
        Args:
            embedding: Raw speaker embedding (will be normalized)
            segment_info: Optional metadata for logging (timestamp, text, etc)
            
        Returns:
            (speaker_id, confidence_score)
        """
        if len(embedding) == 0:
            logger.info("Empty embedding received, cannot identify speaker")
            return "unknown", 0.0
        
        embedding = self._normalize(embedding)
        
        if len(self.centroids) == 0:
            # First speaker
            return self._create_speaker(embedding, segment_info, confidence=1.0)
        
        # Compare against all known speakers
        best_id = None
        best_score = -1.0
        scores = {}
        alt_speakers = {}

        for speaker_id, centroid in self.centroids.items():
            # Skip speakers with too few samples (unstable centroids)
            if self.counts[speaker_id] < self.min_samples_for_match:
                continue
            
            score = self._cosine_similarity(embedding, centroid)
            
            # Recency boost: prefer continuing same speaker
            if speaker_id == self.last_speaker:
                score += self.recency_boost
            
            scores[speaker_id] = score
            
            # Build alt_speakers in the loop
            if score >= self.threshold or score >= (self.threshold - 0.1):
                alt_speakers[speaker_id] = score

            if score > best_score:
                best_score = score
                best_id = speaker_id
        
        # Log for debugging
        self.match_log.append({
            'scores': scores.copy(),
            'threshold': self.threshold,
            'best_id': best_id,
            'best_score': best_score,
            'segment_info': segment_info
        })
        logger.info(f"Speaker match scores: {scores}, best: {best_id} ({best_score:.3f})")
        # Match or create new speaker
        if best_id and best_score >= self.threshold:
            #self._update_speaker(best_id, embedding)
            self.last_speaker = best_id
            self.history.append(best_id)
            return best_id, best_score, alt_speakers
        else:
            speaker_id, confidence = self._create_speaker(embedding, segment_info, confidence=best_score)
            return speaker_id, confidence, alt_speakers
            
    def _create_speaker(
        self, 
        embedding: np.ndarray, 
        segment_info: Optional[dict],
        confidence: float
    ) -> Tuple[str, float]:
        """Create a new speaker profile."""
        speaker_id = f"speaker_{self.speaker_counter}"
        self.speaker_counter += 1
        
        self.centroids[speaker_id] = embedding
        self.counts[speaker_id] = 1
        self.last_speaker = speaker_id
        self.history.append(speaker_id)
        
        return speaker_id, confidence
    
    def _update_speaker(self, speaker_id: str, embedding: np.ndarray):
        """Update speaker centroid with running average."""
        n = self.counts[speaker_id]
        
        # Weighted average: new centroid = (old * n + new) / (n + 1)
        new_centroid = (self.centroids[speaker_id] * n + embedding) / (n + 1)
        
        # Re-normalize to maintain unit length
        self.centroids[speaker_id] = self._normalize(new_centroid)
        self.counts[speaker_id] += 1
    
    def merge_speakers(self, speaker_a: str, speaker_b: str):
        """
        Manually merge two speakers (useful for correcting errors).
        Keeps speaker_a, merges speaker_b into it.
        """
        logger.info(f"Merging speakers: {speaker_a} <- {speaker_b}")
        if speaker_a not in self.centroids or speaker_b not in self.centroids:
            raise ValueError("Both speakers must exist")
        
        # Weighted average of centroids
        n_a = self.counts[speaker_a]
        n_b = self.counts[speaker_b]
        merged = (self.centroids[speaker_a] * n_a + self.centroids[speaker_b] * n_b) / (n_a + n_b)
        
        self.centroids[speaker_a] = self._normalize(merged)
        self.counts[speaker_a] = n_a + n_b
        
        # Remove speaker_b
        del self.centroids[speaker_b]
        del self.counts[speaker_b]
        
        # Update history
        self.history = deque(
            [speaker_a if s == speaker_b else s for s in self.history],
            maxlen=self.history.maxlen
        )
    
    def get_speaker_stats(self) -> dict:
        """Get statistics about tracked speakers."""
        return {
            speaker_id: {
                'count': self.counts[speaker_id],
                'last_seen': speaker_id == self.last_speaker,
                'in_recent_history': speaker_id in self.history
            }
            for speaker_id in self.centroids.keys()
        }
    
    def reset(self):
        """Clear all speaker data."""
        self.centroids.clear()
        self.counts.clear()
        self.history.clear()
        self.last_speaker = None
        self.speaker_counter = 0
        self.match_log.clear()

def diarization_worker(hf_token: str, request_queue, result_queue):
    """
    Worker function for diarization process.
    
    Args:
        hf_token: HuggingFace token for accessing pyannote models
        request_queue: Multiprocessing queue for receiving requests
        result_queue: Multiprocessing queue for sending results
    """
    logger.info("Diarization worker starting")
    
    if Pipeline is None:
        logger.error("pyannote.audio not installed")
        return
    
    try:
        # Initialize pipeline
        import torch
        use_cuda = False #torch.cuda.is_available(), not working on cuda for some reason
        if use_cuda:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token
            ).cuda()
            logger.info("Diarization pipeline initialized on CUDA device")
        else:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=hf_token
            )
            logger.info("Diarization pipeline initialized on CPU")
    except Exception as e:
        logger.error(f"Failed to initialize diarization pipeline: {e}")
        return
    
    speakers = SpeakerMemory(threshold=0.70)

    while True:
        try:
            request = request_queue.get(timeout=1.0)
            logger.info(f"Received diarization request: {request.request_id}, audio: {request.audio_path}")
            if request is None:  # Shutdown signal
                break
            
            try:
                # Run diarization
                diarization_result = pipeline(request.audio_path)
                #logger.info(f"Diarization result for {request.request_id}: {diarization_result}")
                #get speaker ids
                speaker_ids = {}
                for s, speaker in enumerate(diarization_result.speaker_diarization.labels()):
                    speaker_embedding = diarization_result.speaker_embeddings[s]
                    speaker_id, score, alt_speakers = speakers.identify(speaker_embedding)
                    speaker_ids[speaker] = {"id": speaker_id, "score": score, "alt_speakers": alt_speakers}
                # Extract segments
                segments = []
                for turn, speaker in diarization_result.speaker_diarization:
                    segments.append(SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=str(speaker_ids[speaker]["id"]),
                        confidence=str(speaker_ids[speaker]["score"]),
                        alt_speakers=speaker_ids[speaker]["alt_speakers"] if len(speaker_ids[speaker]["alt_speakers"]) > 0 else None
                    ))
                logger.info(f"Extracted {len(segments)} speaker segments")
                if len(segments) > 1:
                    logger.info(f"Speakers identified: {set(s.speaker for s in segments)}")
                    logger.debug(f"diarization result: {diarization_result}")
                
                # Send result
                result = DiarizationResult(
                    request_id=request.request_id,
                    audio_path=request.audio_path,
                    segments=segments
                )
                result_queue.put(result)
                logger.debug(f"Sent diarization result for {request.request_id}")
                
            except Exception as e:
                logger.error(f"Diarization processing error: {e}")
                result_queue.put(DiarizationResult(
                    request_id=request.request_id,
                    audio_path=request.audio_path,
                    segments=[],
                    error=str(e)
                ))
                
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            logger.error(f"Worker error: {e}")


class DiarizationProcess:
    """Manages a separate process for speaker diarization."""
    
    def __init__(self, hf_token: str):
        """
        Initialize the diarization process manager.
        
        Args:
            hf_token: HuggingFace token for accessing pyannote models
        """
        self.hf_token = hf_token
        self._process: Optional[multiprocessing.Process] = None
        self._request_queue: Optional[multiprocessing.Queue] = None
        self._result_queue: Optional[multiprocessing.Queue] = None
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the diarization process."""
        logger.debug("DiarizationProcess.start() called")
        async with self._lock:
            if self._running:
                logger.info("Diarization process already running")
                return
            
            # Create queues
            self._request_queue = multiprocessing.Queue()
            self._result_queue = multiprocessing.Queue()
            
            # Start process
            logger.info("Starting diarization worker process")
            self._process = multiprocessing.Process(
                target=diarization_worker,
                args=(self.hf_token, self._request_queue, self._result_queue),
                daemon=True
            )
            self._process.start()
            self._running = True
            logger.info("Diarization process started")
    
    async def stop(self):
        """Stop the diarization process gracefully."""
        logger.info("DiarizationProcess.stop() called")
        async with self._lock:
            if not self._running:
                return
            
            # Send shutdown signal
            if self._request_queue is not None:
                logger.info("Sending shutdown signal to diarization worker")
                self._request_queue.put(None)
            
            # Wait for process to finish
            if self._process is not None:
                self._process.join(timeout=5.0)
                if self._process.is_alive():
                    self._process.terminate()
            
            # Cleanup
            self._running = False
            self._process = None
            self._request_queue = None
            self._result_queue = None
            logger.info("Diarization process stopped")
    
    async def reset(self):
        """Reset internal state without stopping the process.
        
        This clears queues and resets state for a new stream while
        keeping the diarization worker process running.
        """
        async with self._lock:
            # Clear queues to discard pending requests from previous stream
            if self._request_queue is not None:
                try:
                    while not self._request_queue.empty():
                        self._request_queue.get_nowait()
                except Exception:
                    pass
            if self._result_queue is not None:
                try:
                    while not self._result_queue.empty():
                        self._result_queue.get_nowait()
                except Exception:
                    pass
            logger.info("Diarization process state reset")
    
    async def process_audio(self, audio_path: str, request_id: str):
        """
        Send audio for diarization processing.
        
        Args:
            audio_path: Path to the audio file
            request_id: Unique request identifier
        """
        #logger.info(f"DiarizationProcess.process_audio() called: {audio_path}, {request_id}")
        if not self._running:
            raise RuntimeError("Diarization process not started")
        
        request = DiarizationRequest(
            audio_path=audio_path,
            request_id=request_id
        )
        #logger.info(f"Putting request in queue: {request_id}")
        self._request_queue.put(request)
        #logger.info(f"Request put in queue: {request_id}")
    
    async def get_result(self, timeout: float = 10.0) -> Optional[DiarizationResult]:
        """
        Get a diarization result.
        
        Args:
            timeout: Maximum time to wait for a result
            
        Returns:
            DiarizationResult or None if timeout
        """
        if not self._running:
            logger.warning(f"get_result called but _running=False, request_id would be unknown")
            raise RuntimeError("Diarization process not started")
        
        try:
            return await asyncio.to_thread(self._result_queue.get, timeout=timeout)
        except multiprocessing.queues.Empty:
            logger.debug(f"get_result timeout - queue empty")
            return None
    
    @property
    def is_running(self) -> bool:
        """Check if the process is running."""
        return self._running
