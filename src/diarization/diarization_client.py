"""
Diarization client for speaker diarization using pyannote.audio.

This module provides a client class for running pyannote speaker diarization,
with proper lifecycle management similar to SummaryClient.

Features:
- Speaker embedding quality validation (filters zero/invalid embeddings)
- Segment duration filtering (skips short segments)
- Enhanced pyannote pipeline configuration
- Statistics tracking and monitoring callbacks
"""

import asyncio
import logging
import multiprocessing
import threading
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable, Union
from collections import deque
import tempfile
import wave
import numpy as np

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None

from diarization.embedding_validator import EmbeddingQualityValidator, create_default_validator

logger = logging.getLogger(__name__)

# Special marker for reset signal
RESET_SIGNAL = "RESET_SIGNAL"

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
    - Embedding quality validation (filters zero/invalid embeddings)
    - Statistics tracking and monitoring callbacks
    - Segment duration filtering
    """
    
    def __init__(
        self,
        threshold: float = 0.91,
        recency_boost: float = 0.00,
        history_size: int = 20,
        min_samples_for_match: int = 1,
        min_segment_duration: float = 0.5,
        embedding_validator: Optional[EmbeddingQualityValidator] = None,
        on_invalid_embedding: Optional[Callable[[np.ndarray, str, Optional[dict]], None]] = None,
        on_low_confidence: Optional[Callable[[str, float, Optional[dict]], None]] = None
    ):
        """
        Args:
            threshold: Cosine similarity threshold for speaker match (0.65-0.75 typical)
            recency_boost: Bonus added to most recent speaker's similarity
            history_size: Number of recent speaker IDs to track
            min_samples_for_match: Minimum observations before matching against a speaker
            min_segment_duration: Minimum segment duration in seconds (default: 0.5)
            embedding_validator: Custom embedding validator (uses default if None)
            on_invalid_embedding: Callback for invalid embeddings: (embedding, reason, segment_info)
            on_low_confidence: Callback for low confidence matches: (speaker_id, confidence, segment_info)
        """
        self.threshold = threshold
        self.recency_boost = recency_boost
        self.min_samples_for_match = min_samples_for_match
        self.min_segment_duration = min_segment_duration
        
        # Embedding quality validation
        if embedding_validator is not None:
            self.validator = embedding_validator
        else:
            self.validator = create_default_validator()
        
        # Set up monitoring callbacks
        self.on_invalid_embedding = on_invalid_embedding
        self.on_low_confidence = on_low_confidence
        
        # Configure validator callback if provided
        if on_invalid_embedding:
            self.validator.on_invalid = on_invalid_embedding
        
        # Speaker profiles
        self.centroids: Dict[str, np.ndarray] = {}  # speaker_id -> normalized embedding
        self.counts: Dict[int, int] = {}             # speaker_id -> sample count
        self.speaker_counter = 0
        
        # Temporal tracking
        self.history: deque = deque(maxlen=history_size)
        self.last_speaker: Optional[str] = None
        
        # Debugging
        self.match_log: List[dict] = []
        
        # Statistics tracking
        self._stats = {
            "identifications": 0,
            "valid_embeddings": 0,
            "invalid_embeddings": 0,
            "matches": 0,
            "new_speakers": 0,
            "low_confidence_matches": 0,
            "skipped_short_segments": 0,
        }
    
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
        Returns -1.0 if NaN is detected (will fail threshold check).
        """
        dot_product = np.dot(a, b)
        if np.isnan(dot_product) or np.isinf(dot_product):
            logger.warning(f"Invalid cosine similarity result: {dot_product}")
            return -1.0  # Return invalid value that will fail threshold
        return float(dot_product)
    
    def identify(
        self,
        embedding: np.ndarray,
        segment_info: Optional[dict] = None
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Match embedding to existing speaker or create new one.
        
        Args:
            embedding: Raw speaker embedding (will be normalized)
            segment_info: Optional metadata for logging (timestamp, text, etc)
            
        Returns:
            (speaker_id, confidence_score, alt_speakers)
        """
        self._stats["identifications"] += 1
        
        # Validate embedding quality FIRST
        if not self.validator.validate(embedding, segment_info):
            self._stats["invalid_embeddings"] += 1
            logger.debug(f"Invalid embedding rejected for segment: {segment_info}")
            return "unknown", 0.0, {}
        
        self._stats["valid_embeddings"] += 1
        
        embedding = self._normalize(embedding)
        
        # Clean up any invalid speakers that may have been created before validation
        # This prevents NaN issues from zero embeddings in speaker memory
        self.cleanup_invalid_speakers()
        
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
            self._stats["matches"] += 1
            
            # Check for low confidence match
            low_confidence_threshold = 0.75
            if best_score < low_confidence_threshold:
                self._stats["low_confidence_matches"] += 1
                if self.on_low_confidence:
                    try:
                        self.on_low_confidence(best_id, best_score, segment_info)
                    except Exception as e:
                        logger.warning(f"Error in on_low_confidence callback: {e}")
            
            #self._update_speaker(best_id, embedding)
            self.last_speaker = best_id
            self.history.append(best_id)
            return best_id, best_score, alt_speakers
        else:
            self._stats["new_speakers"] += 1
            speaker_id, confidence, _ = self._create_speaker(embedding, segment_info, confidence=best_score)
            return speaker_id, confidence, alt_speakers
            
    def _create_speaker(
        self,
        embedding: np.ndarray,
        segment_info: Optional[dict],
        confidence: float
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """Create a new speaker profile with validation."""
        # Double-check validation before storage
        is_valid, reason = self.validator.is_valid(embedding)
        if not is_valid:
            logger.warning(f"Attempted to create speaker with invalid embedding: {reason}")
            return "unknown", 0.0, {}
        
        speaker_id = f"speaker_{self.speaker_counter}"
        self.speaker_counter += 1
        
        self.centroids[speaker_id] = embedding
        self.counts[speaker_id] = 1
        self.last_speaker = speaker_id
        self.history.append(speaker_id)
        
        logger.info(f"Created new speaker: {speaker_id} (confidence: {confidence:.3f})")
        return speaker_id, confidence, {}
    
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
        
        # Use internal merge method
        self._merge_speakers_internal(speaker_a, speaker_b)
        
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
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics about speaker memory operations."""
        total = self._stats["identifications"]
        valid = self._stats["valid_embeddings"]
        
        return {
            **self._stats,
            "speaker_count": len(self.centroids),
            "total_samples": sum(self.counts.values()),
            "avg_samples_per_speaker": (
                sum(self.counts.values()) / max(1, len(self.centroids))
            ),
            "validation_rate": valid / max(1, total),
            "validator_stats": self.validator.get_validation_stats(),
        }
    
    def find_similar_speakers(self, similarity_threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """
        Find pairs of speakers with high similarity (potential duplicates).
        
        Args:
            similarity_threshold: Pairs above this threshold are considered similar
            
        Returns:
            List of (speaker_a, speaker_b, similarity) tuples, sorted by similarity
        """
        similar_pairs = []
        speaker_ids = list(self.centroids.keys())
        
        for i, speaker_a in enumerate(speaker_ids):
            for speaker_b in speaker_ids[i + 1:]:
                # Skip if either speaker has too few samples
                if (self.counts.get(speaker_a, 0) < self.min_samples_for_match or
                    self.counts.get(speaker_b, 0) < self.min_samples_for_match):
                    continue
                
                similarity = self._cosine_similarity(
                    self.centroids[speaker_a],
                    self.centroids[speaker_b]
                )
                
                if similarity >= similarity_threshold:
                    similar_pairs.append((speaker_a, speaker_b, similarity))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def auto_merge_similar_speakers(
        self,
        similarity_threshold: float = 0.85,
        max_merges: int = 5
    ) -> int:
        """
        Automatically merge speakers that are very similar (likely duplicates).
        
        This helps consolidate speakers that were incorrectly split during
        diarization. Uses a greedy approach: merge the most similar pair,
        update centroids, then repeat until no more pairs exceed threshold.
        
        Args:
            similarity_threshold: Pairs above this threshold will be merged
            max_merges: Maximum number of merges to perform in one call
            
        Returns:
            Number of merges performed
        """
        merges_performed = 0
        
        for _ in range(max_merges):
            similar_pairs = self.find_similar_speakers(similarity_threshold)
            
            if not similar_pairs:
                break
            
            # Merge the most similar pair
            speaker_a, speaker_b, similarity = similar_pairs[0]
            
            logger.info(f"Auto-merging speakers: {speaker_a} <-> {speaker_b} "
                       f"(similarity: {similarity:.3f})")
            
            # Use speaker_a as the target, merge speaker_b into it
            self._merge_speakers_internal(speaker_a, speaker_b)
            merges_performed += 1
            
            # Update history - replace speaker_b with speaker_a
            self.history = deque(
                [speaker_a if s == speaker_b else s for s in self.history],
                maxlen=self.history.maxlen
            )
        
        if merges_performed > 0:
            logger.info(f"Auto-merge completed: {merges_performed} merges performed")
        
        return merges_performed
    
    def _merge_speakers_internal(self, speaker_a: str, speaker_b: str):
        """
        Internal method to merge speaker_b into speaker_a.
        Called by both merge_speakers() and auto_merge_similar_speakers().
        """
        if speaker_a not in self.centroids or speaker_b not in self.centroids:
            raise ValueError("Both speakers must exist")
        
        # Weighted average of centroids based on sample counts
        n_a = self.counts[speaker_a]
        n_b = self.counts[speaker_b]
        merged = (self.centroids[speaker_a] * n_a + self.centroids[speaker_b] * n_b) / (n_a + n_b)
        
        self.centroids[speaker_a] = self._normalize(merged)
        self.counts[speaker_a] = n_a + n_b
        
        # Remove speaker_b
        del self.centroids[speaker_b]
        del self.counts[speaker_b]
    
    def cleanup_invalid_speakers(self) -> int:
        """
        Remove speakers with invalid embeddings (zeros, NaN, or very low quality).
        This is useful for cleaning up speaker memory that may have been created
        before validation was in place.
        
        Returns:
            Number of invalid speakers removed.
        """
        invalid_speakers = []
        
        for speaker_id, embedding in self.centroids.items():
            # Check if embedding is valid
            is_valid, reason = self.validator.is_valid(embedding)
            if not is_valid:
                invalid_speakers.append(speaker_id)
                logger.warning(f"Removing invalid speaker {speaker_id}: {reason}")
        
        # Remove invalid speakers
        for speaker_id in invalid_speakers:
            del self.centroids[speaker_id]
            del self.counts[speaker_id]
            # Also remove from history
            self.history = deque(
                [s for s in self.history if s != speaker_id],
                maxlen=self.history.maxlen
            )
        
        if invalid_speakers:
            logger.info(f"Cleaned up {len(invalid_speakers)} invalid speakers")
        
        return len(invalid_speakers)
    
    def reset(self):
        """Clear all speaker data and statistics."""
        self.centroids.clear()
        self.counts.clear()
        self.history.clear()
        self.last_speaker = None
        self.speaker_counter = 0
        self.match_log.clear()
        
        # Reset statistics
        self._stats = {
            "identifications": 0,
            "valid_embeddings": 0,
            "invalid_embeddings": 0,
            "matches": 0,
            "new_speakers": 0,
            "low_confidence_matches": 0,
            "skipped_short_segments": 0,
        }
        
        # Reset validator stats
        self.validator.reset_stats()


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
        
        # Note: The community pipeline version doesn't support parameter instantiation
        # with custom clustering parameters. The default pyannote parameters are already
        # optimized for most use cases. If you need custom clustering, consider using
        # the full pyannote-audio library with a custom configuration.
        logger.debug("Using default pyannote pipeline configuration")
        
    except Exception as e:
        logger.error(f"Failed to initialize diarization pipeline: {e}")
        return
    
    # Minimum segment duration to process (skip very short segments)
    MIN_SEGMENT_DURATION = 0.5  # seconds
    
    # Initialize speaker memory with validation
    speakers = SpeakerMemory(threshold=0.70)

    shutdown_received = False
    while True:
        try:
            # Check for shutdown signal with timeout
            try:
                request = request_queue.get(timeout=1.0)
            except multiprocessing.queues.Empty:
                # Queue empty - check if we should shutdown
                if shutdown_received:
                    logger.info("Diarization worker shutting down - all work processed")
                    break
                continue
            
            # Handle shutdown signal
            if request is None:
                # Check if there are pending requests before shutting down
                if not request_queue.empty():
                    # Mark that we received shutdown signal, continue processing
                    shutdown_received = True
                    logger.info("Diarization worker received shutdown signal, processing remaining items")
                    continue
                else:
                    # Queue is empty, safe to exit
                    logger.info("Diarization worker shutting down - queue empty")
                    break
            
            # Handle reset signal for new stream
            if request == RESET_SIGNAL:
                shutdown_received = False  # Reset for new stream
                logger.info("Received reset signal - clearing speaker memory for new stream")
                speakers.reset()
                # CRITICAL FIX: Do NOT continue loop here - we need to re-check queue state
                # The issue is that after clearing queues in reset_process(), the queue is empty
                # If we continue and find queue empty with shutdown_received=False, we won't exit
                # But if shutdown_received was True before, we need to ensure it stays False
                # Actually, the issue is different - let me trace through the flow more carefully
                logger.debug(f"After RESET_SIGNAL: shutdown_received={shutdown_received}, queue.empty()={request_queue.empty()}")

            logger.info(f"Received diarization request: {request.request_id}, audio: {request.audio_path}")
            try:
                # Run diarization
                diarization_result = pipeline(request.audio_path)
                
                # Get speaker embeddings and identify speakers
                speaker_ids = {}
                for s, speaker in enumerate(diarization_result.speaker_diarization.labels()):
                    speaker_embedding = diarization_result.speaker_embeddings[s]
                    segment_info = {
                        "request_id": request.request_id,
                        "speaker_label": speaker,
                        "embedding_index": s
                    }
                    speaker_id, score, alt_speakers = speakers.identify(speaker_embedding, segment_info)
                    speaker_ids[speaker] = {"id": speaker_id, "score": score, "alt_speakers": alt_speakers}
                
                # Extract segments with duration filtering
                segments = []
                skipped_count = 0
                for turn, speaker in diarization_result.speaker_diarization:
                    segment_duration = turn.end - turn.start
                    
                    # Skip very short segments (often noise or artifacts)
                    if segment_duration < MIN_SEGMENT_DURATION:
                        skipped_count += 1
                        logger.debug(f"Skipping short segment: {segment_duration:.2f}s < {MIN_SEGMENT_DURATION}s")
                        continue
                    
                    segments.append(SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=str(speaker_ids[speaker]["id"]),
                        confidence=str(speaker_ids[speaker]["score"]),
                        alt_speakers=speaker_ids[speaker]["alt_speakers"] if len(speaker_ids[speaker]["alt_speakers"]) > 0 else None
                    ))
                
                if skipped_count > 0:
                    logger.info(f"Skipped {skipped_count} short segments (< {MIN_SEGMENT_DURATION}s)")
                
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


class DiarizationClient:
    """Client for speaker diarization using pyannote.audio."""
    
    def __init__(
        self,
        hf_token: str = "",
        threshold: float = 0.70,
        recency_boost: float = 0.00,
        history_size: int = 20,
        min_samples_for_match: int = 1,
        min_segment_duration: float = 0.5,
        embedding_validator: Optional[EmbeddingQualityValidator] = None,
        on_invalid_embedding: Optional[Callable[[np.ndarray, str, Optional[dict]], None]] = None,
        on_low_confidence: Optional[Callable[[str, float, Optional[dict]], None]] = None
    ):
        """
        Initialize the diarization client.
        
        Args:
            hf_token: HuggingFace token for accessing pyannote models
            threshold: Cosine similarity threshold for speaker match
            recency_boost: Bonus added to most recent speaker's similarity
            history_size: Number of recent speaker IDs to track
            min_samples_for_match: Minimum observations before matching against a speaker
            min_segment_duration: Minimum segment duration in seconds (default: 0.5)
            embedding_validator: Custom embedding validator (uses default if None)
            on_invalid_embedding: Callback for invalid embeddings
            on_low_confidence: Callback for low confidence matches
        """
        self.hf_token = hf_token
        self.threshold = threshold
        self.recency_boost = recency_boost
        self.history_size = history_size
        self.min_samples_for_match = min_samples_for_match
        self.min_segment_duration = min_segment_duration
        self.embedding_validator = embedding_validator
        self.on_invalid_embedding = on_invalid_embedding
        self.on_low_confidence = on_low_confidence
        
        # State management
        self._lock: Optional[threading.Lock] = None
        self._speaker_memory: Optional[SpeakerMemory] = None
        self._process: Optional[multiprocessing.Process] = None
        self._request_queue: Optional[multiprocessing.Queue] = None
        self._result_queue: Optional[multiprocessing.Queue] = None
        self._running = False
        
        # In-flight tracking for graceful shutdown
        self.in_flight_requests: set[str] = set()  # Track request IDs being processed
    
    async def initialize(self):
        """Initialize lock and speaker memory."""
        
        if self._lock is None:
            self._lock = threading.Lock()
        
        self._speaker_memory = SpeakerMemory(
            threshold=self.threshold,
            recency_boost=self.recency_boost,
            history_size=self.history_size,
            min_samples_for_match=self.min_samples_for_match,
            min_segment_duration=self.min_segment_duration,
            embedding_validator=self.embedding_validator,
            on_invalid_embedding=self.on_invalid_embedding,
            on_low_confidence=self.on_low_confidence
        )
        
        logger.info(f"DiarizationClient initialized with threshold={self.threshold}, min_segment_duration={self.min_segment_duration}s")
    
    async def start(self):
        """Start the diarization process."""
        logger.debug("DiarizationClient.start() called")
        with self._lock:
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
    
    def update_params(
        self,
        threshold: Optional[float] = None,
        recency_boost: Optional[float] = None,
        history_size: Optional[int] = None,
        min_samples_for_match: Optional[int] = None,
        min_segment_duration: Optional[float] = None
    ):
        """
        Update client parameters dynamically.
        
        Args:
            threshold: New similarity threshold
            recency_boost: New recency boost value
            history_size: New history size
            min_samples_for_match: New min samples for match
            min_segment_duration: New minimum segment duration in seconds
        """
        if threshold is not None:
            self.threshold = threshold
        if recency_boost is not None:
            self.recency_boost = recency_boost
        if history_size is not None:
            self.history_size = history_size
        if min_samples_for_match is not None:
            self.min_samples_for_match = min_samples_for_match
        if min_segment_duration is not None:
            self.min_segment_duration = min_segment_duration
        
        logger.info(f"DiarizationClient params updated: threshold={self.threshold}, recency_boost={self.recency_boost}, min_segment_duration={self.min_segment_duration}")
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics from speaker memory."""
        if self._speaker_memory is not None:
            return self._speaker_memory.get_stats()
        return {}
    
    def reset(self):
        """Reset all accumulated state.
        
        NOTE: This method does NOT stop the worker process anymore.
        The worker process continues running across stream boundaries.
        Only the speaker memory and in-flight requests are cleared.
        """
        logger.info("DiarizationClient.reset() - clearing state WITHOUT stopping worker")
        if self._speaker_memory is not None:
            self._speaker_memory.reset()
        self.in_flight_requests.clear()
        # NOTE: We no longer call reset_process() here because that would:
        # 1. Send RESET_SIGNAL to worker
        # 2. Then stop_process() terminates the worker before it can process the signal
        # The worker should keep running across streams - we only need to clear speaker memory
        logger.info("DiarizationClient state reset (worker still running)")
    
    def add_in_flight_request(self, request_id: str):
        """Add request ID to in-flight tracking."""
        self.in_flight_requests.add(request_id)
    
    def remove_in_flight_request(self, request_id: str):
        """Remove request ID from in-flight tracking."""
        self.in_flight_requests.discard(request_id)
    
    def get_pending_count(self) -> int:
        """Get count of pending diarization requests."""
        return len(self.in_flight_requests)
    
    def send_shutdown_signal(self):
        """Send shutdown signal to worker process."""
        if self._request_queue is not None:
            self._request_queue.put(None)
            logger.info("DiarizationClient sent shutdown signal to worker")
    
    def is_idle(self) -> bool:
        """Check if worker is idle (no pending work and queue empty)."""
        if not self._running:
            return True
        if self._request_queue is None:
            return True
        try:
            return self._request_queue.empty() and len(self.in_flight_requests) == 0
        except Exception:
            return False
    
    async def stop_process(self):
        """Stop the diarization process gracefully."""
        logger.info("DiarizationClient.stop_process() called")
        with self._lock:
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
    
    def reset_process(self):
        """Reset internal state without stopping the process.
        
        This clears queues and resets state for a new stream while
        keeping the diarization worker process running.
        """
        with self._lock:
            # Clear queues to discard pending requests from previous stream
            if self._request_queue is not None:
                try:
                    while not self._request_queue.empty():
                        self._request_queue.get_nowait()
                except Exception:
                    pass
                # Send reset signal to clear speaker memory in worker
                self._request_queue.put(RESET_SIGNAL)
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
        #logger.info(f"DiarizationClient.process_audio() called: {audio_path}, {request_id}")
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