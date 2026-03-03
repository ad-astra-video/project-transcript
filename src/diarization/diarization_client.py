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
from datetime import datetime

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

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
    merged_speakers: Optional[List[Dict[str, str]]] = None  # [{"source": "speaker_1", "target": "speaker_0"}]

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
        threshold: float = 0.72,
        recency_boost: float = 0.05,
        history_size: int = 20,
        min_samples_for_match: int = 5,
        min_segment_duration: float = 0.5,
        ema_alpha: float = 0.15,
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
            ema_alpha: EMA smoothing factor for centroid updates (default: 0.15)
        """
        self.threshold = threshold
        self.recency_boost = recency_boost
        self.min_samples_for_match = min_samples_for_match
        self.min_segment_duration = min_segment_duration
        self.ema_alpha = ema_alpha
        
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
        self._match_log: List[dict] = []
        self._recent_merges_log: List[dict] = []
        
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
    
    def _get_dynamic_threshold(self) -> float:
        """
        Lower threshold as speaker count grows to reduce over-segmentation.
        
        This dynamic adjustment helps prevent creating too many speakers
        in meetings with many participants by lowering the similarity
        threshold when speaker count increases.
        """
        n = len(self.centroids)
        base = self.threshold  # 0.72 (updated from 0.78)
        
        if n <= 4:
            return base
        elif n <= 8:
            return max(base - 0.08, 0.65)   # 0.64
        elif n <= 12:
            return max(base - 0.14, 0.60)   # 0.58
        else:
            return max(base - 0.20, 0.55)   # 0.52 (aggressive latch-on)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Robust cosine similarity with full input validation.
        Both inputs should already be normalized.
        Returns -1.0 if NaN/Inf is detected or inputs are invalid.
        """
        # Defensive input validation
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            return -1.0
        
        # Check for NaN/Inf in inputs
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            logger.warning("Invalid embedding(s) in cosine similarity")
            return -1.0
        
        # Ensure normalized (defensive)
        a = self._normalize(a)
        b = self._normalize(b)
        
        dot = np.dot(a, b)
        if not np.isfinite(dot):
            return -1.0
        
        # Round to 2 decimal places to avoid floating-point precision issues
        # This prevents cases like 0.7799... < 0.78 threshold comparison
        return round(float(np.clip(dot, -1.0, 1.0)), 2)
    
    def _compute_calibrated_confidence(
        self,
        scores: Dict[str, float],
        best_id: Optional[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute calibrated confidence from similarity scores.
        
        Uses softmax to convert raw cosine similarities to probabilities,
        then computes confidence based on the gap between best and second-best.
        
        Returns:
            (confidence_score, normalized_probabilities)
        """
        if not scores or best_id is None:
            return 0.0, {}
        
        # Convert to numpy array for processing
        speaker_ids = list(scores.keys())
        raw_scores = np.array(list(scores.values()))
        
        # Apply temperature scaling (higher temperature = softer distribution)
        # Use 1.0 for more balanced confidence
        temperature = 1.0
        scaled_scores = raw_scores / temperature
        
        # Softmax to convert to probabilities (with numerical stability)
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        probs = exp_scores / np.sum(exp_scores)
        
        # Get probability of best speaker
        max_prob = probs[speaker_ids.index(best_id)]
        
        # Compute gap-based confidence: difference between best and second-best
        sorted_probs = np.sort(probs)[::-1]  # Sort descending
        if len(sorted_probs) > 1:
            gap = sorted_probs[0] - sorted_probs[1]
        else:
            gap = 1.0
        
        # Combined confidence: use max_prob weighted by the gap
        # This rewards both high probability for best match AND clear separation
        confidence = max_prob * (0.7 + 0.3 * gap)
        
        # Return normalized probabilities
        prob_dict = {sid: float(p) for sid, p in zip(speaker_ids, probs)}
        
        return float(confidence), prob_dict
    
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
        
        # DEBUG: Log embedding stats before validation
        embedding_norm = np.linalg.norm(embedding)
        embedding_mean = np.mean(np.abs(embedding)) if len(embedding) > 0 else 0
        logger.debug(f"Embedding stats: norm={embedding_norm:.4f}, mean={embedding_mean:.8f}, shape={embedding.shape}")
        
        # Validate embedding quality FIRST (use validate() to track statistics and invoke callback)
        is_valid = self.validator.validate(embedding, segment_info)
        if not is_valid:
            self._stats["invalid_embeddings"] += 1
            # Get the reason from validator's last check
            _, validation_reason = self.validator.is_valid(embedding)
            logger.warning(f"Embedding validation FAILED: {validation_reason}, norm={embedding_norm:.4f}, mean={embedding_mean:.8f}")
            # Invalid embedding: skip segment entirely, do NOT assign to last speaker
            # Note: callback is already invoked by validator.validate()
            logger.debug(f"Invalid embedding rejected for segment: {segment_info}")
            return "unknown", 0.0, {}
        
        logger.debug(f"Embedding validation PASSED")
        
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

        # Dynamic threshold: use min of configured min_samples_for_match or total identifications
        effective_min = min(self._stats["identifications"], self.min_samples_for_match)
        
        # DEBUG: Log speaker state before matching
        total_samples = sum(self.counts.values())
        logger.info(f"Matching: {len(self.centroids)} speakers, total_samples={total_samples}, min_samples_for_match={self.min_samples_for_match}, effective_min={effective_min}")
        for speaker_id, centroid in self.centroids.items():
            # Skip speakers with too few samples (unstable centroids)
            if self.counts[speaker_id] < effective_min:
                logger.info(f"Skipping {speaker_id}: count={self.counts[speaker_id]} < effective_min={effective_min}")
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
        
        # Get dynamic threshold **before** logging
        effective_threshold = self._get_dynamic_threshold()

        # Log for debugging
        self._match_log.append({
            'scores': scores.copy(),
            'base_threshold': self.threshold,
            'effective_threshold': effective_threshold,
            'best_id': best_id,
            'best_score': best_score,
            'decision': 'match' if (best_id and best_score >= effective_threshold) else 'new_speaker',
            'segment_info': segment_info
        })
        logger.info(f"Speaker match scores: {scores}, best: {best_id} ({best_score:.3f})")
        
        # Get dynamic threshold based on speaker count
        effective_threshold = self._get_dynamic_threshold()
        
        # Match or create new speaker
        if best_id and best_score >= effective_threshold:
            self._stats["matches"] += 1
            
            # Compute calibrated confidence using softmax + entropy
            calibrated_confidence, speaker_probs = self._compute_calibrated_confidence(
                scores, best_id
            )
            
            # Log calibrated confidence for debugging
            logger.info(f"Calibrated confidence: {calibrated_confidence:.3f}, probs: {speaker_probs}")
            
            # Check for low confidence match using calibrated confidence
            # Lower threshold (0.65) since we now have calibrated scores
            low_confidence_threshold = 0.65
            if calibrated_confidence < low_confidence_threshold:
                self._stats["low_confidence_matches"] += 1
                if self.on_low_confidence:
                    try:
                        self.on_low_confidence(best_id, calibrated_confidence, segment_info)
                    except Exception as e:
                        logger.warning(f"Error in on_low_confidence callback: {e}")
            
            # Enable EMA centroid update
            self._update_speaker(best_id, embedding)
            self.last_speaker = best_id
            self.history.append(best_id)
            
            # Periodic consolidation: every 5 identifications or if speaker count exceeds 10
            if (self._stats["identifications"] % 5 == 0) or len(self.centroids) > 10:
                merges = self.auto_merge_similar_speakers(similarity_threshold=0.82)
                if merges:
                    logger.info(f"Periodic consolidation: {merges} merges")
            
            # Emergency cap: if speaker count exceeds 18, aggressive merge
            if len(self.centroids) > 18:
                self.auto_merge_similar_speakers(similarity_threshold=0.73, max_merges=10)
                logger.warning("Emergency merge triggered - speaker count exceeded 18")
            
            return best_id, best_score, alt_speakers
        else:
            self._stats["new_speakers"] += 1
            speaker_id, confidence, _ = self._create_speaker(embedding, segment_info, confidence=1.0)
            
            # Proactive merge: check if new speaker should be merged immediately
            merge_result = self._check_and_merge_after_creation(speaker_id, embedding)
            if merge_result:
                # Return the merged speaker ID
                return merge_result, self._cosine_similarity(embedding, self.centroids[merge_result]), alt_speakers
            
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
        """EMA update with configurable alpha and first-update special case."""
        n = self.counts[speaker_id]
        alpha = getattr(self, 'ema_alpha', 0.28)  # Make configurable
        
        if n == 1:
            # First update: blend equally
            self.centroids[speaker_id] = 0.5 * self.centroids[speaker_id] + 0.5 * embedding
        else:
            # EMA update
            self.centroids[speaker_id] = alpha * embedding + (1 - alpha) * self.centroids[speaker_id]
        
        # Re-normalize to maintain unit length
        self.centroids[speaker_id] = self._normalize(self.centroids[speaker_id])
        self.counts[speaker_id] += 1
    
    def _check_and_merge_after_creation(self, new_speaker_id: str, new_embedding: np.ndarray) -> Optional[str]:
        """Check if new speaker should be merged immediately."""
        for existing_id, centroid in list(self.centroids.items()):
            if existing_id == new_speaker_id:
                continue
            if self.counts.get(existing_id, 0) < 3:
                continue
            sim = self._cosine_similarity(new_embedding, centroid)
            
            # Immediate merge for very high similarity (>= 0.92)
            if sim >= 0.92:
                self._merge_speakers_internal(existing_id, new_speaker_id)
                logger.info(f"Immediate merge: {new_speaker_id} → {existing_id} (sim={sim:.3f})")
                return existing_id
            
            if sim >= 0.76:
                self._merge_speakers_internal(existing_id, new_speaker_id)
                logger.info(f"Proactive merge: {new_speaker_id} → {existing_id} (sim={sim:.3f})")
                return existing_id
        return None
    
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
    
    def _get_similarity_matrix(self) -> List[Dict]:
        """
        Get similarity matrix for top similar speaker pairs.
        
        Returns:
            List of dicts with speaker_a, speaker_b, similarity for top pairs.
        """
        similar_pairs = self.find_similar_speakers(similarity_threshold=0.7)
        return [
            {"speaker_a": a, "speaker_b": b, "similarity": float(sim)}
            for a, b, sim in similar_pairs[:10]  # Top 10 pairs
        ]
    
    def get_diagnostics(self) -> dict:
        """
        Production-grade diagnostics for speaker diarization.
        
        Returns comprehensive diagnostic information including:
        - Speaker count and sample statistics
        - Dynamic threshold value
        - Similarity matrix for potential duplicate speakers
        - Low confidence match rate
        - Recent merge events
        """
        total = self._stats["identifications"]
        matches = self._stats["matches"]
        
        return {
            "speaker_count": len(self.centroids),
            "total_samples": sum(self.counts.values()),
            "avg_samples_per_speaker": (
                sum(self.counts.values()) / max(1, len(self.centroids))
            ),
            "dynamic_threshold": self._get_dynamic_threshold(),
            "similarity_matrix": self._get_similarity_matrix(),
            "low_confidence_rate": self._stats["low_confidence_matches"] / max(1, matches),
            "validator_stats": self.validator.get_validation_stats(),
            "recent_merges": self._recent_merges_log[-10:],
            "config": {
                "threshold": self.threshold,
                "min_samples_for_match": self.min_samples_for_match,
                "ema_alpha": self.ema_alpha,
            },
            # Enhanced diagnostics
            "score_distribution": self._get_score_distribution(),
            "speaker_timeline": list(self.history),
            "avg_confidence": self._compute_avg_confidence(),
        }
    
    def _get_score_distribution(self) -> dict:
        """Get distribution of match scores."""
        if not self._match_log:
            return {}
        
        scores = [m['best_score'] for m in self._match_log if m.get('best_score')]
        if not scores:
            return {}
        
        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'p25': float(np.percentile(scores, 25)),
            'p50': float(np.percentile(scores, 50)),
            'p75': float(np.percentile(scores, 75)),
        }
    
    def _compute_avg_confidence(self) -> float:
        """Compute average confidence from recent match log."""
        if not self._match_log:
            return 1.0
        
        recent = self._match_log[-20:]  # Last 20 matches
        confidences = []
        for entry in recent:
            if entry.get('decision') == 'match':
                scores = entry.get('scores', {})
                best_id = entry.get('best_id')
                if scores and best_id:
                    conf, _ = self._compute_calibrated_confidence(scores, best_id)
                    confidences.append(conf)
        
        return float(np.mean(confidences)) if confidences else 1.0
    
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
        
        # DEBUG: Log all centroid-to-centroid similarities
        all_similarities = []
        
        for i, speaker_a in enumerate(speaker_ids):
            for speaker_b in speaker_ids[i + 1:]:
                # Skip if either speaker has too few samples
                if (self.counts.get(speaker_a, 0) < self.min_samples_for_match or
                    self.counts.get(speaker_b, 0) < self.min_samples_for_match):
                    logger.debug(f"Skipping merge check {speaker_a} vs {speaker_b}: counts {self.counts.get(speaker_a, 0)}/{self.counts.get(speaker_b, 0)} < min_samples_for_match={self.min_samples_for_match}")
                    continue
                
                similarity = self._cosine_similarity(
                    self.centroids[speaker_a],
                    self.centroids[speaker_b]
                )
                
                all_similarities.append((speaker_a, speaker_b, similarity))
                
                if similarity >= similarity_threshold:
                    logger.info(f"Auto-merge candidate: {speaker_a} <-> {speaker_b}, similarity={similarity:.3f}")
                    similar_pairs.append((speaker_a, speaker_b, similarity))
        
        # DEBUG: Log top similarities even if below threshold
        if all_similarities:
            all_similarities.sort(key=lambda x: x[2], reverse=True)
            top_3 = all_similarities[:3]
            logger.info(f"Top centroid similarities: {[(f'{a}<->{b}:{s:.3f}', a, b, s) for a, b, s in top_3]}")
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def auto_merge_similar_speakers(
        self,
        similarity_threshold: float = 0.76,
        max_merges: int = 5
    ) -> int:
        """
        Automatically merge speakers using hierarchical approach.
        
        - Immediate merge: similarity >= 0.92 (very likely same speaker)
        - Standard merge: similarity >= threshold (0.76 default)
        
        This helps consolidate speakers that were incorrectly split during
        diarization. Uses a greedy approach: merge the most similar pair,
        update centroids, then repeat until no more pairs exceed threshold.
        
        Args:
            similarity_threshold: Pairs above this threshold will be merged (default: 0.76)
            max_merges: Maximum number of merges to perform in one call
            
        Returns:
            Number of merges performed
        """
        merges_performed = 0
        
        # Phase 1: Immediate merges for very high similarity (>= 0.92)
        immediate_threshold = 0.92
        for _ in range(max_merges):
            similar_pairs = self.find_similar_speakers(immediate_threshold)
            
            if not similar_pairs:
                break
            
            # Merge the most similar pair
            speaker_a, speaker_b, similarity = similar_pairs[0]
            
            logger.info(f"Immediate auto-merge: {speaker_a} <-> {speaker_b} "
                       f"(similarity: {similarity:.3f})")
            
            # Use speaker_a as the target, merge speaker_b into it
            self._merge_speakers_internal(speaker_a, speaker_b, similarity)
            merges_performed += 1
            
            # Update history - replace speaker_b with speaker_a
            self.history = deque(
                [speaker_a if s == speaker_b else s for s in self.history],
                maxlen=self.history.maxlen
            )
        
        # Phase 2: Standard merges using the provided threshold
        for _ in range(max_merges - merges_performed):
            similar_pairs = self.find_similar_speakers(similarity_threshold)
            
            if not similar_pairs:
                break
            
            # Merge the most similar pair
            speaker_a, speaker_b, similarity = similar_pairs[0]
            
            logger.info(f"Standard auto-merge: {speaker_a} <-> {speaker_b} "
                       f"(similarity: {similarity:.3f})")
            
            # Use speaker_a as the target, merge speaker_b into it
            self._merge_speakers_internal(speaker_a, speaker_b, similarity)
            merges_performed += 1
            
            # Update history - replace speaker_b with speaker_a
            self.history = deque(
                [speaker_a if s == speaker_b else s for s in self.history],
                maxlen=self.history.maxlen
            )
        
        if merges_performed > 0:
            logger.info(f"Auto-merge completed: {merges_performed} merges performed")
        
        return merges_performed
    
    def _merge_speakers_internal(self, speaker_a: str, speaker_b: str, similarity: float = 0.0):
        """
        Internal method to merge speaker_b into speaker_a.
        The speaker created FIRST (lower number) will survive the merge.
        Called by both merge_speakers() and auto_merge_similar_speakers().
        """
        if speaker_a not in self.centroids or speaker_b not in self.centroids:
            raise ValueError("Both speakers must exist")
        
        # Extract speaker numbers to determine which was created first
        def get_speaker_number(speaker_id: str) -> int:
            try:
                return int(speaker_id.split('_')[1])
            except (IndexError, ValueError):
                return 0
        
        num_a = get_speaker_number(speaker_a)
        num_b = get_speaker_number(speaker_b)
        
        # Ensure the first-created speaker (lower number) survives
        if num_b < num_a:
            speaker_a, speaker_b = speaker_b, speaker_a
            num_a, num_b = num_b, num_a
        
        # Weighted average of centroids based on sample counts
        n_a = self.counts[speaker_a]
        n_b = self.counts[speaker_b]
        merged = (self.centroids[speaker_a] * n_a + self.centroids[speaker_b] * n_b) / (n_a + n_b)
        
        self.centroids[speaker_a] = self._normalize(merged)
        self.counts[speaker_a] = n_a + n_b
        
        # Remove speaker_b
        del self.centroids[speaker_b]
        del self.counts[speaker_b]

        #log merge event for debugging
        self._recent_merges_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "from": speaker_b,
            "to": speaker_a,
            "similarity": similarity
        })
        if len(self._recent_merges_log) > 20:
            self._recent_merges_log.pop(0)
    
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
        self._match_log.clear()
        self._recent_merges_log.clear()
        
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

    def get_and_clear_merges(self) -> List[Dict[str, str]]:
        """
        Get recent merge events and clear the log.
        
        Returns:
            List of merge events as [{"source": "speaker_1", "target": "speaker_0"}]
        """
        merges = [
            {"source": m["from"], "target": m["to"]}
            for m in self._recent_merges_log
        ]
        self._recent_merges_log.clear()
        return merges
    
    def get_centroids(self) -> List[dict]:
        """
        Get 2D PCA-projected coordinates of speaker centroids.
        
        Returns:
            List of dicts with speaker_id, x, y, sample_count
        """
        if len(self.centroids) == 0:
            return []
        
        if PCA is None:
            logger.warning("sklearn PCA not available, returning empty centroids")
            return []
        
        # Stack embeddings into matrix
        embeddings = np.array([
            self.centroids[sid] for sid in self.centroids
        ])
        
        # Normalize embeddings (already normalized, but defensive)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # PCA projection to 2D
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        # Build result
        speaker_ids = list(self.centroids.keys())
        return [
            {
                "speaker_id": speaker_ids[i],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "sample_count": self.counts[speaker_ids[i]]
            }
            for i in range(len(speaker_ids))
        ]
    
    def get_pairwise_similarity(self) -> List[dict]:
        """
        Get pairwise cosine similarity between all speakers.
        
        Returns:
            List of dicts with speaker_a, speaker_b, similarity
        """
        if len(self.centroids) < 2:
            return []
        
        speaker_ids = list(self.centroids.keys())
        results = []
        
        for i in range(len(speaker_ids)):
            for j in range(i + 1, len(speaker_ids)):
                sid_a = speaker_ids[i]
                sid_b = speaker_ids[j]
                sim = self._cosine_similarity(
                    self.centroids[sid_a],
                    self.centroids[sid_b]
                )
                results.append({
                    "speaker_a": sid_a,
                    "speaker_b": sid_b,
                    "similarity": sim
                })
        
        return results


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
    
    # Initialize speaker memory with validation (using v2 defaults)
    # min_samples_for_match=1 allows immediate matching after first sample
    # Embedding validation and short segment filtering provide safeguards against false positives
    speakers = SpeakerMemory(threshold=0.78, min_samples_for_match=1, ema_alpha=0.28)

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
                logger.warning("*** RESET SIGNAL RECEIVED - clearing speaker memory ***")
                logger.warning(f"Speakers before reset: {list(speakers.centroids.keys())}")
                speakers.reset()
                logger.warning(f"Speakers after reset: {list(speakers.centroids.keys())}")
                logger.debug(f"After RESET_SIGNAL: shutdown_received={shutdown_received}, queue.empty()={request_queue.empty()}")
                continue  # Skip the rest of the loop - request is a string, not a DiarizationRequest
            
            logger.info(f"WORKER: Received diarization request: {request.request_id}, audio: {request.audio_path}")
            try:
                # Run diarization
                diarization_result = pipeline(request.audio_path)
                
                # Combined loop: validate segment duration BEFORE speaker matching
                # Build a mapping from speaker label to embedding index
                speaker_to_embedding_idx = {speaker: idx for idx, speaker in enumerate(diarization_result.speaker_diarization.labels())}
                
                segments = []
                skipped_short = 0
                skipped_invalid = 0
                speaker_ids = {}  # Track for logging/debugging
                
                for turn, speaker in diarization_result.speaker_diarization:
                    segment_duration = turn.end - turn.start
                    
                    # STEP 1: Check segment duration FIRST - skip if too short
                    if segment_duration < MIN_SEGMENT_DURATION:
                        skipped_short += 1
                        logger.debug(f"Skipping short segment: {segment_duration:.2f}s < {MIN_SEGMENT_DURATION}s")
                        continue  # Skip - do NOT call identify()
                    
                    # STEP 2: Get embedding for this speaker
                    s = speaker_to_embedding_idx.get(speaker)
                    if s is None:
                        logger.warning(f"No embedding index found for speaker {speaker}")
                        continue
                    
                    speaker_embedding = diarization_result.speaker_embeddings[s]
                    emb_norm = np.linalg.norm(speaker_embedding)
                    logger.debug(f"Pyannote embedding[{s}] norm={emb_norm:.4f}, shape={speaker_embedding.shape}")
                    
                    segment_info = {
                        "request_id": request.request_id,
                        "speaker_label": speaker,
                        "embedding_index": s,
                        "segment_duration": segment_duration
                    }
                    
                    # STEP 3: Call identify() only for valid-duration segments
                    speaker_id, score, alt_speakers = speakers.identify(speaker_embedding, segment_info)
                    speaker_ids[speaker] = {"id": speaker_id, "score": score, "alt_speakers": alt_speakers}
                    
                    # STEP 4: Skip if embedding was invalid (returned "unknown")
                    if speaker_id == "unknown":
                        skipped_invalid += 1
                        logger.debug(f"Skipping segment with invalid embedding")
                        continue
                    
                    # STEP 5: Add valid segment to output
                    segments.append(SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=str(speaker_id),
                        confidence=str(score),
                        alt_speakers=alt_speakers if len(alt_speakers) > 0 else None
                    ))
                
                # Log summary of skipped segments
                if skipped_short > 0:
                    logger.info(f"Skipped {skipped_short} short segments (< {MIN_SEGMENT_DURATION}s)")
                if skipped_invalid > 0:
                    logger.info(f"Skipped {skipped_invalid} segments with invalid embeddings")
                
                logger.info(f"Extracted {len(segments)} speaker segments")
                if len(segments) > 1:
                    logger.info(f"Speakers identified: {set(s.speaker for s in segments)}")
                    logger.debug(f"diarization result: {diarization_result}")
                
                # Get merge events that occurred during this processing
                merged_speakers = speakers.get_and_clear_merges()
                
                # Send result
                result = DiarizationResult(
                    request_id=request.request_id,
                    audio_path=request.audio_path,
                    segments=segments,
                    merged_speakers=merged_speakers if merged_speakers else None
                )
                result_queue.put(result)
                logger.info(f"WORKER: Sent diarization result for {request.request_id}, segments={len(segments)}")
                
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
        threshold: float = 0.72,
        recency_boost: float = 0.05,
        history_size: int = 20,
        min_samples_for_match: int = 5,
        min_segment_duration: float = 0.5,
        ema_alpha: float = 0.15,
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
        self.ema_alpha = ema_alpha
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
            ema_alpha=self.ema_alpha,
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
    
    def get_centroids(self) -> dict:
        """
        Get speaker centroids for graph visualization.
        
        Returns:
            Dict with speakers (2D coords) and pairwise_similarity
        """
        if self._speaker_memory is None:
            return {"speakers": [], "pairwise_similarity": []}
        
        return {
            "speakers": self._speaker_memory.get_centroids(),
            "pairwise_similarity": self._speaker_memory.get_pairwise_similarity()
        }
    
    def reset(self):
        """Reset all accumulated state.
        
        This clears both the in-process speaker memory AND sends a RESET_SIGNAL
        to the worker process to clear its speaker memory for the new stream.
        The worker process continues running across stream boundaries.
        """
        logger.info("DiarizationClient.reset() - clearing state and signaling worker")
        if self._speaker_memory is not None:
            self._speaker_memory.reset()
        self.in_flight_requests.clear()
        
        # Send RESET_SIGNAL to worker to clear its speaker memory
        self.reset_process()
        
        logger.info("DiarizationClient state reset (worker signaled)")
    
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
        logger.debug(f"DiarizationClient.process_audio() called: {audio_path}, {request_id}")
        if not self._running:
            raise RuntimeError("Diarization process not started")
        
        request = DiarizationRequest(
            audio_path=audio_path,
            request_id=request_id
        )
        logger.debug(f"Putting request in queue: {request_id}")
        self._request_queue.put(request)
        logger.debug(f"Request put in queue: {request_id}, queue size unknown (multiprocessing)")
    
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
        except asyncio.TimeoutError:
            # This is the actual exception raised when timeout expires
            logger.debug(f"get_result timeout ({timeout}s) - queue empty or not ready")
            return None
    
    @property
    def is_running(self) -> bool:
        """Check if the process is running.
        
        Also checks if the underlying process is still alive. If the process
        has exited (e.g., due to shutdown or error), returns False even if
        _running was previously set to True.
        """
        if not self._running:
            return False
        
        # Check if process is still alive
        if self._process is not None and not self._process.is_alive():
            logger.warning("Diarization process has exited but _running=True, updating state")
            self._running = False
            return False
        
        return True