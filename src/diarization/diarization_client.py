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
import time
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
class MergeSpeakersRequest:
    """Request to merge two speakers in the worker's SpeakerMemory."""
    source: str   # speaker to remove
    target: str   # speaker to keep

@dataclass
class ConsolidateRequest:
    """Request to run a post-session consolidation pass in the worker."""
    relaxed_threshold: float = 0.65
    relaxed_guard: float = 0.55

@dataclass
class ConsolidationResult:
    """Result from a post-session consolidation pass."""
    merges: List[Tuple[str, str]]  # (merged_into, merged_from)
    speaker_count_before: int
    speaker_count_after: int

@dataclass
class WorkerConfigMessage:
    """Configuration message forwarded from DiarizationClient to the worker at startup."""
    threshold: float
    min_samples_for_match: int
    ema_alpha: float
    active_window_seconds: float

@dataclass
class DiarizationResult:
    """Result from diarization processing."""
    request_id: str
    audio_path: str
    segments: List[SpeakerSegment]
    error: Optional[str] = None
    merged_speakers: Optional[List[Dict[str, str]]] = None  # [{"source": "speaker_1", "target": "speaker_0"}]
    speaker_centroids: Optional[List[dict]] = None  # [{"speaker_id": "speaker_0", "x": 0.1, "y": -0.2, "sample_count": 5}]
    pairwise_similarity: Optional[List[dict]] = None  # [{"speaker_a": "speaker_0", "speaker_b": "speaker_1", "similarity": 0.85}]
    pca_processing_time_ms: float = 0.0

class SpeakerMemory:
    """
    Track speakers across audio segments using embedding similarity.
    
    Features:
    - Multi-prototype speaker representation (up to K embeddings per speaker)
      captures within-speaker voice variability instead of averaging it away,
      which is critical for distinguishing similar-sounding speakers.
    - Margin-based scoring penalises matches where the top-2 candidates are
      too close, biasing toward new-speaker creation rather than wrong assignment.
    - Temporal transition model adds a soft prior based on observed turn-taking
      patterns (A→B transitions).
    - Temporal context (recent speaker more likely to continue)
    - Configurable similarity threshold
    - Speaker history for debugging
    - Embedding quality validation (filters zero/invalid embeddings)
    - Statistics tracking and monitoring callbacks
    - Segment duration filtering
    """

    # Maximum number of prototype embeddings per speaker
    MAX_PROTOTYPES = 8
    # Cosine similarity threshold for assigning to an existing prototype vs creating new
    PROTOTYPE_MATCH_THRESHOLD = 0.78
    # When pairwise centroid similarity exceeds this, merge even if speakers
    # co-occurred in the same audio chunk.  Now that co-occurrence uses reliable
    # chunk+label tracking (not wall-clock time), this override should only fire
    # for near-certain same-speaker matches where stale chunk data might block.
    CO_OCCURRENCE_OVERRIDE_SIMILARITY = 0.92
    # Minimum "best-match" across prototype pools for a merge to be allowed.
    # For each prototype of speaker A, find its best match in speaker B's pool;
    # if the *minimum* of these best-matches is below this threshold, at least
    # one vocal mode of A has no counterpart in B → block the merge.
    PROTOTYPE_MERGE_GUARD_THRESHOLD = 0.65
    # Minimum margin between top-2 candidates before margin penalty kicks in
    MIN_MARGIN = 0.02
    # Margin penalty multiplier (kept mild to avoid over-fragmenting)
    MARGIN_PENALTY_SCALE = 1.0
    # When margin is too small AND best score < this, require new speaker creation.
    # Set low so it only fires when the match is truly ambiguous and weak.
    MARGIN_HARD_THRESHOLD = 0.65
    # Temporal transition bonus cap
    TRANSITION_BONUS_CAP = 0.03
    # Minimum time gap (seconds) between segments for same-speaker assignment
    # Two segments closer than this assigned to different speakers triggers re-evaluation
    MIN_SPEAKER_SWITCH_GAP = 0.3
    
    def __init__(
        self,
        threshold: float = 0.71,
        recency_boost: float = 0.02,
        history_size: int = 20,
        min_samples_for_match: int = 2,
        min_segment_duration: float = 0.15,
        min_segment_duration_for_creation: float = 0.40,
        ema_alpha: float = 0.05,
        active_window_seconds: float = 1200.0,
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
            min_segment_duration: Minimum segment duration in seconds — below this the
                segment is dropped entirely by the worker (default: 0.15)
            min_segment_duration_for_creation: Below this duration a new speaker will NOT
                be created; short segments are soft-assigned to the best match (if score
                >= 0.50) or discarded.  Does not affect matching known speakers. (default: 0.40)
            ema_alpha: EMA smoothing factor for prototype updates (default: 0.05,
                lowered from 0.15 to prevent centroid drift between similar speakers)
            active_window_seconds: Seconds since last seen before a speaker is skipped
                during matching. Inactive speakers are retained in memory and can
                become active again if a new segment looks similar. (default: 28800 = 8 hours)
            embedding_validator: Custom embedding validator (uses default if None)
            on_invalid_embedding: Callback for invalid embeddings: (embedding, reason, segment_info)
            on_low_confidence: Callback for low confidence matches: (speaker_id, confidence, segment_info)
        """
        self.threshold = threshold
        self.recency_boost = recency_boost
        self.min_samples_for_match = min_samples_for_match
        self.min_segment_duration = min_segment_duration
        self.min_segment_duration_for_creation = min_segment_duration_for_creation
        self.ema_alpha = ema_alpha
        self.active_window_seconds = active_window_seconds
        
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
        
        # Speaker profiles — multi-prototype representation
        # Each speaker stores a list of (embedding, count, last_seen) prototype tuples
        # plus a primary centroid (weighted average) for backward compatibility.
        self.centroids: Dict[str, np.ndarray] = {}  # speaker_id -> primary centroid (weighted avg)
        self.prototypes: Dict[str, List[Tuple[np.ndarray, int, float]]] = {}  # speaker_id -> [(emb, count, last_seen), ...]
        self.counts: Dict[str, int] = {}             # speaker_id -> total sample count
        self.last_seen: Dict[str, float] = {}        # speaker_id -> time.time() of last match
        self.speaker_counter = 0
        
        # Temporal tracking
        self.history: deque = deque(maxlen=history_size)
        self.last_speaker: Optional[str] = None
        
        # Temporal transition model: (speaker_a, speaker_b) -> count
        self._transitions: Dict[Tuple[str, str], int] = {}
        
        # Segment chunk/label tracking for co-occurrence checks.
        # Stores (chunk_id, pyannote_label) tuples so we know when two
        # SpeakerMemory IDs originated from different pyannote clusters in
        # the *same* audio chunk — meaning they are definitively different people.
        self._segment_chunks: Dict[str, List[Tuple[str, str]]] = {}  # speaker_id -> [(chunk_id, label), ...]
        
        # Debugging
        self._match_log: List[dict] = []
        self._recent_merges_log: List[dict] = []
        self._invalid_embedding_reasons: List[str] = []  # Track reasons for invalid embeddings

        # Per-speaker intra-prototype variance (1 - mean_pairwise_cosine).
        # Used by the variance-adjusted prototype merge guard so speakers with
        # naturally wide vocal range don't block legitimate self-merges.
        self._intra_variance: Dict[str, float] = {}

        # Tracks last segment end time for MIN_SPEAKER_SWITCH_GAP continuity bonus
        self._last_segment_end: float = 0.0

        # Statistics tracking
        self._stats = {
            "identifications": 0,
            "valid_embeddings": 0,
            "invalid_embeddings": 0,
            "matches": 0,
            "new_speakers": 0,
            "low_confidence_matches": 0,
            "skipped_short_segments": 0,
            "margin_penalties_applied": 0,
            "margin_new_speaker_forced": 0,
            "overlap_segments_skipped_update": 0,
            "short_segments_skipped_update": 0,
            "short_segments_soft_assigned": 0,
        }
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _get_dynamic_threshold(self) -> float:
        """
        Raise threshold smoothly as active speaker count grows to prevent
        similar speakers from collapsing into each other.

        More active speakers → higher chance of a close-but-wrong match →
        require stricter similarity to assign to an existing speaker.

        Uses exponential saturation so the threshold never cliffs at arbitrary
        step boundaries.
        Formula: threshold = base + max_increase * (1 - exp(-N / lambda))
          - At N=0:  threshold = base (0.71)
          - At N=4:  threshold ≈ 0.718
          - At N=8:  threshold ≈ 0.723
          - At N=∞:  threshold → base + 0.02 = 0.73
          - Hard ceiling: base + max_increase (0.73)

        Co-occurrence guards and prototype merge guards are the primary
        mechanism for keeping truly different speakers apart.
        """
        # Count speakers seen within 2τ as "effectively active" for threshold scaling.
        # Speakers outside this window still compete but with heavily decayed scores.
        now = time.time()
        tau = self.active_window_seconds
        n = sum(
            1 for sid in self.centroids
            if now - self.last_seen.get(sid, now) <= 2 * tau
        ) if self.last_seen else len(self.centroids)
        base = self.threshold
        max_increase = 0.02
        lam = 8.0  # controls how quickly threshold rises
        threshold = base + max_increase * (1.0 - float(np.exp(-n / lam)))
        return min(threshold, base + max_increase)
    
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
        
        # Use full float64 precision — rounding to 2 decimals destroyed the
        # fine-grained signal needed to distinguish similar-sounding speakers
        # whose similarity differs by <0.01.
        return float(np.clip(dot, -1.0, 1.0))
    
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
    
    def _max_prototype_similarity(self, embedding: np.ndarray, speaker_id: str) -> float:
        """
        Compute maximum cosine similarity between *embedding* and all prototypes
        of *speaker_id*.  Falls back to the primary centroid if no prototypes exist.
        """
        protos = self.prototypes.get(speaker_id)
        if protos:
            return max(self._cosine_similarity(embedding, p[0]) for p in protos)
        # Fallback: no prototypes yet — use primary centroid
        centroid = self.centroids.get(speaker_id)
        if centroid is not None:
            return self._cosine_similarity(embedding, centroid)
        return -1.0

    def _min_cross_prototype_best_match(self, speaker_a: str, speaker_b: str) -> float:
        """Return the minimum "best-match" when comparing prototype pools.

        For each prototype of *speaker_a*, find its highest cosine similarity
        to any prototype of *speaker_b*.  Return the **minimum** of these
        best-matches.

        A low value means at least one vocal mode of speaker_a has no
        counterpart in speaker_b — strong evidence they are different people.
        If either speaker has no prototypes, falls back to centroid comparison
        (returning 1.0 so the guard never blocks for lack of data).
        """
        protos_a = self.prototypes.get(speaker_a, [])
        protos_b = self.prototypes.get(speaker_b, [])
        if not protos_a or not protos_b:
            return 1.0  # no prototype data — do not block merge

        min_best = 1.0
        for emb_a, _, _ in protos_a:
            best = max(self._cosine_similarity(emb_a, emb_b) for emb_b, _, _ in protos_b)
            if best < min_best:
                min_best = best
        return min_best

    def _compute_margin_penalty(self, best_score: float, second_best_score: float) -> float:
        """
        Penalise match score when the gap between top-2 candidates is too small.

        This biases toward new-speaker creation when the system cannot confidently
        distinguish between similar-sounding speakers.

        Returns the penalty to **subtract** from the raw best score.
        """
        margin = best_score - second_best_score
        if margin >= self.MIN_MARGIN:
            return 0.0
        penalty = (self.MIN_MARGIN - margin) * self.MARGIN_PENALTY_SCALE
        return penalty

    def _get_transition_bonus(self, from_speaker: Optional[str], candidate: str) -> float:
        """
        Compute a small bonus for *candidate* if the transition from *from_speaker*
        → *candidate* has been observed frequently.

        Returns a value in [0, TRANSITION_BONUS_CAP].
        """
        if from_speaker is None or from_speaker == candidate:
            return 0.0
        total = sum(
            v for (src, _), v in self._transitions.items() if src == from_speaker
        )
        if total == 0:
            return 0.0
        count = self._transitions.get((from_speaker, candidate), 0)
        ratio = count / total  # [0, 1]
        return min(ratio * self.TRANSITION_BONUS_CAP, self.TRANSITION_BONUS_CAP)

    def _record_transition(self, from_speaker: Optional[str], to_speaker: str):
        """Record a speaker transition for the temporal model."""
        if from_speaker is not None and from_speaker != to_speaker:
            key = (from_speaker, to_speaker)
            self._transitions[key] = self._transitions.get(key, 0) + 1

    def _record_segment_time(self, speaker_id: str, segment_info: Optional[dict] = None):
        """Record the chunk+label of a segment for co-occurrence checks.

        Uses the *request_id* (chunk identifier) and *speaker_label* (pyannote
        per-chunk cluster label) from *segment_info*.  When two SpeakerMemory
        IDs share a chunk_id but have different pyannote labels, pyannote
        explicitly decided they are different speakers within the same audio —
        the strongest evidence that they should not be merged.
        """
        if segment_info is None:
            return
        chunk_id = segment_info.get("request_id")
        label = segment_info.get("speaker_label")
        if chunk_id is None or label is None:
            return
        entry = (str(chunk_id), str(label))
        if speaker_id not in self._segment_chunks:
            self._segment_chunks[speaker_id] = []
        entries = self._segment_chunks[speaker_id]
        entries.append(entry)
        # Keep only the last 50 entries per speaker to bound memory
        if len(entries) > 50:
            self._segment_chunks[speaker_id] = entries[-50:]

    def _speakers_co_occurred(self, speaker_a: str, speaker_b: str) -> bool:
        """Return True if speaker_a and speaker_b were assigned different
        pyannote labels within the *same* audio chunk.

        This is a reliable co-occurrence signal: pyannote explicitly separated
        them in the same audio, so they are by definition different people.
        Unlike the previous wall-clock approach, this is independent of
        processing speed and chunk ordering.
        """
        entries_a = self._segment_chunks.get(speaker_a, [])
        entries_b = self._segment_chunks.get(speaker_b, [])
        if not entries_a or not entries_b:
            return False
        # Build a mapping from chunk_id → set of labels for speaker_a
        chunks_a: Dict[str, set] = {}
        for cid, lbl in entries_a:
            chunks_a.setdefault(cid, set()).add(lbl)
        for cid, lbl in entries_b:
            if cid in chunks_a:
                # Same chunk — did they have *different* pyannote labels?
                if lbl not in chunks_a[cid]:
                    return True
        return False

    def identify(
        self,
        embedding: np.ndarray,
        segment_info: Optional[dict] = None
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Match embedding to existing speaker or create new one.

        Uses multi-prototype similarity (max over K prototypes per speaker),
        margin-based scoring, and temporal transition priors.

        Args:
            embedding: Raw speaker embedding (will be normalized)
            segment_info: Optional metadata; recognised keys:
                - segment_duration (float): length of this audio segment in seconds
                - is_overlap (bool): True if this segment overlaps with another speaker —
                  centroid updates are skipped and no new speaker is created.
                - is_onset_contaminated (bool): True if the onset of this turn is
                  within 500 ms of a previous speaker's end — centroid updates are skipped.
                - start_time (float): turn start in seconds (used for switch-gap bonus)
                - end_time (float): turn end in seconds (used to track _last_segment_end)
                - request_id, speaker_label: chunk-level co-occurrence tracking

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
            _, validation_reason = self.validator.is_valid(embedding)
            # Track the reason for consolidated logging later
            if validation_reason not in self._invalid_embedding_reasons:
                self._invalid_embedding_reasons.append(validation_reason)
            logger.debug(f"Embedding validation FAILED: {validation_reason}, norm={embedding_norm:.4f}, mean={embedding_mean:.8f}")
            logger.debug(f"Invalid embedding rejected for segment: {segment_info}")
            return "unknown", 0.0, {}

        logger.debug(f"Embedding validation PASSED")

        self._stats["valid_embeddings"] += 1

        embedding = self._normalize(embedding)

        # --- Extract quality flags from segment_info ---
        seg = segment_info or {}
        segment_duration = float(seg.get("segment_duration", 1.0))
        is_overlap = bool(seg.get("is_overlap", False))
        is_onset_contaminated = bool(seg.get("is_onset_contaminated", False))
        seg_start_time: Optional[float] = seg.get("start_time")
        seg_end_time: Optional[float] = seg.get("end_time")

        # A segment should update speaker centroids only when it is:
        #   - not from a region where two speakers overlap (embedding is a blend)
        #   - not from a contaminated turn onset (residual bleed from previous speaker)
        #   - long enough to represent the speaker's full vocal character
        should_update_centroid = (
            not is_overlap
            and not is_onset_contaminated
            and segment_duration >= 0.35
        )

        # Overlap-contaminated segments: never create a new speaker, skip update
        if is_overlap:
            logger.info(f"Overlap-contaminated segment (dur={segment_duration:.2f}s): skipping centroid update/creation")

        # Clean up any invalid speakers that may have been created before validation
        self.cleanup_invalid_speakers()

        if len(self.centroids) == 0:
            if is_overlap:
                # Don't bootstrap speaker memory from a contaminated embedding
                return "unknown", 0.0, {}
            # First speaker
            speaker_id, confidence, alt = self._create_speaker(embedding, segment_info, confidence=1.0)
            self._record_segment_time(speaker_id, segment_info)
            self.last_speaker = speaker_id
            if seg_end_time is not None:
                self._last_segment_end = seg_end_time
            return speaker_id, confidence, alt

        # ------- Compare against all known speakers (multi-prototype) -------
        best_id = None
        best_score = -1.0
        second_best_score = -1.0
        scores = {}
        alt_speakers = {}

        total_samples = sum(self.counts.values())
        logger.debug(f"Matching: {len(self.centroids)} speakers, total_samples={total_samples}, "
                      f"min_samples_for_match={self.min_samples_for_match} (threshold gates matching, not count)")
        now = time.time()
        for speaker_id in list(self.centroids.keys()):
            # NOTE: we do NOT skip low-count speakers here — the similarity threshold (0.72)
            # is the gating criterion.  Filtering by count caused fragmentation cascades where
            # newly created speakers (count=1) could never accumulate observations because they
            # were always skipped, forcing yet another new speaker on every segment.
            # min_samples_for_match is still used below to guard *new speaker creation*.

            # Multi-prototype similarity: use the max similarity across all prototypes
            score = self._max_prototype_similarity(embedding, speaker_id)

            # Age-based score decay: score *= exp(-age / τ).
            age = now - self.last_seen.get(speaker_id, now)
            if age > 0 and self.active_window_seconds > 0:
                decay = float(np.exp(-age / self.active_window_seconds))
                score *= decay
                logger.debug(f"{speaker_id}: raw_score before decay={score/decay:.3f}, age={age:.0f}s, "
                              f"decay={decay:.3f}, adjusted={score:.3f}")

            # Recency boost: prefer continuing same speaker
            if speaker_id == self.last_speaker:
                score += self.recency_boost
                # MIN_SPEAKER_SWITCH_GAP: rapid turn-continuation bonus when the
                # new segment starts very shortly after the last one ended.
                if (seg_start_time is not None and self._last_segment_end > 0
                        and 0.0 <= seg_start_time - self._last_segment_end < self.MIN_SPEAKER_SWITCH_GAP):
                    score += 0.02
                    logger.debug(f"{speaker_id}: fast-continuation bonus (+0.02)")

            # Temporal transition bonus
            transition_bonus = self._get_transition_bonus(self.last_speaker, speaker_id)
            if transition_bonus > 0:
                score += transition_bonus
                logger.debug(f"{speaker_id}: transition_bonus={transition_bonus:.4f}")

            scores[speaker_id] = score

            # Build alt_speakers in the loop
            if score >= self.threshold or score >= (self.threshold - 0.1):
                alt_speakers[speaker_id] = score

            # Track top-2 scores
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_id = speaker_id
            elif score > second_best_score:
                second_best_score = score

        # Get dynamic threshold
        effective_threshold = self._get_dynamic_threshold()

        # ------- Margin-based penalty -------
        margin_penalty = 0.0
        if second_best_score > -1.0:
            margin_penalty = self._compute_margin_penalty(best_score, second_best_score)
            if margin_penalty > 0:
                self._stats["margin_penalties_applied"] += 1
                logger.info(f"Margin penalty={margin_penalty:.4f} applied "
                            f"(best={best_score:.4f}, 2nd={second_best_score:.4f}, "
                            f"margin={best_score - second_best_score:.4f})")

        effective_score = best_score - margin_penalty

        # If margin is too small AND effective score is below the hard threshold
        # → force new speaker creation to avoid wrong assignment.
        # Does NOT apply to short segments (they are governed by their own branch below).
        force_new_speaker = (
            margin_penalty > 0
            and effective_score < self.MARGIN_HARD_THRESHOLD
            and segment_duration >= self.min_segment_duration_for_creation
        )
        if force_new_speaker:
            self._stats["margin_new_speaker_forced"] += 1
            logger.info(f"Margin guard: forcing new speaker (effective_score={effective_score:.4f} "
                        f"< {self.MARGIN_HARD_THRESHOLD})")

        # Log for debugging
        self._match_log.append({
            'scores': scores.copy(),
            'base_threshold': self.threshold,
            'effective_threshold': effective_threshold,
            'best_id': best_id,
            'best_score': best_score,
            'margin_penalty': margin_penalty,
            'effective_score': effective_score,
            'force_new_speaker': force_new_speaker,
            'is_overlap': is_overlap,
            'is_onset_contaminated': is_onset_contaminated,
            'segment_duration': segment_duration,
            'decision': 'match' if (best_id and effective_score >= effective_threshold and not force_new_speaker) else 'new_speaker',
            'segment_info': segment_info
        })
        logger.info(f"Speaker match scores: {scores}, best: {best_id} ({best_score:.3f}), "
                     f"effective: {effective_score:.3f}, threshold: {effective_threshold:.3f}")

        # ------- Short-segment soft assignment (Part 2 / Part 4) -------
        # For segments too short to create a reliable new speaker, either:
        #   a) soft-assign to the best existing match (score >= 0.50), or
        #   b) discard (return "unknown") rather than spawning a ghost speaker.
        # This branch is checked BEFORE the standard match/new-speaker decision so that
        # the force_new_speaker flag cannot override it for short segments.
        if segment_duration < self.min_segment_duration_for_creation and not is_overlap:
            if best_id and best_score >= 0.50:
                self._stats["short_segments_soft_assigned"] += 1
                self._record_transition(self.last_speaker, best_id)
                self._record_segment_time(best_id, segment_info)
                self.last_speaker = best_id
                self.history.append(best_id)
                if seg_end_time is not None:
                    self._last_segment_end = seg_end_time
                logger.info(
                    f"Short-segment soft assignment: {best_id} "
                    f"(score={best_score:.3f}, dur={segment_duration:.2f}s)"
                )
                # Reduced confidence tag for downstream consumers
                reduced_conf = best_score * 0.7
                return best_id, reduced_conf, {**alt_speakers, "_short_segment_soft_assignment": 1.0}
            else:
                # Score too weak AND can't create → discard
                self._stats["skipped_short_segments"] += 1
                logger.info(
                    f"Short segment discarded (score={best_score:.3f} < 0.50, "
                    f"dur={segment_duration:.2f}s): no new speaker created"
                )
                return "unknown", 0.0, {}

        # ------- Match or create new speaker -------
        if best_id and effective_score >= effective_threshold and not force_new_speaker:
            self._stats["matches"] += 1

            # Compute calibrated confidence using softmax + entropy
            calibrated_confidence, speaker_probs = self._compute_calibrated_confidence(
                scores, best_id
            )

            logger.debug(f"Calibrated confidence: {calibrated_confidence:.3f}, probs: {speaker_probs}")

            # Check for low confidence match
            low_confidence_threshold = 0.65
            if calibrated_confidence < low_confidence_threshold:
                self._stats["low_confidence_matches"] += 1
                if self.on_low_confidence:
                    try:
                        self.on_low_confidence(best_id, calibrated_confidence, segment_info)
                    except Exception as e:
                        logger.warning(f"Error in on_low_confidence callback: {e}")

            # Update speaker prototypes & centroid — skip for contaminated segments
            if should_update_centroid:
                self._update_speaker(best_id, embedding)
            else:
                if is_overlap:
                    self._stats["overlap_segments_skipped_update"] += 1
                    logger.debug(f"Skipping centroid update for {best_id}: overlap segment")
                else:
                    self._stats["short_segments_skipped_update"] += 1
                    logger.debug(
                        f"Skipping centroid update for {best_id}: "
                        f"{'onset contamination' if is_onset_contaminated else 'short segment'} "
                        f"(dur={segment_duration:.2f}s)"
                    )
                # Still update last_seen so decay stays fresh
                self.last_seen[best_id] = now

            self._record_transition(self.last_speaker, best_id)
            self._record_segment_time(best_id, segment_info)
            self.last_speaker = best_id
            self.history.append(best_id)
            if seg_end_time is not None:
                self._last_segment_end = seg_end_time

            # Periodic consolidation: every 5 identifications or if speaker count exceeds 10.
            if (self._stats["identifications"] % 5 == 0) or len(self.centroids) > 10:
                merges = self.auto_merge_similar_speakers(similarity_threshold=0.615)
                if merges:
                    logger.info(f"Periodic consolidation: {merges} merges")

            return best_id, calibrated_confidence, alt_speakers
        else:
            # Would create a new speaker — block if contaminated or too short
            if is_overlap:
                logger.info(
                    f"Overlap contamination: refusing new speaker creation "
                    f"(best_score={best_score:.3f}, dur={segment_duration:.2f}s)"
                )
                return "unknown", 0.0, {}

            self._stats["new_speakers"] += 1
            speaker_id, confidence, _ = self._create_speaker(embedding, segment_info, confidence=1.0)

            self._record_transition(self.last_speaker, speaker_id)
            self._record_segment_time(speaker_id, segment_info)
            if seg_end_time is not None:
                self._last_segment_end = seg_end_time

            # Proactive merge: check if new speaker should be merged immediately
            merge_result = self._check_and_merge_after_creation(speaker_id, embedding)
            if merge_result:
                return merge_result, self._cosine_similarity(embedding, self.centroids[merge_result]), alt_speakers

            return speaker_id, confidence, alt_speakers
            
    def _create_speaker(
        self,
        embedding: np.ndarray,
        segment_info: Optional[dict],
        confidence: float
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """Create a new speaker profile with validation and prototype initialization."""
        # Double-check validation before storage
        is_valid, reason = self.validator.is_valid(embedding)
        if not is_valid:
            logger.warning(f"Attempted to create speaker with invalid embedding: {reason}")
            return "unknown", 0.0, {}
        
        speaker_id = f"speaker_{self.speaker_counter}"
        self.speaker_counter += 1
        
        now = time.time()
        self.centroids[speaker_id] = embedding.copy()
        self.prototypes[speaker_id] = [(embedding.copy(), 1, now)]
        self.counts[speaker_id] = 1
        self.last_seen[speaker_id] = now
        self.last_speaker = speaker_id
        self.history.append(speaker_id)
        
        logger.info(f"Created new speaker: {speaker_id} (confidence: {confidence:.3f})")
        return speaker_id, confidence, {}
    
    def _update_speaker(self, speaker_id: str, embedding: np.ndarray):
        """Multi-prototype update with EMA.

        For each incoming embedding:
        1. Find the closest existing prototype (cosine >= PROTOTYPE_MATCH_THRESHOLD).
           - If found → EMA-update that prototype.
           - If not → add as a new prototype (up to MAX_PROTOTYPES).
           - If full → replace the oldest (least recently seen) prototype.
        2. Recompute the primary centroid as the weighted average of all prototypes.
        """
        now = time.time()
        alpha = self.ema_alpha  # 0.08 by default

        protos = self.prototypes.get(speaker_id, [])

        # Find closest prototype
        best_idx = -1
        best_sim = -1.0
        for i, (p_emb, p_cnt, p_ts) in enumerate(protos):
            sim = self._cosine_similarity(embedding, p_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= self.PROTOTYPE_MATCH_THRESHOLD:
            # EMA-update the matching prototype
            p_emb, p_cnt, _ = protos[best_idx]
            if p_cnt == 0:
                updated = embedding.copy()
            else:
                updated = alpha * embedding + (1 - alpha) * p_emb
            updated = self._normalize(updated)
            protos[best_idx] = (updated, p_cnt + 1, now)
        elif len(protos) < self.MAX_PROTOTYPES:
            # Add as a new prototype
            protos.append((embedding.copy(), 1, now))
        else:
            # Pool is full: replace the most redundant prototype (the one whose
            # best similarity to any *other* prototype is highest — i.e. it adds
            # the least unique information).  Prefer evicting similar prototypes
            # over older ones so diverse vocal modes are preserved.
            most_redundant_idx = 0
            most_redundant_sim = -1.0
            for i, (p_emb_i, _, _) in enumerate(protos):
                best_other = max(
                    self._cosine_similarity(p_emb_i, protos[j][0])
                    for j in range(len(protos)) if j != i
                )
                if best_other > most_redundant_sim:
                    most_redundant_sim = best_other
                    most_redundant_idx = i
            protos[most_redundant_idx] = (embedding.copy(), 1, now)

        self.prototypes[speaker_id] = protos

        # Recompute primary centroid as weighted average of prototypes
        total_count = sum(p[1] for p in protos)
        if total_count > 0:
            weighted = sum(p[0] * p[1] for p in protos) / total_count
            self.centroids[speaker_id] = self._normalize(weighted)
        else:
            self.centroids[speaker_id] = self._normalize(embedding)

        self.counts[speaker_id] += 1
        self.last_seen[speaker_id] = now

        # Compute intra-prototype variance: 1 - mean(pairwise cosine similarity).
        # High variance → speaker has wide vocal range → relax merge guard.
        if len(protos) >= 2:
            sims = []
            for i in range(len(protos)):
                for j in range(i + 1, len(protos)):
                    sims.append(self._cosine_similarity(protos[i][0], protos[j][0]))
            self._intra_variance[speaker_id] = 1.0 - float(np.mean(sims))
        else:
            self._intra_variance[speaker_id] = 0.0
    
    def _check_and_merge_after_creation(self, new_speaker_id: str, new_embedding: np.ndarray) -> Optional[str]:
        """Check if new speaker should be merged immediately.

        Guarded by chunk-based co-occurrence and variance-adjusted prototype merge guard.
        The co-occurrence check uses chunk+label tracking (reliable), and
        the prototype guard prevents merging speakers with incompatible
        vocal mode distributions, automatically relaxed for speakers with
        naturally wide vocal range (high intra-prototype variance).
        """
        for existing_id, centroid in list(self.centroids.items()):
            if existing_id == new_speaker_id:
                continue
            # Require a small number of observations before a centroid is merge-eligible
            if self.counts.get(existing_id, 0) < 2:
                continue
            sim = self._cosine_similarity(new_embedding, centroid)

            # High-similarity override: nearly identical centroids → merge
            if sim >= self.CO_OCCURRENCE_OVERRIDE_SIMILARITY:
                self._merge_speakers_internal(existing_id, new_speaker_id)
                logger.info(f"Immediate merge (override): {new_speaker_id} → {existing_id} (sim={sim:.3f})")
                return existing_id

            # Co-occurrence guard: pyannote separated them in the same chunk
            if self._speakers_co_occurred(existing_id, new_speaker_id):
                logger.debug(f"Co-occurrence guard: {new_speaker_id} and {existing_id} co-occurred, skipping merge")
                continue

            # Variance-adjusted prototype merge guard.
            # Speakers with wider vocal range (high intra-variance) naturally produce
            # lower cross-prototype similarity → compensate so legitimate merges aren't
            # blocked.  Floor is 0.50 regardless of variance.
            cross_min = self._min_cross_prototype_best_match(existing_id, new_speaker_id)
            avg_variance = (
                self._intra_variance.get(existing_id, 0.0)
                + self._intra_variance.get(new_speaker_id, 0.0)
            ) / 2.0
            effective_guard = max(self.PROTOTYPE_MERGE_GUARD_THRESHOLD - avg_variance, 0.50)
            if cross_min < effective_guard:
                logger.debug(
                    f"Prototype guard: {new_speaker_id} <-> {existing_id} "
                    f"cross_min={cross_min:.3f} < effective_guard={effective_guard:.3f} "
                    f"(base={self.PROTOTYPE_MERGE_GUARD_THRESHOLD}, avg_var={avg_variance:.3f}), skipping merge"
                )
                continue

            if sim >= 0.80:
                self._merge_speakers_internal(existing_id, new_speaker_id)
                logger.info(f"Immediate merge: {new_speaker_id} → {existing_id} (sim={sim:.3f})")
                return existing_id
        return None
    
    def merge_speakers(self, speaker_a: str, speaker_b: str):
        """
        Manually merge two speakers (useful for correcting errors).
        Keeps speaker_a, merges speaker_b into it.
        """
        logger.info(f"Merging speakers: {speaker_b} -> {speaker_a}")
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
        similarity_threshold: float = 0.88,
        max_merges: int = 5,
        post_session: bool = False,
        override_guard: Optional[float] = None,
    ) -> int:
        """
        Automatically merge speakers using hierarchical approach.

        - Immediate merge: similarity >= 0.92 (near-certain same speaker)
        - Standard merge: similarity >= threshold (default: 0.88 for manual calls;
          0.615 when called from identify() for live incremental consolidation)

        Guards:
          1. Chunk-based co-occurrence: pyannote separated them → block
             (bypassed when post_session=True — full-session context available)
          2. Variance-adjusted prototype merge guard: incompatible vocal modes → block
             (threshold = max(base - avg_intra_variance, 0.50))
          3. CO_OCCURRENCE_OVERRIDE_SIMILARITY (0.92): bypass both guards

        Args:
            similarity_threshold: Pairs above this threshold will be merged
            max_merges: Maximum number of merges to perform in one call
            post_session: When True, bypass the co-occurrence guard — used in
              the end-of-stream consolidation pass where the full session context
              makes it safe to merge speakers who happen to share a chunk.
            override_guard: If set, override PROTOTYPE_MERGE_GUARD_THRESHOLD with
              this value for this call (used by consolidate_post_session).

        Returns:
            Number of merges performed
        """
        guard_threshold = override_guard if override_guard is not None else self.PROTOTYPE_MERGE_GUARD_THRESHOLD
        merges_performed = 0

        def _apply_guards(speaker_a: str, speaker_b: str, similarity: float) -> bool:
            """Return True if the merge is allowed after applying guards."""
            # High-similarity override: bypass all guards
            if similarity >= self.CO_OCCURRENCE_OVERRIDE_SIMILARITY:
                return True
            # Co-occurrence guard (skipped in post-session pass)
            if not post_session and self._speakers_co_occurred(speaker_a, speaker_b):
                logger.info(f"Co-occurrence guard blocked merge: {speaker_a} <-> {speaker_b} "
                            f"(sim={similarity:.3f})")
                return False
            # Variance-adjusted prototype guard
            cross_min = self._min_cross_prototype_best_match(speaker_a, speaker_b)
            avg_variance = (
                self._intra_variance.get(speaker_a, 0.0)
                + self._intra_variance.get(speaker_b, 0.0)
            ) / 2.0
            effective_guard = max(guard_threshold - avg_variance, 0.50)
            if cross_min < effective_guard:
                logger.info(
                    f"Prototype guard blocked merge: {speaker_a} <-> {speaker_b} "
                    f"(sim={similarity:.3f}, cross_min={cross_min:.3f}, "
                    f"effective_guard={effective_guard:.3f}, avg_var={avg_variance:.3f})"
                )
                return False
            return True

        # Phase 1: Immediate merges for near-certain same-speaker (>= 0.92)
        immediate_threshold = 0.92
        for _ in range(max_merges):
            similar_pairs = self.find_similar_speakers(immediate_threshold)
            if not similar_pairs:
                break
            merge_pair = next(
                ((a, b, sim) for a, b, sim in similar_pairs if _apply_guards(a, b, sim)),
                None
            )
            if merge_pair is None:
                break
            speaker_a, speaker_b, similarity = merge_pair
            logger.info(f"Immediate auto-merge: {speaker_a} <-> {speaker_b} (similarity: {similarity:.3f})")
            self._merge_speakers_internal(speaker_a, speaker_b, similarity)
            merges_performed += 1
            self.history = deque(
                [speaker_a if s == speaker_b else s for s in self.history],
                maxlen=self.history.maxlen
            )

        # Phase 2: Standard merges using the provided threshold
        for _ in range(max_merges - merges_performed):
            similar_pairs = self.find_similar_speakers(similarity_threshold)
            if not similar_pairs:
                break
            merge_pair = next(
                ((a, b, sim) for a, b, sim in similar_pairs if _apply_guards(a, b, sim)),
                None
            )
            if merge_pair is None:
                break
            speaker_a, speaker_b, similarity = merge_pair
            logger.info(f"Standard auto-merge: {speaker_a} <-> {speaker_b} (similarity: {similarity:.3f})")
            self._merge_speakers_internal(speaker_a, speaker_b, similarity)
            merges_performed += 1
            self.history = deque(
                [speaker_a if s == speaker_b else s for s in self.history],
                maxlen=self.history.maxlen
            )

        if merges_performed > 0:
            logger.info(f"Auto-merge completed: {merges_performed} merges performed")

        return merges_performed
    
    def consolidate_post_session(
        self,
        relaxed_threshold: float = 0.65,
        relaxed_guard: float = 0.55,
        max_merges: int = 20,
    ) -> List[Tuple[str, str]]:
        """Run a post-session consolidation pass with relaxed merge criteria.

        Should be called once, before the session is reset, when full-session
        context is available.  Key differences from the live pass:
          - Co-occurrence guard is bypassed (full session means pyannote chunk
            boundaries are less reliable evidence of speaker identity).
          - Prototype guard base threshold is lowered to ``relaxed_guard`` (0.55).
          - ``relaxed_threshold`` (0.65) is used instead of the live 0.72 floor.
          - Up to max_merges (20) merges are allowed.

        Returns:
            List of (merged_into, merged_from) string pairs for each merge performed.
        """
        log_before = len(self._recent_merges_log)
        n = self.auto_merge_similar_speakers(
            similarity_threshold=relaxed_threshold,
            max_merges=max_merges,
            post_session=True,
            override_guard=relaxed_guard,
        )
        new_entries = self._recent_merges_log[log_before:]
        result = [(m["to"], m["from"]) for m in new_entries]
        if n > 0:
            logger.info(
                f"Post-session consolidation: {n} merge(s) "
                f"({[f'{src}→{tgt}' for tgt, src in result]})"
            )
        else:
            logger.info("Post-session consolidation: no merges needed")
        return result

    def _merge_speakers_internal(self, speaker_a: str, speaker_b: str, similarity: float = 0.0):
        """
        Internal method to merge speaker_b into speaker_a.
        The speaker created FIRST (lower number) will survive the merge.
        Prototype pools are combined (keeping up to MAX_PROTOTYPES, dropping
        the most similar pair to avoid redundancy).
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
        
        # ---- Merge prototype pools ----
        protos_a = self.prototypes.get(speaker_a, [])
        protos_b = self.prototypes.get(speaker_b, [])
        combined = protos_a + protos_b
        
        # If too many prototypes, keep the top-K by count (most observed)
        if len(combined) > self.MAX_PROTOTYPES:
            combined.sort(key=lambda p: p[1], reverse=True)
            combined = combined[:self.MAX_PROTOTYPES]
        
        self.prototypes[speaker_a] = combined
        
        # Recompute primary centroid from merged prototypes
        n_a = self.counts[speaker_a]
        n_b = self.counts[speaker_b]
        if combined:
            total_p_count = sum(p[1] for p in combined)
            if total_p_count > 0:
                weighted = sum(p[0] * p[1] for p in combined) / total_p_count
                self.centroids[speaker_a] = self._normalize(weighted)
            else:
                # Fallback to weighted average of old centroids
                merged = (self.centroids[speaker_a] * n_a + self.centroids[speaker_b] * n_b) / (n_a + n_b)
                self.centroids[speaker_a] = self._normalize(merged)
        else:
            merged = (self.centroids[speaker_a] * n_a + self.centroids[speaker_b] * n_b) / (n_a + n_b)
            self.centroids[speaker_a] = self._normalize(merged)
        
        self.counts[speaker_a] = n_a + n_b
        self.last_seen[speaker_a] = max(
            self.last_seen.get(speaker_a, 0),
            self.last_seen.get(speaker_b, 0)
        )

        # ---- Merge segment chunk records ----
        entries_a = self._segment_chunks.get(speaker_a, [])
        entries_b = self._segment_chunks.get(speaker_b, [])
        merged_entries = (entries_a + entries_b)[-50:]
        self._segment_chunks[speaker_a] = merged_entries
        self._segment_chunks.pop(speaker_b, None)

        # ---- Merge transition counts ----
        # Remap all transitions from/to speaker_b → speaker_a
        keys_to_update = [(src, dst) for (src, dst) in self._transitions if src == speaker_b or dst == speaker_b]
        for src, dst in keys_to_update:
            count = self._transitions.pop((src, dst), 0)
            new_src = speaker_a if src == speaker_b else src
            new_dst = speaker_a if dst == speaker_b else dst
            if new_src != new_dst:
                self._transitions[(new_src, new_dst)] = self._transitions.get((new_src, new_dst), 0) + count

        # Remove speaker_b
        del self.centroids[speaker_b]
        del self.counts[speaker_b]
        self.last_seen.pop(speaker_b, None)
        self.prototypes.pop(speaker_b, None)
        self._intra_variance.pop(speaker_b, None)
        # Clear speaker_a's variance so it gets recomputed on next _update_speaker call
        self._intra_variance.pop(speaker_a, None)

        # Log merge event for debugging
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
            self.last_seen.pop(speaker_id, None)
            self.prototypes.pop(speaker_id, None)
            self._segment_chunks.pop(speaker_id, None)
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
        self.prototypes.clear()
        self.counts.clear()
        self.last_seen.clear()
        self.history.clear()
        self.last_speaker = None
        self.speaker_counter = 0
        self._match_log.clear()
        self._recent_merges_log.clear()
        self._transitions.clear()
        self._segment_chunks.clear()
        self._intra_variance.clear()
        self._last_segment_end = 0.0
        
        # Reset statistics
        self._stats = {
            "identifications": 0,
            "valid_embeddings": 0,
            "invalid_embeddings": 0,
            "matches": 0,
            "new_speakers": 0,
            "low_confidence_matches": 0,
            "skipped_short_segments": 0,
            "margin_penalties_applied": 0,
            "margin_new_speaker_forced": 0,
            "overlap_segments_skipped_update": 0,
            "short_segments_skipped_update": 0,
            "short_segments_soft_assigned": 0,
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
    
    def get_and_clear_invalid_reasons(self) -> List[str]:
        """
        Get invalid embedding reasons tracked during identify() calls and clear the log.
        
        Returns:
            List of unique reasons (e.g., ["all_zeros", "nan_detected"])
        """
        reasons = self._invalid_embedding_reasons.copy()
        self._invalid_embedding_reasons.clear()
        return reasons
    
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
        
        # Need at least 2 speakers for PCA 2D projection
        if len(self.centroids) < 2:
            # Return single speaker with default coordinates
            speaker_ids = list(self.centroids.keys())
            return [
                {
                    "speaker_id": speaker_ids[0],
                    "x": 0.0,
                    "y": 0.0,
                    "sample_count": self.counts[speaker_ids[0]]
                }
            ]
        
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
    
    # Minimum segment duration to process (drop entirely — no identify() call)
    MIN_SEGMENT_DURATION = 0.15  # seconds
    # Below this duration a new speaker will NOT be created; short segments are
    # soft-assigned to the best existing match (if score >= 0.50) or discarded.
    MIN_DURATION_FOR_CREATION = 0.40  # seconds

    # Initialize speaker memory with defaults; a WorkerConfigMessage sent at
    # startup by DiarizationClient.start() can override these values.
    # min_samples_for_match=2 requires at least 2 observations of a speaker before matching
    # before their centroid can be a merge target, reducing ghost-speaker merges
    # from short contaminated segments.
    speakers = SpeakerMemory(
        threshold=0.71,
        min_samples_for_match=2,
        min_segment_duration_for_creation=MIN_DURATION_FOR_CREATION,
        ema_alpha=0.05,
        active_window_seconds=28800.0,
    )

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
            
            # Handle worker config message (sent by DiarizationClient.start())
            if isinstance(request, WorkerConfigMessage):
                speakers = SpeakerMemory(
                    threshold=request.threshold,
                    min_samples_for_match=request.min_samples_for_match,
                    min_segment_duration_for_creation=MIN_DURATION_FOR_CREATION,
                    ema_alpha=request.ema_alpha,
                    active_window_seconds=request.active_window_seconds,
                )
                logger.info(
                    f"WORKER: Applied config: threshold={request.threshold}, "
                    f"min_samples={request.min_samples_for_match}, "
                    f"ema_alpha={request.ema_alpha}, "
                    f"active_window={request.active_window_seconds}"
                )
                continue

            # Handle post-session consolidation request
            if isinstance(request, ConsolidateRequest):
                count_before = len(speakers.centroids)
                merges = speakers.consolidate_post_session(
                    relaxed_threshold=request.relaxed_threshold,
                    relaxed_guard=request.relaxed_guard,
                )
                count_after = len(speakers.centroids)
                logger.info(
                    f"WORKER: Post-session consolidation: "
                    f"{len(merges)} merge(s), speakers {count_before} → {count_after}"
                )
                result_queue.put(ConsolidationResult(
                    merges=merges,
                    speaker_count_before=count_before,
                    speaker_count_after=count_after,
                ))
                continue

            # Handle speaker merge request
            if isinstance(request, MergeSpeakersRequest):
                try:
                    speakers.merge_speakers(request.target, request.source)
                    logger.info(f"WORKER: Merged {request.source} -> {request.target}")
                except ValueError as e:
                    logger.warning(f"WORKER: Could not merge {request.source} -> {request.target}: {e}")
                continue

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

                # --- Build overlap and onset-contamination maps for this chunk ---
                # Collect all raw pyannote (start, end, speaker) turns from this chunk.
                raw_turns = [
                    (t.start, t.end, spk)
                    for t, spk in diarization_result.speaker_diarization
                ]
                # Overlap intervals: time regions where 2+ different pyannote labels coexist.
                overlap_intervals: List[Tuple[float, float]] = []
                for i in range(len(raw_turns)):
                    s1, e1, spk1 = raw_turns[i]
                    for j in range(i + 1, len(raw_turns)):
                        s2, e2, spk2 = raw_turns[j]
                        if spk1 == spk2:
                            continue
                        ov_s = max(s1, s2)
                        ov_e = min(e1, e2)
                        if ov_e > ov_s:
                            overlap_intervals.append((ov_s, ov_e))
                # Onset-contamination: a turn whose start is within 500 ms of
                # another speaker's end is considered onset-contaminated (residual
                # bleed from previous speaker).
                other_speaker_ends: Dict[str, List[float]] = {}
                for ts, te, tspk in raw_turns:
                    other_speaker_ends.setdefault(tspk, [])  # ensure key exists
                for ts, te, tspk in raw_turns:
                    for ospk, ends in other_speaker_ends.items():
                        if ospk != tspk:
                            ends.append(te)

                # Combined loop: validate segment duration BEFORE speaker matching
                # Build a mapping from speaker label to embedding index
                speaker_to_embedding_idx = {speaker: idx for idx, speaker in enumerate(diarization_result.speaker_diarization.labels())}
                
                segments = []
                skipped_short = 0
                skipped_short_durations = []
                skipped_invalid = 0
                speaker_ids = {}  # Track for logging/debugging
                
                for turn, speaker in diarization_result.speaker_diarization:
                    segment_duration = turn.end - turn.start
                    
                    # STEP 1: Check segment duration FIRST - skip if too short
                    if segment_duration < MIN_SEGMENT_DURATION:
                        skipped_short += 1
                        skipped_short_durations.append(segment_duration)
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

                    # Overlap: this turn's time range intersects a known overlap interval
                    is_overlap = any(
                        min(turn.end, ov_e) > max(turn.start, ov_s)
                        for ov_s, ov_e in overlap_intervals
                    )
                    # Onset contamination: this turn starts within 500 ms of
                    # a different-speaker's segment end in this chunk.
                    other_ends_for_this_spk = other_speaker_ends.get(speaker, [])
                    is_onset_contaminated = any(
                        0.0 <= turn.start - e <= 0.5
                        for e in other_ends_for_this_spk
                    )

                    segment_info = {
                        "request_id": request.request_id,
                        "speaker_label": speaker,
                        "embedding_index": s,
                        "segment_duration": segment_duration,
                        "start_time": turn.start,
                        "end_time": turn.end,
                        "is_overlap": is_overlap,
                        "is_onset_contaminated": is_onset_contaminated,
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
                    durations_str = ", ".join(f"{d:.2f}s" for d in skipped_short_durations)
                    logger.info(f"Skipped {skipped_short} short segments (< {MIN_SEGMENT_DURATION}s): [{durations_str}]")
                if skipped_invalid > 0:
                    # Get invalid embedding reasons tracked during processing
                    reasons = speakers.get_and_clear_invalid_reasons()
                    if reasons:
                        logger.info(f"Skipped {skipped_invalid} segments with invalid embeddings: {', '.join(reasons)}")
                    else:
                        logger.info(f"Skipped {skipped_invalid} segments with invalid embeddings")
                
                logger.info(f"Extracted {len(segments)} speaker segments")
                if len(segments) > 1:
                    logger.info(f"Speakers identified: {set(s.speaker for s in segments)}")
                    logger.debug(f"diarization result: {diarization_result}")
                
                # Get merge events that occurred during this processing
                merged_speakers = speakers.get_and_clear_merges()
                
                # Get centroid data for the current speakers with timing
                import time
                start_time = time.monotonic()
                speaker_centroids = speakers.get_centroids()
                pairwise_similarity = speakers.get_pairwise_similarity()
                pca_time_ms = (time.monotonic() - start_time) * 1000
                
                # Send result
                result = DiarizationResult(
                    request_id=request.request_id,
                    audio_path=request.audio_path,
                    segments=segments,
                    merged_speakers=merged_speakers if merged_speakers else None,
                    speaker_centroids=speaker_centroids,
                    pairwise_similarity=pairwise_similarity,
                    pca_processing_time_ms=pca_time_ms
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
        threshold: float = 0.71,
        recency_boost: float = 0.02,
        history_size: int = 20,
        min_samples_for_match: int = 2,
        min_segment_duration: float = 0.15,
        ema_alpha: float = 0.05,
        active_window_seconds: float = 28800.0,
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
            min_samples_for_match: Minimum samples before including a speaker in auto-merge
                checks. No longer gates the matching loop (threshold does that).
            min_segment_duration: Minimum segment duration in seconds (default: 0.5)
            ema_alpha: EMA smoothing factor for centroid updates
            active_window_seconds: Decay time constant τ for absence penalty. A speaker
                last seen τ seconds ago has their score multiplied by e^⁻¹ (≈0.37).
                Set to 0 to disable decay entirely. (default: 28800 = 8 hours)
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
        self.active_window_seconds = active_window_seconds
        self.embedding_validator = embedding_validator
        self.on_invalid_embedding = on_invalid_embedding
        self.on_low_confidence = on_low_confidence
        
        # State management
        self._lock: Optional[threading.Lock] = None
        # Cache of the latest speaker data received from the worker process.
        # Populated by record_result() after each DiarizationResult arrives.
        self._speaker_memory_data: dict = {"speakers": [], "pairwise_similarity": []}
        self._process: Optional[multiprocessing.Process] = None
        self._request_queue: Optional[multiprocessing.Queue] = None
        self._result_queue: Optional[multiprocessing.Queue] = None
        self._running = False
        
        # In-flight tracking for graceful shutdown
        self.in_flight_requests: set[str] = set()  # Track request IDs being processed
    
    async def initialize(self):
        """Initialize lock."""
        if self._lock is None:
            self._lock = threading.Lock()
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

            # Forward client configuration to the worker so its SpeakerMemory
            # uses the same parameters as the DiarizationClient.
            self._request_queue.put(WorkerConfigMessage(
                threshold=self.threshold,
                min_samples_for_match=self.min_samples_for_match,
                ema_alpha=self.ema_alpha,
                active_window_seconds=self.active_window_seconds,
            ))
            logger.info("Diarization process started")
    
    def update_params(
        self,
        threshold: Optional[float] = None,
        recency_boost: Optional[float] = None,
        history_size: Optional[int] = None,
        min_samples_for_match: Optional[int] = None,
        min_segment_duration: Optional[float] = None,
        active_window_seconds: Optional[float] = None,
        merge_speakers: Optional[List[Dict[str, str]]] = None
    ):
        """
        Update client parameters dynamically.
        
        Args:
            threshold: New similarity threshold
            recency_boost: New recency boost value
            history_size: New history size
            min_samples_for_match: New min samples for match
            min_segment_duration: New minimum segment duration in seconds
            active_window_seconds: New active window duration in seconds
            merge_speakers: List of merge requests [{"source": "speaker_1", "target": "speaker_0"}]
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
        if active_window_seconds is not None:
            self.active_window_seconds = active_window_seconds

        # Send merge requests directly to the worker so they take effect
        if merge_speakers and self._request_queue is not None:
            for merge in merge_speakers:
                source_id = merge.get("source")
                target_id = merge.get("target")
                if source_id and target_id:
                    self._request_queue.put(MergeSpeakersRequest(source=source_id, target=target_id))
                    logger.info(f"Queued merge {source_id} -> {target_id} for worker")
        elif merge_speakers:
            logger.warning("merge_speakers requested but worker not running; ignoring")

        logger.info(f"DiarizationClient params updated: threshold={self.threshold}, recency_boost={self.recency_boost}, min_segment_duration={self.min_segment_duration}")
    
    def get_stats(self) -> dict:
        """Get the latest speaker statistics cached from the worker."""
        speakers = self._speaker_memory_data.get("speakers", [])
        return {
            "speaker_count": len(speakers),
            "speakers": speakers,
            "pairwise_similarity": self._speaker_memory_data.get("pairwise_similarity", []),
        }

    def record_result(self, result: "DiarizationResult") -> None:
        """Update cached speaker data from an incoming worker result."""
        if result.speaker_centroids is not None:
            self._speaker_memory_data["speakers"] = result.speaker_centroids
        if result.pairwise_similarity is not None:
            self._speaker_memory_data["pairwise_similarity"] = result.pairwise_similarity
    
    def get_centroids(self) -> Tuple[dict, float]:
        """
        Get speaker centroids for graph visualization.

        Returns the latest data cached from the worker (PCA and similarity
        are computed inside the worker process and delivered via record_result).

        Returns:
            Tuple of (dict with speakers and pairwise_similarity, processing_time_ms)
            processing_time_ms is 0.0 because computation is done in the worker.
        """
        return dict(self._speaker_memory_data), 0.0
    
    def reset(self):
        """Reset all accumulated state.
        
        This clears both the in-process speaker memory AND sends a RESET_SIGNAL
        to the worker process to clear its speaker memory for the new stream.
        The worker process continues running across stream boundaries.
        """
        logger.info("DiarizationClient.reset() - clearing state and signaling worker")
        self._speaker_memory_data = {"speakers": [], "pairwise_similarity": []}
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

        Triggers a post-session consolidation pass in the worker (so speakers
        like 41→2, 39→38 etc. are merged before the session state is cleared),
        then drains stale audio requests and sends a RESET_SIGNAL to clear the
        worker's speaker memory for the new stream.
        """
        with self._lock:
            if self._request_queue is not None:
                # Request post-session consolidation BEFORE draining — it was
                # queued before any pending audio so it runs first-in-first-out.
                # (The drain below clears the rest; ConsolidateRequest was already
                # in the queue before any audio requests for the current stream.)
                self._request_queue.put(ConsolidateRequest())
                # Drain remaining audio requests from the previous stream
                try:
                    while not self._request_queue.empty():
                        item = self._request_queue.get_nowait()
                        if isinstance(item, ConsolidateRequest):
                            # Re-queue so it executes before RESET_SIGNAL
                            self._request_queue.put(item)
                            break
                except Exception:
                    pass
                # Signal worker to clear speaker memory for the new stream
                self._request_queue.put(RESET_SIGNAL)
            if self._result_queue is not None:
                try:
                    while not self._result_queue.empty():
                        self._result_queue.get_nowait()
                except Exception:
                    pass
            logger.info("Diarization process state reset (consolidation queued)")

    def consolidate(self,
                    relaxed_threshold: float = 0.65,
                    relaxed_guard: float = 0.55) -> None:
        """Queue a post-session consolidation pass in the worker.

        The result arrives asynchronously as a :class:`ConsolidationResult` on
        the result queue and is handled by the main polling loop, which emits a
        ``speakers_merged`` event to the client.

        Call this before :meth:`reset` (or during a graceful stream-end) so that
        the consolidation always executes ahead of the RESET_SIGNAL.
        """
        if self._request_queue is not None:
            self._request_queue.put(ConsolidateRequest(
                relaxed_threshold=relaxed_threshold,
                relaxed_guard=relaxed_guard,
            ))
            logger.info("DiarizationClient: consolidation request queued")
        else:
            logger.warning("consolidate() called but worker not running; ignoring")
    
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
    
    async def get_result(self, timeout: float = 10.0) -> Optional[Union[DiarizationResult, ConsolidationResult]]:
        """
        Get a diarization or consolidation result.
        
        Args:
            timeout: Maximum time to wait for a result
            
        Returns:
            DiarizationResult, ConsolidationResult, or None if timeout
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