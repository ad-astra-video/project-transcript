# Speaker Diarization System

This document describes the speaker diarization system implemented in this project, focusing on the mechanics of speaker memory, embedding tracking, merging analysis, and merge guards.

## Overview

The diarization system uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization with a custom `SpeakerMemory` class that tracks speakers across audio segments using embedding similarity. The system is designed to handle streaming audio with proper lifecycle management.

## Architecture

### Core Components

1. **[`DiarizationClient`](diarization_client.py:1)** - Main client for running pyannote speaker diarization
2. **[`SpeakerMemory`](diarization_client.py:78)** - Tracks speakers across audio segments using embedding similarity
3. **[`EmbeddingQualityValidator`](embedding_validator.py:16)** - Filters invalid/low-quality embeddings

---

## SpeakerMemory

The `SpeakerMemory` class is the heart of the speaker tracking system. It maintains speaker profiles and handles speaker identification, merging, and cleanup.

### Multi-Prototype Speaker Representation

Instead of using a single centroid per speaker, the system stores up to **K prototypes** (default: 8) per speaker. This captures within-speaker voice variability instead of averaging it away, which is critical for distinguishing similar-sounding speakers.

```python
# Each speaker stores:
self.centroids: Dict[str, np.ndarray]  # Primary centroid (weighted average)
self.prototypes: Dict[str, List[Tuple[np.ndarray, int, float]]]  # [(embedding, count, last_seen), ...]
self.counts: Dict[str, int]  # Total sample count per speaker
self.last_seen: Dict[str, float]  # Last time seen
```

### Configuration Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_PROTOTYPES` | 8 | Maximum number of prototype embeddings per speaker |
| `PROTOTYPE_MATCH_THRESHOLD` | 0.78 | Cosine similarity threshold for assigning to an existing prototype vs creating new |
| `CO_OCCURRENCE_OVERRIDE_SIMILARITY` | 0.92 | When centroid similarity exceeds this, merge even if speakers co-occurred |
| `PROTOTYPE_MERGE_GUARD_THRESHOLD` | 0.65 | Minimum cross-prototype similarity required for merge |
| `MIN_MARGIN` | 0.02 | Minimum gap between top-2 candidates before margin penalty |
| `MARGIN_PENALTY_SCALE` | 1.0 | Margin penalty multiplier |
| `MARGIN_HARD_THRESHOLD` | 0.65 | When margin is too small AND best score < this, require new speaker |
| `TRANSITION_BONUS_CAP` | 0.03 | Maximum temporal transition bonus |
| `MIN_SPEAKER_SWITCH_GAP` | 0.3 | Minimum time gap (seconds) between segments for same-speaker assignment |

---

## Speaker Identification Process

The [`identify()`](diarization_client.py:459) method matches an embedding to an existing speaker or creates a new one.

### Step 1: Embedding Validation

Before processing, embeddings are validated using [`EmbeddingQualityValidator`](embedding_validator.py:16):

```python
is_valid = self.validator.validate(embedding, segment_info)
if not is_valid:
    return "unknown", 0.0, {}
```

**Validation checks:**
- Empty array detection
- All-zero embeddings
- NaN or Inf values
- Self-dot product sanity check
- Extreme values check
- L2 norm threshold (default: 0.1)
- Mean absolute value threshold (default: 1e-6)
- Zero ratio threshold (default: 50%)

### Step 2: Multi-Prototype Similarity Scoring

For each known speaker, the system computes the **maximum cosine similarity** across all prototypes:

```python
score = self._max_prototype_similarity(embedding, speaker_id)
```

This approach captures the full voice variability of each speaker rather than relying on a single average.

### Step 3: Age-Based Score Decay

Scores are decayed based on how long since the speaker was last seen:

```python
age = now - self.last_seen.get(speaker_id, now)
decay = float(np.exp(-age / self.active_window_seconds))
score *= decay
```

The `active_window_seconds` (default: 28800 = 8 hours) controls the decay rate.

### Step 4: Recency Boost

A small bonus is added if the candidate matches the most recent speaker:

```python
if speaker_id == self.last_speaker:
    score += self.recency_boost  # Default: 0.02
```

### Step 5: Temporal Transition Bonus

Based on observed turn-taking patterns, a bonus is applied for common transitions:

```python
transition_bonus = self._get_transition_bonus(self.last_speaker, speaker_id)
```

This models the fact that speakers often alternate in conversation.

### Step 6: Dynamic Threshold

The threshold is raised smoothly as the number of active speakers grows:

```python
effective_threshold = self._get_dynamic_threshold()
# Formula: threshold = base + max_increase * (1 - exp(-N / lambda))
# At N=0: threshold = base (0.65)
# At N=20: threshold ≈ 0.76
```

This prevents similar speakers from collapsing into each other when many speakers are present.

### Step 7: Margin-Based Penalty

If the gap between top-2 candidates is too small, a penalty is applied:

```python
margin_penalty = self._compute_margin_penalty(best_score, second_best_score)
effective_score = best_score - margin_penalty
```

This biases toward new-speaker creation when the system cannot confidently distinguish between similar-sounding speakers.

### Step 8: Hard Margin Threshold

If the margin is too small AND the effective score is below `MARGIN_HARD_THRESHOLD` (0.65), a new speaker is forced:

```python
force_new_speaker = (
    margin_penalty > 0
    and effective_score < self.MARGIN_HARD_THRESHOLD
)
```

---

## Embedding Tracking

### Prototype Update Mechanism

When a speaker is matched, their prototypes are updated using Exponential Moving Average (EMA):

```python
def _update_speaker(self, speaker_id: str, embedding: np.ndarray):
    # Find closest prototype
    best_idx, best_sim = find_closest_prototype(embedding, self.prototypes[speaker_id])
    
    if best_idx >= 0 and best_sim >= PROTOTYPE_MATCH_THRESHOLD:
        # EMA-update the matching prototype
        updated = alpha * embedding + (1 - alpha) * p_emb
    elif len(protos) < MAX_PROTOTYPES:
        # Add as new prototype
        protos.append((embedding.copy(), 1, now))
    else:
        # Replace oldest prototype
        protos[oldest_idx] = (embedding.copy(), 1, now)
```

The `ema_alpha` parameter (default: 0.05, lowered from 0.15 to prevent centroid drift) controls how quickly prototypes adapt to new voice characteristics.

### Primary Centroid Computation

The primary centroid is recomputed as the weighted average of all prototypes:

```python
total_count = sum(p[1] for p in protos)
weighted = sum(p[0] * p[1] for p in protos) / total_count
self.centroids[speaker_id] = self._normalize(weighted)
```

---

## Speaker Merging

The system supports both automatic and manual speaker merging.

### Automatic Merging

Automatic merging happens in two phases:

1. **Immediate merges** (similarity >= 0.92): For near-certain same-speaker matches
2. **Standard merges** (similarity >= threshold, default 0.88): For likely same-speaker matches

The [`auto_merge_similar_speakers()`](diarization_client.py:955) method is called:
- Every 5 identifications
- When speaker count exceeds 10

### Manual Merging

```python
speaker_memory.merge_speakers(speaker_a, speaker_b)
# Keeps speaker_a, merges speaker_b into it
```

---

## Merge Guards

The system implements multiple guards to prevent merging different speakers:

### 1. Co-Occurrence Guard

This is the **strongest evidence** that two speakers are different. When pyannote.audio separates two speakers within the *same* audio chunk, they are definitively different people.

```python
def _speakers_co_occurred(self, speaker_a: str, speaker_b: str) -> bool:
    # Check if both speakers were assigned different pyannote labels
    # in the same audio chunk
    entries_a = self._segment_chunks.get(speaker_a, [])
    entries_b = self._segment_chunks.get(speaker_b, [])
    
    # Build chunk_id → set of labels mapping
    for cid, lbl in entries_a:
        if cid in chunks_a and lbl not in chunks_a[cid]:
            return True  # Co-occurred with different labels
    return False
```

The system tracks chunk+label pairs in [`_segment_chunks`](diarization_client.py:195):

```python
self._segment_chunks: Dict[str, List[Tuple[str, str]]]  # speaker_id -> [(chunk_id, label), ...]
```

### 2. Prototype Merge Guard

This guard prevents merging speakers with incompatible vocal mode distributions. For each prototype of speaker A, it finds the best match in speaker B's pool. If the **minimum** of these best-matches is below `PROTOTYPE_MERGE_GUARD_THRESHOLD` (0.65), at least one vocal mode of A has no counterpart in B → block the merge.

```python
def _min_cross_prototype_best_match(self, speaker_a: str, speaker_b: str) -> float:
    protos_a = self.prototypes.get(speaker_a, [])
    protos_b = self.prototypes.get(speaker_b, [])
    
    min_best = 1.0
    for emb_a, _, _ in protos_a:
        best = max(cosine_similarity(emb_a, emb_b) for emb_b, _, _ in protos_b)
        min_best = min(min_best, best)
    return min_best
```

### 3. High-Similarity Override

When centroid similarity exceeds `CO_OCCURRENCE_OVERRIDE_SIMILARITY` (0.92), both guards are bypassed. This handles cases where stale chunk data might block a near-certain same-speaker match.

### Merge Decision Flow

```
                    ┌─────────────────────────────┐
                    │  Check similarity >= 0.92  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │ YES                                       │ NO
              ▼                                           ▼
    ┌─────────────────┐                    ┌─────────────────────────────┐
    │ Override merge  │                    │ Check co-occurrence guard   │
    │ (bypass guards) │                    └──────────────┬──────────────┘
    └─────────────────┘                                   │
                         ┌────────────────────────────────┴────────────────┐
                         │ NO (didn't co-occur)                           │ YES
                         ▼                                                ▼
          ┌──────────────────────────┐                    ┌─────────────────────┐
          │ Check prototype guard    │                    │ BLOCK merge         │
          │ (cross_min >= 0.65)      │                    │ (different people) │
          └──────────────┬───────────┘                    └─────────────────────┘
                        │
           ┌────────────┴────────────┐
           │ YES                      │ NO
           ▼                          ▼
    ┌─────────────┐        ┌─────────────────────┐
    │ Merge       │        │ BLOCK merge         │
    │ speakers    │        │ (incompatible modes)│
    └─────────────┘        └─────────────────────┘
```

---

## Post-Creation Merge Check

After creating a new speaker, the system immediately checks if it should be merged with an existing speaker:

```python
def _check_and_merge_after_creation(self, new_speaker_id: str, new_embedding: np.ndarray):
    for existing_id, centroid in self.centroids.items():
        # High-similarity override
        if sim >= self.CO_OCCURRENCE_OVERRIDE_SIMILARITY:
            return existing_id
        
        # Co-occurrence guard
        if self._speakers_co_occurred(existing_id, new_speaker_id):
            continue
        
        # Prototype merge guard
        cross_min = self._min_cross_prototype_best_match(existing_id, new_speaker_id)
        if cross_min < self.PROTOTYPE_MERGE_GUARD_THRESHOLD:
            continue
        
        # Standard merge at 0.80
        if sim >= 0.80:
            return existing_id
```

---

## Internal Merge Process

The [`_merge_speakers_internal()`](diarization_client.py:1072) method handles the actual merge:

1. **Speaker survival**: The first-created speaker (lower number) survives
2. **Prototype pool merging**: Combined and trimmed to MAX_PROTOTYPES
3. **Centroid recomputation**: Weighted average of merged prototypes
4. **Chunk records merging**: Combined segment chunk history
5. **Transition remapping**: All transitions from/to speaker_b → speaker_a

---

## Statistics and Diagnostics

### Tracking Statistics

The system tracks comprehensive statistics:

```python
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
}
```

### Diagnostics

```python
speaker_memory.get_diagnostics()
# Returns:
# - speaker_count
# - total_samples
# - avg_samples_per_speaker
# - dynamic_threshold
# - similarity_matrix (top 10 similar pairs)
# - low_confidence_rate
# - validator_stats
# - recent_merges
# - score_distribution
# - speaker_timeline
# - avg_confidence
```

---

## Reset and Cleanup

### Reset Signal

The system supports a reset signal:

```python
RESET_SIGNAL = "RESET_SIGNAL"
```

### Invalid Speaker Cleanup

```python
def cleanup_invalid_speakers(self) -> int:
    # Remove speakers with invalid embeddings (zeros, NaN, or very low quality)
```

---

## Usage Example

```python
from diarization import SpeakerMemory

# Create speaker memory with custom configuration
speaker_memory = SpeakerMemory(
    threshold=0.65,           # Base similarity threshold
    recency_boost=0.02,       # Bonus for same speaker continuation
    history_size=20,          # Number of recent speakers to track
    min_samples_for_match=3, # Minimum observations before matching
    min_segment_duration=0.15, # Minimum segment duration
    ema_alpha=0.05,           # EMA smoothing factor
    active_window_seconds=1200.0, # Active window (20 minutes)
)

# Identify a speaker from an embedding
speaker_id, confidence, alt_speakers = speaker_memory.identify(
    embedding=np.array([...]),  # Speaker embedding
    segment_info={"timestamp": 123.45, "text": "Hello"}
)

# Get diagnostics
diagnostics = speaker_memory.get_diagnostics()

# Manual merge (if needed)
speaker_memory.merge_speakers("speaker_0", "speaker_1")
```

---

## Integration with pyannote.audio

The `DiarizationClient` wraps pyannote.audio's pipeline:

1. Loads the pyannote speaker diarization pipeline
2. Processes audio files to extract speaker segments
3. Extracts speaker embeddings using pyannote's embedding model
4. Uses `SpeakerMemory` to track and merge speakers across segments

---

## Key Design Decisions

1. **Multi-prototype representation**: Captures within-speaker variability for better discrimination
2. **Chunk-based co-occurrence**: Reliable signal that's independent of processing speed
3. **Prototype merge guard**: Prevents merging speakers with incompatible vocal modes
4. **Dynamic threshold**: Prevents speaker collapse when many speakers are present
5. **Margin-based scoring**: Biases toward new speaker creation when ambiguous
6. **Age-based decay**: Allows speakers to "expire" when not seen for extended periods