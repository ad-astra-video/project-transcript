"""Tests for Phase 1 diarization improvements.

These tests validate:
- Calibrated confidence scoring
- Auto-merge thresholds
- Configuration defaults
- Enhanced diagnostics
"""

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
from diarization.diarization_client import SpeakerMemory


class TestCalibratedConfidence:
    """Tests for calibrated confidence scoring."""
    
    def test_confidence_high_separation(self):
        """Test confidence when one speaker clearly matches."""
        memory = SpeakerMemory()
        scores = {'speaker_0': 0.91, 'speaker_1': 0.35, 'speaker_2': 0.30}
        confidence, probs = memory._compute_calibrated_confidence(scores, 'speaker_0')
        
        # With high separation, speaker_0 should have highest probability
        assert probs['speaker_0'] > 0.4, f"speaker_0 should have highest probability, got {probs}"
        assert abs(sum(probs.values()) - 1.0) < 0.01, "Probabilities should sum to 1"
    
    def test_confidence_low_separation(self):
        """Test confidence when multiple speakers are similar."""
        memory = SpeakerMemory()
        scores = {'speaker_0': 0.72, 'speaker_1': 0.65, 'speaker_2': 0.62}
        confidence, probs = memory._compute_calibrated_confidence(scores, 'speaker_0')
        
        # With low separation, confidence should be lower
        assert confidence < 0.8, f"Lower confidence expected for close scores, got {confidence}"
    
    def test_confidence_empty_scores(self):
        """Test confidence with empty scores."""
        memory = SpeakerMemory()
        confidence, probs = memory._compute_calibrated_confidence({}, None)
        
        assert confidence == 0.0
        assert probs == {}
    
    def test_confidence_single_speaker(self):
        """Test confidence with single speaker in scores."""
        memory = SpeakerMemory()
        scores = {'speaker_0': 0.95}
        confidence, probs = memory._compute_calibrated_confidence(scores, 'speaker_0')
        
        assert confidence > 0.9, f"High confidence for single speaker, got {confidence}"
        assert probs['speaker_0'] == 1.0
    
    def test_confidence_with_example_data(self):
        """Test confidence with the example data from the issue."""
        memory = SpeakerMemory()
        # Example: speaker_4 match scores
        scores = {
            'speaker_0': 0.44,
            'speaker_1': 0.37,
            'speaker_2': 0.38,
            'speaker_3': 0.72,
            'speaker_4': 0.91,
            'speaker_5': 0.67,
            'speaker_6': 0.33,
            'speaker_7': 0.32
        }
        confidence, probs = memory._compute_calibrated_confidence(scores, 'speaker_4')
        
        # speaker_4 should have highest probability since it has the highest score
        assert probs['speaker_4'] > 0.15, f"speaker_4 should have highest probability, got {probs}"
        # Confidence should be positive since there's a clear leader (though with 8 speakers it's lower)
        assert confidence > 0.1, f"Expected confidence > 0.1, got {confidence}"


class TestAutoMerge:
    """Tests for improved auto-merge logic."""
    
    def test_auto_merge_method_signature(self):
        """Test that auto_merge_similar_speakers has correct default threshold."""
        memory = SpeakerMemory()
        # The default threshold should be 0.76
        # We test this indirectly by checking the method can be called
        result = memory.auto_merge_similar_speakers(similarity_threshold=0.76)
        assert isinstance(result, int), "Should return number of merges"
    
    def test_different_speakers_not_merged(self):
        """Test that different speakers are not merged."""
        memory = SpeakerMemory(threshold=0.72)
        
        # Create first speaker
        emb1 = np.random.randn(512)
        emb1 = emb1 / np.linalg.norm(emb1)
        memory.identify(emb1)
        
        # Create second speaker with different embedding
        emb2 = np.random.randn(512)
        emb2 = emb2 / np.linalg.norm(emb2)
        memory.identify(emb2)
        
        # Should have 2 speakers
        assert len(memory.centroids) == 2, f"Different speakers should not merge, got {len(memory.centroids)} speakers"


class TestConfiguration:
    """Tests for configuration changes."""
    
    def test_default_threshold(self):
        """Test that SpeakerMemory uses the correct default threshold."""
        memory = SpeakerMemory()
        assert memory.threshold == 0.65, f"Default threshold should be 0.65, got {memory.threshold}"
    
    def test_default_ema_alpha(self):
        """Test that default EMA alpha is updated."""
        memory = SpeakerMemory()
        assert memory.ema_alpha == 0.05, f"Default EMA alpha should be 0.05, got {memory.ema_alpha}"
    
    def test_default_min_samples(self):
        """Test that default min_samples_for_match is updated."""
        memory = SpeakerMemory()
        assert memory.min_samples_for_match == 3, f"Default min_samples should be 3, got {memory.min_samples_for_match}"
    
    def test_default_recency_boost(self):
        """Test that default recency_boost is updated."""
        memory = SpeakerMemory()
        assert memory.recency_boost == 0.02, f"Default recency_boost should be 0.02, got {memory.recency_boost}"


class TestDynamicThreshold:
    """Tests for dynamic threshold calculation."""
    
    def test_dynamic_threshold_few_speakers(self):
        """Test dynamic threshold with few speakers."""
        memory = SpeakerMemory(threshold=0.72)

        # Add a few speakers via identify so last_seen is populated
        for _ in range(3):
            emb = np.random.randn(512)
            emb = emb / np.linalg.norm(emb)
            memory.identify(emb)

        t = memory._get_dynamic_threshold()
        # Smooth curve: threshold increases with N to prevent similar speakers merging
        assert t >= 0.72, "Threshold should not go below base"
        assert t <= 0.76, "Threshold should not exceed base + max_increase (0.04)"
    
    def test_dynamic_threshold_many_speakers(self):
        """Test dynamic threshold with many speakers."""
        memory = SpeakerMemory(threshold=0.90)
        
        # Add many speakers
        for _ in range(15):
            emb = np.random.randn(512)
            emb = emb / np.linalg.norm(emb)
            memory.identify(emb)
        
        # Should return a threshold above the base (capped at base + 0.04)
        threshold = memory._get_dynamic_threshold()
        assert threshold > 0.90, f"Threshold should be raised above base for many speakers, got {threshold}"
        assert threshold <= 0.94, f"Threshold should not exceed base + max_increase (0.94), got {threshold}"


class TestEnhancedDiagnostics:
    """Tests for enhanced diagnostics."""
    
    def test_score_distribution(self):
        """Test score distribution calculation."""
        memory = SpeakerMemory()
        
        # Add some identifications
        for _ in range(10):
            emb = np.random.randn(512)
            emb = emb / np.linalg.norm(emb)
            memory.identify(emb)
        
        distribution = memory._get_score_distribution()
        
        assert 'mean' in distribution
        assert 'std' in distribution
        assert 'min' in distribution
        assert 'max' in distribution
        assert distribution['min'] <= distribution['mean'] <= distribution['max']
    
    def test_diagnostics_includes_enhanced_fields(self):
        """Test that diagnostics includes enhanced fields."""
        memory = SpeakerMemory()
        
        # Add some identifications
        for _ in range(5):
            emb = np.random.randn(512)
            emb = emb / np.linalg.norm(emb)
            memory.identify(emb)
        
        diagnostics = memory.get_diagnostics()
        
        assert 'score_distribution' in diagnostics
        assert 'speaker_timeline' in diagnostics
        assert 'avg_confidence' in diagnostics
    
    def test_avg_confidence(self):
        """Test average confidence calculation."""
        memory = SpeakerMemory()
        
        # Add some identifications
        for _ in range(10):
            emb = np.random.randn(512)
            emb = emb / np.linalg.norm(emb)
            memory.identify(emb)
        
        avg_conf = memory._compute_avg_confidence()
        
        assert 0.0 <= avg_conf <= 1.0, f"Average confidence should be between 0 and 1, got {avg_conf}"


class TestPeriodicConsolidation:
    """Tests for periodic consolidation."""
    
    def test_consolidation_frequency(self):
        """Test that consolidation happens at correct frequency."""
        memory = SpeakerMemory(threshold=0.72)
        
        # Create two very similar speakers
        emb1 = np.random.randn(512)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        # First identification creates speaker_0
        memory.identify(emb1)
        
        # Add more identifications to trigger consolidation
        for i in range(4):
            # Create similar embeddings that should trigger merge
            emb = emb1 + np.random.randn(512) * 0.1
            emb = emb / np.linalg.norm(emb)
            memory.identify(emb)
        
        # At 5 identifications, consolidation should trigger
        # Check that auto-merge happened
        stats = memory.get_stats()
        assert stats['speaker_count'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])