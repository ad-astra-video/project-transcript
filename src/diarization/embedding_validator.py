"""
Embedding Quality Validator for Speaker Diarization.

This module provides validation for speaker embeddings to filter out
low-quality or invalid embeddings (zeros, noise, silence) before they
are processed and stored in speaker memory.
"""

import logging
from typing import Tuple, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingQualityValidator:
    """
    Validates speaker embeddings for quality before processing.
    
    This validator checks embeddings for:
    - Empty arrays
    - All-zero embeddings
    - Low L2 norm (indicates silence or noise)
    - Low mean absolute value
    - High ratio of zero values
    
    These checks prevent invalid embeddings from polluting speaker memory
    and degrading diarization quality.
    """
    
    def __init__(
        self,
        min_embedding_norm: float = 0.1,
        min_embedding_mean: float = 1e-6,
        max_zero_ratio: float = 0.5,
        log_invalid: bool = True,
        on_invalid: Optional[Callable[[np.ndarray, str, Optional[dict]], None]] = None
    ):
        """
        Initialize the embedding quality validator.
        
        Args:
            min_embedding_norm: Minimum L2 norm for valid embedding.
                Embeddings with norm below this are considered low-quality.
                Default: 0.1
            min_embedding_mean: Minimum mean absolute value for valid embedding.
                Default: 1e-6
            max_zero_ratio: Maximum ratio of zero values allowed in embedding.
                Default: 0.5 (50%)
            log_invalid: Whether to log invalid embeddings with details.
            on_invalid: Optional callback when embedding is invalid.
                Called with (embedding, reason, segment_info)
        """
        self.min_embedding_norm = min_embedding_norm
        self.min_embedding_mean = min_embedding_mean
        self.max_zero_ratio = max_zero_ratio
        self.log_invalid = log_invalid
        self.on_invalid = on_invalid
        
        # Statistics tracking
        self._total_checked = 0
        self._invalid_count = 0
        self._invalid_reasons = {}
    
    def is_valid(self, embedding: np.ndarray) -> Tuple[bool, str]:
        """
        Check if embedding passes quality thresholds.
        
        Args:
            embedding: The speaker embedding to validate.
            
        Returns:
            Tuple of (is_valid, reason_for_rejection).
            reason_for_rejection is empty string if valid.
        """
        # Check for empty array
        if len(embedding) == 0:
            return False, "empty_embedding"
        
        # Check for all zeros
        if np.allclose(embedding, 0):
            return False, "all_zeros"
        
        # NEW: Check for NaN or Inf values
        if not np.all(np.isfinite(embedding)):
            return False, "contains_nan_or_inf"
        
        # NEW: Self-dot product sanity check
        dot_self = np.dot(embedding, embedding)
        if not np.isfinite(dot_self) or dot_self < 1e-8:
            return False, "invalid_self_dot"
        
        # NEW: Extreme values check
        if np.max(np.abs(embedding)) > 100.0:
            return False, "extreme_values"
        
        # Check L2 norm
        norm = np.linalg.norm(embedding)
        if norm < self.min_embedding_norm:
            return False, f"low_norm_{norm:.6f}"
        
        # Check mean absolute value
        mean_abs = np.mean(np.abs(embedding))
        if mean_abs < self.min_embedding_mean:
            return False, f"low_mean_{mean_abs:.10f}"
        
        # Check zero ratio
        zero_ratio = np.sum(np.isclose(embedding, 0)) / len(embedding)
        if zero_ratio > self.max_zero_ratio:
            return False, f"high_zero_ratio_{zero_ratio:.2f}"
        
        return True, ""
    
    def validate(
        self,
        embedding: np.ndarray,
        segment_info: Optional[dict] = None
    ) -> bool:
        """
        Validate embedding and handle invalid cases.
        
        Args:
            embedding: The speaker embedding to validate.
            segment_info: Optional metadata for logging.
            
        Returns:
            True if embedding is valid, False otherwise.
        """
        self._total_checked += 1
        
        is_valid, reason = self.is_valid(embedding)
        
        if not is_valid:
            self._invalid_count += 1
            self._invalid_reasons[reason] = self._invalid_reasons.get(reason, 0) + 1
            
            if self.log_invalid:
                self._log_invalid_embedding(embedding, reason, segment_info)
            
            if self.on_invalid:
                try:
                    self.on_invalid(embedding, reason, segment_info)
                except Exception as e:
                    logger.warning(f"Error in on_invalid callback: {e}")
        
        return is_valid
    
    def _log_invalid_embedding(
        self,
        embedding: np.ndarray,
        reason: str,
        segment_info: Optional[dict] = None
    ):
        """Log details about an invalid embedding."""
        try:
            norm = float(np.linalg.norm(embedding))
            mean_abs = float(np.mean(np.abs(embedding)))
            zero_ratio = float(np.sum(np.isclose(embedding, 0)) / len(embedding))
            
            log_data = {
                "reason": reason,
                "embedding_norm": norm,
                "embedding_mean": mean_abs,
                "zero_ratio": zero_ratio,
                "embedding_dim": len(embedding),
            }
            
            if segment_info:
                log_data["segment_info"] = segment_info
            
            logger.warning(
                f"Invalid embedding rejected: {reason}",
                extra={"diarization_embedding_quality": log_data}
            )
        except Exception as e:
            logger.warning(f"Error logging invalid embedding: {e}")
    
    def get_validation_stats(self) -> dict:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics.
        """
        total = self._total_checked
        invalid = self._invalid_count
        
        return {
            "total_checked": total,
            "invalid_count": invalid,
            "valid_count": total - invalid,
            "validity_rate": (total - invalid) / max(1, total),
            "invalid_reasons": self._invalid_reasons.copy(),
        }
    
    def reset_stats(self):
        """Reset validation statistics."""
        self._total_checked = 0
        self._invalid_count = 0
        self._invalid_reasons.clear()
    
    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "min_embedding_norm": self.min_embedding_norm,
            "min_embedding_mean": self.min_embedding_mean,
            "max_zero_ratio": self.max_zero_ratio,
            "log_invalid": self.log_invalid,
        }


def create_default_validator() -> EmbeddingQualityValidator:
    """
    Create a default EmbeddingQualityValidator with standard settings.
    
    Returns:
        Configured EmbeddingQualityValidator instance.
    """
    return EmbeddingQualityValidator(
        min_embedding_norm=0.1,
        min_embedding_mean=1e-6,
        max_zero_ratio=0.5,
        log_invalid=True
    )


def create_strict_validator() -> EmbeddingQualityValidator:
    """
    Create a stricter EmbeddingQualityValidator for high-quality audio.
    
    Returns:
        Stricter EmbeddingQualityValidator instance.
    """
    return EmbeddingQualityValidator(
        min_embedding_norm=0.2,
        min_embedding_mean=1e-5,
        max_zero_ratio=0.3,
        log_invalid=True
    )


def create_lenient_validator() -> EmbeddingQualityValidator:
    """
    Create a more lenient EmbeddingQualityValidator for challenging audio.
    
    Returns:
        More lenient EmbeddingQualityValidator instance.
    """
    return EmbeddingQualityValidator(
        min_embedding_norm=0.05,
        min_embedding_mean=1e-7,
        max_zero_ratio=0.7,
        log_invalid=True
    )