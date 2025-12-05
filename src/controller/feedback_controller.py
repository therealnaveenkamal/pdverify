"""
Acceptance-aware feedback controller for dynamic draft length adjustment.
"""

import time
from collections import deque
from typing import Optional
import logging
import numpy as np

from ..utils.config import ControllerConfig

logger = logging.getLogger(__name__)


class FeedbackController:
    """
    Dynamically adjusts draft length (L) based on:
    - Token acceptance ratio
    - Verify lane queue depth
    - Decode lane p95 latency
    """
    
    def __init__(self, config: ControllerConfig):
        """
        Initialize feedback controller.
        
        Args:
            config: Controller configuration
        """
        self.config = config
        self.current_draft_length = config.initial_draft_length
        
        # Metrics tracking
        self.acceptance_ratios: deque = deque(maxlen=100)  # Last 100 requests
        self.decode_latencies: deque = deque(maxlen=100)   # Last 100 decode ops
        
        # Update tracking
        self.requests_since_update = 0
        self.last_update_time = time.time()
        
        # Statistics
        self.total_adjustments = 0
        self.length_increases = 0
        self.length_decreases = 0
        
        logger.info(f"Initialized controller with L={self.current_draft_length}")
    
    def record_acceptance(self, acceptance_ratio: float):
        """
        Record acceptance ratio for a request.
        
        Args:
            acceptance_ratio: Ratio of accepted tokens (0.0 to 1.0)
        """
        self.acceptance_ratios.append(acceptance_ratio)
        self.requests_since_update += 1
    
    def record_decode_latency(self, latency_ms: float):
        """
        Record decode latency.
        
        Args:
            latency_ms: Decode latency in milliseconds
        """
        self.decode_latencies.append(latency_ms)
    
    def update(self, verify_queue_depth: int) -> Optional[int]:
        """
        Update draft length based on current metrics.
        
        Args:
            verify_queue_depth: Current depth of verify queue
            
        Returns:
            New draft length if changed, None otherwise
        """
        # Only update periodically
        if self.requests_since_update < self.config.update_interval_requests:
            return None
        
        old_length = self.current_draft_length
        
        # Calculate metrics
        avg_acceptance = self._get_average_acceptance()
        decode_p95 = self._get_decode_p95()
        
        # Decision logic
        should_decrease = (
            verify_queue_depth > self.config.max_verify_queue_depth or
            decode_p95 > self.config.target_decode_p95_ms or
            avg_acceptance < self.config.length_decrease_threshold
        )
        
        should_increase = (
            verify_queue_depth == 0 and
            decode_p95 < self.config.target_decode_p95_ms * 0.8 and  # 20% headroom
            avg_acceptance > self.config.length_increase_threshold
        )
        
        # Adjust draft length
        if should_decrease and self.current_draft_length > self.config.min_draft_length:
            self.current_draft_length -= 1
            self.length_decreases += 1
            logger.info(f"Decreased L to {self.current_draft_length} "
                       f"(queue={verify_queue_depth}, p95={decode_p95:.1f}ms, "
                       f"accept={avg_acceptance:.2f})")
        
        elif should_increase and self.current_draft_length < self.config.max_draft_length:
            self.current_draft_length += 1
            self.length_increases += 1
            logger.info(f"Increased L to {self.current_draft_length} "
                       f"(queue={verify_queue_depth}, p95={decode_p95:.1f}ms, "
                       f"accept={avg_acceptance:.2f})")
        
        # Reset counter
        self.requests_since_update = 0
        self.last_update_time = time.time()
        
        if self.current_draft_length != old_length:
            self.total_adjustments += 1
            return self.current_draft_length
        
        return None
    
    def get_draft_length(self) -> int:
        """Get current draft length."""
        return self.current_draft_length
    
    def _get_average_acceptance(self) -> float:
        """Calculate average acceptance ratio."""
        if not self.acceptance_ratios:
            return self.config.target_acceptance_ratio
        return float(np.mean(self.acceptance_ratios))
    
    def _get_decode_p95(self) -> float:
        """Calculate p95 decode latency."""
        if not self.decode_latencies:
            return 0.0
        return float(np.percentile(self.decode_latencies, 95))
    
    def get_stats(self) -> dict:
        """Get controller statistics."""
        return {
            "current_draft_length": self.current_draft_length,
            "average_acceptance": self._get_average_acceptance(),
            "decode_p95_ms": self._get_decode_p95(),
            "total_adjustments": self.total_adjustments,
            "length_increases": self.length_increases,
            "length_decreases": self.length_decreases,
            "samples_collected": len(self.acceptance_ratios)
        }
    
    def reset(self):
        """Reset controller state."""
        self.current_draft_length = self.config.initial_draft_length
        self.acceptance_ratios.clear()
        self.decode_latencies.clear()
        self.requests_since_update = 0
        logger.info("Reset feedback controller")
