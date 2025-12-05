"""
Metrics collection for performance monitoring.
"""

import time
import json
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Optional
import logging
import numpy as np

from ..utils.config import MetricsConfig

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and tracks performance metrics."""
    
    def __init__(self, config: MetricsConfig):
        """
        Initialize metrics collector.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
        
        # Latency tracking per lane
        self.prefill_latencies: deque = deque(maxlen=1000)
        self.decode_latencies: deque = deque(maxlen=1000)
        self.verify_latencies: deque = deque(maxlen=1000)
        self.e2e_latencies: deque = deque(maxlen=1000)  # End-to-end
        
        # Throughput tracking
        self.tokens_per_second: deque = deque(maxlen=100)
        self.requests_completed = 0
        self.start_time = time.time()
        
        # Queue depths over time
        self.queue_depths: Dict[str, deque] = {
            "prefill": deque(maxlen=1000),
            "decode": deque(maxlen=1000),
            "verify": deque(maxlen=1000)
        }
        
        # Acceptance rate tracking
        self.acceptance_rates: deque = deque(maxlen=1000)
        
        # Draft length history
        self.draft_lengths: deque = deque(maxlen=1000)
        
        # Last log time
        self.last_log_time = time.time()
        
        # Output directory
        if config.save_to_file:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Metrics will be saved to {self.output_dir}")
    
    def record_latency(self, lane: str, latency_ms: float):
        """
        Record latency for a lane.
        
        Args:
            lane: Lane name (prefill, decode, verify, e2e)
            latency_ms: Latency in milliseconds
        """
        if lane == "prefill":
            self.prefill_latencies.append(latency_ms)
        elif lane == "decode":
            self.decode_latencies.append(latency_ms)
        elif lane == "verify":
            self.verify_latencies.append(latency_ms)
        elif lane == "e2e":
            self.e2e_latencies.append(latency_ms)
    
    def record_queue_depth(self, lane: str, depth: int):
        """Record queue depth for a lane."""
        if lane in self.queue_depths:
            self.queue_depths[lane].append(depth)
    
    def record_acceptance_rate(self, rate: float):
        """Record token acceptance rate."""
        self.acceptance_rates.append(rate)
    
    def record_draft_length(self, length: int):
        """Record current draft length."""
        self.draft_lengths.append(length)
    
    def record_request_completion(self, num_tokens: int, duration_s: float):
        """
        Record completed request.
        
        Args:
            num_tokens: Number of tokens generated
            duration_s: Total duration in seconds
        """
        self.requests_completed += 1
        if duration_s > 0:
            tps = num_tokens / duration_s
            self.tokens_per_second.append(tps)
    
    def get_percentiles(self, data: deque, percentiles: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate percentiles for data.
        
        Args:
            data: Data to analyze
            percentiles: List of percentiles (0-1), defaults to config
            
        Returns:
            Dictionary of percentile values
        """
        if not data:
            return {}
        
        percentiles = percentiles or self.config.percentiles
        values = np.array(data)
        
        result = {}
        for p in percentiles:
            key = f"p{int(p * 100)}"
            result[key] = float(np.percentile(values, p * 100))
        
        result["mean"] = float(np.mean(values))
        result["min"] = float(np.min(values))
        result["max"] = float(np.max(values))
        
        return result
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        uptime = time.time() - self.start_time
        
        return {
            "uptime_seconds": uptime,
            "requests_completed": self.requests_completed,
            "latency_ms": {
                "prefill": self.get_percentiles(self.prefill_latencies),
                "decode": self.get_percentiles(self.decode_latencies),
                "verify": self.get_percentiles(self.verify_latencies),
                "e2e": self.get_percentiles(self.e2e_latencies)
            },
            "throughput": {
                "tokens_per_second": self.get_percentiles(self.tokens_per_second),
                "requests_per_second": self.requests_completed / uptime if uptime > 0 else 0
            },
            "queue_depths": {
                lane: self.get_percentiles(depths)
                for lane, depths in self.queue_depths.items()
            },
            "acceptance_rate": self.get_percentiles(self.acceptance_rates),
            "draft_length": self.get_percentiles(self.draft_lengths)
        }
    
    def log_metrics(self, force: bool = False):
        """
        Log current metrics.
        
        Args:
            force: Force logging regardless of interval
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_log_time) < self.config.log_interval_seconds:
            return
        
        summary = self.get_summary()
        
        logger.info("=" * 80)
        logger.info("METRICS SUMMARY")
        logger.info(f"Requests completed: {summary['requests_completed']}")
        logger.info(f"Uptime: {summary['uptime_seconds']:.1f}s")
        
        if summary['latency_ms']['e2e']:
            logger.info(f"E2E Latency - p50: {summary['latency_ms']['e2e'].get('p50', 0):.1f}ms, "
                       f"p95: {summary['latency_ms']['e2e'].get('p95', 0):.1f}ms, "
                       f"p99: {summary['latency_ms']['e2e'].get('p99', 0):.1f}ms")
        
        if summary['throughput']['tokens_per_second']:
            logger.info(f"Throughput - mean: {summary['throughput']['tokens_per_second'].get('mean', 0):.1f} tokens/s")
        
        if summary['acceptance_rate']:
            logger.info(f"Acceptance rate - mean: {summary['acceptance_rate'].get('mean', 0):.2f}")
        
        logger.info("=" * 80)
        
        self.last_log_time = current_time
        
        # Save to file
        if self.config.save_to_file:
            self.save_metrics(summary)
    
    def save_metrics(self, summary: Optional[dict] = None):
        """Save metrics to file."""
        if not self.config.save_to_file:
            return
        
        summary = summary or self.get_summary()
        
        output_file = self.output_dir / f"metrics_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.debug(f"Saved metrics to {output_file}")
    
    def reset(self):
        """Reset all metrics."""
        self.prefill_latencies.clear()
        self.decode_latencies.clear()
        self.verify_latencies.clear()
        self.e2e_latencies.clear()
        self.tokens_per_second.clear()
        self.acceptance_rates.clear()
        self.draft_lengths.clear()
        
        for depths in self.queue_depths.values():
            depths.clear()
        
        self.requests_completed = 0
        self.start_time = time.time()
        
        logger.info("Reset all metrics")
