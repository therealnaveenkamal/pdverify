"""
Poisson arrival benchmark for realistic load testing.
"""

import time
import random
import logging
from typing import List, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRequest:
    """Represents a benchmark request."""
    request_id: str
    prompt: str
    arrival_time: float
    max_tokens: int = 100


class PoissonBenchmark:
    """
    Generates requests following Poisson arrival pattern.
    Simulates realistic production traffic.
    """
    
    def __init__(
        self,
        prompts: List[str],
        arrival_rate: float = 1.0,
        duration_seconds: Optional[int] = None,
        num_requests: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize Poisson benchmark.
        
        Args:
            prompts: List of prompts to use
            arrival_rate: Average requests per second (lambda)
            duration_seconds: Run for this duration (alternative to num_requests)
            num_requests: Generate this many requests (alternative to duration)
            seed: Random seed
        """
        self.prompts = prompts
        self.arrival_rate = arrival_rate
        self.duration_seconds = duration_seconds
        self.num_requests = num_requests
        
        random.seed(seed)
        np.random.seed(seed)
        
        if not duration_seconds and not num_requests:
            raise ValueError("Must specify either duration_seconds or num_requests")
        
        logger.info(f"Poisson benchmark initialized: rate={arrival_rate} req/s")
    
    def generate_requests(self) -> List[BenchmarkRequest]:
        """
        Generate benchmark requests with Poisson arrivals.
        
        Returns:
            List of benchmark requests with arrival times
        """
        requests = []
        current_time = 0.0
        request_count = 0
        
        while True:
            # Generate inter-arrival time (exponential distribution)
            inter_arrival = np.random.exponential(1.0 / self.arrival_rate)
            current_time += inter_arrival
            
            # Check termination conditions
            if self.duration_seconds and current_time > self.duration_seconds:
                break
            if self.num_requests and request_count >= self.num_requests:
                break
            
            # Create request
            prompt = random.choice(self.prompts)
            request = BenchmarkRequest(
                request_id=f"req_{request_count:04d}",
                prompt=prompt,
                arrival_time=current_time,
                max_tokens=random.randint(50, 150)  # Varying output lengths
            )
            requests.append(request)
            request_count += 1
        
        logger.info(f"Generated {len(requests)} requests over {current_time:.1f} seconds")
        return requests
    
    def run_benchmark(self, engine, requests: Optional[List[BenchmarkRequest]] = None):
        """
        Run benchmark on an engine.
        
        Args:
            engine: Engine to test (must have process_request method)
            requests: Pre-generated requests, or None to generate
            
        Returns:
            Benchmark results
        """
        if requests is None:
            requests = self.generate_requests()
        
        logger.info(f"Starting benchmark with {len(requests)} requests")
        
        results = []
        start_time = time.time()
        last_arrival_time = 0.0
        
        for request in requests:
            # Wait for request arrival time
            elapsed = time.time() - start_time
            wait_time = request.arrival_time - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Process request
            from ..scheduler import Request, LaneType
            
            req = Request(
                request_id=request.request_id,
                prompt=request.prompt,
                stage=LaneType.PREFILL,
                created_at=time.time()
            )
            
            req_start = time.time()
            try:
                result_text = engine.process_request(req)
                req_end = time.time()
                
                results.append({
                    "request_id": request.request_id,
                    "latency_ms": (req_end - req_start) * 1000,
                    "tokens_generated": req.tokens_generated,
                    "tokens_accepted": req.tokens_accepted,
                    "acceptance_ratio": req.get_acceptance_ratio()
                })
                
                logger.debug(f"Completed {request.request_id} in {(req_end - req_start) * 1000:.1f}ms")
            
            except Exception as e:
                logger.error(f"Error processing {request.request_id}: {e}")
                results.append({
                    "request_id": request.request_id,
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.1f} seconds")
        
        return self._analyze_results(results, total_time)
    
    def _analyze_results(self, results: List[dict], total_time: float) -> dict:
        """Analyze benchmark results."""
        successful = [r for r in results if "error" not in r]
        
        if not successful:
            return {"error": "No successful requests"}
        
        latencies = [r["latency_ms"] for r in successful]
        acceptance_ratios = [r["acceptance_ratio"] for r in successful]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "total_time_seconds": total_time,
            "throughput_rps": len(successful) / total_time,
            "latency_ms": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies)
            },
            "acceptance_ratio": {
                "mean": np.mean(acceptance_ratios),
                "median": np.median(acceptance_ratios),
                "min": np.min(acceptance_ratios),
                "max": np.max(acceptance_ratios)
            }
        }


def get_sharegpt_prompts(num_samples: int = 100) -> List[str]:
    """
    Get sample prompts (placeholder for ShareGPT dataset).
    In production, this would load from actual ShareGPT dataset.
    
    Args:
        num_samples: Number of prompts to return
        
    Returns:
        List of prompts
    """
    # Sample prompts for testing
    sample_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the best practices for Python programming?",
        "Describe the water cycle.",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Explain the theory of relativity.",
        "Write a poem about nature.",
        "What are the causes of climate change?",
        "Describe how a car engine works."
    ]
    
    # Repeat and shuffle to get desired number
    prompts = sample_prompts * (num_samples // len(sample_prompts) + 1)
    random.shuffle(prompts)
    return prompts[:num_samples]
