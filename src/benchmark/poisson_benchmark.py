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
    
    def run_benchmark(self, engine, requests: Optional[List[BenchmarkRequest]] = None, max_concurrent: int = 5, no_delay: bool = False):
        """
        Run benchmark on an engine with concurrent request processing.
        Respects Poisson arrival times and enforces max_concurrent limit.

        Args:
            engine: Engine to test (must have submit_request_async method)
            requests: Pre-generated requests, or None to generate
            max_concurrent: Maximum number of concurrent requests
            no_delay: If True, ignore arrival times and submit as fast as possible (for fair comparison with baseline)

        Returns:
            Benchmark results
        """
        if requests is None:
            requests = self.generate_requests()

        # Sort by arrival time
        requests = sorted(requests, key=lambda r: r.arrival_time)

        logger.info(f"Starting benchmark with {len(requests)} requests (max_concurrent={max_concurrent})")

        from ..scheduler import Request, LaneType
        import threading

        results = []
        active_requests = {}  # request_id -> (benchmark_req, start_time)
        active_count = 0
        request_idx = 0

        # Thread-safe synchronization
        result_lock = threading.Lock()
        active_count_lock = threading.Lock()

        def completion_callback(engine_req: Request, result_text: str):
            """Callback called when a request completes."""
            nonlocal active_count

            with result_lock:
                benchmark_req_data = active_requests.pop(engine_req.request_id, None)
                if benchmark_req_data is None:
                    logger.error(f"Completion callback for unknown request {engine_req.request_id}")
                    return

                benchmark_req, req_start = benchmark_req_data
                end_time = time.time()
                latency_ms = (end_time - req_start) * 1000

                result = {
                    "request_id": benchmark_req.request_id,
                    "latency_ms": latency_ms,
                    "tokens_generated": getattr(engine_req, 'tokens_generated', 0),
                    "tokens_accepted": getattr(engine_req, 'tokens_accepted', 0),
                    "acceptance_ratio": getattr(engine_req, 'get_acceptance_ratio', lambda: 0.0)()
                }

                if hasattr(engine_req, 'error') and engine_req.error:
                    result["error"] = engine_req.error

                results.append(result)
                logger.debug(f"Completed {benchmark_req.request_id} in {latency_ms:.1f}ms")

            # Decrement active count
            with active_count_lock:
                active_count -= 1

        # Submit requests at their arrival times
        benchmark_start = time.time()

        while request_idx < len(requests):
            current_time = time.time() - benchmark_start
            benchmark_req = requests[request_idx]

            # Wait until arrival time (unless no_delay mode)
            if not no_delay and benchmark_req.arrival_time > current_time:
                time.sleep(benchmark_req.arrival_time - current_time)

            # Enforce concurrency limit - wait for a slot
            while True:
                with active_count_lock:
                    if active_count < max_concurrent:
                        break
                time.sleep(0.001)  # Reduced from 0.01 for faster response

            # Submit request
            engine_req = Request(
                request_id=benchmark_req.request_id,
                prompt=benchmark_req.prompt,
                stage=LaneType.PREFILL,
                created_at=time.time()
            )

            req_start = time.time()
            try:
                with result_lock:
                    active_requests[benchmark_req.request_id] = (benchmark_req, req_start)

                with active_count_lock:
                    active_count += 1

                engine.submit_request_async(engine_req, callback=completion_callback)

                actual_submit_time = time.time() - benchmark_start
                logger.debug(f"Submitted {benchmark_req.request_id} at t={actual_submit_time:.2f}s "
                           f"(scheduled: {benchmark_req.arrival_time:.2f}s, active: {active_count})")
            except Exception as e:
                logger.error(f"Error submitting {benchmark_req.request_id}: {e}")
                with result_lock:
                    results.append({
                        "request_id": benchmark_req.request_id,
                        "error": str(e),
                        "latency_ms": 0.0
                    })
                with active_count_lock:
                    active_count -= 1

            request_idx += 1

        # Wait for all requests to complete
        logger.info(f"All requests submitted, waiting for completion...")
        while active_count > 0:
            time.sleep(0.001)  # Reduced from 0.01 for faster completion detection

        total_time = time.time() - benchmark_start
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
