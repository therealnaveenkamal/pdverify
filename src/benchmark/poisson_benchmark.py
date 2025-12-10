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
    
    def run_benchmark(self, engine, requests: Optional[List[BenchmarkRequest]] = None, max_concurrent: int = 5):
        """
        Run benchmark on an engine with concurrent request processing.
        
        Args:
            engine: Engine to test (must have submit_request_async method)
            requests: Pre-generated requests, or None to generate
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Benchmark results
        """
        if requests is None:
            requests = self.generate_requests()
        
        logger.info(f"Starting benchmark with {len(requests)} requests (max_concurrent={max_concurrent})")
        
        from ..scheduler import Request, LaneType
        import threading

        results = []
        start_time = time.time()
        active_requests = {}  # request_id -> (benchmark_req, start_time)
        remaining_requests = list(requests)
        request_idx = 0

        # Thread-safe result collection
        result_lock = threading.Lock()

        def completion_callback(engine_req: Request, result_text: str):
            """Callback called when a request completes."""
            with result_lock:
                benchmark_req = active_requests.pop(engine_req.request_id, None)
                if benchmark_req is None:
                    logger.error(f"Completion callback for unknown request {engine_req.request_id}")
                    return

                benchmark_req, start_time = benchmark_req
                end_time = time.time()
                request_latency_ms = (end_time - start_time) * 1000

                # Calculate per-token latencies
                token_latencies_ms = []
                if hasattr(engine_req, 'token_timestamps') and engine_req.token_timestamps:
                    request_start = engine_req.request_start_time if hasattr(engine_req, 'request_start_time') and engine_req.request_start_time > 0 else start_time
                    for token_time in engine_req.token_timestamps:
                        token_latency = (token_time - request_start) * 1000
                        token_latencies_ms.append(token_latency)

                result = {
                    "request_id": benchmark_req.request_id,
                    "request_latency_ms": request_latency_ms,
                    "token_latencies_ms": token_latencies_ms,
                    "tokens_generated": getattr(engine_req, 'tokens_generated', 0),
                    "tokens_accepted": getattr(engine_req, 'tokens_accepted', 0),
                    "acceptance_ratio": getattr(engine_req, 'get_acceptance_ratio', lambda: 0.0)()
                }

                if hasattr(engine_req, 'error') and engine_req.error:
                    result["error"] = engine_req.error

                results.append(result)
                logger.debug(f"Completed {benchmark_req.request_id} in {request_latency_ms:.1f}ms")

        # Submit requests at their Poisson arrival times (asynchronously)
        benchmark_start = start_time

        while request_idx < len(remaining_requests):
            benchmark_req = remaining_requests[request_idx]
            request_idx += 1

            # Wait until this request's arrival time
            target_time = benchmark_start + benchmark_req.arrival_time
            sleep_duration = target_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            req_start = time.time()
            engine_req = Request(
                request_id=benchmark_req.request_id,
                prompt=benchmark_req.prompt,
                stage=LaneType.PREFILL,
                created_at=req_start
            )
            # Set request start time for token latency calculation
            engine_req.request_start_time = req_start

            try:
                active_requests[benchmark_req.request_id] = (benchmark_req, req_start)
                engine.submit_request_async(engine_req, callback=completion_callback)
                actual_time = time.time() - benchmark_start
                logger.debug(f"Submitted {benchmark_req.request_id} at t={actual_time:.3f}s (scheduled: {benchmark_req.arrival_time:.3f}s)")
            except Exception as e:
                logger.error(f"Error submitting {benchmark_req.request_id}: {e}")
                with result_lock:
                    results.append({
                        "request_id": benchmark_req.request_id,
                        "error": str(e),
                        "latency_ms": 0.0
                    })

        # Wait for all requests to complete
        while len(active_requests) > 0:
            time.sleep(0.01)  # Small sleep to avoid busy waiting

        total_time = time.time() - start_time
        logger.info(f"Benchmark completed in {total_time:.1f} seconds")
        
        return self._analyze_results(results, total_time)
    
    def _analyze_results(self, results: List[dict], total_time: float) -> dict:
        """Analyze benchmark results."""
        successful = [r for r in results if "error" not in r]
        
        if not successful:
            return {"error": "No successful requests"}
        
        # Collect all token latencies for per-token metrics
        all_token_latencies = []
        total_tokens = 0
        request_latencies = []
        acceptance_ratios = []
        
        for r in successful:
            request_latencies.append(r["request_latency_ms"])
            acceptance_ratios.append(r["acceptance_ratio"])
            
            # Collect per-token latencies
            if "token_latencies_ms" in r and r["token_latencies_ms"]:
                all_token_latencies.extend(r["token_latencies_ms"])
                total_tokens += len(r["token_latencies_ms"])
        
        # Calculate token-based metrics
        token_latency_stats = {}
        if all_token_latencies:
            token_latency_stats = {
                "mean": float(np.mean(all_token_latencies)),
                "median": float(np.median(all_token_latencies)),
                "p95": float(np.percentile(all_token_latencies, 95)),
                "p99": float(np.percentile(all_token_latencies, 99)),
                "min": float(np.min(all_token_latencies)),
                "max": float(np.max(all_token_latencies))
            }
        
        # Calculate throughput in tokens per second
        throughput_tps = total_tokens / total_time if total_time > 0 else 0.0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(results) - len(successful),
            "total_time_seconds": total_time,
            "total_tokens": total_tokens,
            "throughput_tps": throughput_tps,  # tokens per second
            "token_latency_ms": token_latency_stats,
            "request_latency_ms": {  # Keep for reference
                "mean": float(np.mean(request_latencies)),
                "median": float(np.median(request_latencies)),
                "p95": float(np.percentile(request_latencies, 95)),
                "p99": float(np.percentile(request_latencies, 99)),
                "min": float(np.min(request_latencies)),
                "max": float(np.max(request_latencies))
            },
            "acceptance_ratio": {
                "mean": float(np.mean(acceptance_ratios)),
                "median": float(np.median(acceptance_ratios)),
                "min": float(np.min(acceptance_ratios)),
                "max": float(np.max(acceptance_ratios))
            }
        }


def load_question_jsonl(path: str) -> List[str]:
    """
    Load prompts from question.jsonl file (specbench format).
    Extracts first turn from each entry.
    
    Args:
        path: Path to question.jsonl file
        
    Returns:
        List of prompts (first turn from each entry)
    """
    import json
    prompts = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Extract first turn
                    if "turns" in data and len(data["turns"]) > 0:
                        prompts.append(data["turns"][0])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {path}: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        raise
    
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


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
