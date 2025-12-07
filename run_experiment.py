#!/usr/bin/env python3
"""
Experiment runner comparing baseline vs Verify-PD.
"""

import argparse
import logging
import json
from pathlib import Path

import torch

from src.engine import SpeculativeEngine
from src.benchmark import BaselineSpeculativeDecoding, PoissonBenchmark, get_sharegpt_prompts
from src.utils import get_cpu_config, get_default_config, get_test_config, get_fast_config, get_performance_config
from dotenv import load_dotenv

load_dotenv()

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Verify-PD experiments")
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use (auto detects CUDA if available)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        help="Dataset to use (sample or sharegpt)"
    )
    
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to benchmark"
    )
    
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=1.0,
        help="Average requests per second (Poisson lambda)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use small models for testing"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast iteration mode: fewer tokens and smaller batches"
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Performance demonstration mode: fast draft + accurate verifier"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate per request (overrides config)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent requests in benchmark"
    )
    
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline comparison"
    )
    
    return parser.parse_args()


def run_baseline_benchmark(config, prompts, num_requests):
    """Run baseline speculative decoding benchmark."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Running BASELINE Speculative Decoding")
    logger.info("=" * 80)
    
    baseline = BaselineSpeculativeDecoding(
        model_config=config.model,
        device=config.hardware.device
    )
    baseline.load_models()
    
    results = []
    for i, prompt in enumerate(prompts[:num_requests]):
        logger.info(f"Processing request {i+1}/{num_requests}")
        # Reset per-request stats
        baseline.total_tokens_generated = 0
        baseline.total_tokens_accepted = 0
        
        text, latency_ms = baseline.generate(prompt, max_tokens=config.model.max_new_tokens)
        
        # Calculate per-request acceptance rate
        per_request_acceptance = baseline.get_acceptance_rate()
        
        results.append({
            "request_id": f"baseline_{i}",
            "latency_ms": latency_ms,
            "acceptance_rate": per_request_acceptance,
            "tokens_generated": baseline.total_tokens_generated,
            "tokens_accepted": baseline.total_tokens_accepted
        })
    
    # Calculate statistics
    import numpy as np
    latencies = [r["latency_ms"] for r in results]
    acceptance_rates = [r["acceptance_rate"] for r in results]
    
    # Calculate overall acceptance rate from all tokens
    total_generated = sum(r["tokens_generated"] for r in results)
    total_accepted = sum(r["tokens_accepted"] for r in results)
    overall_acceptance = total_accepted / total_generated if total_generated > 0 else 0.0
    
    stats = {
        "system": "baseline",
        "num_requests": len(results),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        },
        "acceptance_rate": overall_acceptance,
        "acceptance_rate_per_request": {
            "mean": float(np.mean(acceptance_rates)),
            "median": float(np.median(acceptance_rates)),
            "min": float(np.min(acceptance_rates)),
            "max": float(np.max(acceptance_rates))
        }
    }
    
    logger.info(f"Baseline p95 latency: {stats['latency_ms']['p95']:.1f}ms")
    logger.info(f"Baseline p99 latency: {stats['latency_ms']['p99']:.1f}ms")
    logger.info(f"Baseline acceptance rate: {stats['acceptance_rate']:.2%}")
    logger.info(f"Baseline per-request acceptance (mean): {stats['acceptance_rate_per_request']['mean']:.2%}")
    
    return stats


def run_verifypd_benchmark(config, prompts, num_requests, arrival_rate, max_concurrent=5):
    """Run Verify-PD benchmark."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Running VERIFY-PD Speculative Decoding")
    logger.info("=" * 80)

    # Create Poisson benchmark
    benchmark = PoissonBenchmark(
        prompts=prompts,
        arrival_rate=arrival_rate,
        num_requests=num_requests,
        seed=42
    )

    # Run benchmark with no_delay=True for fair comparison with baseline
    with SpeculativeEngine(config) as engine:
        results = benchmark.run_benchmark(engine, max_concurrent=max_concurrent, no_delay=True)

    # Check for errors
    if "error" in results:
        logger.error(f"Verify-PD benchmark failed: {results['error']}")
        # Return minimal results structure
        return {
            "system": "verify-pd",
            "error": results["error"],
            "latency_ms": {"p95": 0.0, "p99": 0.0},
            "acceptance_ratio": {"mean": 0.0}
        }

    logger.info(f"Verify-PD p95 latency: {results['latency_ms']['p95']:.1f}ms")
    logger.info(f"Verify-PD p99 latency: {results['latency_ms']['p99']:.1f}ms")
    logger.info(f"Verify-PD acceptance rate: {results['acceptance_ratio']['mean']:.2%}")

    results["system"] = "verify-pd"
    return results


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Verify-PD Experiment Runner")
    logger.info("=" * 80)
    
    # Auto-detect device if requested
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {args.device}")
    
    # Get configuration
    if args.test_mode:
        config = get_test_config()
        logger.info("Using TEST configuration")
    elif args.performance:
        config = get_performance_config()
        logger.info("Using PERFORMANCE demonstration configuration")
    elif args.fast:
        config = get_fast_config()
        logger.info("Using FAST iteration configuration")
    elif args.device == "cpu":
        config = get_cpu_config()
        logger.info("Using CPU configuration")
    else:
        config = get_default_config()
        logger.info("Using GPU configuration")
    
    config.hardware.device = args.device
    
    # Override max_new_tokens if specified
    if args.max_tokens is not None:
        config.model.max_new_tokens = args.max_tokens
        logger.info(f"Using {args.max_tokens} max tokens per request")
    
    # Get prompts
    logger.info(f"Loading {args.num_requests} prompts...")
    prompts = get_sharegpt_prompts(args.num_requests)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    
    if not args.skip_baseline:
        baseline_results = run_baseline_benchmark(config, prompts, args.num_requests)
        all_results["baseline"] = baseline_results
    
    verifypd_results = run_verifypd_benchmark(config, prompts, args.num_requests, args.arrival_rate, max_concurrent=args.max_concurrent)
    all_results["verify-pd"] = verifypd_results
    
    # Save results
    output_file = output_dir / "experiment_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Print comparison
    if not args.skip_baseline:
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON")
        logger.info("=" * 80)
        
        baseline_p95 = baseline_results["latency_ms"]["p95"]
        verifypd_p95 = verifypd_results["latency_ms"]["p95"]
        improvement = ((baseline_p95 - verifypd_p95) / baseline_p95) * 100
        
        logger.info(f"Baseline p95: {baseline_p95:.1f}ms")
        logger.info(f"Verify-PD p95: {verifypd_p95:.1f}ms")
        logger.info(f"Improvement: {improvement:+.1f}%")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
