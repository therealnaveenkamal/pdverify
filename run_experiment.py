#!/usr/bin/env python3
"""
Experiment runner comparing baseline vs Verify-PD.
"""

import argparse
import logging
import json
from pathlib import Path

from src.engine import SpeculativeEngine
from src.benchmark import BaselineSpeculativeDecoding, PoissonBenchmark, get_sharegpt_prompts
from src.utils import get_cpu_config, get_default_config, get_test_config


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
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
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
        text, latency_ms = baseline.generate(prompt, max_tokens=100)
        results.append({
            "request_id": f"baseline_{i}",
            "latency_ms": latency_ms,
            "acceptance_rate": baseline.get_acceptance_rate()
        })
    
    # Calculate statistics
    import numpy as np
    latencies = [r["latency_ms"] for r in results]
    
    stats = {
        "system": "baseline",
        "num_requests": len(results),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        },
        "acceptance_rate": baseline.get_acceptance_rate()
    }
    
    logger.info(f"Baseline p95 latency: {stats['latency_ms']['p95']:.1f}ms")
    logger.info(f"Baseline p99 latency: {stats['latency_ms']['p99']:.1f}ms")
    logger.info(f"Baseline acceptance rate: {stats['acceptance_rate']:.2%}")
    
    return stats


def run_verifypd_benchmark(config, prompts, num_requests, arrival_rate):
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
    
    # Run benchmark
    with SpeculativeEngine(config) as engine:
        results = benchmark.run_benchmark(engine)
    
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
    
    # Get configuration
    if args.test_mode:
        config = get_test_config()
        logger.info("Using TEST configuration")
    elif args.device == "cpu":
        config = get_cpu_config()
        logger.info("Using CPU configuration")
    else:
        config = get_default_config()
        logger.info("Using GPU configuration")
    
    config.hardware.device = args.device
    
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
    
    verifypd_results = run_verifypd_benchmark(config, prompts, args.num_requests, args.arrival_rate)
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
