#!/usr/bin/env python3
"""
Three-way comprehensive benchmark: Baseline vs PD vs PD-Verify.
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch

from src.engine import SpeculativeEngine, BaselineEngine, PDEngine
from src.benchmark import PoissonBenchmark, get_sharegpt_prompts
from src.scheduler import Request, LaneType
from src.utils import get_performance_config


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class ThreeWayBenchmark:
    """Three-way benchmark runner for comprehensive comparison."""
    
    def __init__(self, config, output_dir: str = "results/three_way/"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.results = {
            "test_scenarios": [],
            "summary": {}
        }
    
    def run_all_tests(self):
        """Run all test scenarios."""
        self.logger.info("="*80)
        self.logger.info("THREE-WAY COMPREHENSIVE BENCHMARK")
        self.logger.info("Baseline vs PD (2-lane) vs PD-Verify (3-lane)")
        self.logger.info("="*80)
        
        # Test scenarios
        scenarios = [
            {
                "name": "Single Request",
                "num_requests": 1,
                "max_concurrent": 1,
                "arrival_rate": 1.0,
                "description": "Measure per-request latency without concurrency"
            },
            {
                "name": "Low Concurrency",
                "num_requests": 5,
                "max_concurrent": 3,
                "arrival_rate": 2.0,
                "description": "Test basic concurrent handling (2-5 requests)"
            },
            {
                "name": "Medium Concurrency",
                "num_requests": 20,
                "max_concurrent": 10,
                "arrival_rate": 5.0,
                "description": "Test performance under typical load"
            },
            {
                "name": "High Concurrency",
                "num_requests": 50,
                "max_concurrent": 25,
                "arrival_rate": 10.0,
                "description": "Stress test and scalability"
            }
        ]
        
        for scenario in scenarios:
            self.logger.info("\n" + "="*80)
            self.logger.info(f"SCENARIO: {scenario['name']}")
            self.logger.info(f"Description: {scenario['description']}")
            self.logger.info("="*80)
            
            result = self.run_scenario(scenario)
            self.results["test_scenarios"].append(result)
            
            # Print summary
            self._print_scenario_summary(result)
        
        # Generate final summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        self.logger.info("\n" + "="*80)
        self.logger.info("THREE-WAY BENCHMARK COMPLETE")
        self.logger.info("="*80)
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """Run a single test scenario on all 3 systems."""
        prompts = get_sharegpt_prompts(scenario["num_requests"])
        
        # Run baseline
        self.logger.info("\n--- Running BASELINE ---")
        baseline_results = self._run_baseline(
            prompts, 
            scenario["num_requests"],
            scenario["arrival_rate"],
            scenario["max_concurrent"]
        )
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
        
        # Run PD (2-lane)
        self.logger.info("\n--- Running PD (2-lane) ---")
        pd_results = self._run_pd(
            prompts,
            scenario["num_requests"],
            scenario["arrival_rate"],
            scenario["max_concurrent"]
        )
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
        
        # Run PD-Verify (3-lane)
        self.logger.info("\n--- Running PD-VERIFY (3-lane) ---")
        pdverify_results = self._run_pdverify(
            prompts,
            scenario["num_requests"],
            scenario["arrival_rate"],
            scenario["max_concurrent"]
        )
        
        return {
            "scenario": scenario,
            "baseline": baseline_results,
            "pd": pd_results,
            "pdverify": pdverify_results
        }
    
    def _run_baseline(self, prompts: List[str], num_requests: int, arrival_rate: float, max_concurrent: int) -> Dict:
        """Run baseline benchmark."""
        benchmark = PoissonBenchmark(
            prompts=prompts,
            arrival_rate=arrival_rate,
            num_requests=num_requests,
            seed=42
        )
        
        with BaselineEngine(self.config) as engine:
            results = benchmark.run_benchmark(engine, max_concurrent=max_concurrent)
        
        if "error" not in results:
            self.logger.info(f"Baseline - p95: {results['latency_ms']['p95']:.1f}ms, "
                           f"throughput: {results['throughput_rps']:.2f} req/s")
        
        return results
    
    def _run_pd(self, prompts: List[str], num_requests: int, arrival_rate: float, max_concurrent: int) -> Dict:
        """Run PD (2-lane) benchmark."""
        benchmark = PoissonBenchmark(
            prompts=prompts,
            arrival_rate=arrival_rate,
            num_requests=num_requests,
            seed=42
        )
        
        with PDEngine(self.config) as engine:
            results = benchmark.run_benchmark(engine, max_concurrent=max_concurrent)
        
        if "error" not in results:
            self.logger.info(f"PD (2-lane) - p95: {results['latency_ms']['p95']:.1f}ms, "
                           f"throughput: {results['throughput_rps']:.2f} req/s")
        
        return results
    
    def _run_pdverify(self, prompts: List[str], num_requests: int, arrival_rate: float, max_concurrent: int) -> Dict:
        """Run PD-Verify (3-lane) benchmark."""
        benchmark = PoissonBenchmark(
            prompts=prompts,
            arrival_rate=arrival_rate,
            num_requests=num_requests,
            seed=42
        )
        
        with SpeculativeEngine(self.config) as engine:
            results = benchmark.run_benchmark(engine, max_concurrent=max_concurrent)
        
        if "error" not in results:
            self.logger.info(f"PD-Verify (3-lane) - p95: {results['latency_ms']['p95']:.1f}ms, "
                           f"throughput: {results['throughput_rps']:.2f} req/s")
        
        return results
    
    def _print_scenario_summary(self, result: Dict):
        """Print summary for a scenario."""
        scenario = result["scenario"]
        baseline = result["baseline"]
        pd = result["pd"]
        pdverify = result["pdverify"]
        
        self.logger.info("\n" + "-"*80)
        self.logger.info(f"SUMMARY: {scenario['name']}")
        self.logger.info("-"*80)
        
        if "error" in baseline or "error" in pd or "error" in pdverify:
            self.logger.error("Error in one or more systems")
            return
        
        # P95 latencies
        baseline_p95 = baseline["latency_ms"]["p95"]
        pd_p95 = pd["latency_ms"]["p95"]
        pdverify_p95 = pdverify["latency_ms"]["p95"]
        
        # Throughputs
        baseline_tput = baseline["throughput_rps"]
        pd_tput = pd["throughput_rps"]
        pdverify_tput = pdverify["throughput_rps"]
        
        self.logger.info(f"Latency (p95):")
        self.logger.info(f"  Baseline:         {baseline_p95:>10.1f}ms")
        self.logger.info(f"  PD (2-lane):      {pd_p95:>10.1f}ms ({((baseline_p95-pd_p95)/baseline_p95*100):+.1f}% vs baseline)")
        self.logger.info(f"  PD-Verify (3-ln): {pdverify_p95:>10.1f}ms ({((baseline_p95-pdverify_p95)/baseline_p95*100):+.1f}% vs baseline)")
        
        self.logger.info(f"\nThroughput:")
        self.logger.info(f"  Baseline:         {baseline_tput:>10.2f} req/s")
        self.logger.info(f"  PD (2-lane):      {pd_tput:>10.2f} req/s ({((pd_tput-baseline_tput)/baseline_tput*100):+.1f}% vs baseline)")
        self.logger.info(f"  PD-Verify (3-ln): {pdverify_tput:>10.2f} req/s ({((pdverify_tput-baseline_tput)/baseline_tput*100):+.1f}% vs baseline)")
        self.logger.info("-"*80)
    
    def _generate_summary(self):
        """Generate overall summary."""
        summary = {
            "total_scenarios": len(self.results["test_scenarios"]),
            "pd_vs_baseline": {"wins": 0, "losses": 0},
            "pdverify_vs_baseline": {"wins": 0, "losses": 0},
            "pdverify_vs_pd": {"wins": 0, "losses": 0},
            "scenarios": []
        }
        
        for result in self.results["test_scenarios"]:
            baseline = result["baseline"]
            pd = result["pd"]
            pdverify = result["pdverify"]
            
            if "error" in baseline or "error" in pd or "error" in pdverify:
                continue
            
            baseline_p95 = baseline["latency_ms"]["p95"]
            pd_p95 = pd["latency_ms"]["p95"]
            pdverify_p95 = pdverify["latency_ms"]["p95"]
            
            # Determine winners
            if pd_p95 < baseline_p95:
                summary["pd_vs_baseline"]["wins"] += 1
            else:
                summary["pd_vs_baseline"]["losses"] += 1
            
            if pdverify_p95 < baseline_p95:
                summary["pdverify_vs_baseline"]["wins"] += 1
            else:
                summary["pdverify_vs_baseline"]["losses"] += 1
            
            if pdverify_p95 < pd_p95:
                summary["pdverify_vs_pd"]["wins"] += 1
            else:
                summary["pdverify_vs_pd"]["losses"] += 1
            
            # Find overall winner
            if pdverify_p95 < pd_p95 and pdverify_p95 < baseline_p95:
                winner = "PD-Verify (3-lane)"
            elif pd_p95 < baseline_p95 and pd_p95 < pdverify_p95:
                winner = "PD (2-lane)"
            else:
                winner = "Baseline"
            
            summary["scenarios"].append({
                "name": result["scenario"]["name"],
                "winner": winner,
                "baseline_p95": baseline_p95,
                "pd_p95": pd_p95,
                "pdverify_p95": pdverify_p95
            })
        
        self.results["summary"] = summary
    
    def _save_results(self):
        """Save results to file."""
        output_file = self.output_dir / "three_way_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"\nResults saved to {output_file}")
        
        # Generate markdown report
        self._generate_markdown_report()
    
    def _generate_markdown_report(self):
        """Generate markdown report."""
        report_file = self.output_dir / "three_way_comparison.md"
        
        with open(report_file, 'w') as f:
            f.write("# Three-Way Comparison: Baseline vs PD vs PD-Verify\n\n")
            f.write("**Fair apples-to-apples comparison of all three systems.**\n\n")
            
            f.write("## Systems Compared\n\n")
            f.write("1. **Baseline** - Standard speculative decoding with concurrent workers\n")
            f.write("2. **PD (2-lane)** - Prefill-Decode disaggregation\n")
            f.write("3. **PD-Verify (3-lane)** - Full disaggregation with separate Verify lane\n\n")
            
            f.write("## Test Configuration\n\n")
            f.write(f"- Models: {self.config.model.draft_model_name} (draft) + {self.config.model.verifier_model_name} (verifier)\n")
            f.write(f"- Device: {self.config.hardware.device}\n")
            f.write(f"- Max tokens per request: {self.config.model.max_new_tokens}\n\n")
            
            f.write("## Overall Summary\n\n")
            summary = self.results["summary"]
            f.write(f"- Total scenarios tested: {summary['total_scenarios']}\n\n")
            
            f.write("**Head-to-Head Results:**\n\n")
            f.write(f"- PD (2-lane) vs Baseline: {summary['pd_vs_baseline']['wins']} wins, {summary['pd_vs_baseline']['losses']} losses\n")
            f.write(f"- PD-Verify (3-lane) vs Baseline: {summary['pdverify_vs_baseline']['wins']} wins, {summary['pdverify_vs_baseline']['losses']} losses\n")
            f.write(f"- PD-Verify (3-lane) vs PD (2-lane): {summary['pdverify_vs_pd']['wins']} wins, {summary['pdverify_vs_pd']['losses']} losses\n\n")
            
            f.write("## Detailed Results\n\n")
            
            for result in self.results["test_scenarios"]:
                scenario = result["scenario"]
                baseline = result["baseline"]
                pd = result["pd"]
                pdverify = result["pdverify"]
                
                f.write(f"### {scenario['name']}\n\n")
                f.write(f"**Description:** {scenario['description']}\n\n")
                f.write(f"**Configuration:**\n")
                f.write(f"- Requests: {scenario['num_requests']}\n")
                f.write(f"- Max concurrent: {scenario['max_concurrent']}\n")
                f.write(f"- Arrival rate: {scenario['arrival_rate']} req/s\n\n")
                
                if "error" in baseline or "error" in pd or "error" in pdverify:
                    f.write("**Error during execution**\n\n")
                    continue
                
                # Create comparison table
                f.write("| Metric | Baseline | PD (2-lane) | PD-Verify (3-lane) |\n")
                f.write("|--------|----------|-------------|--------------------|\n")
                
                # Latencies
                for metric in ["p95", "p99"]:
                    b_val = baseline["latency_ms"][metric]
                    p_val = pd["latency_ms"][metric]
                    v_val = pdverify["latency_ms"][metric]
                    f.write(f"| Latency {metric.upper()} (ms) | {b_val:.1f} | {p_val:.1f} | {v_val:.1f} |\n")
                
                # Throughput
                b_tput = baseline["throughput_rps"]
                p_tput = pd["throughput_rps"]
                v_tput = pdverify["throughput_rps"]
                f.write(f"| Throughput (req/s) | {b_tput:.2f} | {p_tput:.2f} | {v_tput:.2f} |\n")
                
                # Acceptance rate
                b_acc = baseline["acceptance_ratio"]["mean"]
                p_acc = pd["acceptance_ratio"]["mean"]
                v_acc = pdverify["acceptance_ratio"]["mean"]
                f.write(f"| Acceptance Rate | {b_acc:.1%} | {p_acc:.1%} | {v_acc:.1%} |\n")
                
                f.write("\n")
        
        self.logger.info(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run three-way fair comparison")
    parser.add_argument("--output", type=str, default="results/three_way/", help="Output directory")
    args = parser.parse_args()
    
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Three-Way Comprehensive Benchmark")
    
    # Get configuration
    config = get_performance_config()
    
    # Auto-detect device
    if torch.cuda.is_available():
        config.hardware.device = "cuda"
        logger.info(f"Using CUDA device")
    else:
        config.hardware.device = "cpu"
        logger.info(f"Using CPU device")
    
    # Run benchmark
    benchmark = ThreeWayBenchmark(config, output_dir=args.output)
    benchmark.run_all_tests()


if __name__ == "__main__":
    main()
