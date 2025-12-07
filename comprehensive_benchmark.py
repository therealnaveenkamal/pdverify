#!/usr/bin/env python3
"""
Comprehensive benchmark comparing Baseline and Verify-PD under identical conditions.
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

import torch

from src.engine import SpeculativeEngine, BaselineEngine
from src.benchmark import PoissonBenchmark, get_sharegpt_prompts
from src.scheduler import Request, LaneType
from src.utils import get_performance_config


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class ComprehensiveBenchmark:
    """Comprehensive benchmark runner for fair comparison."""
    
    def __init__(self, config, output_dir: str = "results/comprehensive/"):
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
        self.logger.info("COMPREHENSIVE BENCHMARK - Fair Comparison")
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
        self.logger.info("COMPREHENSIVE BENCHMARK COMPLETE")
        self.logger.info("="*80)
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """Run a single test scenario."""
        prompts = get_sharegpt_prompts(scenario["num_requests"])
        
        # Run baseline
        self.logger.info("\n--- Running BASELINE ---")
        baseline_results = self._run_baseline(
            prompts, 
            scenario["num_requests"],
            scenario["arrival_rate"],
            scenario["max_concurrent"]
        )
        
        # Clear GPU cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2)
        
        # Run Verify-PD
        self.logger.info("\n--- Running VERIFY-PD ---")
        verifypd_results = self._run_verifypd(
            prompts,
            scenario["num_requests"],
            scenario["arrival_rate"],
            scenario["max_concurrent"]
        )
        
        return {
            "scenario": scenario,
            "baseline": baseline_results,
            "verify_pd": verifypd_results
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
                           f"throughput: {results['throughput_rps']:.2f} req/s, "
                           f"acceptance: {results['acceptance_ratio']['mean']:.2%}")
        
        return results
    
    def _run_verifypd(self, prompts: List[str], num_requests: int, arrival_rate: float, max_concurrent: int) -> Dict:
        """Run Verify-PD benchmark."""
        benchmark = PoissonBenchmark(
            prompts=prompts,
            arrival_rate=arrival_rate,
            num_requests=num_requests,
            seed=42
        )
        
        with SpeculativeEngine(self.config) as engine:
            results = benchmark.run_benchmark(engine, max_concurrent=max_concurrent)
        
        if "error" not in results:
            self.logger.info(f"Verify-PD - p95: {results['latency_ms']['p95']:.1f}ms, "
                           f"throughput: {results['throughput_rps']:.2f} req/s, "
                           f"acceptance: {results['acceptance_ratio']['mean']:.2%}")
        
        return results
    
    def _print_scenario_summary(self, result: Dict):
        """Print summary for a scenario."""
        scenario = result["scenario"]
        baseline = result["baseline"]
        verifypd = result["verify_pd"]
        
        self.logger.info("\n" + "-"*80)
        self.logger.info(f"SUMMARY: {scenario['name']}")
        self.logger.info("-"*80)
        
        if "error" in baseline or "error" in verifypd:
            self.logger.error("Error in one or both systems")
            return
        
        # Latency comparison
        baseline_p95 = baseline["latency_ms"]["p95"]
        verifypd_p95 = verifypd["latency_ms"]["p95"]
        latency_improvement = ((baseline_p95 - verifypd_p95) / baseline_p95) * 100
        
        # Throughput comparison
        baseline_tput = baseline["throughput_rps"]
        verifypd_tput = verifypd["throughput_rps"]
        tput_improvement = ((verifypd_tput - baseline_tput) / baseline_tput) * 100
        
        self.logger.info(f"Latency (p95):")
        self.logger.info(f"  Baseline:  {baseline_p95:>10.1f}ms")
        self.logger.info(f"  Verify-PD: {verifypd_p95:>10.1f}ms")
        self.logger.info(f"  Improvement: {latency_improvement:>+8.1f}%")
        
        self.logger.info(f"\nThroughput:")
        self.logger.info(f"  Baseline:  {baseline_tput:>10.2f} req/s")
        self.logger.info(f"  Verify-PD: {verifypd_tput:>10.2f} req/s")
        self.logger.info(f"  Improvement: {tput_improvement:>+8.1f}%")
        
        self.logger.info(f"\nAcceptance Rate:")
        self.logger.info(f"  Baseline:  {baseline['acceptance_ratio']['mean']:>10.1%}")
        self.logger.info(f"  Verify-PD: {verifypd['acceptance_ratio']['mean']:>10.1%}")
        self.logger.info("-"*80)
    
    def _generate_summary(self):
        """Generate overall summary."""
        summary = {
            "total_scenarios": len(self.results["test_scenarios"]),
            "verifypd_wins": 0,
            "baseline_wins": 0,
            "scenarios": []
        }
        
        for result in self.results["test_scenarios"]:
            baseline = result["baseline"]
            verifypd = result["verify_pd"]
            
            if "error" in baseline or "error" in verifypd:
                continue
            
            # Compare on p95 latency
            if verifypd["latency_ms"]["p95"] < baseline["latency_ms"]["p95"]:
                summary["verifypd_wins"] += 1
                winner = "Verify-PD"
            else:
                summary["baseline_wins"] += 1
                winner = "Baseline"
            
            summary["scenarios"].append({
                "name": result["scenario"]["name"],
                "winner": winner,
                "baseline_p95": baseline["latency_ms"]["p95"],
                "verifypd_p95": verifypd["latency_ms"]["p95"]
            })
        
        self.results["summary"] = summary
    
    def _save_results(self):
        """Save results to file."""
        output_file = self.output_dir / "comprehensive_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"\nResults saved to {output_file}")
        
        # Generate markdown report
        self._generate_markdown_report()
    
    def _generate_markdown_report(self):
        """Generate markdown report."""
        report_file = self.output_dir / "comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Comparison: Baseline vs Verify-PD\n\n")
            f.write("Fair apples-to-apples comparison under identical conditions.\n\n")
            
            f.write("## Test Configuration\n\n")
            f.write(f"- Models: {self.config.model.draft_model_name} (draft) + {self.config.model.verifier_model_name} (verifier)\n")
            f.write(f"- Device: {self.config.hardware.device}\n")
            f.write(f"- Max tokens per request: {self.config.model.max_new_tokens}\n\n")
            
            f.write("## Summary\n\n")
            summary = self.results["summary"]
            f.write(f"- Total scenarios tested: {summary['total_scenarios']}\n")
            f.write(f"- Verify-PD wins: {summary['verifypd_wins']}\n")
            f.write(f"- Baseline wins: {summary['baseline_wins']}\n\n")
            
            f.write("## Detailed Results\n\n")
            
            for result in self.results["test_scenarios"]:
                scenario = result["scenario"]
                baseline = result["baseline"]
                verifypd = result["verify_pd"]
                
                f.write(f"### {scenario['name']}\n\n")
                f.write(f"**Description:** {scenario['description']}\n\n")
                f.write(f"**Configuration:**\n")
                f.write(f"- Requests: {scenario['num_requests']}\n")
                f.write(f"- Max concurrent: {scenario['max_concurrent']}\n")
                f.write(f"- Arrival rate: {scenario['arrival_rate']} req/s\n\n")
                
                if "error" in baseline or "error" in verifypd:
                    f.write("**Error during execution**\n\n")
                    continue
                
                # Create comparison table
                f.write("| Metric | Baseline | Verify-PD | Improvement |\n")
                f.write("|--------|----------|-----------|-------------|\n")
                
                # Latency metrics
                for metric in ["p50", "p95", "p99"]:
                    b_val = baseline["latency_ms"][metric]
                    v_val = verifypd["latency_ms"][metric]
                    imp = ((b_val - v_val) / b_val) * 100
                    f.write(f"| Latency {metric.upper()} (ms) | {b_val:.1f} | {v_val:.1f} | {imp:+.1f}% |\n")
                
                # Throughput
                b_tput = baseline["throughput_rps"]
                v_tput = verifypd["throughput_rps"]
                tput_imp = ((v_tput - b_tput) / b_tput) * 100
                f.write(f"| Throughput (req/s) | {b_tput:.2f} | {v_tput:.2f} | {tput_imp:+.1f}% |\n")
                
                # Acceptance rate
                b_acc = baseline["acceptance_ratio"]["mean"]
                v_acc = verifypd["acceptance_ratio"]["mean"]
                f.write(f"| Acceptance Rate | {b_acc:.1%} | {v_acc:.1%} | - |\n")
                
                f.write("\n")
        
        self.logger.info(f"Report saved to {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive fair comparison")
    parser.add_argument("--output", type=str, default="results/comprehensive/", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer scenarios")
    args = parser.parse_args()
    
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Comprehensive Benchmark")
    
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
    benchmark = ComprehensiveBenchmark(config, output_dir=args.output)
    benchmark.run_all_tests()


if __name__ == "__main__":
    main()
