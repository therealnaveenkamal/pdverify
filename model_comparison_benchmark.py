#!/usr/bin/env python3
"""
Model Comparison Benchmark
Compares Baseline vs VerifyPD across multiple model pairs with visualization
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import configuration and engines
from src.utils import VerifyPDConfig, ModelConfig, get_performance_config
from src.engine import BaselineEngine, SpeculativeEngine
from src.benchmark.poisson_benchmark import PoissonBenchmark, get_sharegpt_prompts

from dotenv import load_dotenv

load_dotenv()

class ModelComparisonBenchmark:
    """Benchmark comparing Baseline vs VerifyPD across different models."""

    def __init__(self, output_dir: str = "results/models", hf_token: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.hf_token = hf_token

        # Define model pairs
        self.model_pairs = {
            "TinyLlama": {
                "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "verifier": "meta-llama/Llama-2-7b-hf",
                "color": "blue"
            },
            "Qwen": {
                "draft": "Qwen/Qwen2-1.5B",
                "verifier": "Qwen/Qwen2-7B",
                "color": "green"
            },
            # "Llama3": {
            #     "draft": "meta-llama/Llama-3.2-1B",
            #     "verifier": "meta-llama/Meta-Llama-3-8B",
            #     "color": "orange"
            # },
            "Gemma": {
                "draft": "google/gemma-2b",
                "verifier": "google/gemma-7b",
                "color": "red"
            }
        }

        # Define scenarios (medium & high only)
        self.scenarios = [
            {
                "name": "Low Concurrency",
                "num_requests": 5,
                "max_concurrent": 3,
                "arrival_rate": 2.0,
            },
            {
                "name": "Medium Concurrency",
                "num_requests": 20,
                "max_concurrent": 10,
                "arrival_rate": 5.0
            }
        ]

    def _setup_logger(self):
        """Setup logging."""
        logger = logging.getLogger("ModelComparison")
        logger.setLevel(logging.INFO)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        detailed_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        return logger

    def run_all(self):
        """Run benchmark for all model pairs and scenarios."""
        all_results = {}

        for model_name, model_config in self.model_pairs.items():
            self.logger.info("=" * 80)
            self.logger.info(f"TESTING MODEL PAIR: {model_name}")
            self.logger.info(f"Draft: {model_config['draft']}")
            self.logger.info(f"Verifier: {model_config['verifier']}")
            self.logger.info("=" * 80)

            model_results = []

            for scenario in self.scenarios:
                self.logger.info(f"\nRunning scenario: {scenario['name']}")
                scenario_result = self.run_scenario(model_name, model_config, scenario)
                model_results.append(scenario_result)

                # GPU cleanup between scenarios
                torch.cuda.empty_cache()
                time.sleep(2)

            all_results[model_name] = model_results

            # GPU cleanup between model pairs
            torch.cuda.empty_cache()
            time.sleep(5)

        # Save results
        self._save_results(all_results)

        # Generate visualizations
        self._generate_graphs(all_results)

        # Print summary
        self._print_summary(all_results)

        return all_results

    def run_scenario(self, model_name: str, model_config: Dict, scenario: Dict) -> Dict:
        """Run a single scenario for one model pair."""
        prompts = get_sharegpt_prompts(scenario["num_requests"])

        # Run Baseline
        self.logger.info("Running Baseline...")
        baseline_results = self._run_baseline(model_config, prompts, scenario)

        # GPU cleanup
        torch.cuda.empty_cache()
        time.sleep(2)

        # Run VerifyPD (3-lane)
        self.logger.info("Running VerifyPD (3-lane)...")
        verifypd_results = self._run_verifypd(model_config, prompts, scenario)

        return {
            "scenario": scenario,
            "model": model_name,
            "baseline": baseline_results,
            "verifypd": verifypd_results
        }

    def _run_baseline(self, model_config: Dict, prompts: List[str], scenario: Dict) -> Dict:
        """Run baseline benchmark."""
        config = get_performance_config()
        config.model.draft_model_name = model_config["draft"]
        config.model.verifier_model_name = model_config["verifier"]
        config.model.hf_token = self.hf_token

        benchmark = PoissonBenchmark(
            prompts=prompts,
            arrival_rate=scenario["arrival_rate"],
            num_requests=scenario["num_requests"],
            seed=42
        )

        with BaselineEngine(config) as engine:
            results = benchmark.run_benchmark(
                engine,
                max_concurrent=scenario["max_concurrent"]
            )

        return results

    def _run_verifypd(self, model_config: Dict, prompts: List[str], scenario: Dict) -> Dict:
        """Run VerifyPD (3-lane) benchmark."""
        config = get_performance_config()
        config.model.draft_model_name = model_config["draft"]
        config.model.verifier_model_name = model_config["verifier"]
        config.model.hf_token = self.hf_token

        benchmark = PoissonBenchmark(
            prompts=prompts,
            arrival_rate=scenario["arrival_rate"],
            num_requests=scenario["num_requests"],
            seed=42
        )

        with SpeculativeEngine(config) as engine:
            results = benchmark.run_benchmark(
                engine,
                max_concurrent=scenario["max_concurrent"]
            )

        return results

    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        output_file = self.output_dir / "model_comparison_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {output_file}")

    def _generate_graphs(self, results: Dict):
        """Generate visualization graphs."""
        # Create figure with subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Comparison: Baseline vs VerifyPD (3-lane)', fontsize=16, fontweight='bold')

        # Extract data for plotting
        concurrency_levels = ["Low", "Medium"]
        concurrency_values = [3, 10]  # max_concurrent values matching scenarios

        # Plot 1: P95 Latency
        ax1 = axes[0, 0]
        self._plot_metric(ax1, results, concurrency_values, "latency_ms", "p95",
                         "Concurrency vs P95 Latency", "P95 Latency (ms)")

        # Plot 2: Mean Latency
        ax2 = axes[0, 1]
        self._plot_metric(ax2, results, concurrency_values, "latency_ms", "mean",
                         "Concurrency vs Mean Latency", "Mean Latency (ms)")

        # Plot 3: Throughput
        ax3 = axes[1, 0]
        self._plot_throughput(ax3, results, concurrency_values)

        # Plot 4: Acceptance Ratio (VerifyPD only)
        ax4 = axes[1, 1]
        self._plot_acceptance_ratio(ax4, results, concurrency_values)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / "model_comparison_graphs.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        self.logger.info(f"Graphs saved to: {output_file}")

        plt.close()

    def _plot_metric(self, ax, results: Dict, concurrency_values: List[int],
                     metric_category: str, metric_name: str, title: str, ylabel: str):
        """Plot a specific metric for all models."""
        for model_name, model_results in results.items():
            color = self.model_pairs[model_name]["color"]

            # Baseline (dotted)
            baseline_values = [
                r["baseline"][metric_category][metric_name]
                for r in model_results
            ]
            ax.plot(concurrency_values, baseline_values,
                   label=f"{model_name} Baseline",
                   color=color, linestyle='--', marker='o', linewidth=2)

            # VerifyPD (solid)
            verifypd_values = [
                r["verifypd"][metric_category][metric_name]
                for r in model_results
            ]
            ax.plot(concurrency_values, verifypd_values,
                   label=f"{model_name} VerifyPD",
                   color=color, linestyle='-', marker='s', linewidth=2)

        ax.set_xlabel('Max Concurrent Requests', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_throughput(self, ax, results: Dict, concurrency_values: List[int]):
        """Plot throughput comparison."""
        for model_name, model_results in results.items():
            color = self.model_pairs[model_name]["color"]

            # Baseline (dotted)
            baseline_values = [r["baseline"]["throughput_rps"] for r in model_results]
            ax.plot(concurrency_values, baseline_values,
                   label=f"{model_name} Baseline",
                   color=color, linestyle='--', marker='o', linewidth=2)

            # VerifyPD (solid)
            verifypd_values = [r["verifypd"]["throughput_rps"] for r in model_results]
            ax.plot(concurrency_values, verifypd_values,
                   label=f"{model_name} VerifyPD",
                   color=color, linestyle='-', marker='s', linewidth=2)

        ax.set_xlabel('Max Concurrent Requests', fontsize=12)
        ax.set_ylabel('Throughput (req/s)', fontsize=12)
        ax.set_title('Concurrency vs Throughput', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_acceptance_ratio(self, ax, results: Dict, concurrency_values: List[int]):
        """Plot acceptance ratio (VerifyPD only)."""
        for model_name, model_results in results.items():
            color = self.model_pairs[model_name]["color"]

            # VerifyPD acceptance ratio (solid)
            verifypd_values = [
                r["verifypd"]["acceptance_ratio"]["mean"]
                for r in model_results
            ]
            ax.plot(concurrency_values, verifypd_values,
                   label=f"{model_name} VerifyPD",
                   color=color, linestyle='-', marker='s', linewidth=2)

        ax.set_xlabel('Max Concurrent Requests', fontsize=12)
        ax.set_ylabel('Mean Acceptance Ratio', fontsize=12)
        ax.set_title('Concurrency vs Token Acceptance (VerifyPD)', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    def _print_summary(self, results: Dict):
        """Print comprehensive summary table."""
        self.logger.info("\n" + "=" * 120)
        self.logger.info("COMPREHENSIVE MODEL COMPARISON SUMMARY")
        self.logger.info("=" * 120)

        # Create markdown table
        markdown_lines = [
            "# Model Comparison Results: Baseline vs VerifyPD (3-lane)\n",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "\n## Summary Table\n",
            "| Model | Concurrency | System | P95 Latency (ms) | Mean Latency (ms) | Throughput (req/s) | Acceptance Ratio |",
            "|-------|-------------|--------|------------------|-------------------|--------------------|--------------------|"
        ]

        for model_name, model_results in results.items():
            for result in model_results:
                concurrency = result["scenario"]["name"]

                # Baseline row
                baseline = result["baseline"]
                markdown_lines.append(
                    f"| {model_name} | {concurrency} | Baseline | "
                    f"{baseline['latency_ms']['p95']:.1f} | "
                    f"{baseline['latency_ms']['mean']:.1f} | "
                    f"{baseline['throughput_rps']:.2f} | "
                    f"N/A |"
                )

                # VerifyPD row
                verifypd = result["verifypd"]
                improvement = ((baseline['latency_ms']['p95'] - verifypd['latency_ms']['p95'])
                              / baseline['latency_ms']['p95'] * 100)
                markdown_lines.append(
                    f"| {model_name} | {concurrency} | VerifyPD | "
                    f"{verifypd['latency_ms']['p95']:.1f} ({improvement:+.1f}%) | "
                    f"{verifypd['latency_ms']['mean']:.1f} | "
                    f"{verifypd['throughput_rps']:.2f} | "
                    f"{verifypd['acceptance_ratio']['mean']:.3f} |"
                )

        # Save markdown
        markdown_file = self.output_dir / "model_comparison_summary.md"
        with open(markdown_file, 'w') as f:
            f.write('\n'.join(markdown_lines))

        self.logger.info(f"Summary table saved to: {markdown_file}")

        # Print to console
        for line in markdown_lines[3:]:  # Skip header for console
            self.logger.info(line)

        self.logger.info("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default="results/models",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    # Get HuggingFace token from args or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: No HuggingFace token found. Gated models (Llama, Gemma) will fail.")
        print("   Set HF_TOKEN environment variable or use --hf-token argument")

    # Set device
    if args.device == "cpu" or not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Run benchmark
    benchmark = ModelComparisonBenchmark(output_dir=args.output, hf_token=hf_token)
    results = benchmark.run_all()

    print("\n✅ Benchmark completed successfully!")
    print(f"📊 Results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()
