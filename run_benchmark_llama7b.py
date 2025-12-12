#!/usr/bin/env python3
"""
Benchmark script for comparing Baseline vs PD vs PDV engines.
Uses TinyLlama-1.1B as draft model and Llama-2-7B as verifier.

Run this script to generate comparison graphs.
"""

import os
# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Llama7BBenchmark")

def run_benchmark():
    """Run benchmark for all 3 engines with TinyLlama-Llama7B configuration."""
    from src.benchmark.concurrency_benchmark import run_concurrency_benchmark, GPUMonitor
    from src.benchmark.poisson_benchmark import PoissonBenchmark, get_sharegpt_prompts
    from src.utils.config import get_default_config
    from src.engine.baseline_engine import BaselineEngine
    from src.engine.pd_engine import PDEngine
    from src.engine.speculative_engine import PDVLiteEngine
    import torch
    import time
    
    # Configuration for TinyLlama -> Llama-2-7B
    config = get_default_config()
    config.model.draft_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config.model.verifier_model_name = "meta-llama/Llama-2-7b-hf"
    config.model.max_new_tokens = 30  # Shorter for faster benchmarking
    config.model.dtype = "bfloat16"
    
    # Concurrency levels to test
    concurrency_levels = [1, 4, 32, 128]
    
    # Prompts - Load real ShareGPT data from HuggingFace
    print("Loading prompts from ShareGPT dataset...")
    prompts = get_sharegpt_prompts(200, use_real_data=True)  # Get 200 for variety
    print(f"Loaded {len(prompts)} prompts")
    
    # Duration per test (seconds)
    duration_seconds = 20
    
    # Engines to test (order matters - run Baseline first so it doesn't benefit from warm GPU)
    engines = {
        "Baseline": BaselineEngine,
        "PD": PDEngine,
        "PDV": PDVLiteEngine
    }
    
    all_results = []
    
    print("=" * 80)
    print("BENCHMARK: Baseline vs PD vs PDV")
    print("Model: TinyLlama-1.1B (draft) -> Llama-2-7B (verifier)")
    print("=" * 80)
    
    for engine_name, engine_cls in engines.items():
        print(f"\n{'='*40}")
        print(f"Testing Engine: {engine_name}")
        print(f"{'='*40}")
        
        for concurrency in concurrency_levels:
            print(f"\n  Concurrency: {concurrency}")
            
            # Update config
            current_config = copy.deepcopy(config)
            # For Baseline, cap workers at 16 to avoid thread explosion
            if engine_name == "Baseline":
                current_config.scheduler.batch_size = min(concurrency, 16)
            else:
                current_config.scheduler.batch_size = concurrency
            current_config.scheduler.max_queue_size = concurrency * 50
            
            # Initialize Engine
            engine = engine_cls(current_config)
            gpu_monitor = GPUMonitor(0.1)
            
            try:
                engine.start()
                gpu_monitor.start()
                
                # Generate requests with Poisson arrivals
                # Higher arrival rate to better saturate the system
                arrival_rate = max(1.0, concurrency * 0.5)
                
                benchmark = PoissonBenchmark(
                    prompts=prompts,
                    arrival_rate=arrival_rate,
                    duration_seconds=duration_seconds,
                    seed=42
                )
                
                requests = benchmark.generate_requests()
                print(f"    Generated {len(requests)} requests (rate={arrival_rate:.2f}/s)")
                
                # Run benchmark
                stats = benchmark.run_benchmark(
                    engine, 
                    requests=requests,
                    max_concurrent=concurrency 
                )
                
                if "error" in stats:
                    print(f"    ERROR: {stats['error']}")
                    continue
                
                gpu_util = gpu_monitor.get_avg_utilization()
                
                result = {
                    "Engine": engine_name,
                    "Concurrency": concurrency,
                    "Throughput (TPS)": float(f"{stats['throughput_tps']:.2f}"),
                    "Latency P50 (ms)": float(f"{stats['token_latency_ms'].get('median', 0):.2f}"),
                    "Latency P99 (ms)": float(f"{stats['token_latency_ms'].get('p99', 0):.2f}"),
                    "Acceptance Rate": f"{stats['acceptance_ratio'].get('mean', 0):.2%}",
                    "GPU Util %": float(f"{gpu_util:.1f}"),
                    "Avg Request Latency (ms)": float(f"{stats['request_latency_ms'].get('mean', 0):.2f}")
                }
                
                all_results.append(result)
                print(f"    Throughput: {result['Throughput (TPS)']} TPS")
                print(f"    Latency: {result['Avg Request Latency (ms)']} ms")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                gpu_monitor.stop()
                engine.stop()
                
                # Clear GPU memory and wait for cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Small delay between tests to ensure clean state
                time.sleep(2)
    
    # Save results
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    df = pd.DataFrame(all_results)
    output_path = results_dir / "TinyLlama-Llama7B_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return df


def plot_results(df=None):
    """Generate comparison plots."""
    results_dir = Path("benchmark_results")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Load results if not provided
    if df is None:
        csv_path = results_dir / "TinyLlama-Llama7B_summary.csv"
        if not csv_path.exists():
            print(f"No results found at {csv_path}. Run benchmark first.")
            return
        df = pd.read_csv(csv_path)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Colors for 3 engines
    colors = {'Baseline': '#2ca02c', 'PD': '#1f77b4', 'PDV': '#ff7f0e'}
    markers = {'Baseline': '^', 'PD': 'o', 'PDV': 's'}
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Throughput comparison
    ax = axes[0, 0]
    for engine_name in ['Baseline', 'PD', 'PDV']:
        engine_data = df[df['Engine'] == engine_name]
        if not engine_data.empty:
            ax.plot(engine_data['Concurrency'], engine_data['Throughput (TPS)'], 
                    marker=markers[engine_name], linewidth=2, markersize=8, 
                    label=engine_name, color=colors[engine_name])
    
    ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Throughput (TPS)', fontsize=11, fontweight='bold')
    ax.set_title('Throughput vs Concurrency', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # 2. Average Latency comparison
    ax = axes[0, 1]
    for engine_name in ['Baseline', 'PD', 'PDV']:
        engine_data = df[df['Engine'] == engine_name]
        if not engine_data.empty:
            ax.plot(engine_data['Concurrency'], engine_data['Avg Request Latency (ms)'], 
                    marker=markers[engine_name], linewidth=2, markersize=8, 
                    label=engine_name, color=colors[engine_name])
    
    ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Avg Latency (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Average Latency vs Concurrency', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    # 3. GPU Utilization
    ax = axes[1, 0]
    for engine_name in ['Baseline', 'PD', 'PDV']:
        engine_data = df[df['Engine'] == engine_name]
        if not engine_data.empty:
            ax.plot(engine_data['Concurrency'], engine_data['GPU Util %'], 
                    marker=markers[engine_name], linewidth=2, markersize=8, 
                    label=engine_name, color=colors[engine_name])
    
    ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('GPU Utilization (%)', fontsize=11, fontweight='bold')
    ax.set_title('GPU Utilization vs Concurrency', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_ylim([0, 100])
    
    # 4. Acceptance Rate
    ax = axes[1, 1]
    for engine_name in ['Baseline', 'PD', 'PDV']:
        engine_data = df[df['Engine'] == engine_name]
        if not engine_data.empty:
            accept_rate = engine_data['Acceptance Rate'].str.rstrip('%').astype(float)
            ax.plot(engine_data['Concurrency'], accept_rate, 
                    marker=markers[engine_name], linewidth=2, markersize=8, 
                    label=engine_name, color=colors[engine_name])
    
    ax.set_xlabel('Concurrency Level', fontsize=11, fontweight='bold')
    ax.set_ylabel('Acceptance Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Token Acceptance Rate vs Concurrency', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Add title
    fig.suptitle('Baseline vs PD vs PDV Comparison\n(TinyLlama-1.1B → Llama-2-7B)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path = plots_dir / 'baseline_pd_pdv_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Baseline vs PD vs PDV")
    parser.add_argument("--run", action="store_true", help="Run the benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate plots from existing results")
    args = parser.parse_args()
    
    if args.run:
        df = run_benchmark()
        plot_results(df)
    elif args.plot:
        plot_results()
    else:
        print("Usage:")
        print("  python run_benchmark_llama7b.py --run   # Run benchmark and generate plots")
        print("  python run_benchmark_llama7b.py --plot  # Generate plots from existing results")

