#!/usr/bin/env python3
"""
Comprehensive benchmark runner for PDV analysis.
Runs benchmarks across multiple concurrency levels and model configurations.
"""
import subprocess
import json
import pandas as pd
from pathlib import Path
import sys

# Model configurations for Set A
MODEL_CONFIGS = {
    "SetA-1": {
        "name": "TinyLlama-TinyLlama",
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "verifier": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "SetA-2": {
        "name": "TinyLlama-Llama7B",
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "verifier": "meta-llama/Llama-2-7b-hf"
    },
    "SetA-3": {
        "name": "TinyLlama-CodeLlama34B",
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "verifier": "codellama/CodeLlama-34b-hf"
    }
}

# Concurrency levels to test
CONCURRENCY_LEVELS = [1, 2, 4, 6, 8, 10, 12, 16, 32, 64]

# Benchmark duration (seconds)
DURATION = 30

def update_config(draft_model, verifier_model):
    """Update the config file with new model names."""
    config_path = Path("src/utils/config.py")
    content = config_path.read_text()
    
    # Replace draft model
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if 'draft_model_name: str =' in line:
            new_lines.append(f'    draft_model_name: str = "{draft_model}"')
        elif 'verifier_model_name: str =' in line:
            new_lines.append(f'    verifier_model_name: str = "{verifier_model}"')
        else:
            new_lines.append(line)
    
    config_path.write_text('\n'.join(new_lines))
    print(f"✓ Updated config: draft={draft_model}, verifier={verifier_model}")

def run_benchmark(concurrency_levels, duration):
    """Run benchmark for given concurrency levels."""
    concurrency_str = ','.join(map(str, concurrency_levels))
    cmd = [
        sys.executable, "-m", "src.benchmark.concurrency_benchmark",
        "--duration", str(duration),
        "--concurrency", concurrency_str
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Benchmark failed: {result.stderr}")
        return False
    
    print(f"✓ Benchmark completed")
    return True

def main():
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK ANALYSIS - SET A")
    print("=" * 80)
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Duration per test: {DURATION}s")
    print("=" * 80)
    
    for config_id, config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"{config_id}: {config['name']}")
        print(f"{'=' * 80}")
        
        # Update configuration
        update_config(config['draft'], config['verifier'])
        
        # Run benchmark
        success = run_benchmark(CONCURRENCY_LEVELS, DURATION)
        
        if success:
            # Save results with config name
            summary_file = Path("concurrency_benchmark_summary.csv")
            detailed_file = Path("concurrency_benchmark_detailed.csv")
            
            if summary_file.exists():
                new_summary = results_dir / f"{config['name']}_summary.csv"
                summary_file.rename(new_summary)
                print(f"✓ Saved: {new_summary}")
            
            if detailed_file.exists():
                new_detailed = results_dir / f"{config['name']}_detailed.csv"
                detailed_file.rename(new_detailed)
                print(f"✓ Saved: {new_detailed}")
        else:
            print(f"✗ Failed to complete {config_id}")
            continue
    
    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETED")
    print("=" * 80)
    print(f"Results saved in: {results_dir.absolute()}")

if __name__ == "__main__":
    main()

