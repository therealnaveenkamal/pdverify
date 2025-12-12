#!/usr/bin/env python3
"""
Run high concurrency benchmarks and merge with existing results.
"""
import subprocess
import pandas as pd
from pathlib import Path
import sys

# High concurrency levels to test
HIGH_CONCURRENCY = [72, 84, 96, 128]

# Model configurations
MODEL_CONFIGS = {
    "TinyLlama-TinyLlama": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "verifier": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    },
    "TinyLlama-Llama7B": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "verifier": "meta-llama/Llama-2-7b-hf"
    },
    "TinyLlama-CodeLlama34B": {
        "draft": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "verifier": "codellama/CodeLlama-34b-hf"
    }
}

DURATION = 30

def update_config(draft_model, verifier_model):
    """Update the config file with new model names."""
    config_path = Path("src/utils/config.py")
    content = config_path.read_text()
    
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
    print(f"✓ Updated config: {draft_model} -> {verifier_model}")

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

def merge_results(config_name):
    """Merge new results with existing results."""
    results_dir = Path("benchmark_results")
    
    # Load existing results
    existing_summary = results_dir / f"{config_name}_summary.csv"
    existing_detailed = results_dir / f"{config_name}_detailed.csv"
    
    # Load new results
    new_summary = Path("concurrency_benchmark_summary.csv")
    new_detailed = Path("concurrency_benchmark_detailed.csv")
    
    if not new_summary.exists():
        print(f"✗ No new results found for {config_name}")
        return False
    
    # Merge summary
    if existing_summary.exists():
        old_df = pd.read_csv(existing_summary)
        new_df = pd.read_csv(new_summary)
        
        # Remove duplicates (keep new data)
        merged = pd.concat([old_df, new_df]).drop_duplicates(
            subset=['Engine', 'Concurrency'], keep='last'
        ).sort_values(['Engine', 'Concurrency'])
        
        merged.to_csv(existing_summary, index=False)
        print(f"✓ Merged summary: {len(old_df)} -> {len(merged)} rows")
    else:
        new_summary.rename(existing_summary)
        print(f"✓ Created summary")
    
    # Merge detailed
    if existing_detailed.exists():
        old_df = pd.read_csv(existing_detailed)
        new_df = pd.read_csv(new_detailed)
        merged = pd.concat([old_df, new_df]).drop_duplicates(
            subset=['Engine', 'Concurrency', 'RequestID'], keep='last'
        ).sort_values(['Engine', 'Concurrency', 'RequestID'])
        merged.to_csv(existing_detailed, index=False)
        print(f"✓ Merged detailed")
    else:
        new_detailed.rename(existing_detailed)
        print(f"✓ Created detailed")
    
    # Clean up temp files
    if new_summary.exists():
        new_summary.unlink()
    if new_detailed.exists():
        new_detailed.unlink()
    
    return True

def main():
    print("=" * 80)
    print("HIGH CONCURRENCY BENCHMARK EXTENSION")
    print("=" * 80)
    print(f"Concurrency levels: {HIGH_CONCURRENCY}")
    print(f"Duration per test: {DURATION}s")
    print("=" * 80)
    
    for config_name, config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"{config_name}")
        print(f"{'=' * 80}")
        
        # Update configuration
        update_config(config['draft'], config['verifier'])
        
        # Run benchmark
        success = run_benchmark(HIGH_CONCURRENCY, DURATION)
        
        if success:
            # Merge with existing results
            merge_results(config_name)
        else:
            print(f"✗ Failed to complete {config_name}")
            continue
    
    print("\n" + "=" * 80)
    print("HIGH CONCURRENCY BENCHMARKS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()

