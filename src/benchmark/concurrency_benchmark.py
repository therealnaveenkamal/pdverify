
import logging
import argparse
import time
import threading
import subprocess
import copy
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate

from src.benchmark.poisson_benchmark import PoissonBenchmark, get_sharegpt_prompts
from src.utils.config import get_default_config, get_test_config, get_cpu_config
from src.engine.baseline_engine import BaselineEngine
from src.engine.pd_engine import PDEngine
from src.engine.speculative_engine import PDVLiteEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConcurrencyBenchmark")

class GPUMonitor:
    """Monitors GPU utilization."""
    def __init__(self, interval_sec: float = 0.5):
        self.interval_sec = interval_sec
        self.running = False
        self.thread = None
        self.utilization_history = []
        self.memory_history = []
        
    def start(self):
        self.running = True
        self.utilization_history = []
        self.memory_history = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        while self.running:
            try:
                # Query nvidia-smi
                # utilizing --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8"
                )
                lines = result.strip().split('\n')
                total_util = 0.0
                total_mem = 0.0
                count = 0
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        util = float(parts[0].strip())
                        mem = float(parts[1].strip())
                        total_util += util
                        total_mem += mem
                        count += 1
                
                if count > 0:
                    self.utilization_history.append(total_util / count)
                    self.memory_history.append(total_mem / count)
                else:
                    self.utilization_history.append(0.0)
                    self.memory_history.append(0.0)
                    
            except FileNotFoundError:
                # No nvidia-smi
                self.utilization_history.append(0.0)
                self.memory_history.append(0.0)
            except Exception as e:
                # logger.error(f"GPU Monitor error: {e}")
                self.utilization_history.append(0.0)
                self.memory_history.append(0.0)
                
            time.sleep(self.interval_sec)
            
    def get_avg_utilization(self) -> float:
        if not self.utilization_history:
            return 0.0
        return sum(self.utilization_history) / len(self.utilization_history)
        
    def get_max_memory(self) -> float:
        if not self.memory_history:
            return 0.0
        return max(self.memory_history)

def run_concurrency_benchmark(
    concurrency_levels: List[int],
    duration_seconds: int = 30,
    prompts: List[str] = None,
    config = None
):
    """
    Run benchmark across different concurrency levels for all engines.
    """
    if prompts is None:
        prompts = get_sharegpt_prompts(100)
    
    # Engines to test (all 3 for fair comparison)
    engines = {
        "Baseline": BaselineEngine,
        "PD": PDEngine,
        "PDV": PDVLiteEngine
    }
    
    summary_results = []
    detailed_results = [] # Per request stats
    
    # Config
    if config is None: # If config is not passed, determine it here
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            base_config = get_default_config()
        else:
            base_config = get_cpu_config()
            
        base_config.model.max_new_tokens = 50 # Keep it short for faster benchmarking
    else:
        base_config = config # Use the passed config
    
    for engine_name, engine_cls in engines.items():
        logger.info(f" Benchmarking Engine: {engine_name}")
        
        for concurrency in concurrency_levels:
            if concurrency == 0:
                continue 
                
            logger.info(f"  Running at Concurrency: {concurrency}")
            
            # Update config
            current_config = copy.deepcopy(base_config)
            current_config.scheduler.batch_size = concurrency
            current_config.scheduler.max_queue_size = concurrency * 100
            
            # Initialize Engine
            engine = engine_cls(current_config)
            gpu_monitor = GPUMonitor(0.1)
            
            try:
                print(f"DEBUG: Starting engine for C={concurrency}...")
                engine.start()
                if not engine.is_running:
                     logger.error("Engine failed to start!")
                     continue
                print("DEBUG: Engine started. Starting GPU monitor...")
                gpu_monitor.start()
                
                # Flood with requests
                # arrival_rate = concurrency * 4.0 # High load - Too high for 24s latency
                arrival_rate = concurrency * 0.05 # Reduced load to ensure completion
                print(f"DEBUG: Generating requests with rate {arrival_rate}...")
                
                benchmark = PoissonBenchmark(
                    prompts=prompts,
                    arrival_rate=arrival_rate,
                    duration_seconds=duration_seconds,
                    seed=42
                )
                
                requests = benchmark.generate_requests()
                print(f"DEBUG: Generated {len(requests)} requests. Running benchmark...")
                
                stats = benchmark.run_benchmark(
                    engine, 
                    requests=requests,
                    max_concurrent=concurrency 
                )
                print("DEBUG: Benchmark run finished.")
                
                if "error" in stats:
                     logger.error(f"   Benchmark Failed: {stats['error']}")
                     continue
                
                gpu_util = gpu_monitor.get_avg_utilization()
                gpu_mem = gpu_monitor.get_max_memory()
                
                # Summary
                summary_results.append({
                    "Engine": engine_name,
                    "Concurrency": concurrency,
                    "Throughput (TPS)": f"{stats['throughput_tps']:.2f}",
                    "Latency P50 (ms)": f"{stats['token_latency_ms'].get('median', 0):.2f}",
                    "Latency P99 (ms)": f"{stats['token_latency_ms'].get('p99', 0):.2f}",
                    "Acceptance Rate": f"{stats['acceptance_ratio'].get('mean', 0):.2%}",
                    "GPU Util %": f"{gpu_util:.1f}",
                    "GPU Mem (MB)": f"{gpu_mem:.0f}",
                    "Avg Request Latency (ms)": f"{stats['request_latency_ms'].get('mean', 0):.2f}"
                })
                
                # Detailed
                if "detailed_list" in stats:
                    for r in stats['detailed_list']:
                        # Calculate prefield/decode duration
                        # We need access to those fields. They are in the dict if we put them there.
                        detailed_results.append({
                            "Engine": engine_name,
                            "Concurrency": concurrency,
                            "RequestID": r.get('request_id'),
                            "TotalLatency": r.get('request_latency_ms'),
                            "PrefillDuration": r.get('prefill_ms', 0),
                            "DecodeDuration": r.get('decode_ms', 0),
                            "TokensGenerated": r.get('tokens_generated'),
                            "TokensAccepted": r.get('tokens_accepted')
                        })
                
            except Exception as e:
                logger.error(f"   Error: {e}")
            finally:
                gpu_monitor.stop()
                engine.stop()
                
    # Print Summary
    print("\n" + "="*100)
    print("CONCURRENCY BENCHMARK SUMMARY")
    print("="*100)
    print(tabulate(summary_results, headers="keys", tablefmt="pretty"))
    print("="*100)
    
    # Save CSVs
    pd.DataFrame(summary_results).to_csv("concurrency_benchmark_summary.csv", index=False)
    if detailed_results:
        pd.DataFrame(detailed_results).to_csv("concurrency_benchmark_detailed.csv", index=False)
        
    print("Results saved to concurrency_benchmark_summary.csv and concurrency_benchmark_detailed.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10, help="Duration per test in seconds")
    parser.add_argument("--concurrency", type=str, default="2,8,16,32,64", help="Comma separated concurrency levels")
    args = parser.parse_args()
    
    levels = [int(x) for x in args.concurrency.split(",") if int(x) > 0]
    
    print(f"Running benchmark for levels: {levels}")
    
    from src.utils.config import get_performance_config # Use performance config for realistic benchmarking
    config = get_performance_config()
    
    # Allow overriding max_new_tokens for faster benchmark if needed
    config.model.max_new_tokens = 20 # Reduced to 20 to allow C=64 to finish within timeout
    
    run_concurrency_benchmark(levels, duration_seconds=args.duration, config=config)
