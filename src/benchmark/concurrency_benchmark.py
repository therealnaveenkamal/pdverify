
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
from src.engine.speculative_engine import SpeculativeEngine

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
    prompts: List[str] = None
):
    """
    Run benchmark across different concurrency levels for all engines.
    """
    if prompts is None:
        prompts = get_sharegpt_prompts(100)
    
    # Engines to test
    engines = {
        "Baseline": BaselineEngine,
        "PD": PDEngine,
        "PDV": SpeculativeEngine
    }
    
    summary_results = []
    detailed_results = [] # Per request stats
    
    # Config
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        base_config = get_default_config()
    else:
        base_config = get_cpu_config()
        
    base_config.model.max_new_tokens = 50 # Keep it short for faster benchmarking
    
    for engine_name, engine_cls in engines.items():
        logger.info(f" Benchmarking Engine: {engine_name}")
        
        for concurrency in concurrency_levels:
            if concurrency == 0:
                continue 
                
            logger.info(f"  Running at Concurrency: {concurrency}")
            
            # Update config
            current_config = copy.deepcopy(base_config)
            current_config.scheduler.batch_size = concurrency
            current_config.scheduler.max_queue_size = concurrency * 10
            
            # Initialize Engine
            engine = engine_cls(current_config)
            gpu_monitor = GPUMonitor(0.1)
            
            try:
                engine.start()
                gpu_monitor.start()
                
                # Flood with requests
                arrival_rate = float(concurrency) * 4.0 
                
                benchmark = PoissonBenchmark(
                    prompts=prompts,
                    arrival_rate=arrival_rate,
                    duration_seconds=duration_seconds,
                    seed=42
                )
                
                requests = benchmark.generate_requests()
                
                # Inject a custom collector into run_benchmark to capture detailed stats?
                # PoissonBenchmark.run_benchmark returns aggregate list. 
                # BUT the `results` list inside run_benchmark stores limited fields.
                # We need to hack it or update PoissonBenchmark to return raw request objects?
                # Actually, run_benchmark returns `token_latency_ms`.
                # We need timestamps.
                # Let's Modify `PoissonBenchmark` simply by monkey-patching or just subclassing?
                # Easier: Modify run_benchmark output in `poisson_benchmark.py`? No, let's keep it clean.
                
                # We will access the raw stats from the return if we modify `PoissonBenchmark`.
                # Wait, I cannot modify `poisson_benchmark.py` easily inside this script.
                # I will subclass `PoissonBenchmark` here.
                
                stats = benchmark.run_benchmark(
                    engine, 
                    requests=requests,
                    max_concurrent=concurrency 
                )
                
                # Wait. run_benchmark returns a dict of aggregates only.
                # I need granular data.
                # I will assume `run_benchmark` logic is mostly insufficient for per-request *breakdown* 
                # unless I parse the raw results. But `run_benchmark` encapsulates collecting them.
                # I will ignore `stats` (summary) for the detailed file and try to extract more?
                # No, I can't.
                
                # Ok, I will accept that I can't get the per-request breakdown WITHOUT modifying `poisson_benchmark.py`
                # to return the raw list.
                # But wait! I edited `Request` object to store timestamps.
                # Those requests are passed IN to `run_benchmark`.
                # Does `run_benchmark` modify them in place? Yes.
                # `requests` list still holds the Request objects?
                # `requests` holds `BenchmarkRequest`.
                # Inside `run_benchmark`, `active_requests` holds the mapping.
                # The engine modifies the `Request` (engine object), not `BenchmarkRequest`.
                # The `Request` object is created INSIDE `run_benchmark`.
                
                # SOLUTION: I must modify `poisson_benchmark.py` to return the detailed result list, NOT just the summary dict.
                # I will do that in the NEXT step.
                # For now, I will create this script assuming `stats` contains a "raw_results" key.
                
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
    
    run_concurrency_benchmark(levels, duration_seconds=args.duration)
